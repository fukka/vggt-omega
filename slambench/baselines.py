# Copyright (c) 2026.
"""The two lens strategies — the only thing that differs between the arms.

A baseline is asked one question and answers it one way:

    predict(frame, points) -> one depth per ground-truth point

The harness hands over the **raw fisheye frame** and the **native ground-truth
point list** and takes back an array. What happened in between is the baseline's
business, and that is the whole design: the evaluation does not privilege one
lens treatment by building it into the measurement.

    raw           the frame goes to the model as it is. No camera model needed,
                  no geometry anywhere, nothing resampled but the resize to the
                  network's own token grid.

    rect_derect   the baseline rectifies to a co-axial pinhole, runs the model on
                  that, and maps the predicted depth **back** onto the fisheye
                  points. Every step is inside here; the harness never learns it
                  happened.

Why de-rectification is a gather, not an image
-----------------------------------------------
The obvious implementation renders a full de-rectified depth map and then samples
it — two resamplings, the second of which invents values between pixels. There is
no need: the score is per point, so each ground-truth point is taken *through*
the rectification individually —

    fisheye pixel --unproject--> ray --project--> pinhole pixel --sample--> depth

which resamples once, at the only place a value is actually needed.

Depth conventions
-----------------
The rectification is **co-axial**: the pinhole shares the fisheye's optical
centre and axis, differing only in how rays meet the image plane. Planar z is
measured about that shared axis, so it is *unchanged* by rectifying and no
conversion is applied. This is not an assumption inherited from a document — it
is measured on the release itself, where the producer's own rectified and fisheye
point sets carry identical depth values for the same points (99.7-99.8 % of
rectified depths appear verbatim among the fisheye depths, with identical
ranges). ``tests/test_baselines.py`` pins the invariant against an analytic
scene, so a future non-co-axial rectifier cannot slip through silently.

Support
-------
A 110 deg pinhole cannot cover a 110+ deg fisheye cone: points near the rim
unproject to rays outside the rectified frame and get **NaN**, not a guess. Those
points are then excluded from *both* arms by ``metrics.common_support``, because
scoring two baselines over different point sets compares the point sets as much
as the baselines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from slambench import _REPO  # noqa: F401  (import registers sys.path)

from slambench import data as D  # noqa: E402
from slambench.camera import Fisheye624  # noqa: E402

#: Field of view of the pinhole ``rect_derect`` rectifies to. 110 deg is what
#: ego-synth's own producer chose for its rectified stream, so a run at this
#: value is comparable with that render; it is a knob, not a constant of nature.
DEFAULT_RECT_FOV_DEG = 110.0

RAW = "raw"
RECT_DERECT = "rect_derect"
BASELINES = (RAW, RECT_DERECT)


@dataclass
class Pinhole:
    """The co-axial perspective camera ``rect_derect`` rectifies to."""

    size: int
    fov_deg: float

    @property
    def focal(self) -> float:
        return (self.size / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)

    @property
    def c(self) -> float:
        return (self.size - 1) / 2.0

    def rays(self) -> np.ndarray:
        """Unit ray per pixel, ``(size, size, 3)``, about the shared axis."""
        j = np.arange(self.size, dtype=np.float64)
        x = (j - self.c) / self.focal
        xv, yv = np.meshgrid(x, x)
        d = np.stack([xv, yv, np.ones_like(xv)], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def project(self, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unit rays ``(N, 3)`` -> pinhole pixels. Rays at or behind the plane
        of the lens have no image and come back NaN."""
        z = dirs[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.focal * (dirs[:, 0] / z) + self.c
            v = self.focal * (dirs[:, 1] / z) + self.c
        bad = ~np.isfinite(u) | ~np.isfinite(v) | (z <= 1e-6)
        return np.where(bad, np.nan, u), np.where(bad, np.nan, v)


class Baseline:
    """Common interface. ``name`` is what the report prints."""

    name = "?"

    def predict(self, frame: np.ndarray, pts: "D.FramePoints") -> np.ndarray:
        raise NotImplementedError

    @property
    def needs_camera(self) -> bool:
        return False


class RawBaseline(Baseline):
    """The fisheye frame, straight to the model.

    The only processing is the resize to the network's own token grid — the same
    rule the rest of this repo uses, so that nothing is resampled between the
    frame and the network — after which the prediction is gathered at the point
    list. No camera model is involved, which is why this arm runs today and
    ``rect_derect`` waits on calibration.
    """

    name = RAW

    def __init__(self, model, input_size: Optional[int] = None):
        self.model = model
        self.input_size = int(input_size or model.input_size)

    def predict(self, frame: np.ndarray, pts: "D.FramePoints") -> np.ndarray:
        raw = self.model.predict(D.resize_frame(frame, self.input_size),
                                 gt=pts.d)
        raw = np.asarray(raw)
        if raw.ndim == 1:                       # the oracle stand-in
            return raw.astype(np.float32)
        return D.sample_prediction(raw, pts)


class RectDerectBaseline(Baseline):
    """Rectify, run the model, map the depth back onto the fisheye points.

    All three steps are here rather than in the harness, so that "was rectifying
    worth it" is a property of this baseline and not of the benchmark.

    The camera model is per take and comes from the calibration fetched by
    ``tools/fetch_egosynth_calibration.py``; there is no fallback to a nominal
    Aria calibration, because a nominal one is wrong by more than the effect
    being measured.
    """

    name = RECT_DERECT

    def __init__(self, model, cam: Fisheye624,
                 fov_deg: float = DEFAULT_RECT_FOV_DEG,
                 rect_size: Optional[int] = None):
        self.model = model
        self.cam = cam
        self.pin = Pinhole(int(rect_size or model.input_size), float(fov_deg))
        self._map = None                        # built lazily, reused per take

    @property
    def needs_camera(self) -> bool:
        return True

    # -- rectification ------------------------------------------------------ #
    def _remap(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(mapx, mapy, in_cone)`` taking the pinhole grid to fisheye pixels.

        Depends only on the camera and the pinhole, so it is built once per take
        rather than per frame — it is the most expensive thing here.
        """
        if self._map is None:
            d = self.pin.rays().reshape(-1, 3)
            u, v = self.cam.project(d)
            ok = (np.isfinite(u) & np.isfinite(v)
                  & (u >= 0) & (u <= self.cam.width - 1)
                  & (v >= 0) & (v <= self.cam.height - 1))
            n = self.pin.size
            self._map = (np.where(ok, u, -1).reshape(n, n).astype(np.float32),
                         np.where(ok, v, -1).reshape(n, n).astype(np.float32),
                         ok.reshape(n, n))
        return self._map

    def rectify(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The fisheye frame as a co-axial pinhole render, plus its valid mask.

        Zero-filled outside the imaged cone rather than border-replicated: a
        replicated pixel is a fabricated observation, and the model would be
        scored on its response to one.
        """
        import cv2
        mapx, mapy, in_cone = self._remap()
        out = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out, in_cone

    # -- the round trip ----------------------------------------------------- #
    def predict(self, frame: np.ndarray, pts: "D.FramePoints") -> np.ndarray:
        rect, _ = self.rectify(frame)
        raw = np.asarray(self.model.predict(rect, gt=None))
        if raw.ndim != 2:
            raise SystemExit(
                f"[slambench] the {self.model.key!r} stand-in answers per point, "
                f"so it cannot be run through {RECT_DERECT!r}, which has to "
                f"sample a rectified map. Use it with {RAW!r}.")
        return self.derectify(raw, pts)

    def derectify(self, pred_rect: np.ndarray,
                  pts: "D.FramePoints") -> np.ndarray:
        """Sample a pinhole depth map at the *fisheye* points.

        One resampling, at the points that need a value:

            fisheye pixel -> ray -> pinhole pixel -> bilinear sample

        Planar z needs no conversion because the pinhole is co-axial with the
        fisheye — see the module docstring, where that is measured rather than
        assumed. Points whose ray leaves the pinhole's field come back NaN.
        """
        d = self.cam.unproject(pts.u.astype(np.float64),
                               pts.v.astype(np.float64))
        u, v = self.pin.project(d)
        n = self.pin.size
        inside = (np.isfinite(u) & np.isfinite(v)
                  & (u >= 0) & (u <= n - 1) & (v >= 0) & (v <= n - 1))
        out = np.full(u.shape, np.nan, np.float32)
        if not inside.any():
            return out
        out[inside] = _bilinear(pred_rect, u[inside], v[inside])
        return out


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    H, W = img.shape
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    p = img.astype(np.float32)
    top = p[y0, x0] * (1 - fx) + p[y0, x1] * fx
    bot = p[y1, x0] * (1 - fx) + p[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def build(name: str, model, cam: Optional[Fisheye624] = None,
          fov_deg: float = DEFAULT_RECT_FOV_DEG,
          rect_size: Optional[int] = None) -> Baseline:
    if name == RAW:
        return RawBaseline(model)
    if name == RECT_DERECT:
        if cam is None:
            raise SystemExit(
                f"[slambench] {RECT_DERECT!r} needs this take's camera model and "
                f"none was loaded. Fetch it with "
                f"tools/fetch_egosynth_calibration.py (ticket 012) and pass "
                f"--calib-root, or run --baselines raw, which needs no camera.")
        return RectDerectBaseline(model, cam, fov_deg, rect_size)
    raise SystemExit(f"[slambench] unknown baseline {name!r}; "
                     f"choose from {list(BASELINES)}")
