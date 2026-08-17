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

    vggt360       this repository's VGGT-360-fisheye port: split the frame into a
                  centre view plus an 8-direction ring of 60 deg tangent views,
                  reconstruct all nine in one VGGT pass, fuse back onto the
                  fisheye grid, gather at the points. A third lens strategy, in
                  the same slot as the other two, which is the whole reason this
                  file is written the way it is.

The third arm is not a third model
----------------------------------
``vggt360`` brings its own backbone — the vendored VGGT-1B, because its
attention-weighted fusion reads frame attention off a 37x37 patch grid that no
model-zoo adapter exposes. So it is a lens strategy *for that one model*, and
``run.py`` refuses to pair it with any other: a row labelled ``dav2_large ×
vggt360`` would contain no Depth-Anything at all, which is a full plausible table
saying something that never happened.

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
centre and axis, differing only in how rays meet the image plane. A depth
measured about that shared axis is therefore *unchanged* by rectifying, and no
conversion is applied on the way back. That invariance is measured on the release
itself — the producer's own rectified and fisheye point sets carry identical
depth values for the same points (99.7-99.8 % of rectified depths appear verbatim
among the fisheye depths, with identical ranges) — and
``tests/test_baselines.py`` pins it against an analytic scene, so a future
non-co-axial rectifier cannot slip through silently.

**That measurement establishes the invariance and not the convention**, and the
distinction cost this repo a result once already (CONTEXT.md: the ERP-era "depth
is range" assumption is what made the fisheye port score range against z-GT).
Euclidean range is a property of the *ray*, so it too is unchanged by a co-axial
rectification: the two streams agreeing is exactly as consistent with ``d`` being
range as with ``d`` being planar z. Whether ``pts.d`` is one or the other is a
separate question, it is worth up to 1.74x across this lens, and nothing in this
package can currently answer it — see ``data.FramePoints.d`` and ticket 016.

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
VGGT360 = "vggt360"

#: Every arm this module can build.
BASELINES = (RAW, RECT_DERECT, VGGT360)

#: What ``--baselines`` defaults to. Deliberately **not** :data:`BASELINES`:
#: every published SLAM number was produced by the two-arm default, and silently
#: adding a third would change what the bare command measures and what a
#: re-run of it reproduces. ``vggt360`` is asked for by name.
DEFAULT_BASELINES = (RAW, RECT_DERECT)


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


def _window(frames) -> list:
    """A window as a list, whether one frame or several were handed over.

    One RGB frame is itself a 3-d array, so ``list(frame)`` would silently
    iterate its rows and hand the model 896 one-pixel-tall images.
    """
    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        return [frames]
    return list(frames)


class Baseline:
    """Common interface. ``name`` is what the report prints.

    ``predict`` takes a **window** of frames and the index within it of the one
    being scored, so that a multi-view model can be handed a temporal context.
    A window of one is the ordinary case and is not a special path.
    """

    name = "?"

    def predict(self, frames, pts: "D.FramePoints",
                target: int = -1) -> np.ndarray:
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

    def predict(self, frames, pts: "D.FramePoints",
                target: int = -1) -> np.ndarray:
        stack = [D.resize_frame(f, self.input_size) for f in _window(frames)]
        raw = np.asarray(self.model.predict_stack(stack, target, gt=pts.d))
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

        ``projectaria_tools.distort_by_calibration`` does the resampling when it
        is importable. That is the call the Aria ecosystem rectifies with — the
        Ego-Exo4D undistortion tutorial and the AEA release both use it — so a
        model fed by this baseline sees what it would see anywhere else. The
        ``cv2.remap`` path below is the fallback, and ``test_baselines.py`` pins
        the two together.

        Zero-filled outside the imaged cone rather than border-replicated: a
        replicated pixel is a fabricated observation, and the model would be
        scored on its response to one.
        """
        _, _, in_cone = self._remap()
        out = _library_rectify(frame, self.cam, self.pin)
        if out is None:
            import cv2
            mapx, mapy, _ = self._remap()
            out = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out, in_cone

    # -- the round trip ----------------------------------------------------- #
    def predict(self, frames, pts: "D.FramePoints",
                target: int = -1) -> np.ndarray:
        # Every frame of the window is rectified, not just the scored one: a
        # model handed one pinhole render among nine fisheye frames would be
        # asked to match features across two different lenses, and whatever it
        # then did would be about that mismatch.
        stack = [self.rectify(f)[0] for f in _window(frames)]
        raw = np.asarray(self.model.predict_stack(stack, target, gt=None))
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

        The addresses are computed on ``self.pin``'s grid, so a map that arrived
        on a different one would be sampled at the wrong place *silently* — the
        failure mode this whole class is written against, since it reads as a
        merely mediocre model rather than as an error. Both adapters in the zoo
        resize their output back to the frame they were handed, so the contract
        holds today; it is asserted because it is a cross-module one.
        """
        n_pin = self.pin.size
        if pred_rect.shape != (n_pin, n_pin):
            raise SystemExit(
                f"[slambench] {RECT_DERECT!r} rectified to {n_pin}x{n_pin} but "
                f"{getattr(self.model, 'key', 'the model')!r} answered with a "
                f"{pred_rect.shape[0]}x{pred_rect.shape[1]} map. Every point "
                f"would be sampled at the wrong pixel and the run would look "
                f"like a bad model rather than a broken one. The adapter must "
                f"return depth on the grid it was given.")
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


class VGGT360Baseline(Baseline):
    """VGGT-360-fisheye: nine tangent views, one pass, fused back to the lens.

    The pipeline itself is ``VGGT-360-fisheye/utils/pipeline.py``, shared with
    ``main_adt.py`` and with the ADT-FOV test, so this class holds none of the
    method — only the three things that are about *this* evaluation:

    **The lens.** ego-synth is FISHEYE624, not the KB4 the port was written on,
    and the take's own camera answers both the forward warp and the inverse ray
    field rather than a KB4 fitted to it (``slambench/vggt360.py``).

    **The depth convention.** Fusion produces euclidean range along each ray and
    ``pts.d`` is planar z, so this arm converts by ``cos(theta)`` — 1.00 on axis,
    1.74x at 55 deg, radial and therefore not absorbable by the alignment affine.
    It is the only arm that converts, because it is the only one that leaves the
    camera axis: ``raw`` and ``rect_derect`` both answer about the same axis the
    ground truth is measured about, so neither needs to.

    That ``pts.d`` is planar z is **measured**, not assumed — ticket 016, closed
    2026-08-14, ``slambench/verify_depth_convention.py``: the residual against z
    is 0.0002 and flat across incidence angle (the float16 noise floor of the
    stored value), while the range hypothesis is wrong by exactly ``1 - cos
    theta`` in all eight bins on both aea and nymeria. Being flat is the load
    bearing part: a convention error is radial by construction, so it cannot hide
    in a flat residual.

    **The rim.** The pipeline answers for the imaged cone and NaN outside it.
    That is left as NaN on purpose: ``run.py`` intersects support across arms and
    reports what each gave up, so an honest hole is information here. (The
    ADT-FOV harness owns its own mask and cannot take a NaN, which is why
    ``fovbench`` fills and this does not — same pipeline, two callers, and the
    difference is in the callers.)
    """

    name = VGGT360

    def __init__(self, model, cam: Fisheye624, pipe, cfg,
                 theta_max_deg: Optional[float] = None):
        from slambench.vggt360 import Fisheye624Lens
        self.model = model
        self.pipe = pipe
        self.cfg = cfg
        self.lens = Fisheye624Lens(cam, theta_max_deg=theta_max_deg)

    @property
    def needs_camera(self) -> bool:
        return True

    def predict(self, frames, pts: "D.FramePoints",
                target: int = -1) -> np.ndarray:
        from utils.pipeline import range_to_planar_z

        window = _window(frames)
        if len(window) != 1:
            # One frame in, nine views out. A temporal window would be nine
            # views per frame in a single pass — a different experiment, and
            # not one to perform by accident because a sweep flag was set.
            raise SystemExit(
                f"[slambench] {VGGT360!r} takes one frame at a time: it already "
                f"hands VGGT a {self.cfg.n_ring + 1}-view reconstruction of that "
                f"frame, and a {len(window)}-frame context would silently make "
                f"that {(self.cfg.n_ring + 1) * len(window)} views in one pass. "
                f"Run this arm with --context-frames 1.")

        ff = self.pipe.range_map(window[0], self.lens,
                                 project=self.lens.project,
                                 ray_lut=self.lens.ray_lut(),
                                 maps_for=self.lens.maps_for)
        z = range_to_planar_z(ff.range_map, self.lens.cos_theta())
        # Outside the cone, or covered by no view: no answer, not a guess.
        z = np.where(ff.covered, z, np.nan).astype(np.float32)
        return D.sample_prediction(z, pts)


def _library_rectify(frame: np.ndarray, cam: Fisheye624,
                     pin: Pinhole) -> Optional[np.ndarray]:
    """Rectify with ``projectaria_tools``, or ``None`` if it is not importable.

    The library has no rotated fisheye calibration to offer — its
    ``rotate_camera_calib_cw90deg`` is Linear-model-only — so the frame is turned
    back to the sensor's own orientation, rectified there, and turned upright
    again. The pinhole is square and centred, so it is unchanged by that turn
    and ``pin`` describes the result either way.
    """
    try:
        from projectaria_tools.core import calibration as pcal
        from projectaria_tools.core.sophus import SE3
    except ImportError:
        return None
    if not hasattr(pcal, "distort_by_calibration"):
        return None
    args = ("camera-rgb", pcal.CameraModelType.FISHEYE624,
            np.asarray(cam.params, float), SE3(), cam.width, cam.height,
            None, np.pi)
    try:
        src = pcal.CameraCalibration(*args, "")
    except TypeError:                       # older binding, no serial argument
        src = pcal.CameraCalibration(*args)
    dst = pcal.get_linear_camera_calibration(pin.size, pin.size, pin.focal,
                                             "pinhole")
    # np.rot90's exponent is the opposite sign to camera._rot_pixel's quarter
    # turns: rot90(m, k)[i, j] == m[j, N - i], which is _rot_pixel's inverse.
    turned = np.ascontiguousarray(np.rot90(frame, cam.rotation))
    out = pcal.distort_by_calibration(turned, dst, src)
    return np.ascontiguousarray(np.rot90(np.asarray(out), -cam.rotation))


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
          rect_size: Optional[int] = None,
          vggt360_pipe=None, vggt360_cfg=None) -> Baseline:
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
    if name == VGGT360:
        if cam is None:
            raise SystemExit(
                f"[slambench] {VGGT360!r} needs this take's camera model: it "
                f"renders tangent views through the lens and fuses back through "
                f"its inverse. Fetch it with "
                f"tools/fetch_egosynth_calibration.py (ticket 012) and pass "
                f"--calib-root.")
        if vggt360_pipe is None:
            raise SystemExit(
                f"[slambench] {VGGT360!r} was built without a loaded pipeline. "
                f"It is created once per run, not once per take — nine views of "
                f"VGGT-1B is not a per-clip construction cost.")
        return VGGT360Baseline(model, cam, vggt360_pipe, vggt360_cfg)
    raise SystemExit(f"[slambench] unknown baseline {name!r}; "
                     f"choose from {list(BASELINES)}")
