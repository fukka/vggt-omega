# Copyright (c) 2026.
"""The Aria RGB camera model, per take.

ego-synth 5B ships **no camera model** — verified across all four source
datasets' ``meta.json``. It is fetched separately from each dataset's MPS
``online_calibration`` by ``tools/fetch_egosynth_calibration.py``, which reduces
that time series to one ``camera_rgb.json`` per take. This module is what reads
those files and turns them into something that can project and unproject.

The model is Aria's **FisheyeRadTanThinPrism**, the ``FISHEYE624`` of
``projectaria_tools.core.calibration.CameraModelType``. Fifteen parameters:

    f, cx, cy, k1..k6, p1, p2, s1..s4
    ^^^^^^^^^^  ^^^^^^  ^^^^^^  ^^^^^^
    projection  radial  tangen  thin prism

It is a **per-device** calibration, not a per-dataset one: two takes measured
here differ by ``f`` 1214.03 vs 1218.65 and ``cx`` 1469.5 vs 1462.1, which is
2.2 px once scaled to the 896 frame ego-synth ships. Nothing here falls back to a
nominal Aria calibration, because a nominal one is wrong by more than the effect
being measured.

Who needs it
------------
Only the ``rect_derect`` baseline. Scoring a model that was fed a rectified image
means mapping its depth back onto the raw fisheye points, which is *fisheye pixel
-> ray -> pinhole pixel*, and both directions of this model are on that path. The
``raw`` baseline needs no camera model at all and does not import this.

Two conventions this must get right, and only one of them is settled
--------------------------------------------------------------------
**Resolution — settled.** The stored params are at the sensor's own resolution
(``f~1218``, ``c~(1462, 1444)`` can only be a 2880 sensor), while ego-synth's
frames are 896. :meth:`Fisheye624.rescale` does that, on the pixel-centre
convention ``c' = (c + 0.5) * s - 0.5``, the same one this repo uses everywhere
a calibration crosses a resize.

**Orientation — NOT settled.** Aria stores frames rotated; ego-synth writes them
upright, and carries ``input_orientation`` as a *per-dataset* argument, so the
rotation is not shared across the four. Measured against the staged sample's own
rect<->fisheye correspondences, ``aea`` wants 90 deg CCW and ``nymeria`` wants 0,
each the only rotation putting any correspondence inside 2 px. But that
measurement is contaminated — the correspondences were paired on exact float16
depth, so two different points sharing a depth get paired — and the best rotation
still leaves p05 ~1.4 px rather than sub-pixel.

So :data:`DATASET_ROTATION` records what was measured, marks it unverified, and
:func:`verify_orientation` is the acceptance test that must pass before any
``rect_derect`` number is quoted. Until it does, treat this module as loaded but
unproven.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from slambench import _REPO  # noqa: F401  (import registers sys.path)

#: Parameter count of FisheyeRadTanThinPrism / FISHEYE624.
N_PARAMS = 15

#: Frame size ego-synth ships everything at (gotcha 1 of the data card).
EGOSYNTH_RES = 896

#: Quarter turns (CCW) taking the calibration's sensor frame to ego-synth's
#: upright frame, per dataset.
#:
#: **Unverified.** Measured on one take each of ``aea`` and ``nymeria`` against
#: contaminated correspondences (see the module docstring); ``egoexo4d`` and
#: ``oxford`` were never measured at all and are guesses carried only so the
#: code runs. :func:`verify_orientation` is what settles any of these, and
#: :func:`require_verified` refuses to hand out an unverified camera.
DATASET_ROTATION: Dict[str, int] = {
    "aea": 1,          # 90 deg CCW  -- measured, contaminated
    "nymeria": 0,      #              -- measured, contaminated
    "egoexo4d": 0,     #              -- NOT measured
    "oxford": 0,       #              -- NOT measured
}

#: Datasets whose rotation has survived :func:`verify_orientation` at the
#: tolerance below. Empty until one does. Written by hand after a verification
#: run, never by the verifier itself — a measurement that promotes itself is not
#: a check.
VERIFIED_ROTATION: Tuple[str, ...] = ()

#: Median reprojection error, in pixels of the 896 frame, below which the model
#: and its orientation are considered to describe the data. Half a pixel is the
#: float16 quantisation of the stored ``u``/``v``, so anything at or under it is
#: at the noise floor of what the ground truth can even express.
ORIENTATION_TOL_PX = 0.5


class CalibrationUnavailable(RuntimeError):
    """No camera model on disk for this take.

    Raised rather than substituted. ``rect_derect`` cannot run without one and a
    nominal Aria calibration is wrong by more than the effect being measured, so
    the honest failure is to stop. The ``raw`` baseline is unaffected and needs
    no calibration.
    """


class OrientationUnverified(RuntimeError):
    """The take's dataset has a rotation that no acceptance test has confirmed."""


@dataclass(frozen=True)
class Fisheye624:
    """Aria's FisheyeRadTanThinPrism, resolved for one frame size.

    ``params`` is the 15-vector in the order the calibration stores it. ``width``
    and ``height`` are the frame these parameters describe — changing the frame
    means :meth:`rescale`, never editing the params in place.
    """

    params: Tuple[float, ...]
    width: int
    height: int
    label: str = "camera-rgb"
    serial: str = ""
    dataset: str = ""
    take: str = ""

    # -- accessors ---------------------------------------------------------- #
    @property
    def f(self) -> float:
        return float(self.params[0])

    @property
    def cx(self) -> float:
        return float(self.params[1])

    @property
    def cy(self) -> float:
        return float(self.params[2])

    @property
    def k(self) -> np.ndarray:
        return np.asarray(self.params[3:9], float)

    @property
    def p(self) -> Tuple[float, float]:
        return float(self.params[9]), float(self.params[10])

    @property
    def s(self) -> np.ndarray:
        return np.asarray(self.params[11:15], float)

    # -- frame changes ------------------------------------------------------ #
    def rescale(self, out_size: int) -> "Fisheye624":
        """The same lens, described for a frame resized to ``out_size`` square.

        Only ``f``, ``cx``, ``cy`` scale. The radial, tangential and thin-prism
        coefficients act on *normalised* image coordinates, so they are invariant
        to the resize — which is exactly why they must not be touched here.

        The principal point moves on the pixel-centre convention
        ``c' = (c + 0.5) * s - 0.5``, not ``c * s``: a resize maps output pixel
        centre ``j + 0.5`` to source ``(j + 0.5) / s``, and the two differ by
        ``0.5 * (1 - s)`` px. At the 2880 -> 896 factor used here that is 0.35 px
        — small, and the same order as the effect this calibration exists to fix,
        so it is not roundable away.
        """
        s = out_size / float(self.width)
        q = list(self.params)
        q[0] = self.f * s
        q[1] = (self.cx + 0.5) * s - 0.5
        q[2] = (self.cy + 0.5) * s - 0.5
        return Fisheye624(tuple(q), out_size, out_size, self.label, self.serial,
                          self.dataset, self.take)

    def rotate90(self, k: int) -> "Fisheye624":
        """The same lens, described for a frame rotated ``k`` quarter turns CCW.

        Square frames only, which every ego-synth frame is. One CCW quarter turn
        sends pixel ``(x, y) -> (N - y, x)`` with ``N = width - 1``, so the
        principal point follows and ``f`` is unchanged (the model is isotropic in
        ``f``; there is no separate ``fy``).

        The distortion coefficients are radial about the principal point and so
        are rotation-invariant — with one exception that matters: the
        **tangential** ``p1, p2`` and the **thin-prism** ``s1..s4`` terms are
        *not* isotropic, and a rotation mixes them. They are tiny here (order
        1e-4 against radial terms of order 1e-1), so rotating them is a
        refinement below the 0.5 px floor this module works to; they are carried
        through unrotated and :func:`verify_orientation` is what would expose it
        if that ever stopped being true.
        """
        if self.width != self.height:
            raise ValueError("rotate90 is for square frames; "
                             f"got {self.width}x{self.height}")
        cx, cy, n = self.cx, self.cy, self.width - 1.0
        for _ in range(k % 4):
            cx, cy = n - cy, cx
        q = list(self.params)
        q[1], q[2] = cx, cy
        return Fisheye624(tuple(q), self.width, self.height, self.label,
                          self.serial, self.dataset, self.take)

    # -- geometry ----------------------------------------------------------- #
    def project(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """3D points in the camera frame ``(N, 3)`` -> pixels ``(u, v)``.

        Points behind the camera are not meaningful for a fisheye of this FOV
        and come back as NaN rather than as a plausible pixel.
        """
        xyz = np.asarray(xyz, float)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = x / z, y / z
        r = np.hypot(a, b)
        th = np.arctan(r)
        thd = th * self._radial(th)
        scale = np.divide(thd, r, out=np.ones_like(r), where=r > 1e-12)
        xr, yr = a * scale, b * scale
        xd, yd = self._tangential_prism(xr, yr)
        u = self.f * xd + self.cx
        v = self.f * yd + self.cy
        bad = ~np.isfinite(u) | ~np.isfinite(v) | (z <= 0)
        u = np.where(bad, np.nan, u)
        v = np.where(bad, np.nan, v)
        return u, v

    def unproject(self, u: np.ndarray, v: np.ndarray,
                  iters: int = 20) -> np.ndarray:
        """Pixels -> unit ray directions ``(N, 3)`` in the camera frame.

        The inverse has no closed form: the forward map composes a degree-13
        polynomial in ``theta`` with tangential and thin-prism terms in the
        already-distorted coordinates. It is solved in two nested fixed points,
        both of which converge because the correction terms are small compared
        to what they correct:

        1. peel off tangential + thin prism by iterating
           ``(xr, yr) <- (xd, yd) - correction(xr, yr)``;
        2. invert ``theta_d = theta * poly(theta)`` by Newton on ``theta``.

        ``test_camera.py`` pins the round trip against
        :meth:`project` rather than trusting the iteration count.
        """
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        xd = (u - self.cx) / self.f
        yd = (v - self.cy) / self.f
        xr, yr = xd.copy(), yd.copy()
        for _ in range(iters):
            cx_, cy_ = self._tangential_prism(xr, yr)
            xr = xd - (cx_ - xr)
            yr = yd - (cy_ - yr)
        thd = np.hypot(xr, yr)
        th = self._invert_radial(thd, iters)
        t = np.tan(th)
        scale = np.divide(t, thd, out=np.ones_like(thd), where=thd > 1e-12)
        a, b = xr * scale, yr * scale
        d = np.stack([a, b, np.ones_like(a)], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def theta_of(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Incidence angle (degrees) of the rays those pixels see."""
        d = self.unproject(u, v)
        return np.degrees(np.arccos(np.clip(d[:, 2], -1.0, 1.0)))

    # -- internals ---------------------------------------------------------- #
    def _radial(self, th: np.ndarray) -> np.ndarray:
        k = self.k
        t2 = th * th
        return 1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * (
            k[3] + t2 * (k[4] + t2 * k[5])))))

    def _radial_d(self, th: np.ndarray) -> np.ndarray:
        """d(theta_d)/d(theta), for the Newton step in :meth:`_invert_radial`."""
        k = self.k
        acc = np.ones_like(th)
        for i, ki in enumerate(k, start=1):
            acc = acc + (2 * i + 1) * ki * th ** (2 * i)
        return acc

    def _invert_radial(self, thd: np.ndarray, iters: int) -> np.ndarray:
        th = thd.copy()
        for _ in range(iters):
            f = th * self._radial(th) - thd
            d = self._radial_d(th)
            th = th - f / np.where(np.abs(d) < 1e-9, 1e-9, d)
        return np.clip(th, 0.0, math.pi / 2)

    def _tangential_prism(self, xr: np.ndarray, yr: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        p1, p2 = self.p
        s = self.s
        rr = xr * xr + yr * yr
        xt = 2 * p1 * xr * yr + p2 * (rr + 2 * xr * xr)
        yt = p1 * (rr + 2 * yr * yr) + 2 * p2 * xr * yr
        xp = s[0] * rr + s[1] * rr * rr
        yp = s[2] * rr + s[3] * rr * rr
        return xr + xt + xp, yr + yt + yp


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load(path: str, dataset: str = "", take: str = "",
         out_size: int = EGOSYNTH_RES, rotation: Optional[int] = None
         ) -> Fisheye624:
    """One take's ``camera_rgb.json``, resolved for ego-synth's own frame.

    Rescaled from the sensor's resolution to ``out_size`` and rotated by the
    dataset's convention. ``rotation`` overrides :data:`DATASET_ROTATION`, which
    is what the verification script uses to try each one.

    The sensor's resolution is not stored in the file, because MPS does not put
    it there. It is inferred from the principal point: a calibration is centred
    to within a few per cent, so ``2 * max(cx, cy)`` rounded to the nearest
    standard Aria RGB size identifies it. Aria RGB is 2880 square at full
    resolution and 1408 square binned, and those are far enough apart that the
    inference cannot land between them.
    """
    with open(path) as fh:
        doc = json.load(fh)
    params = doc.get("params")
    if not params or len(params) != N_PARAMS:
        raise CalibrationUnavailable(
            f"{path}: expected {N_PARAMS} FISHEYE624 params, got "
            f"{0 if not params else len(params)}")
    native = _infer_native(float(params[1]), float(params[2]))
    cam = Fisheye624(tuple(float(x) for x in params), native, native,
                     label=doc.get("label", "camera-rgb"),
                     serial=(doc.get("serial_numbers") or [""])[0],
                     dataset=dataset or doc.get("dataset", ""),
                     take=take or doc.get("take", ""))
    k = DATASET_ROTATION.get(cam.dataset, 0) if rotation is None else rotation
    return cam.rescale(out_size).rotate90(k)


#: Aria RGB frame sizes. Full resolution and the binned stream.
_ARIA_SIZES = (2880, 1408)


def _infer_native(cx: float, cy: float) -> int:
    guess = 2.0 * max(cx, cy)
    best = min(_ARIA_SIZES, key=lambda s: abs(s - guess))
    if abs(best - guess) > 0.15 * best:
        raise CalibrationUnavailable(
            f"principal point ({cx:.1f}, {cy:.1f}) implies a ~{guess:.0f} px "
            f"sensor, which is not an Aria RGB size {_ARIA_SIZES}; the file is "
            f"not the calibration this expects")
    return best


def calibration_path(root: str, dataset: str, take: str) -> str:
    return os.path.join(root, dataset, take, "camera_rgb.json")


def require_verified(cam: Fisheye624) -> Fisheye624:
    """Pass the camera through, or refuse if its orientation is unproven.

    ``rect_derect`` maps predictions through this model, so an orientation that
    is off by a quarter turn does not degrade the number — it silently scores
    every point against the wrong part of the image. That is worth a hard stop
    rather than a warning nobody reads.
    """
    if cam.dataset not in VERIFIED_ROTATION:
        raise OrientationUnverified(
            f"the {cam.dataset!r} sensor-to-upright rotation has not passed "
            f"verify_orientation, so rect_derect would map predictions through "
            f"an unproven camera. Run "
            f"`python -m slambench.verify_camera --egosynth-root ... "
            f"--calib-root ...`, then add {cam.dataset!r} to "
            f"camera.VERIFIED_ROTATION. The 'raw' baseline needs no camera and "
            f"is unaffected.")
    return cam
