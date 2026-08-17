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

The frame the calibration describes is *not* the frame ego-synth ships
----------------------------------------------------------------------
This is the whole difficulty of this module, and it is a **known, documented
discrepancy in the Aria ecosystem**, not something peculiar to ego-synth:

    projectaria_tools issue #322, "ADT online_calibration.jsonl RGB is off by
    factor of 2x" — the MPS ``online_calibration`` RGB intrinsics are expressed
    at full sensor resolution, while the VRS ships the binned stream.

Three facts compose into the chain this module implements:

1. The Aria RGB sensor is **2880** square, and that is the frame MPS writes the
   calibration in. :func:`_sensor_frame` reads it off the principal point.
2. The **1408** stream every Aria recording in this evaluation was captured at
   is not a plain downsample of it: it is a 2816 centre crop, binned 2x. That
   the window is 2816 rather than 2880 is what issue #322 reports, and its own
   numbers pin the offset — the reporter's full-resolution
   ``f, cx, cy = 1221.88, 1462.73, 1465.93`` and their locally regenerated
   1408 values ``610.94, 715.11, 716.71`` agree with a 32 px crop then a halving
   to 0.005 px, and disagree with a plain halving by exactly 16 px.
3. ego-synth resized that 1408 stream to **896** — ``meta.json`` records
   ``source_width = 1408`` on all four datasets.

So the transform is one crop-then-scale, in ``CameraCalibration.rescale``'s own
order ("(1) shift -> (2) scaling") and on its pixel-centre convention:

    896 = rescale(scale = 896/2816, origin_offset = 32)

:func:`_rescale_params` is that formula, and ``test_camera.py`` pins it against
the library's own ``rescale`` — exactly, not approximately. It is written out
here rather than called because ``projectaria_tools`` needs Python >= 3.9 and
this repo's own interpreter is older; the library is the authority, the test is
what enforces it.

The rotation is a rotation of the *ray*, not of the model
---------------------------------------------------------
ego-synth's frames are upright, the calibration's are not. The obvious move is
to rotate the model's principal point, and this module used to. It is wrong:
``p1, p2`` and ``s1..s4`` are not isotropic and a quarter turn mixes them.
``projectaria_tools`` makes the same judgement structurally — its
``rotate_camera_calib_cw90deg`` accepts the **Linear model only**, and there is
no fisheye equivalent.

So :attr:`Fisheye624.rotation` is carried as a quarter-turn count and applied to
the 3D ray on the way in and to the pixel on the way out, which is exact and
touches no coefficient. Measured against the true-twin correspondences of
``verify_camera``, over 9 frames x 3 clips x 2 datasets:

    rotating the model   1.5 px median
    rotating the ray     1.0 px median

Where this stands
-----------------
Settled, and ``verify_camera`` passes. Three corrections, in the order they were
found, each worth roughly what the one before it left behind:

    twin residual (px)        aea    nymeria
    as it was                6.74     6.96
    + the 2816 crop          1.37     1.51
    + rotating the ray       1.05     0.99
    + the rectified centre   0.31     0.29     <- verify_camera's own bug

**Not one fitted parameter.** Every number in the chain is a sensor
specification, a published report, or a pixel-centre convention the reference
implementation already uses. The third correction was in the verifier rather than
here: it back-projected through the rectified render using ``N / 2`` where the
whole ecosystem uses ``(N - 1) / 2``, and that half pixel arrives at the fisheye
magnified by the ratio of the focal lengths. See ``verify_camera.rect_centre``.

At 0.29-0.31 px the residual is under the 0.5 px float16 quantisation of the
stored coordinates, which is the floor this data can express. ``aea`` and
``nymeria`` are in :data:`VERIFIED_ROTATION`; ``oxford`` and ``egoexo4d`` are
not, and :func:`require_verified` still refuses those.
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

#: The Aria RGB sensor, and the frame MPS writes ``online_calibration`` in.
SENSOR_FULL = 2880

#: The binned RGB stream every recording in this evaluation was captured at, and
#: the resolution ``meta.json`` reports as ``source_width`` on all four datasets.
STREAM_RES = 1408

#: The window of the sensor that stream reads out, before the 2x binning. Not
#: :data:`SENSOR_FULL`: see the module docstring and projectaria_tools #322.
SENSOR_WINDOW = 2 * STREAM_RES                      # 2816

#: Offset of that window's origin inside the full sensor frame, per axis. This
#: is ``origin_offset`` in ``CameraCalibration.rescale``.
CROP_ORIGIN = (SENSOR_FULL - SENSOR_WINDOW) / 2.0   # 32.0

#: Quarter turns (CCW) taking the calibration's sensor frame to ego-synth's
#: upright frame, per dataset.
#:
#: ``verify_camera``'s twin residual separates the four turns by a factor of 40
#: and gives the same answer on both datasets measured:
#:
#:     twin residual (px)   rot 0    rot 90   rot 180   rot 270
#:     aea                  13.20      0.31     13.41     18.83
#:     nymeria              10.11      0.29     10.41     14.56
#:
#: ``oxford`` has never been measured and ``egoexo4d`` is paused; both carry the
#: same value only because it is the one with evidence behind it anywhere, and
#: neither is in :data:`VERIFIED_ROTATION`.
DATASET_ROTATION: Dict[str, int] = {
    "aea": 1,          # 90 deg CCW -- twin 0.31 px, 43x margin
    "nymeria": 1,      # 90 deg CCW -- twin 0.29 px, 35x margin
    "egoexo4d": 1,     #             -- PAUSED, never measured
    "oxford": 1,       #             -- never measured
}

#: Datasets whose rotation has passed ``verify_camera`` at the tolerance below.
#: Written by hand after a verification run, never by the verifier itself — a
#: measurement that promotes itself is not a check.
#:
#: Both entries passed at NN median 0.29 px with 96.9 % of points inside 1 px,
#: against a 80 % bar and a 0.5-2 % chance rate, and beat the runner-up rotation
#: by 14x. **On one take each** — the local sample is one take per dataset. The
#: rotation is a per-dataset orientation convention rather than a per-device
#: quantity, and a quarter-turn error shows up as 4-5 px against 0.29, so one
#: take is genuinely decisive *for the rotation*. Re-run the verifier across
#: takes when ticket 012 lands the calibrations at scale; that is issue #17.
VERIFIED_ROTATION: Tuple[str, ...] = ("aea", "nymeria")

#: Median reprojection error, in pixels of the 896 frame, below which the model
#: and its orientation are considered to describe the data. Half a pixel is the
#: float16 quantisation of the stored ``u``/``v``, so anything at or under it is
#: at the noise floor of what the ground truth can even express.
ORIENTATION_TOL_PX = 0.5


_REF_CACHE: Dict[tuple, object] = {}


def _reference_projection(params: Tuple[float, ...]):
    """``projectaria_tools``' own FISHEYE624 projection, if the package is here.

    Optional on purpose. The package is the reference implementation of this
    model and is preferred wherever a sub-pixel answer matters, but requiring it
    would make the whole SLAM evaluation — including the ``raw`` baseline, which
    needs no camera at all — depend on an Aria SDK.
    """
    if params in _REF_CACHE:
        return _REF_CACHE[params]
    try:
        from projectaria_tools.core import calibration as _pcal
        obj = _pcal.CameraProjection(_pcal.CameraModelType.FISHEYE624,
                                     np.asarray(params, float))
    except Exception:                       # noqa: BLE001 — absent or too old
        obj = None
    _REF_CACHE[params] = obj
    return obj


def reference_available() -> bool:
    """Whether the reference FISHEYE624 implementation can be used."""
    return _reference_projection((1.0,) * N_PARAMS) is not None


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

    ``rotation`` is a count of quarter turns CCW taking the frame ``params``
    describes to the frame the caller works in. It is deliberately *not* folded
    into ``params``: see the module docstring on why a quarter turn may not touch
    this model's coefficients. :meth:`project` and :meth:`unproject` apply it.
    """

    params: Tuple[float, ...]
    width: int
    height: int
    rotation: int = 0
    label: str = "camera-rgb"
    serial: str = ""
    dataset: str = ""
    take: str = ""

    def _with(self, **kw) -> "Fisheye624":
        """A copy with some fields replaced, carrying the metadata across."""
        f = dict(params=self.params, width=self.width, height=self.height,
                 rotation=self.rotation, label=self.label, serial=self.serial,
                 dataset=self.dataset, take=self.take)
        f.update(kw)
        return Fisheye624(**f)

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
    def rescale(self, out_size: int,
                origin_offset: float = 0.0) -> "Fisheye624":
        """The same lens, described for a cropped and resized square frame.

        ``origin_offset`` px are taken off every side first, then what is left is
        resized to ``out_size``. That is ``CameraCalibration.rescale``'s own
        order — its docstring reads "transform is done in the order of (1) shift
        -> (2) scaling" — and :func:`_rescale_params` is its arithmetic.
        ``test_camera.py`` pins the two together.

        Only ``f``, ``cx``, ``cy`` move. The radial, tangential and thin-prism
        coefficients act on *normalised* image coordinates, so they are invariant
        to both steps — which is exactly why they must not be touched here, and
        the library leaves them alone too.
        """
        window = self.width - 2.0 * origin_offset
        if window <= 0:
            raise ValueError(f"origin_offset {origin_offset} crops away the "
                             f"whole {self.width} px frame")
        q = _rescale_params(self.params, out_size / window, origin_offset)
        return self._with(params=q, width=out_size, height=out_size)

    def rotated(self, k: int) -> "Fisheye624":
        """The same lens, read in a frame rotated ``k`` quarter turns CCW.

        Records the turn; it is applied to rays and pixels by :meth:`project` and
        :meth:`unproject`, never to the parameters. Square frames only, which
        every ego-synth frame is.
        """
        if self.width != self.height:
            raise ValueError("a quarter turn needs a square frame; "
                             f"got {self.width}x{self.height}")
        return self._with(rotation=k % 4)

    # -- geometry ----------------------------------------------------------- #
    def project(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """3D points in the camera frame ``(N, 3)`` -> pixels ``(u, v)``.

        Uses ``projectaria_tools``' own FISHEYE624 projection when that package
        is importable, and the implementation below otherwise. The two now agree
        to floating point: ``test_camera.py`` pins the fallback against a
        transcription of the public reference model, at 1e-9 px and out to 62 deg
        of incidence, on a machine with neither the package nor torch.

        It did not always. The fallback carried OpenCV's tangential convention
        rather than this model's — a p1/p2 swap, self-consistent between
        ``project`` and ``unproject``, so the round trip closed and the drift was
        visible only from outside: 2.9 px at the rim of a 2880 sensor, 0.90 px at
        ego-synth's 896 frame, against the 0.29 px ``verify_camera`` accepts a
        camera at. Verification runs on a box *with* the package and
        ``rect_derect`` may run on one without, so a fallback that merely tracks
        the reference "closely enough" is not a safe state for this module.

        Points behind the camera are not meaningful for a fisheye of this FOV
        and come back as NaN rather than as a plausible pixel.

        ``xyz`` is in the camera frame of the **rotated** view, so the turn is
        undone on the ray, the unrotated model projects, and the turn is redone
        on the pixel. No coefficient is touched — see the module docstring.
        """
        xyz = _rot_xyz(np.asarray(xyz, float), -self.rotation)
        ref = _reference_projection(self.params)
        if ref is not None:
            out = np.full((xyz.shape[0], 2), np.nan)
            for i, p in enumerate(xyz):
                if p[2] > 0:
                    out[i] = ref.project(p)
            u, v = out[:, 0], out[:, 1]
        else:
            u, v = self._pure_project(xyz)
        return _rot_pixel(u, v, self.rotation, self.width)

    def _pure_project(self, xyz: np.ndarray):
        """The numpy FISHEYE624 forward model, used when the reference is absent."""
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
        u, v = _rot_pixel(np.asarray(u, float), np.asarray(v, float),
                          -self.rotation, self.width)
        ref = _reference_projection(self.params)
        if ref is not None:
            # Both directions must come from the same implementation or the
            # round trip does not close: with the reference forward model and
            # the fallback inverse, it opens to 0.95 px on this lens.
            out = np.empty((u.size, 3))
            for i, (uu, vv) in enumerate(zip(u.ravel(), v.ravel())):
                out[i] = ref.unproject(np.array([uu, vv], float))
            out = out / np.linalg.norm(out, axis=-1, keepdims=True)
            return _rot_xyz(out, self.rotation)
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
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return _rot_xyz(d, self.rotation)

    def theta_of(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Incidence angle (degrees) of the rays those pixels see."""
        d = self.unproject(u, v)
        return np.degrees(np.arccos(np.clip(d[:, 2], -1.0, 1.0)))

    # -- bulk geometry, for warps rather than point lists -------------------- #
    #
    # :meth:`project` and :meth:`unproject` call ``projectaria_tools`` **one
    # point at a time**, because its binding takes one point. That is right for
    # this evaluation's own work — a frame's ground truth is a few thousand
    # points — and unusable for a warp: rendering the VGGT-360 view layout asks
    # for ~21 million rays per frame, where a Python-level loop is the whole run
    # rather than a cost inside it.
    #
    # The two methods below take the vectorised numpy path unconditionally. That
    # is safe *because of* ``test_camera.py``, which pins that path against a
    # transcription of the public reference model to 1e-9 px out to 62 deg of
    # incidence — the same pin that exists so ``rect_derect`` may run on a box
    # without the package. They are not a second camera model; they are the same
    # one, called in a shape a warp can afford. A number produced through them is
    # a number about this lens.

    def project_bulk(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """:meth:`project` for many rays at once, shape-preserving.

        Accepts any ``(..., 3)`` and returns ``u, v`` with that leading shape,
        so a ``(H, W, 3)`` ray field comes back as two ``(H, W)`` maps.
        """
        a = np.asarray(xyz, float)
        lead = a.shape[:-1]
        u, v = self._pure_project(_rot_xyz(a.reshape(-1, 3), -self.rotation))
        u, v = _rot_pixel(u, v, self.rotation, self.width)
        return u.reshape(lead), v.reshape(lead)

    def unproject_bulk(self, u: np.ndarray, v: np.ndarray,
                       iters: int = 20) -> np.ndarray:
        """:meth:`unproject` for many pixels at once, shape-preserving.

        ``(H, W)`` in, ``(H, W, 3)`` out.
        """
        u = np.asarray(u, float)
        lead = u.shape
        uu, vv = _rot_pixel(u.reshape(-1), np.asarray(v, float).reshape(-1),
                            -self.rotation, self.width)
        xd = (uu - self.cx) / self.f
        yd = (vv - self.cy) / self.f
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
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return _rot_xyz(d, self.rotation).reshape(lead + (3,))

    def max_imaged_theta(self, n: int = 4096) -> float:
        """Largest incidence angle this lens actually images, in **radians**.

        The FISHEYE624 calibration carries no valid radius, so it is derived the
        way ``VGGT-360-fisheye``'s KB4 model derives its own (``fisheye_cam.
        inscribed_valid_theta``): walk theta outward and stop at the first of two
        things — the projected pixel leaving the frame, or the radial polynomial
        turning over, past which the projection is non-injective and sampling it
        aliases distant rays onto in-cone pixels.

        Averaged over eight azimuths, because this model is **not** radially
        symmetric: the tangential and thin-prism terms make the imaged region
        slightly non-circular, so a single azimuth would report one radius of an
        oval as if it were the cone. The minimum over azimuths is taken, which is
        the largest angle imaged in *every* direction — the conservative reading,
        and the one a view layout should be sized against.

        Computed in the sensor's **own** orientation, not the caller's. The cone
        is a property of the lens, and a quarter turn of a square frame permutes
        the four principal-point margins without changing their minimum — the
        same argument ``fisheye_cam.inscribed_valid_theta`` makes. Projecting
        through the rotation instead would test rotated pixels against an
        unrotated principal point and report the on-axis ray as already outside.
        """
        th = np.linspace(0.0, math.pi / 2 - 1e-3, n)
        best = math.pi / 2
        for az in np.linspace(0.0, 2 * math.pi, 8, endpoint=False):
            d = np.stack([np.sin(th) * math.cos(az),
                          np.sin(th) * math.sin(az),
                          np.cos(th)], axis=-1)
            u, v = self._pure_project(d)
            r = np.hypot(u - self.cx, v - self.cy)
            inside = (np.isfinite(u) & np.isfinite(v)
                      & (u >= 0) & (u <= self.width - 1)
                      & (v >= 0) & (v <= self.height - 1))
            # First index that is either out of frame or past the turnover.
            turn = np.r_[False, np.diff(r) <= 0]
            bad = (~inside) | turn
            i = int(np.argmax(bad)) if bad.any() else n - 1
            best = min(best, float(th[max(i - 1, 0)]))
        return best

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
        """FISHEYE624's tangential + thin-prism terms, in the reference's order.

        Aria's model is **not** OpenCV's tangential convention: ``p1`` carries
        the ``(r^2 + 2x^2)`` term on *x* and ``p2`` carries it on *y*, where
        OpenCV has them the other way round. Written OpenCV-style this was a
        clean p1/p2 swap — self-consistent, so the project/unproject round trip
        still closed to 1e-9 and every test but the reference comparison passed,
        while the projection sat up to 2.9 px off at the rim of a 2880 sensor.
        The authority is ``projectaria_tools``' ``FisheyeRadTanThinPrism``, and
        ``test_camera.py`` pins this against a transcription of it that needs no
        optional package.
        """
        p1, p2 = self.p
        s = self.s
        rr = xr * xr + yr * yr
        xt = p1 * (rr + 2 * xr * xr) + 2 * p2 * xr * yr
        yt = p2 * (rr + 2 * yr * yr) + 2 * p1 * xr * yr
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

    Cropped and rescaled from the frame MPS wrote it in down to ``out_size``,
    and rotated by the dataset's convention. ``rotation`` overrides
    :data:`DATASET_ROTATION`, which is what the verification script uses to try
    each one.

    The crop is the substance of this function and the module docstring is where
    it is justified. If the calibration is at :data:`SENSOR_FULL`, the frame
    ego-synth resized was the 2816 window inside it, so 32 px come off each side
    before the scale. If it is already at :data:`STREAM_RES` then it describes
    that window directly and there is nothing to crop.
    """
    with open(path) as fh:
        doc = json.load(fh)
    params = doc.get("params")
    if not params or len(params) != N_PARAMS:
        raise CalibrationUnavailable(
            f"{path}: expected {N_PARAMS} FISHEYE624 params, got "
            f"{0 if not params else len(params)}")
    sensor = _sensor_frame(float(params[1]), float(params[2]))
    offset = CROP_ORIGIN if sensor == SENSOR_FULL else 0.0
    cam = Fisheye624(tuple(float(x) for x in params), sensor, sensor,
                     label=doc.get("label", "camera-rgb"),
                     serial=(doc.get("serial_numbers") or [""])[0],
                     dataset=dataset or doc.get("dataset", ""),
                     take=take or doc.get("take", ""))
    k = DATASET_ROTATION.get(cam.dataset, 0) if rotation is None else rotation
    return cam.rescale(out_size, origin_offset=offset).rotated(k)


#: Aria RGB frame sizes a calibration can arrive in.
_ARIA_SIZES = (SENSOR_FULL, STREAM_RES)


def _sensor_frame(cx: float, cy: float) -> int:
    """Which frame the calibration's parameters are expressed in.

    MPS does not record it, so it is read off the principal point: a calibration
    is centred to within a few per cent, so ``2 * max(cx, cy)`` rounded to the
    nearest Aria RGB size identifies it, and the two sizes are far enough apart
    that the inference cannot land between them.
    """
    guess = 2.0 * max(cx, cy)
    best = min(_ARIA_SIZES, key=lambda s: abs(s - guess))
    if abs(best - guess) > 0.15 * best:
        raise CalibrationUnavailable(
            f"principal point ({cx:.1f}, {cy:.1f}) implies a ~{guess:.0f} px "
            f"sensor, which is not an Aria RGB size {_ARIA_SIZES}; the file is "
            f"not the calibration this expects")
    return best


def _rescale_params(params: Tuple[float, ...], scale: float,
                    origin_offset: float) -> Tuple[float, ...]:
    """``CameraCalibration.rescale``'s arithmetic, on the 15-vector.

    Shift then scale, on the pixel-centre convention: a resize maps output pixel
    centre ``j + 0.5`` to source ``(j + 0.5) / s``, so ``c' = (c + 0.5) * s -
    0.5`` and not ``c * s``. The two differ by ``0.5 * (1 - s)``, which is
    0.34 px at the factor used here — the same order as the effect this
    calibration exists to correct, so it is not roundable away.

    ``test_camera.py`` asserts this agrees with the library exactly.
    """
    q = list(params)
    q[0] = params[0] * scale
    q[1] = (params[1] - origin_offset + 0.5) * scale - 0.5
    q[2] = (params[2] - origin_offset + 0.5) * scale - 0.5
    return tuple(q)


def _rot_xyz(xyz: np.ndarray, k: int) -> np.ndarray:
    """Rotate camera-frame points ``k`` quarter turns CCW about the optic axis."""
    out = np.asarray(xyz, float)
    for _ in range(k % 4):
        out = np.stack([-out[..., 1], out[..., 0], out[..., 2]], axis=-1)
    return out


def _rot_pixel(u: np.ndarray, v: np.ndarray, k: int, size: int):
    """Rotate pixels ``k`` quarter turns CCW in a ``size`` square frame.

    One turn sends ``(u, v) -> (N - v, u)`` with ``N = size - 1``, the pixel-grid
    counterpart of :func:`_rot_xyz`.
    """
    n = size - 1.0
    for _ in range(k % 4):
        u, v = n - v, u
    return u, v


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
            f"verify_camera, so rect_derect would map predictions through "
            f"an unproven camera. Run "
            f"`python -m slambench.verify_camera --egosynth-root ... "
            f"--calib-root ...`, then add {cam.dataset!r} to "
            f"camera.VERIFIED_ROTATION. The 'raw' baseline needs no camera and "
            f"is unaffected.")
    return cam
