# Copyright (c) 2026.
"""Aria 214-1 KB4 fisheye camera model (forward + inverse projection).

This is the geometric foundation of the VGGT-360-fisheye port.  The upstream
VGGT-360 assumes a full 360x180 equirectangular (ERP) panorama, where
"pixel <-> ray" is a linear longitude/latitude mapping.  On ADT the input is a
single Aria RGB fisheye frame instead, so every place upstream converts between
pixels and rays through ERP trigonometry must go through the Kannala-Brandt
(KB4 / OPENCV_FISHEYE) model implemented here.

Camera model
------------
For a unit ray ``(X, Y, Z)`` in the camera frame (x right, y down, z forward =
optical axis), with incidence angle ``theta = arccos(Z)`` and image-plane
azimuth ``phi = atan2(Y, X)``:

    theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
    u = cx + fx * theta_d * cos(phi)
    v = cy + fy * theta_d * sin(phi)

Two different angular limits
----------------------------
Conflating these two is a bug, and this module deliberately keeps them apart:

    * **fold-back turnover** — ``kb4_max_incidence(k)`` ~ 62.33 deg.  The KB4
      polynomial is only monotonic up to this angle; beyond it the projection
      is non-injective and naively sampling those rays aliases onto wrong
      in-cone pixels ("fold-back ghosting").  This is a property of the fitted
      *model* and is a hard guard for every warp and for the inverse LUT.
    * **usable FOV** — ``aria_valid_theta_max()`` ~ 54.83 deg.  The angle the
      lens/sensor actually *images*.  Two independent measurements agree:
      (a) the image circle is inscribed in the 1408^2 frame — the smallest
      principal-point-to-border margin is 690.3 px, which unprojects to
      54.83 deg (while the circle at the turnover would need 745.8 px); and
      (b) photometrically, on a real ADT frame the p90 brightness per incidence
      band falls 139 -> 92 -> 52 -> 12 across 48 -> 55 -> 57 -> 61 deg, and the
      60-64 deg band is 98.6% black.

``FisheyeCam.theta_max()`` returns the ``min`` of the two, so every consumer
(ray LUT, view rendering, view-layout cone clamp, fusion, eval masks) inherits
the usable limit.  Using the turnover as the usable FOV — as this port
originally did — feeds ~5-12% dead vignette pixels per ring view into the
network flagged as valid, and puts ~10.9% dead pixels into the eval mask.

Provenance
----------
Constants and the KB4 math are vendored from this repo's
``finetune/eval/baselines/aria_fisheye.py`` (calibration validated there to
<0.22 deg against projectaria_tools) so that VGGT-360-fisheye stays a
self-contained subproject.  If you change the calibration, change it in both
places.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Aria 214-1 RGB calibration at native 1408x1408 (see finetune/data/rectify.py).
_ARIA_F_NATIVE = 610.94
_ARIA_CX_NATIVE = 715.11
_ARIA_CY_NATIVE = 716.71
_ARIA_NATIVE = 1408.0
_ARIA_KB4 = (0.3852, -0.4442, 0.5591, -0.3254)

# Optional path to an Aria device-calibration JSON.  When it is set AND
# projectaria_tools is importable, the lens' own "valid radius" is preferred
# over the geometric derivation in ``inscribed_valid_theta``.
_ARIA_CALIB_JSON_ENV = "ARIA_CALIB_JSON"


@dataclass
class FisheyeCam:
    """KB4 fisheye intrinsics resolved for a specific (square) frame size.

    All attributes are in pixels for an ``H x W`` frame.  ``k`` is the KB4
    radial vector ``(k1, k2, k3, k4)`` — rotation- and resolution-invariant.
    ``valid_theta`` is the lens' usable half-FOV in radians (see module
    docstring); ``None`` means "unknown, fall back to the fold-back turnover".
    """

    H: int
    W: int
    fx: float
    fy: float
    cx: float
    cy: float
    k: Tuple[float, float, float, float]
    valid_theta: Optional[float] = None

    def theta_max(self) -> float:
        """Max **usable** incidence angle (radians).

        ``min(usable FOV, fold-back turnover)`` — see the module docstring for
        why those are two different things.  Every downstream consumer
        (``fisheye_ray_lut``, ``fisheye_to_persp``, the view-layout cone clamp,
        fusion, the eval cone mask) goes through here, so this is the single
        place the usable cone is defined.
        """
        turnover = kb4_max_incidence(self.k)
        if self.valid_theta is None:
            return turnover
        return min(float(self.valid_theta), turnover)


def inscribed_valid_theta(fx: float, fy: float, cx: float, cy: float,
                          H: float, W: float,
                          k: Tuple[float, float, float, float]) -> float:
    """Usable half-FOV (radians) from the *inscribed* image circle.

    A fisheye sensor is normally sized so the image circle just fits inside the
    frame, so the smallest principal-point-to-border margin IS the circle
    radius.  Unprojecting that radius therefore gives the lens' usable
    half-FOV without needing the vendor calibration:

        theta_d = min(cx, W-1-cx, cy, H-1-cy) / f   ->   theta = KB4^-1(theta_d)

    For Aria 214-1 the limiting margin is 690.3 px and this returns
    **54.83 deg** — matching the photometric falloff measured on real ADT
    frames (see module docstring).  Being a ratio of pixels to focal length the
    result is resolution-invariant, and rotating a square frame permutes the
    margins without changing their minimum, so it is orientation-invariant too.
    """
    theta_d = min(cx / fx, (W - 1.0 - cx) / fx,
                  cy / fy, (H - 1.0 - cy) / fy)
    return float(kb4_unproject_theta(np.array([theta_d], dtype=np.float64), k)[0])


def _calibrated_valid_theta(k: Tuple[float, float, float, float],
                            calib_json: Optional[str] = None) -> Optional[float]:
    """Usable half-FOV (radians) from Aria's own calibration "valid radius".

    Aria camera calibrations carry an explicit valid-pixel radius, which is the
    authoritative answer.  Reaching it needs both ``projectaria_tools`` and a
    calibration source (``calib_json``, or the ``ARIA_CALIB_JSON`` env var), so
    this returns ``None`` — never raises — whenever the package is missing, the
    file is absent, or the API shape is not what we expect; the caller then
    uses ``inscribed_valid_theta``, which lands on the same answer to within a
    fraction of a degree.  The radius is stored at native resolution, so it is
    divided by the native focal length.
    """
    path = calib_json or os.environ.get(_ARIA_CALIB_JSON_ENV)
    if not path or not os.path.exists(path):
        return None
    try:
        from projectaria_tools.core import calibration as _calib

        with open(path, "r") as fh:
            device = _calib.device_calibration_from_json_string(fh.read())
        radius = device.get_camera_calib("camera-rgb").get_valid_radius()
        if radius is None:
            return None
        theta_d = float(radius) / _ARIA_F_NATIVE
        return float(kb4_unproject_theta(np.array([theta_d], dtype=np.float64), k)[0])
    except Exception:      # missing package, broken extension, changed API
        return None


_VALID_THETA_CACHE: dict = {}


def aria_valid_theta_max(calib_json: Optional[str] = None) -> float:
    """Usable half-FOV of the Aria 214-1 RGB lens (radians), ~54.83 deg.

    Calibration first, inscribed-circle geometry as the fallback.  Cached — the
    inverse-LUT interpolation is not free and the answer never changes.
    """
    key = calib_json or os.environ.get(_ARIA_CALIB_JSON_ENV) or ""
    if key not in _VALID_THETA_CACHE:
        th = _calibrated_valid_theta(_ARIA_KB4, calib_json=calib_json)
        if th is None:
            th = inscribed_valid_theta(_ARIA_F_NATIVE, _ARIA_F_NATIVE,
                                       _ARIA_CX_NATIVE, _ARIA_CY_NATIVE,
                                       _ARIA_NATIVE, _ARIA_NATIVE, _ARIA_KB4)
        _VALID_THETA_CACHE[key] = th
    return _VALID_THETA_CACHE[key]


def aria_intrinsics(H: int, W: int, rotated: bool = True,
                    calib_json: Optional[str] = None) -> FisheyeCam:
    """Aria 214-1 intrinsics scaled to an ``H x W`` frame.

    Parameters
    ----------
    H, W    : frame size actually being processed (square for Aria).
    rotated : True if the frame has had the ADT 270-deg-CCW rotation applied
              (the ADT eval convention — Aria stores frames 90 deg CW).
              Rotating a square frame swaps fx/fy and moves the principal
              point: for 270 CCW, pixel ``(x, y) -> (H-1-y, x)`` so
              ``cx' = (H-1) - cy`` and ``cy' = cx``.
    calib_json : optional Aria device-calibration JSON, used to read the lens'
              own valid radius instead of deriving it (see
              ``aria_valid_theta_max``).
    """
    sx, sy = W / _ARIA_NATIVE, H / _ARIA_NATIVE
    fx = _ARIA_F_NATIVE * sx
    fy = _ARIA_F_NATIVE * sy
    cx = _ARIA_CX_NATIVE * sx
    cy = _ARIA_CY_NATIVE * sy
    if rotated:
        fx, fy = fy, fx
        cx, cy = (H - 1) - cy, cx
    return FisheyeCam(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, k=_ARIA_KB4,
                      valid_theta=aria_valid_theta_max(calib_json))


# --------------------------------------------------------------------------- #
# KB4 polynomial: forward, turnover, inverse
# --------------------------------------------------------------------------- #

def kb4_forward_theta(theta: np.ndarray,
                      k: Tuple[float, float, float, float]) -> np.ndarray:
    """KB4 forward distortion ``theta -> theta_d`` (vectorised, closed form).

    theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8).
    Only meaningful for ``theta <= kb4_max_incidence(k)``; the caller is
    responsible for masking rays beyond the turnover.
    """
    k1, k2, k3, k4 = k
    t2 = theta * theta
    return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))


def kb4_max_incidence(k: Tuple[float, float, float, float], n: int = 8192) -> float:
    """Fold-back guard (radians) = the forward-polynomial turnover, ~62.33 deg.

    Beyond this angle ``theta_d(theta)`` decreases again, so the projection is
    non-injective and sampling such rays aliases onto wrong in-cone pixels.
    Every warp must exclude them.

    This is **not** the lens' usable FOV — the polynomial is fitted only inside
    the imaged circle, so its turnover sits ~7.5 deg beyond where Aria actually
    images (see ``aria_valid_theta_max``, and the module docstring for the two
    measurements).  Use ``FisheyeCam.theta_max()``, which takes the min of both.
    """
    tg = np.linspace(0.0, np.pi / 2 + 0.2, n)
    td = kb4_forward_theta(tg, k)
    dec = np.diff(td) <= 0
    if dec.any():
        return float(tg[int(np.argmax(dec))])
    return float(tg[-1])


def kb4_unproject_theta(theta_d: np.ndarray,
                        k: Tuple[float, float, float, float],
                        n: int = 8192) -> np.ndarray:
    """Invert ``theta_d = kb4_forward_theta(theta)`` for theta.

    The forward polynomial is tabulated on a dense grid, truncated to its
    monotonic prefix (the physically valid FOV), and inverted by linear
    interpolation.  This is robust where Newton iteration diverges (near and
    past the turnover).  ``theta_d`` beyond the lens maximum clamps to the max
    valid angle — such pixels lie outside the fisheye image circle and are
    masked out downstream.
    """
    tg = np.linspace(0.0, np.pi / 2 + 0.2, n)
    td = kb4_forward_theta(tg, k)
    dec = np.diff(td) <= 0
    if dec.any():
        turn = int(np.argmax(dec)) + 1
        tg, td = tg[:turn], td[:turn]
    return np.interp(np.clip(theta_d, 0.0, td[-1]), td, tg)


# --------------------------------------------------------------------------- #
# Per-pixel ray field (the fisheye analogue of upstream's ERP lon/lat grid)
# --------------------------------------------------------------------------- #

def fisheye_ray_lut(cam: FisheyeCam,
                    theta_max: Optional[float] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Unit ray direction + validity for every pixel of the fisheye frame.

    This replaces the first ~8 lines of upstream
    ``depth_set_to_equirect_attention`` (the ERP longitude/latitude ray grid):
    fusion iterates over *output rays*, and on ADT the output grid is the
    fisheye frame itself, whose rays come from KB4 unprojection.

    Returns
    -------
    rays  : ``(H, W, 3)`` float32 unit rays (x right, y down, z forward).
    valid : ``(H, W)`` bool — True where the pixel lies inside the physically
            imaged cone (``theta <= theta_max``).  Pixels outside (vignette
            corners of the square sensor) get a clamped-but-meaningless ray.
    """
    if theta_max is None:
        theta_max = cam.theta_max()
    us, vs = np.meshgrid(np.arange(cam.W, dtype=np.float64),
                         np.arange(cam.H, dtype=np.float64))
    x = (us - cam.cx) / cam.fx
    y = (vs - cam.cy) / cam.fy
    theta_d = np.sqrt(x * x + y * y)
    theta = kb4_unproject_theta(theta_d, cam.k)
    sin_t = np.sin(theta)
    inv = np.where(theta_d > 1e-9, 1.0 / theta_d, 0.0)
    rays = np.stack([sin_t * x * inv,
                     sin_t * y * inv,
                     np.cos(theta)], axis=-1).astype(np.float32)
    # On-axis pixel: theta_d ~ 0 -> ray is exactly +z.
    rays[theta_d <= 1e-9] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    valid = theta <= (theta_max - 1e-6)
    return rays, valid.astype(bool)


def ray_cos_incidence(cam: FisheyeCam) -> np.ndarray:
    """Per-pixel ``cos(theta)`` — converts euclidean range <-> planar z-depth.

    VGGT-360 fuses the *radial distance* ``||world_points||`` (euclidean range
    along the ray), while **ADT GT is planar z** (measured — see
    ``checks/check_gt_depth_domain.py``).  One side must therefore always be
    converted before scoring: ``z = range * cos(theta)`` to bring the
    prediction down, or ``range = z / cos(theta)`` to bring the GT up
    (``main_adt.py --eval-domains``, which does both by default).  At the Aria
    FOV edge ``cos(62 deg) ~ 0.47``, i.e. a >2x difference — the conversion is
    NOT optional, and being radial it cannot be absorbed by affine alignment.
    """
    rays, _ = fisheye_ray_lut(cam)
    return rays[..., 2].astype(np.float32)
