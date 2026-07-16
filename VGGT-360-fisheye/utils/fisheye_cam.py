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

The polynomial is only monotonic up to a lens-dependent *turnover* angle
(~62.3 deg for the Aria 214-1 coefficients).  Beyond it the projection folds
back and is non-injective: those rays are physically never imaged, and naively
sampling them aliases onto wrong in-cone pixels ("fold-back ghosting").  All
code here therefore clamps validity to ``theta <= kb4_max_incidence(k)``.

Provenance
----------
Constants and the KB4 math are vendored from this repo's
``finetune/eval/baselines/aria_fisheye.py`` (calibration validated there to
<0.22 deg against projectaria_tools) so that VGGT-360-fisheye stays a
self-contained subproject.  If you change the calibration, change it in both
places.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Aria 214-1 RGB calibration at native 1408x1408 (see finetune/data/rectify.py).
_ARIA_F_NATIVE = 610.94
_ARIA_CX_NATIVE = 715.11
_ARIA_CY_NATIVE = 716.71
_ARIA_NATIVE = 1408.0
_ARIA_KB4 = (0.3852, -0.4442, 0.5591, -0.3254)


@dataclass
class FisheyeCam:
    """KB4 fisheye intrinsics resolved for a specific (square) frame size.

    All attributes are in pixels for an ``H x W`` frame.  ``k`` is the KB4
    radial vector ``(k1, k2, k3, k4)`` — rotation- and resolution-invariant.
    """

    H: int
    W: int
    fx: float
    fy: float
    cx: float
    cy: float
    k: Tuple[float, float, float, float]

    def theta_max(self) -> float:
        """Max valid incidence angle (radians) — the KB4 forward turnover."""
        return kb4_max_incidence(self.k)


def aria_intrinsics(H: int, W: int, rotated: bool = True) -> FisheyeCam:
    """Aria 214-1 intrinsics scaled to an ``H x W`` frame.

    Parameters
    ----------
    H, W    : frame size actually being processed (square for Aria).
    rotated : True if the frame has had the ADT 270-deg-CCW rotation applied
              (the ADT eval convention — Aria stores frames 90 deg CW).
              Rotating a square frame swaps fx/fy and moves the principal
              point: for 270 CCW, pixel ``(x, y) -> (H-1-y, x)`` so
              ``cx' = (H-1) - cy`` and ``cy' = cx``.
    """
    sx, sy = W / _ARIA_NATIVE, H / _ARIA_NATIVE
    fx = _ARIA_F_NATIVE * sx
    fy = _ARIA_F_NATIVE * sy
    cx = _ARIA_CX_NATIVE * sx
    cy = _ARIA_CY_NATIVE * sy
    if rotated:
        fx, fy = fy, fx
        cx, cy = (H - 1) - cy, cx
    return FisheyeCam(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, k=_ARIA_KB4)


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
    """Max valid incidence angle (radians) = the forward-polynomial turnover.

    Beyond this angle ``theta_d(theta)`` decreases again (the projection folds
    back), so such rays cannot be imaged by the lens and must be excluded from
    every warp.  ~62.33 deg for the Aria 214-1 coefficients.
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
    along the ray).  If the evaluation GT is planar z-depth, convert with
    ``z = range * cos(theta)`` before scoring (``main_adt.py --pred-domain z``).
    At the Aria FOV edge ``cos(62 deg) ~ 0.47``, i.e. a >2x difference — this
    conversion is NOT optional if the domains differ.
    """
    rays, _ = fisheye_ray_lut(cam)
    return rays[..., 2].astype(np.float32)
