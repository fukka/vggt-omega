# Copyright (c) 2026.
"""Aria 214-1 fisheye camera helpers shared by the UniK3D and DAC baselines.

Both baselines are evaluated on the **raw (non-rectified) Aria fisheye** frame —
the whole point of testing "any-camera" depth models. To do that correctly each
model needs the Aria fisheye intrinsics:

  * UniK3D  takes a ``Fisheye624`` camera (16 params: fx,fy,cx,cy + 6 radial +
            2 tangential + 4 thin-prism).
  * DAC     warps the fisheye image into an equirectangular (ERP) patch using an
            ``OPENCV_FISHEYE`` ``cam_params`` dict (fx,fy,cx,cy + k1..k4).

The Aria 214-1 RGB calibration (native 1408×1408, validated in ``test_run.py`` /
``finetune/data/rectify.py`` to <0.22° against projectaria_tools) is a
Kannala-Brandt KB4 model::

    f = 610.94,  cx = 715.11,  cy = 716.71      (pixels @ 1408)
    KB4 = [0.3852, -0.4442, 0.5591, -0.3254]

Why the *same* KB4 coefficients work for both models
----------------------------------------------------
OpenCV-fisheye (DAC) distorts the incidence angle θ as

    θ_d = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
        = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹

UniK3D ``Fisheye624.project`` uses

    θ_k = θ + Σ_i k_i·θ^(3+2i)

i.e. the identical polynomial. So the Aria KB4 vector drops straight into both
(UniK3D gets k5,k6,p1,p2,s1..s4 = 0). projectaria's own KB4 uses this same form,
so the whole chain is consistent.

Rotation
--------
The ADT loader (``adt_depth.ADTWindowDataset``) feeds frames **already rotated
270° CCW** (Aria stores frames 90° CW; 270° CCW makes them upright). Radial
distortion is rotation-invariant, but the principal point moves. For a 270°-CCW
(= 90°-CW) rotation of a square H×W frame, pixel (x, y) → (H-1-y, x), so

    cx' = (H-1) - cy ,   cy' = cx ,   fx' = fy ,   fy' = fx

(matching ``rectify._ARIA_214_1_CX_ROT_NORM = 1 - cy_norm`` etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# Aria 214-1 RGB, native 1408×1408 (see finetune/data/rectify.py).
_ARIA_F_NATIVE = 610.94
_ARIA_CX_NATIVE = 715.11
_ARIA_CY_NATIVE = 716.71
_ARIA_NATIVE = 1408.0
_ARIA_KB4 = (0.3852, -0.4442, 0.5591, -0.3254)


@dataclass
class AriaFisheye:
    """Aria fisheye intrinsics resolved for a specific (square) frame size.

    Attributes are in pixels for an ``H×W`` frame. ``k`` is the KB4 radial vector
    ``(k1, k2, k3, k4)`` (rotation-invariant, resolution-invariant).
    """

    H: int
    W: int
    fx: float
    fy: float
    cx: float
    cy: float
    k: Tuple[float, float, float, float]

    # -- exporters to the two model conventions ----------------------------- #
    def fisheye624_params(self) -> list:
        """16-vector for ``unik3d.utils.camera.Fisheye624``.

        ``[fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]``;
        Aria uses only the first 4 radial terms, the rest are 0.
        """
        return [self.fx, self.fy, self.cx, self.cy,
                self.k[0], self.k[1], self.k[2], self.k[3],
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def usable_theta_max(self) -> float:
        """Max incidence angle (radians) this camera actually images (~54.83°).

        See ``usable_max_incidence`` — this is the usable FOV, *not* the KB4
        fold-back turnover, and it is what every cone mask should use.
        """
        return usable_max_incidence(self.k, self.fx, self.fy,
                                    self.cx, self.cy, self.H, self.W)

    def opencv_fisheye_params(self) -> Dict[str, object]:
        """``cam_params`` dict for ``dac.utils.erp_geometry.cam_to_erp_patch_fast``."""
        return {
            "camera_model": "OPENCV_FISHEYE",
            "fl_x": self.fx, "fl_y": self.fy, "cx": self.cx, "cy": self.cy,
            "k1": self.k[0], "k2": self.k[1], "k3": self.k[2], "k4": self.k[3],
        }


def aria_intrinsics(H: int, W: int, rotated: bool = True) -> AriaFisheye:
    """Aria 214-1 intrinsics scaled to an ``H×W`` frame.

    Parameters
    ----------
    H, W     : frame size actually fed to the model (square for Aria).
    rotated  : True if the frame has had the ADT 270°-CCW rotation applied
               (``ADTWindowDataset`` default). Swaps fx/fy and moves the
               principal point accordingly. Set False for a frame in native
               sensor orientation (e.g. an already-upright EgoExo render).
    """
    sx, sy = W / _ARIA_NATIVE, H / _ARIA_NATIVE
    fx = _ARIA_F_NATIVE * sx
    fy = _ARIA_F_NATIVE * sy
    cx = _ARIA_CX_NATIVE * sx
    cy = _ARIA_CY_NATIVE * sy
    if rotated:
        # 270° CCW (= 90° CW): (x,y) -> (H-1-y, x); fx/fy swap.
        fx, fy = fy, fx
        cx, cy = (H - 1) - cy, cx
    return AriaFisheye(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, k=_ARIA_KB4)


def aria_centered(H: int, W: int) -> AriaFisheye:
    """Aria KB4 with a *centered* principal point (cx=W/2, cy=H/2).

    Use for renders whose fisheye circle is centered and whose exact per-device
    principal point is unknown (e.g. the EgoExo4D ego renders, used qualitatively
    so a small principal-point offset does not matter).
    """
    f = _ARIA_F_NATIVE * (max(H, W) / _ARIA_NATIVE)
    return AriaFisheye(H=H, W=W, fx=f, fy=f, cx=W / 2.0, cy=H / 2.0, k=_ARIA_KB4)


# --------------------------------------------------------------------------- #
# Per-pixel ray grid (KB4 unprojection) and z <-> euclidean-range conversion.
#
# Most monocular MDE models (and the VGGT/DAv2 eval) treat "depth" as the planar
# z coordinate. Spherical/ERP models (DAC) work in euclidean range (distance
# along the ray). These helpers let a caller convert between the two when needed;
# by default the runner matches the existing pipeline and does NOT convert
# (compares the model's native depth output directly to the ADT GT).
# --------------------------------------------------------------------------- #

def _kb4_unproject_theta(theta_d: np.ndarray, k: Tuple[float, float, float, float],
                         n: int = 8192) -> np.ndarray:
    """Invert θ_d = θ + k1θ³ + k2θ⁵ + k3θ⁷ + k4θ⁹ for θ.

    The forward KB4 polynomial is only monotonic up to the lens's max incidence
    angle (it turns over beyond ~62° for the Aria 214-1 coeffs), and Newton
    iteration diverges there. We instead tabulate the forward map on a dense grid,
    truncate it to the monotonic prefix (the valid FOV), and invert by
    interpolation — robust and overflow-free. Pixels whose θ_d exceeds the lens
    maximum (the vignette corners, outside the fisheye circle) clamp to the max
    valid angle and are masked out downstream anyway.
    """
    k1, k2, k3, k4 = k
    tg = np.linspace(0.0, np.pi / 2 + 0.2, n)
    t2 = tg * tg
    td = tg * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    # keep the monotonic-increasing prefix (up to the polynomial turnover).
    dec = np.diff(td) <= 0
    if dec.any():
        turn = int(np.argmax(dec)) + 1
        tg, td = tg[:turn], td[:turn]
    return np.interp(np.clip(theta_d, 0.0, td[-1]), td, tg)


def kb4_max_incidence(k: Tuple[float, float, float, float], n: int = 8192) -> float:
    """Max valid incidence angle (radians) for a KB4 lens = the forward turnover.

    ``θ_d = θ + k1θ³ + k2θ⁵ + k3θ⁷ + k4θ⁹`` is only monotonic up to the lens's
    maximum incidence angle (~62° for the Aria 214-1 coeffs). Beyond it the
    polynomial folds back — the projection is non-injective, so any ray past this
    angle *cannot be imaged* and must be excluded from an ERP warp (otherwise it
    aliases onto a wrong, in-cone source pixel → ghosting). Returns the turnover
    angle, or the table's upper bound (~π/2) if the polynomial stays monotonic.
    """
    k1, k2, k3, k4 = k
    tg = np.linspace(0.0, np.pi / 2 + 0.2, n)
    t2 = tg * tg
    td = tg * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    dec = np.diff(td) <= 0
    if dec.any():
        return float(tg[int(np.argmax(dec))])
    return float(tg[-1])


def inscribed_valid_theta(fx: float, fy: float, cx: float, cy: float,
                          H: float, W: float,
                          k: Tuple[float, float, float, float]) -> float:
    """Usable half-FOV (radians) from the *inscribed* image circle, ~54.83°.

    A fisheye sensor is sized so the image circle just fits inside the frame, so
    the smallest principal-point-to-border margin IS the circle radius:

        θ_d = min(cx, W-1-cx, cy, H-1-cy) / f    →    θ = KB4⁻¹(θ_d)

    For Aria 214-1 the limiting margin is 690.3 px → **54.83°**, whereas the
    circle at the fold-back turnover would need 745.8 px. Being a pixel/focal
    ratio the result is resolution-invariant, and rotating a square frame
    permutes the margins without changing their minimum.
    """
    theta_d = min(cx / fx, (W - 1.0 - cx) / fx,
                  cy / fy, (H - 1.0 - cy) / fy)
    return float(_kb4_unproject_theta(np.array([theta_d], dtype=np.float64), k)[0])


def usable_max_incidence(k: Tuple[float, float, float, float],
                         fx: float, fy: float, cx: float, cy: float,
                         H: float, W: float) -> float:
    """Max incidence angle (radians) that the lens actually **images**.

    Distinct from ``kb4_max_incidence``, and the two must not be confused:

    * ``kb4_max_incidence`` is the polynomial **fold-back turnover** (62.33°) — a
      property of the *fit*. It is the right guard against aliasing, and the
      right truncation point for the inverse LUT.
    * this is the **usable FOV** (~54.83°). The KB4 coefficients are fitted only
      inside the imaged circle, so the turnover sits ~7.5° past where Aria
      images anything. Two independent measurements agree: the inscribed-circle
      geometry above, and photometry on a real ADT frame, where p90 brightness
      per incidence band falls 139 → 92 → 52 → 12 across 48° → 55° → 57° → 61°
      and the 60-64° band is 98.6% black.

    Cone masks — for the ERP warps here and for the tangent-view port in
    ``VGGT-360-fisheye/utils/fisheye_cam.py`` — must use this one, otherwise dead
    vignette pixels are scored against GT (~10.9% of the Aria cone mask).
    Returns ``min`` of the two so it is always a safe warp bound.
    """
    return min(inscribed_valid_theta(fx, fy, cx, cy, H, W, k),
               kb4_max_incidence(k))


def aria_ray_z(cam: AriaFisheye) -> np.ndarray:
    """Per-pixel ray z-component ``cos(θ)`` for an ``H×W`` Aria frame.

    Returns a ``(H, W)`` float32 array; ``planar_z = range · cos(θ)`` so
    ``range = z / cos(θ)`` and ``z = range · cos(θ)``.
    """
    H, W = cam.H, cam.W
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    x = (us - cam.cx) / cam.fx
    y = (vs - cam.cy) / cam.fy
    theta_d = np.sqrt(x * x + y * y)
    theta = _kb4_unproject_theta(theta_d.astype(np.float64), cam.k)
    return np.cos(theta).astype(np.float32)


def planar_to_range(depth_z: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    """planar z → euclidean range (distance along ray)."""
    return depth_z / np.clip(cos_theta, 1e-6, None)


def range_to_planar(depth_r: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    """euclidean range → planar z."""
    return depth_r * cos_theta
