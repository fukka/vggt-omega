# Copyright (c) 2026.
"""Fisheye -> pinhole rectification for the training loader.

The self-supervised geometric/photometric losses assume a PINHOLE camera
(``geometry.decode_pose_encoding`` builds a pinhole K). Raw Aria RGB (214-1) is
fisheye, so warping with a pinhole model is wrong toward the periphery and
injects bad gradients. This rectifier (Kannala-Brandt KB4, same preset that
``test_run.py`` validated against projectaria_tools to <0.22 deg) maps frames to
a linear camera up front.

Operates on HWC float images in [0,1] (RGB). Remap maps are cached per (H, W).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# The Aria 214-1 calibration and the storage-rotation rule come from
# ``finetune/aria_calibration.py``, which is the single description of this lens.
#
# This file used to hold its own copy, and its rotation was a pixel out: it put
# the rotated principal point at ``W - cy`` where the other two consumers use
# ``(H-1) - cy``. A pixel's centre is at its integer coordinate, so the last
# column of an H-wide frame is at H-1; np.rot90(m, 3) sends (u, v) to
# ((H-1) - v, u). Correcting it moves the ADT-FOV rectified arm's per-bin AbsRel
# by 0.1-0.7% and its ``pen`` by -0.4% (measured), and it removes a one-pixel
# disagreement between the two arms of an experiment whose whole purpose is that
# they be comparable.
from finetune.aria_calibration import KB4 as _ARIA_214_1_KB4, intrinsics as _aria

_ARIA_214_1_D_KB4 = np.array(_ARIA_214_1_KB4, np.float64)
_ARIA_214_1_FOCAL_OUT_NORM = 0.55  # output focal / max(H,W): avoids black borders

# Output-focal presets, as focal/max(H,W). The KB4 imaged cone is theta_max =
# 62.33 deg, i.e. a disc of radius tan(62.33 deg) = 1.907 focal lengths in the
# pinhole plane, so this single number sweeps the whole crop/black trade-off:
#
#   focal_out_norm   hFoV     black px   imaged cone kept
#   0.262            124.7    21.5%      99.9%   <- circumscribed: keeps the cone,
#                                                   pays with four black wedges
#   0.371            106.9     0.0%      83.3%   <- inscribed: black-free by
#                                                   construction, drops the rim
#   0.55 (default)    84.6     0.0%      ~55%    <- the historical default, chosen
#                                                   to "avoid black borders"
#
# The default is therefore already a black-free crop -- a conservative one. Any
# comparison that wants a rectified frame WITH black regions must set 0.262.
FOCAL_OUT_CIRCUMSCRIBED = 0.262
FOCAL_OUT_INSCRIBED = 0.371


def _kb4_theta_d(theta: np.ndarray, D: np.ndarray) -> np.ndarray:
    """KB4 forward: theta -> distorted radius (in focal-length units)."""
    t2 = theta * theta
    return theta * (1.0 + D[0] * t2 + D[1] * t2**2 + D[2] * t2**3 + D[3] * t2**4)


def kb4_max_incidence(D: np.ndarray, n: int = 8192) -> float:
    """Largest incidence angle the lens actually images, in radians.

    The KB4 polynomial is only invertible up to its turnover -- the first theta
    where d(theta_d)/d(theta) <= 0. Past it the mapping folds back on itself and
    "rectifying" those directions produces garbage rather than a wider view, so
    the turnover is the true edge of the imaged cone.
    """
    th = np.linspace(0.0, np.pi / 2.0, n)
    td = _kb4_theta_d(th, D)
    drop = np.nonzero(np.diff(td) <= 0)[0]
    return float(th[drop[0]]) if len(drop) else float(th[-1])


class FisheyeRectifier:
    """Callable HWC-float -> HWC-float pinhole rectifier with per-size map cache."""

    def __init__(self, preset: str = "none", fisheye_k: str = "", fisheye_d: str = "",
                 focal_out_norm: Optional[float] = None, fill: str = "black") -> None:
        self.preset = preset
        self.fisheye_k = fisheye_k
        self.fisheye_d = fisheye_d
        # focal_out_norm overrides the preset's output focal (see the table above).
        # Lower = wider = more of the imaged cone kept = more black to fill.
        self.focal_out_norm = focal_out_norm
        # What goes in the invalid region. "black" reproduces the historical
        # behaviour exactly; see finetune.data.fill for the alternatives.
        self.fill = fill
        self._maps: Dict[Tuple[int, int], tuple] = {}
        self._valid: Dict[tuple, np.ndarray] = {}

    def _intrinsics(self, H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.fisheye_k:
            fx, fy, cx, cy = (float(v) for v in self.fisheye_k.split(","))
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], np.float64)
        elif self.preset == "aria-214-1":
            # rotated=True: every ADT loader in this repo applies np.rot90(k=3)
            # before anything else touches the frame.
            fx, fy, cx, cy = _aria(H, W, rotated=True)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], np.float64)
        else:
            f = 0.43 * max(H, W)
            K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]], np.float64)

        if self.fisheye_d:
            D = np.array([float(v) for v in self.fisheye_d.split(",")], np.float64)
        elif self.preset == "aria-214-1":
            D = _ARIA_214_1_D_KB4.copy()
        else:
            D = np.zeros(4, np.float64)

        fon = self.focal_out_norm
        if fon is None:
            fon = _ARIA_214_1_FOCAL_OUT_NORM if self.preset == "aria-214-1" else 0.5
        focal_out = fon * max(H, W)
        Knew = np.array([[focal_out, 0, W / 2.0], [0, focal_out, H / 2.0], [0, 0, 1.0]], np.float64)
        return K, D.reshape(4, 1), Knew

    def _get_maps(self, H: int, W: int):
        if (H, W) not in self._maps:
            import cv2

            K, D4, Knew = self._intrinsics(H, W)
            self._maps[(H, W)] = cv2.fisheye.initUndistortRectifyMap(
                K, D4, np.eye(3, dtype=np.float64), Knew, (W, H), cv2.CV_16SC2
            )
        return self._maps[(H, W)]

    def valid_mask(self, H: int, W: int) -> np.ndarray:
        """``(H, W)`` bool: output pixels that have a real source pixel behind them.

        Two conditions, and BOTH matter:

        1. the ray is inside the imaged cone (``theta <= theta_max``);
        2. the source pixel it maps to is inside the source frame.

        Condition 2 is not redundant. The Aria disc has radius
        ``f * theta_d(theta_max)`` ~= 271 px on a 512 px frame, so the disc is
        *clipped by the sensor's square edges*: along the axes the image stops at
        256 px, well before theta_max. Directions between those two radii are
        inside the cone yet were never imaged. Dropping this check leaves ~10% of
        a circumscribed frame marked valid-but-black -- which a fill would then
        skip, silently leaving black in the cell that is supposed to have none.
        (``VGGT-360-fisheye/utils/fisheye_views.py`` guards the same way, with the
        same half-pixel tolerance for the off-centre principal point.)

        Computed analytically rather than by testing the remapped pixel for
        blackness: past the KB4 turnover the undistort map folds back and samples
        real-but-wrong pixels instead of leaving the frame, so an ``rgb.sum() > 0``
        test would mark garbage as valid -- and would also misfire on genuinely
        dark scene content and on vignetting.
        """
        if (H, W) not in self._valid:
            K, D4, Knew = self._intrinsics(H, W)
            D = D4.reshape(-1)
            theta_max = kb4_max_incidence(D)
            ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
            xn = (xs - Knew[0, 2]) / Knew[0, 0]
            yn = (ys - Knew[1, 2]) / Knew[1, 1]
            r = np.sqrt(xn * xn + yn * yn)
            theta = np.arctan(r)
            # KB4 forward -> source pixel, so we can bounds-check it.
            theta_d = _kb4_theta_d(theta, D)
            inv = np.divide(1.0, r, out=np.zeros_like(r), where=r > 1e-12)
            u = K[0, 2] + K[0, 0] * theta_d * xn * inv
            v = K[1, 2] + K[1, 1] * theta_d * yn * inv
            self._valid[(H, W)] = (
                (theta <= theta_max)
                & (u > -0.5) & (u < W - 0.5)
                & (v > -0.5) & (v < H - 0.5)
            )
        return self._valid[(H, W)]

    def source_valid_mask(self, H: int, W: int) -> np.ndarray:
        """``(H, W)`` bool: validity in the RAW fisheye frame (before rectification).

        The sensor is square but the lens images a disc, so the four corners hold
        no image. The disc's pixel radius is ``f * theta_d(theta_max)`` about the
        principal point -- note this uses the *distorted* radius, since that is
        where the imaged cone actually lands on the sensor.
        """
        key = ("src", H, W)
        if key not in self._valid:
            K, D4, _ = self._intrinsics(H, W)
            D = D4.reshape(-1)
            theta_max = kb4_max_incidence(D)
            r_max = float(K[0, 0] * _kb4_theta_d(np.array([theta_max]), D)[0])
            ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
            r = np.hypot(xs - K[0, 2], ys - K[1, 2])
            self._valid[key] = r <= r_max
        return self._valid[key]

    def geometry(self, H: int, W: int) -> Dict[str, float]:
        """Reportable geometry of this rectification: FoV, black fraction, cone kept."""
        _, D4, Knew = self._intrinsics(H, W)
        theta_max = kb4_max_incidence(D4.reshape(-1))
        valid = self.valid_mask(H, W)
        fx, fy = Knew[0, 0], Knew[1, 1]
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        xn = (xs - Knew[0, 2]) / fx
        yn = (ys - Knew[1, 2]) / fy
        # Solid angle of one pinhole pixel: dA_normalised / (1 + xn^2 + yn^2)^(3/2),
        # with dA_normalised = (1/fx)(1/fy). The 1/cos^3 factor is why the black
        # wedges are 21.5% of the PIXELS but only ~6.7% of the SOLID ANGLE.
        w = (1.0 / (fx * fy)) / (1.0 + xn * xn + yn * yn) ** 1.5
        cone_sr = 2.0 * np.pi * (1.0 - np.cos(theta_max))
        return {
            "theta_max_deg": float(np.degrees(theta_max)),
            "hfov_deg": float(np.degrees(2.0 * np.arctan(0.5 * W / fx))),
            "diag_fov_deg": float(np.degrees(2.0 * np.arctan(0.5 * np.hypot(W, H) / fx))),
            "black_frac_px": float(1.0 - valid.mean()),
            "black_frac_sr": float(1.0 - w[valid].sum() / w.sum()),
            "cone_kept": float(w[valid].sum() / cone_sr),
        }

    def __call__(self, img_hwc: np.ndarray) -> np.ndarray:
        """img_hwc: float32 HxWx3 in [0,1] -> rectified float32 HxWx3 in [0,1].

        Invalid pixels (outside the imaged cone) are filled per ``self.fill``.
        """
        import cv2

        H, W = img_hwc.shape[:2]
        map1, map2 = self._get_maps(H, W)
        out = cv2.remap(img_hwc, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        out = np.ascontiguousarray(out, dtype=np.float32)
        # Applied unconditionally: even for fill="black" this is not a no-op, since
        # BORDER_CONSTANT only zeroes rays that leave the frame, while rays past the
        # KB4 turnover fold back and sample real (wrong) pixels. The analytic mask
        # catches those; the remap's own border handling cannot.
        from .fill import apply_fill
        out = apply_fill(out, self.valid_mask(H, W), self.fill)
        return np.ascontiguousarray(out, dtype=np.float32)

    def rectify_depth(self, depth_hw: np.ndarray) -> np.ndarray:
        """depth_hw: float32 HxW -> rectified float32 HxW (same units).

        Uses the SAME maps as the RGB path so a rectified depth map stays pixel-
        aligned with its rectified image. NEAREST interpolation (never blend depth
        across discontinuities into spurious values) and a 0 border so pixels that
        fall outside the fisheye field of view become 0 == invalid. The depth value
        itself is preserved: rectification shares the camera centre/axis (identity
        rotation), so z-depth / range is unchanged — only the pixel grid is remapped.
        """
        import cv2

        H, W = depth_hw.shape[:2]
        map1, map2 = self._get_maps(H, W)
        out = cv2.remap(
            depth_hw.astype(np.float32), map1, map2,
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        return np.ascontiguousarray(out, dtype=np.float32)


def looks_like_fisheye(clip_pattern: str, data_root: str) -> bool:
    """Heuristic used only to decide whether to warn about missing rectification."""
    hay = (clip_pattern + " " + data_root).lower()
    return ("214-1" in hay) or ("aria" in hay)
