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

from typing import Dict, Tuple

import numpy as np

# Aria RGB 214-1 calibration (native 1408x1408, KB4), as validated in test_run.py.
_ARIA_214_1_D_KB4 = np.array([0.3852, -0.4442, 0.5591, -0.3254], np.float64)
_ARIA_214_1_F_NORM = 610.94 / 1408.0
_ARIA_214_1_CX_NORM = 715.11 / 1408.0
_ARIA_214_1_CY_NORM = 716.71 / 1408.0
# After the standard 90 deg CW storage rotation:
_ARIA_214_1_CX_ROT_NORM = 1.0 - _ARIA_214_1_CY_NORM
_ARIA_214_1_CY_ROT_NORM = _ARIA_214_1_CX_NORM
_ARIA_214_1_FOCAL_OUT_NORM = 0.55  # output focal / max(H,W): avoids black borders


class FisheyeRectifier:
    """Callable HWC-float -> HWC-float pinhole rectifier with per-size map cache."""

    def __init__(self, preset: str = "none", fisheye_k: str = "", fisheye_d: str = "") -> None:
        self.preset = preset
        self.fisheye_k = fisheye_k
        self.fisheye_d = fisheye_d
        self._maps: Dict[Tuple[int, int], tuple] = {}

    def _intrinsics(self, H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.fisheye_k:
            fx, fy, cx, cy = (float(v) for v in self.fisheye_k.split(","))
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], np.float64)
        elif self.preset == "aria-214-1":
            f = _ARIA_214_1_F_NORM * max(H, W)
            cx = _ARIA_214_1_CX_ROT_NORM * W
            cy = _ARIA_214_1_CY_ROT_NORM * H
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]], np.float64)
        else:
            f = 0.43 * max(H, W)
            K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]], np.float64)

        if self.fisheye_d:
            D = np.array([float(v) for v in self.fisheye_d.split(",")], np.float64)
        elif self.preset == "aria-214-1":
            D = _ARIA_214_1_D_KB4.copy()
        else:
            D = np.zeros(4, np.float64)

        focal_out = (_ARIA_214_1_FOCAL_OUT_NORM if self.preset == "aria-214-1" else 0.5) * max(H, W)
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

    def __call__(self, img_hwc: np.ndarray) -> np.ndarray:
        """img_hwc: float32 HxWx3 in [0,1] -> rectified float32 HxWx3 in [0,1]."""
        import cv2

        H, W = img_hwc.shape[:2]
        map1, map2 = self._get_maps(H, W)
        out = cv2.remap(img_hwc, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
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
