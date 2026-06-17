# Copyright (c) 2026.
"""Visualization helpers shared by training and the test-run tools.

OpenCV-only (no matplotlib) so it works on headless training nodes. All
functions accept either torch tensors or numpy arrays for the image/depth
inputs and return ``[H,W,3]`` uint8 BGR arrays ready for ``cv2.imwrite``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:  # torch is always present in this package, but keep viz importable without it
    import torch

    _Tensor = torch.Tensor
except Exception:  # pragma: no cover
    torch = None
    _Tensor = ()  # isinstance(x, ()) is always False


def _to_numpy(x):
    if torch is not None and isinstance(x, _Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def chw_to_bgr(img) -> np.ndarray:
    """``[3,H,W]`` float in ``[0,1]`` RGB -> ``[H,W,3]`` uint8 BGR."""
    import cv2

    arr = _to_numpy(img)
    rgb = (np.clip(arr, 0, 1).transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def colorize_depth(
    depth_hw, vmin: Optional[float] = None, vmax: Optional[float] = None
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """``[H,W]`` depth -> (``[H,W,3]`` BGR uint8 TURBO, (vmin, vmax, median)).

    Percentile (2/98) normalization per frame unless ``vmin``/``vmax`` given.
    Invalid (non-finite or <= 0) pixels are painted black.
    """
    import cv2

    d = _to_numpy(depth_hw).astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.zeros((*d.shape, 3), np.uint8), (0.0, 0.0, 0.0)
    lo = float(np.percentile(d[valid], 2)) if vmin is None else vmin
    hi = float(np.percentile(d[valid], 98)) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1e-6
    u8 = (np.clip((d - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color, (lo, hi, float(np.median(d[valid])))


def mask_to_bgr(mask_hw) -> np.ndarray:
    """``[H,W]`` in ``[0,1]`` -> grayscale BGR (white = 1 = kept / static)."""
    m = _to_numpy(mask_hw).astype(np.float32)
    u8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    return np.repeat(u8[:, :, None], 3, axis=2)


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Overlay a small caption strip in the top-left corner."""
    import cv2

    out = img.copy()
    cv2.rectangle(out, (0, 0), (max(120, 9 * len(text)), 20), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def training_montage(
    images,
    vggt_depth,
    dav2_depth,
    dyn_mask=None,
    num_frames: int = 4,
    caption: str = "",
) -> np.ndarray:
    """Build a grid montage for one clip.

    Rows are frames (evenly spaced across the window), columns are
    ``input | VGGT depth | DAv2 depth | dyn-mask``. Shapes:
        images:     ``[S,3,H,W]`` float [0,1]
        vggt_depth: ``[S,H,W]``
        dav2_depth: ``[S,H,W]``
        dyn_mask:   ``[S,H,W]`` in [0,1] or None
    """
    S = images.shape[0]
    n = min(num_frames, S)
    idxs = np.linspace(0, S - 1, n).round().astype(int).tolist()

    rows = []
    for s in idxs:
        inp = label(chw_to_bgr(images[s]), f"input f{s}")
        v_col, v_stats = colorize_depth(vggt_depth[s])
        d_col, d_stats = colorize_depth(dav2_depth[s])
        panels = [
            inp,
            label(v_col, f"VGGT {v_stats[0]:.2f}-{v_stats[1]:.2f}"),
            label(d_col, f"DAv2 {d_stats[0]:.2f}-{d_stats[1]:.2f}"),
        ]
        if dyn_mask is not None:
            panels.append(label(mask_to_bgr(dyn_mask[s]), "dyn-mask"))
        rows.append(np.hstack(panels))
    montage = np.vstack(rows)

    if caption:
        bar = np.zeros((24, montage.shape[1], 3), np.uint8)
        montage = np.vstack([label(bar, caption), montage])
    return montage
