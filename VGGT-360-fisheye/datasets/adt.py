# Copyright (c) 2026.
"""ADT (Aria Digital Twin) single-frame fisheye loader for VGGT-360-fisheye.

The upstream VGGT-360 ``main.py`` imports ``from Datasets import Stanford2D3D``
— a module the upstream release does not actually ship — so a loader had to be
written from scratch anyway.  This one targets the ADT layout (same convention
as this repo's ``finetune/eval/adt_depth.ADTWindowDataset``, which it mirrors
in the essentials):

    <seq_dir>/
      videos_synthetic/  *.jpg|*.png   rendered RGB, pixel-aligned with GT depth
                                       (the default eval stream)
      videos_rgb/        *.jpg|*.png   real-sensor RGB (NOT perfectly registered
                                       to GT — use for qualitative runs)
      depth_npy/         *.npy         GT depth, uint16 millimetres

Key conventions (identical to the rest of this repo):
  * Aria frames are stored rotated 90 deg CW -> a 270 deg CCW ``np.rot90(k=3)``
    is applied to RGB and depth, and the intrinsics use
    ``aria_intrinsics(rotated=True)`` which swaps fx/fy and moves the
    principal point accordingly.
  * Depth is uint16 mm -> metres via ``depth_scale=0.001``; validity is
    ``0 < d <= depth_max_m``.
  * NO rectification and NO resizing: this pipeline consumes the raw fisheye
    frame at native resolution (perspective views are sampled from it with
    full acuity), unlike ``ADTWindowDataset`` which resizes/rectifies for
    pinhole models.  Optional ``working_size`` downsamples fisheye+depth
    together (nearest for depth) if memory-bound.

RGB<->depth pairing is anchored on the depth files (the limiting set) and
matched by exact stem, falling back to the ``frame_XXXXXX`` id when the
timestamp suffix differs between streams — the same logic as ``adt_depth.py``.

Plain-python iterable (no torch dependency): each item is a dict of numpy
arrays, so the geometry checker can reuse the loader on machines without a
usable torch install.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_FRAME_ID_RE = re.compile(r"frame[_-]?(\d+)")


def _frame_id(path: str) -> Optional[str]:
    """``frame_000400_87552837212850.jpg`` -> ``"400"`` (zero-pad-insensitive)."""
    name = os.path.splitext(os.path.basename(path))[0]
    m = _FRAME_ID_RE.search(name) or re.search(r"(\d+)", name)
    return str(int(m.group(1))) if m else None


def find_adt_sequences(adt_root: str,
                       rgb_subdir: str = "videos_synthetic",
                       depth_subdir: str = "depth_npy") -> List[str]:
    """Discover sequence dirs under ``adt_root`` that have both streams."""
    seqs = []
    for d in sorted(glob.glob(os.path.join(adt_root, "*"))):
        if (os.path.isdir(os.path.join(d, rgb_subdir))
                and os.path.isdir(os.path.join(d, depth_subdir))):
            seqs.append(d)
    return seqs


def _pair_frames(rgb_dir: str, depth_dir: str) -> List[Tuple[str, str]]:
    """Pair depth maps (the limiting set) with their RGB frames."""
    all_rgb = sorted(sum((glob.glob(os.path.join(rgb_dir, ext))
                          for ext in ("*.png", "*.jpg", "*.jpeg")), []))
    all_depth = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))

    rgb_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in all_rgb}
    rgb_by_id: Dict[str, str] = {}
    for p in all_rgb:
        fid = _frame_id(p)
        if fid is not None and fid not in rgb_by_id:
            rgb_by_id[fid] = p

    pairs, seen = [], set()
    for dp in all_depth:
        stem = os.path.splitext(os.path.basename(dp))[0]
        fid = _frame_id(dp)
        rp = rgb_by_stem.get(stem) or (rgb_by_id.get(fid) if fid else None)
        if rp is None:
            continue
        key = fid or stem
        if key in seen:
            continue
        seen.add(key)
        pairs.append((rp, dp))
    return pairs


class ADTFisheyeFrames:
    """Iterable of single raw-fisheye ADT frames with GT depth.

    Parameters
    ----------
    seq_dirs     : sequence directories (see ``find_adt_sequences``).
    rgb_subdir   : ``videos_synthetic`` (GT-aligned, default) or ``videos_rgb``.
    depth_subdir : GT depth dir (uint16 mm .npy).
    depth_scale  : raw GT multiplier (0.001: mm -> m).
    depth_max_m  : validity ceiling in metres (ADT interiors: 10 m is safe).
    rotation_k   : ``np.rot90`` count; 3 = 270 deg CCW = the ADT convention.
    max_frames   : cap per sequence (None = all).
    frame_stride : take every Nth paired frame.
    working_size : optional square size to downsample fisheye+GT to (e.g. 704);
                   None keeps native 1408.

    Item dict
    ---------
    "rgb"       : (H, W, 3) uint8 — upright raw fisheye frame
    "depth"     : (H, W) float32 metres — GT (0 where missing)
    "valid"     : (H, W) bool — GT validity
    "rgb_path"  : source path (for debugging / titles)
    """

    def __init__(self,
                 seq_dirs: List[str],
                 rgb_subdir: str = "videos_synthetic",
                 depth_subdir: str = "depth_npy",
                 depth_scale: float = 0.001,
                 depth_max_m: float = 10.0,
                 rotation_k: int = 3,
                 max_frames: Optional[int] = None,
                 frame_stride: int = 1,
                 working_size: Optional[int] = None) -> None:
        self.depth_scale = depth_scale
        self.depth_max_m = depth_max_m
        self.rotation_k = rotation_k
        self.working_size = working_size

        self.frames: List[Tuple[str, str]] = []
        for seq in seq_dirs:
            pairs = _pair_frames(os.path.join(seq, rgb_subdir),
                                 os.path.join(seq, depth_subdir))
            pairs = pairs[::max(1, frame_stride)]
            if max_frames is not None:
                pairs = pairs[:max_frames]
            print(f"  [ADT] {len(pairs)} frames <- {seq}")
            self.frames.extend(pairs)
        if not self.frames:
            raise RuntimeError(
                f"No paired frames found (rgb_subdir={rgb_subdir!r}, "
                f"depth_subdir={depth_subdir!r}) under: {seq_dirs}")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        rgb_path, depth_path = self.frames[idx]

        with Image.open(rgb_path) as im:
            rgb = np.array(im.convert("RGB"), dtype=np.uint8)
        d = np.load(depth_path).astype(np.float32)
        if d.ndim == 3:
            d = d.squeeze(-1)
        d = np.where(np.isfinite(d), d, 0.0) * self.depth_scale

        if self.rotation_k:
            rgb = np.rot90(rgb, k=self.rotation_k).copy()
            d = np.rot90(d, k=self.rotation_k).copy()

        if self.working_size is not None and rgb.shape[0] != self.working_size:
            s = self.working_size
            rgb = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA)
            d = cv2.resize(d, (s, s), interpolation=cv2.INTER_NEAREST)

        valid = (d > 0) & (d <= self.depth_max_m)
        return {"rgb": rgb, "depth": d, "valid": valid, "rgb_path": rgb_path}
