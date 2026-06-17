# Copyright (c) 2026.
"""Egocentric video dataset: sliding windows of frames, no labels.

Expected layout (frames already extracted)::

    data_root/
        clip_0001/   frame_000000.jpg frame_000001.jpg ...
        clip_0002/   ...

Each item is a window of ``seq_len`` frames sampled with ``stride``, returned as
``images [S,3,H,W]`` in ``[0,1]``. For Aria / fisheye captures, rectify to a
pinhole model upstream (the model and losses assume the pinhole intrinsics that
VGGT-Omega predicts).
"""
from __future__ import annotations

import os
import re
from fnmatch import fnmatch
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _list_frames(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(_IMG_EXTS)]
    files.sort(key=_natural_key)
    return [os.path.join(folder, f) for f in files]


def _round_to_patch(value: float, patch: int) -> int:
    return max(patch, int(round(value / patch)) * patch)


def _target_hw(w: int, h: int, resolution: int, patch: int) -> Tuple[int, int]:
    # longest side -> resolution, keep aspect, round to patch multiple
    if h >= w:
        th = resolution
        tw = _round_to_patch(resolution * w / max(h, 1), patch)
    else:
        tw = resolution
        th = _round_to_patch(resolution * h / max(w, 1), patch)
    return th, tw


class EgocentricVideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        seq_len: int = 8,
        stride: int = 2,
        image_resolution: int = 512,
        patch_size: int = 16,
        window_stride: int = 1,
        clip_pattern: str = "",
        rectifier=None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride
        self.resolution = image_resolution
        self.patch = patch_size
        # Optional fisheye->pinhole rectifier (FisheyeRectifier or any HWC-float
        # callable). The losses assume pinhole; leave None only if frames are
        # already rectified (e.g. prepare_egoexo4d.py --undistort).
        self.rectifier = rectifier
        # Gap (in frames) between consecutive window starts. 1 = maximally
        # overlapping (each frame appears in up to `span` windows — heavy
        # redundancy); set to seq_len*stride for non-overlapping windows.
        self.window_stride = max(1, window_stride)
        self.clip_pattern = clip_pattern

        # A "clip" is any directory (at any depth) that directly contains image
        # frames. This handles a flat folder, data_root/clip/*.jpg, and nested
        # layouts like Ego-Exo4D <root>/train/<seq>/aria01_214-1/*.jpg.
        #
        # An Ego-Exo4D take holds MANY cameras side by side (egocentric RGB
        # aria*_214-1, Aria SLAM grayscale *_1201-1/2, exocentric cam*/gp*); set
        # clip_pattern (fnmatch glob, e.g. "*214-1") to keep only the egocentric
        # RGB stream. A dir qualifies if ANY of its path components (relative to
        # data_root) matches the pattern; empty pattern = accept every dir.
        clips, skipped = [], 0
        for dirpath, dirnames, filenames in os.walk(data_root):
            dirnames.sort()
            if not any(f.lower().endswith(_IMG_EXTS) for f in filenames):
                continue
            if clip_pattern:
                rel = os.path.relpath(dirpath, data_root)
                parts = [p for p in rel.split(os.sep) if p not in ("", ".")]
                if not any(fnmatch(p, clip_pattern) for p in parts):
                    skipped += 1
                    continue
            clips.append(dirpath)
        clips.sort()
        self.clips = clips
        self.num_skipped_dirs = skipped
        if not clips:
            raise FileNotFoundError(
                f"No image clips under {data_root!r} matching clip_pattern={clip_pattern!r} "
                f"({skipped} dirs with images were skipped by the filter; "
                f"pass clip_pattern='' to disable filtering)"
            )

        self.windows: List[Tuple[str, List[str]]] = []
        span = (seq_len - 1) * stride + 1
        for clip in clips:
            frames = _list_frames(clip)
            for start in range(0, max(1, len(frames) - span + 1), self.window_stride):
                window = frames[start : start + span : stride]
                if len(window) == seq_len:
                    self.windows.append((clip, window))
        if not self.windows:
            raise ValueError(
                f"No windows of length {seq_len} (stride {stride}) could be built from {data_root!r}"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def _load(self, path: str, th: int, tw: int) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((tw, th), Image.BICUBIC)
            arr_np = _to_float_array(im)            # [H,W,3] in [0,1]
        if self.rectifier is not None:
            arr_np = self.rectifier(arr_np)         # fisheye -> pinhole
        return torch.from_numpy(arr_np).permute(2, 0, 1).contiguous()  # [3,H,W]

    def __getitem__(self, idx: int):
        clip, paths = self.windows[idx]
        with Image.open(paths[0]) as im0:
            w, h = im0.size
        th, tw = _target_hw(w, h, self.resolution, self.patch)
        frames = [self._load(p, th, tw) for p in paths]
        images = torch.stack(frames, dim=0)  # [S,3,H,W]
        return {"images": images, "clip": clip, "paths": paths}


def _to_float_array(im: "Image.Image"):
    import numpy as np

    return np.asarray(im, dtype="float32") / 255.0


def collate_windows(batch):
    images = torch.stack([b["images"] for b in batch], dim=0)  # [B,S,3,H,W]
    return {"images": images, "clips": [b["clip"] for b in batch]}


def random_egocentric_batch(batch_size=1, seq_len=6, height=64, width=96, seed=0):
    """Smooth random window for tests (no files needed)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(batch_size, 1, 3, height, width, generator=g)
    # low-frequency content + per-frame jitter so warping is non-degenerate
    drift = torch.linspace(0, 0.1, seq_len).view(1, seq_len, 1, 1, 1)
    imgs = (base + drift + 0.02 * torch.rand(batch_size, seq_len, 3, height, width, generator=g)).clamp(0, 1)
    return {"images": imgs}
