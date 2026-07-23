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
_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg")


def _frame_id(path: str) -> Optional[str]:
    """``frame_000400_87552837212850.jpg`` -> ``"400"`` (zero-pad-insensitive)."""
    name = os.path.splitext(os.path.basename(path))[0]
    m = _FRAME_ID_RE.search(name) or re.search(r"(\d+)", name)
    return str(int(m.group(1))) if m else None


def _has_files(d: str) -> bool:
    """True iff ``d`` is a dir containing at least one image or .npy file.

    Used by ``find_adt_sequences`` so an EMPTY ``videos_synthetic/`` (which
    passes ``os.path.isdir``) is treated as "stream not available" instead of
    being discovered and then silently yielding 0 frames.
    """
    if not os.path.isdir(d):
        return False
    for ext in _IMG_EXTS + ("*.npy",):
        if glob.glob(os.path.join(d, ext)):
            return True
    return False


def _rgb_ids(rgb_dir: str) -> set:
    """Set of frame-ids that have an RGB image in ``rgb_dir``."""
    ids = set()
    for ext in _IMG_EXTS:
        for p in glob.glob(os.path.join(rgb_dir, ext)):
            fid = _frame_id(p)
            if fid is not None:
                ids.add(fid)
    return ids


def common_frame_ids(seq_dir: str, rgb_subdirs: List[str],
                     depth_subdir: str = "depth_npy") -> set:
    """Frame-ids present in GT depth AND in every listed RGB stream.

    Pass the result as ``ADTFisheyeFrames(frame_ids=...)`` to force a
    synthetic-vs-real comparison onto the IDENTICAL frame set (so the only
    difference between the two runs is the pixels, i.e. blur — not which
    frames were scored).
    """
    ids = {_frame_id(p) for p in glob.glob(os.path.join(seq_dir, depth_subdir, "*.npy"))}
    ids.discard(None)
    for sub in rgb_subdirs:
        ids &= _rgb_ids(os.path.join(seq_dir, sub))
    return ids


def find_adt_sequences(adt_root: str,
                       rgb_subdir: str = "videos_synthetic",
                       depth_subdir: str = "depth_npy",
                       require_subdirs: Optional[List[str]] = None,
                       verbose: bool = True) -> List[str]:
    """Discover usable sequence dirs under ``adt_root``, transparently.

    A sequence is usable iff it has the depth dir AND the requested
    ``rgb_subdir`` AND every dir in ``require_subdirs`` — each present **and
    non-empty** (an empty ``videos_synthetic/`` used to pass ``os.path.isdir``
    and then silently contribute 0 frames).

    ``require_subdirs`` lets a caller demand extra streams so the sequence set
    is IDENTICAL regardless of which one is being rendered — e.g. pass
    ``["videos_synthetic", "videos_rgb"]`` for a fair synthetic-vs-real
    (blur) A/B: both runs then evaluate exactly the sequences that have both.

    Every candidate is logged as included or skipped-with-reason (``verbose``),
    so a stream that is missing for some sequences is never dropped silently —
    the root cause of "videos_rgb uses all sequences but videos_synthetic does
    not".
    """
    needed = [depth_subdir, rgb_subdir] + list(require_subdirs or [])
    seqs, skipped = [], []
    for d in sorted(glob.glob(os.path.join(adt_root, "*"))):
        if not os.path.isdir(d):
            continue
        missing = [s for s in needed if not _has_files(os.path.join(d, s))]
        if missing:
            skipped.append((os.path.basename(d), missing))
        else:
            seqs.append(d)
    if verbose:
        print(f"  [ADT] {len(seqs)} usable sequence(s) "
              f"(need: {', '.join(needed)})")
        for name, missing in skipped:
            print(f"  [ADT]   SKIP {name}: missing/empty {missing}")
    return seqs


def _pair_frames(rgb_dir: str, depth_dir: str
                 ) -> Tuple[List[Tuple[str, str, str]], int]:
    """Pair depth maps (the anchor set) with their RGB frames.

    Anchored on the GT depth files in sorted (frame-id) order, so the pair
    ORDER is the depth order — identical across RGB streams.  Returns
    ``(pairs, n_depth)`` where each pair is ``(rgb_path, depth_path,
    frame_id)`` and ``n_depth`` is the total depth-map count (to report how
    many had no RGB partner in this stream).
    """
    all_rgb = sorted(sum((glob.glob(os.path.join(rgb_dir, ext))
                          for ext in _IMG_EXTS), []))
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
        pairs.append((rp, dp, key))
    return pairs, len(all_depth)


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
    frame_ids    : optional set of frame-ids to keep (global filter).
                   Applied BEFORE stride/max_frames.
    common_streams : optional list of OTHER rgb subdirs; per sequence, keep only
                   frame-ids present in GT depth AND ``rgb_subdir`` AND every one
                   of these (computed PER sequence, so frame-id collisions
                   between sequences can't leak). This is the airtight way to
                   make a synthetic-vs-real (blur) A/B score identical frames:
                   pass the other stream here in both runs.
    working_size : optional square size to downsample fisheye+GT to (e.g. 704);
                   None keeps native 1408.

    Determinism: frames are selected in GT-depth (frame-id) order, so the same
    ``frame_stride``/``max_frames`` pick the same frame-ids across RGB streams
    (given both streams contain them — otherwise pass ``frame_ids``).

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
                 frame_ids: Optional[set] = None,
                 common_streams: Optional[List[str]] = None,
                 working_size: Optional[int] = None) -> None:
        self.depth_scale = depth_scale
        self.depth_max_m = depth_max_m
        self.rotation_k = rotation_k
        self.working_size = working_size

        # store (rgb_path, depth_path); frame_id kept only for logging
        self.frames: List[Tuple[str, str]] = []
        for seq in seq_dirs:
            name = os.path.basename(seq)
            all_pairs, n_depth = _pair_frames(os.path.join(seq, rgb_subdir),
                                              os.path.join(seq, depth_subdir))
            no_partner = n_depth - len(all_pairs)   # depth maps lacking RGB

            pairs = all_pairs
            if common_streams:
                cids = common_frame_ids(
                    seq, [rgb_subdir] + list(common_streams), depth_subdir)
                pairs = [p for p in pairs if p[2] in cids]
            if frame_ids is not None:
                pairs = [p for p in pairs if p[2] in frame_ids]
            pairs = pairs[::max(1, frame_stride)]
            if max_frames is not None:
                pairs = pairs[:max_frames]

            note = f" ({no_partner} depth maps had no RGB partner)" if no_partner else ""
            print(f"  [ADT] {len(pairs)} frames <- {name}{note}")
            if not pairs:
                print(f"  [ADT]   WARN {name} yielded 0 frames for "
                      f"rgb_subdir={rgb_subdir!r} — excluded from evaluation")
            self.frames.extend((p[0], p[1]) for p in pairs)
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
