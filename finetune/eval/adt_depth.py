# Copyright (c) 2026.
"""ADT (Aria Digital Twin) dense-GT depth evaluation for VGGT-Omega.

ADT provides Aria frames with paired dense GT depth maps (from the digital-twin
simulation), giving absolute-accuracy eval on the same fisheye camera that
Ego-Exo4D uses.

Expected ADT sequence layout
-----------------------------
  <seq_dir>/
    videos_synthetic/  *.jpg | *.png  (rendered RGB, used by default — pixel-aligned
                                        with the rendered GT depth; ~1408×1408)
    videos_rgb/        *.jpg | *.png  (real-sensor RGB — NOT perfectly registered to
                                        the GT depth; opt in via rgb_subdir=videos_rgb)
    depth_npy/         *.npy          (GT depth maps, matching stems; uint16 millimetres)
    groundtruth/
      aria_trajectory.csv            (GT camera pose: world-from-device, XYZW, us timestamps)

Important: Aria frames are rotated 90° CW in storage; apply 270° CCW rotation before
use (same as ADT evaluation convention). Depth values are in millimetres — pass
depth_scale=0.001 to convert to metres (the default). The RGB and GT depth are both
fisheye (KB4); rectify to pinhole (default) so the pinhole-trained model is fed
pinhole input and scored against pinhole GT (set rectify=False for raw fisheye).

Dataset and eval runner
-----------------------
  ADTWindowDataset  -- consecutive windows of seq_len frames with GT depth
  run_adt_eval      -- runs VGGT-Omega, compares predicted vs GT depth (and poses)
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .metrics import (
    aggregate_metrics,
    align_depth,
    depth_metrics,
    pose_ate_rpe,
    print_depth_summary,
    print_pose_summary,
)


# --------------------------------------------------------------------------- #
# Frame collection helpers
# --------------------------------------------------------------------------- #

# ADT filenames look like ``frame_000400_87552837212850.jpg`` — the zero-padded
# integer right after ``frame`` is the *frame id*, the stable key shared between
# the RGB frame and its GT depth.  The trailing token is the capture timestamp,
# which is NOT guaranteed to be identical between videos_rgb/ and depth_npy/, so
# matching on the full stem can silently drop frames.  We sync on the frame id.
_FRAME_ID_RE = re.compile(r"frame[_-]?(\d+)")


def _frame_id(path: str) -> Optional[str]:
    """Extract the frame id from a filename, normalized to a canonical integer.

    ``frame_000400_87552837212850.jpg`` -> ``"400"``.  Normalizing via ``int``
    makes zero-padded and unpadded ids compare equal (``frame_000400`` matches
    ``frame_400``).  Falls back to the first integer run if there's no ``frame``
    prefix, and to None if the name has no digits at all.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    m = _FRAME_ID_RE.search(name) or re.search(r"(\d+)", name)
    return str(int(m.group(1))) if m else None


def _collect_paired_frames(rgb_dir: str, depth_dir: str) -> List[Tuple[str, str]]:
    """Pair GT depth maps with their RGB frame, anchored on the depth range.

    GT depth typically covers a *subset* of the RGB stream (e.g. RGB starts at
    frame_000000 while depth starts at frame_000282), so we iterate over the
    depth files — the limiting set — and find each one's RGB partner.  Overlapping
    frames share the same filename, so the match is by exact stem; a frame-id
    fallback covers a differing timestamp suffix just in case.

    Frames are returned in depth (frame-id) order; duplicates are dropped.
    """
    all_rgb = sorted(
        glob.glob(os.path.join(rgb_dir, "*.png"))
        + glob.glob(os.path.join(rgb_dir, "*.jpg"))
        + glob.glob(os.path.join(rgb_dir, "*.jpeg"))
    )
    all_depth = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))

    # Index RGB files by stem and by frame id.
    rgb_by_stem: Dict[str, str] = {}
    rgb_by_id: Dict[str, str] = {}
    for rp in all_rgb:
        rgb_by_stem[os.path.splitext(os.path.basename(rp))[0]] = rp
        fid = _frame_id(rp)
        if fid is not None:
            rgb_by_id.setdefault(fid, rp)  # first wins (stable order)

    pairs: List[Tuple[str, str]] = []
    seen_ids: set = set()
    n_stem, n_id = 0, 0
    for depth_path in all_depth:                      # anchor on the depth range
        stem = os.path.splitext(os.path.basename(depth_path))[0]
        fid = _frame_id(depth_path)
        rgb_path = rgb_by_stem.get(stem)
        if rgb_path is not None:
            n_stem += 1
        elif fid is not None:
            rgb_path = rgb_by_id.get(fid)
            if rgb_path is not None:
                n_id += 1
        if rgb_path is None:
            continue
        # Dedup on frame id so a depth matched twice is only included once.
        key = fid if fid is not None else stem
        if key in seen_ids:
            continue
        seen_ids.add(key)
        pairs.append((rgb_path, depth_path))

    if n_id and not n_stem:
        print(f"  [ADT] matched {len(pairs)} frames by frame-id "
              f"(stems differ between videos_rgb/ and depth_npy/)")
    elif n_id:
        print(f"  [ADT] matched {n_stem} by exact stem, {n_id} by frame-id")
    return pairs


ADT_RGB_SUBDIR = "videos_synthetic"   # GT-aligned rendered RGB (see ADTWindowDataset)


def _gather_real_frames(
    seq_dir: str,
    rgb_subdir: str = ADT_RGB_SUBDIR,
    depth_subdir: str = "depth_npy",
) -> List[Tuple[str, str]]:
    """Return paired (rgb, depth) frames from a sequence directory.

    Layout: <seq_dir>/<rgb_subdir>/*.jpg  +  <seq_dir>/<depth_subdir>/*.npy

    ``rgb_subdir`` defaults to ``videos_synthetic`` — the rendered RGB, which is
    pixel-exactly aligned with the rendered GT depth. The real-sensor stream
    (``videos_rgb``) is *not* perfectly registered to the GT depth (small pose /
    timing / distortion differences), so scoring against it mixes real depth error
    with registration error. Use ``videos_rgb`` only if you specifically want the
    real-image domain and accept that misalignment.

    ``depth_subdir`` defaults to ``depth_npy`` (ADT convention, uint16 mm). For
    blender-rendered pinhole data use ``depth_maps`` (float32 metres).
    """
    rgb_dir = os.path.join(seq_dir, rgb_subdir)
    depth_dir = os.path.join(seq_dir, depth_subdir)
    if not os.path.isdir(rgb_dir):
        print(f"  [ADT] WARN {rgb_subdir} not found: {rgb_dir}")
        return []
    if not os.path.isdir(depth_dir):
        print(f"  [ADT] WARN {depth_subdir} not found: {depth_dir}")
        return []

    n_rgb = len(glob.glob(os.path.join(rgb_dir, "*.png"))
                + glob.glob(os.path.join(rgb_dir, "*.jpg"))
                + glob.glob(os.path.join(rgb_dir, "*.jpeg")))
    n_depth = len(glob.glob(os.path.join(depth_dir, "*.npy")))
    pairs = _collect_paired_frames(rgb_dir, depth_dir)
    print(f"  [ADT] {len(pairs)} frames from {n_depth} {depth_subdir} "
          f"/ {n_rgb} {rgb_subdir} — {seq_dir}")
    if len(pairs) < n_depth:
        print(f"  [ADT] WARN {n_depth - len(pairs)} depth map(s) had no RGB partner")
    # Verification: show the first few pairings so RGB↔depth sync is auditable,
    # and assert the frame ids actually line up.
    for rgb_p, dep_p in pairs[:3]:
        rid, did = _frame_id(rgb_p), _frame_id(dep_p)
        flag = "" if rid == did else "  <-- FRAME-ID MISMATCH"
        print(f"  [ADT]   rgb={os.path.basename(rgb_p)}  "
              f"depth={os.path.basename(dep_p)}{flag}")
    return pairs


def _colorize_depth(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """depth [H,W] float → [H,W,3] uint8 using viridis colormap."""
    try:
        import matplotlib.cm as cm
        normed = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        return (cm.viridis(normed)[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        normed = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        g = (normed * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)


def _save_qual_grid(
    save_path: str,
    images_np: np.ndarray,   # [S,3,H,W] float32 in [0,1]
    pred_depth: np.ndarray,  # [S,H,W]  aligned to GT scale
    gt_depth: np.ndarray,    # [S,H,W]  GT metres
    valid_mask: np.ndarray,  # [S,H,W]  bool
    label: str = "",
) -> None:
    """Save horizontal strip  RGB | pred | GT | abs_error  for the centre frame."""
    s = images_np.shape[0] // 2
    rgb = (images_np[s].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

    g = gt_depth[s]
    p = pred_depth[s]
    m = valid_mask[s]

    valid_g = g[m]
    vmin = float(np.percentile(valid_g, 2)) if valid_g.size > 0 else 0.0
    vmax = float(np.percentile(valid_g, 98)) if valid_g.size > 0 else 10.0

    p_vis = p.copy(); p_vis[~m] = 0.0
    g_vis = g.copy(); g_vis[~m] = 0.0
    err = np.abs(p - g); err[~m] = 0.0

    pred_col = _colorize_depth(p_vis, vmin, vmax)
    gt_col   = _colorize_depth(g_vis, vmin, vmax)
    err_col  = _colorize_depth(err,   0.0, max((vmax - vmin) * 0.3, 0.1))
    for panel in (pred_col, gt_col, err_col):
        panel[~m] = 24   # dark background for invalid pixels

    strip = np.concatenate([rgb, pred_col, gt_col, err_col], axis=1)

    # Header bar with column labels
    H, W = rgb.shape[:2]
    bar_h = max(18, H // 20)
    bar = np.full((bar_h, strip.shape[1], 3), 40, dtype=np.uint8)
    try:
        from PIL import ImageDraw, ImageFont
        bar_img = Image.fromarray(bar)
        draw = ImageDraw.Draw(bar_img)
        cols = ["RGB", "Pred depth", "GT depth", "Abs error"]
        for i, col_name in enumerate(cols):
            draw.text((i * W + 4, 2), col_name, fill=(220, 220, 220))
        if label:
            draw.text((4, bar_h // 2), label, fill=(255, 200, 100))
        bar = np.array(bar_img)
    except Exception:
        pass

    out = np.concatenate([bar, strip], axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    Image.fromarray(out).save(save_path)
    print(f"  [ADT] qual → {save_path}")


def _load_gt_trajectory(traj_csv: str) -> np.ndarray:
    """Load ADT groundtruth/aria_trajectory.csv → (N, 3) world positions (metres)."""
    import csv
    positions = []
    with open(traj_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append([
                float(row["tx_world_device"]),
                float(row["ty_world_device"]),
                float(row["tz_world_device"]),
            ])
    return np.array(positions, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class ADTWindowDataset(Dataset):
    """Consecutive windows of seq_len frames from ADT.

    Each item
    ---------
    "images"      : (S, 3, H, W) float32 in [0, 1]  (VGGT input format)
    "depths"      : (S, H, W) float32 metres (GT, resized to model resolution)
    "valid_masks" : (S, H, W) bool — True where GT depth is valid
    "rgb_paths"   : list[str] of length S (for debugging)

    Notes
    -----
    • Input RGB is read from ``rgb_subdir`` (default ``videos_synthetic`` — the
      rendered RGB that is pixel-aligned with the rendered GT depth).
    • Aria frames are stored 90° CW; a 270° CCW rotation is applied to both RGB
      and depth, matching the ADT evaluation convention.
    • Rectification (default ON): ADT RGB + GT depth are fisheye (KB4). To match
      training (rectify=true), each frame and its depth are remapped fisheye→pinhole
      with the SAME maps (RGB bilinear, depth nearest) so they stay aligned and the
      pinhole-trained model is fed pinhole input. Set rectify=False to evaluate raw
      fisheye (e.g. for a model trained with rectify=false).
    • Depth values in depth_npy/ are uint16 millimetres; depth_scale=0.001 converts
      them to metres (the default).
    • VGGT-Omega takes [0, 1] images (no ImageNet normalization).
    • Windows within one sequence are non-overlapping by default (stride=seq_len).
    • depth_max_m controls the valid-pixel ceiling (ADT interiors: 10 m is safe).
    """

    def __init__(
        self,
        seq_dirs: List[str],
        seq_len: int = 8,
        window_stride: Optional[int] = None,
        image_resolution: int = 512,
        patch_size: int = 16,
        depth_scale: float = 0.001,   # uint16 mm → metres; use 1.0 for float32-metre blender maps
        depth_max_m: float = 10.0,
        rotation: int = 270,          # CCW degrees; corrects Aria sensor orientation
        max_frames: Optional[int] = 100,  # cap frames per sequence (None = all)
        rgb_subdir: str = ADT_RGB_SUBDIR,
        depth_subdir: str = "depth_npy",  # "depth_maps" for blender-rendered pinhole data
        rectify: bool = True,             # fisheye→pinhole (match training); False for pinhole data
        camera_preset: str = "aria-214-1",
        fisheye_k: str = "",
        fisheye_d: str = "",
    ) -> None:
        self.seq_len = seq_len
        self.resolution = image_resolution
        self.patch = patch_size
        self.depth_scale = depth_scale
        self.depth_max_m = depth_max_m
        self.rot_k = {0: 0, 90: 1, 180: 2, 270: 3}[rotation]
        # Default: non-overlapping windows so each frame is evaluated once
        self.window_stride = window_stride if window_stride is not None else seq_len

        # Fisheye→pinhole rectifier (same class training uses); None = raw fisheye.
        self.rectifier = None
        if rectify:
            from ..data.rectify import FisheyeRectifier
            self.rectifier = FisheyeRectifier(camera_preset, fisheye_k, fisheye_d)
            print(f"  [ADT] rectifying fisheye→pinhole (preset={camera_preset!r})")

        self.depth_subdir = depth_subdir

        self.windows: List[List[Tuple[str, str]]] = []
        for seq_dir in seq_dirs:
            pairs = _gather_real_frames(seq_dir, rgb_subdir, depth_subdir)
            if not pairs:
                continue
            if max_frames is not None and len(pairs) > max_frames:
                pairs = pairs[:max_frames]
                print(f"  [ADT] capped to first {max_frames} frames")
            for start in range(0, len(pairs) - seq_len + 1, self.window_stride):
                self.windows.append(pairs[start : start + seq_len])

        if not self.windows:
            seq_list = "\n  ".join(seq_dirs)
            raise RuntimeError(
                f"No windows of length {seq_len} found in ADT seq dirs:\n  {seq_list}\n"
                f"Ensure {rgb_subdir}/ and depth_npy/ exist in each sequence dir."
            )
        print(f"  [ADT] {len(self.windows)} windows from {len(seq_dirs)} sequence(s)")

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _target_hw(w: int, h: int, resolution: int, patch: int) -> Tuple[int, int]:
        if h >= w:
            th = resolution
            tw = max(patch, int(round(resolution * w / max(h, 1) / patch)) * patch)
        else:
            tw = resolution
            th = max(patch, int(round(resolution * h / max(w, 1) / patch)) * patch)
        return th, tw

    def __getitem__(self, idx: int) -> dict:
        pairs = self.windows[idx]
        rgb_paths = [p[0] for p in pairs]
        depth_paths = [p[1] for p in pairs]

        # Determine target size from the first frame AFTER rotation.
        # np.rot90 with k=3 (270° CCW) swaps H and W if the image is not square.
        with Image.open(rgb_paths[0]) as im0:
            w0_raw, h0_raw = im0.size   # pre-rotation (W, H) PIL convention
        if self.rot_k % 2 == 1:
            # 90° or 270°: H and W swap after rotation
            h0, w0 = w0_raw, h0_raw
        else:
            h0, w0 = h0_raw, w0_raw
        th, tw = self._target_hw(w0, h0, self.resolution, self.patch)

        images, depths, masks = [], [], []
        for rgb_p, dep_p in zip(rgb_paths, depth_paths):
            # ── RGB ──────────────────────────────────────────────────────────
            with Image.open(rgb_p) as im:
                img = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
            if self.rot_k:
                img = np.rot90(img, k=self.rot_k).copy()
            img_t = torch.from_numpy(img).permute(2, 0, 1)  # [3,H,W]
            img_t = F.interpolate(
                img_t.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
            ).squeeze(0)
            if self.rectifier is not None:
                # Rectify after resize (mirrors training: resize → rectify), HWC float.
                img_np = self.rectifier(img_t.permute(1, 2, 0).contiguous().numpy())
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
            images.append(img_t)

            # ── Depth ─────────────────────────────────────────────────────────
            d = np.load(dep_p).astype(np.float32)
            if d.ndim == 3:
                d = d.squeeze(-1)
            # Replace inf (blender background) with 0 so interpolation is safe;
            # they are excluded by the valid-pixel mask (d > 0 & d <= depth_max_m).
            if np.any(np.isinf(d)):
                d = np.where(np.isinf(d), 0.0, d)
            if self.rot_k:
                d = np.rot90(d, k=self.rot_k).copy()
            d = d * self.depth_scale
            d_t = torch.from_numpy(d)
            d_t = F.interpolate(
                d_t.unsqueeze(0).unsqueeze(0), size=(th, tw), mode="nearest"
            ).squeeze(0).squeeze(0)
            if self.rectifier is not None:
                # Same maps as RGB (nearest) → depth stays pixel-aligned; out-of-FOV → 0.
                d_t = torch.from_numpy(self.rectifier.rectify_depth(d_t.numpy()))
            depths.append(d_t)
            masks.append((d_t > 0) & (d_t <= self.depth_max_m))

        return {
            "images":      torch.stack(images),       # [S,3,H,W]
            "depths":      torch.stack(depths),        # [S,H,W]
            "valid_masks": torch.stack(masks),         # [S,H,W] bool
            "rgb_paths":   rgb_paths,
        }


# --------------------------------------------------------------------------- #
# Eval runner
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_adt_eval(
    predict_fn,
    label: str,
    seq_dirs: List[str],
    device: torch.device,
    seq_len: int = 8,
    image_resolution: int = 512,
    batch_size: int = 1,
    depth_scale: float = 0.001,   # uint16 mm → metres; use 1.0 for float32-metre blender maps
    depth_max_m: float = 10.0,
    align_modes: Tuple[str, ...] = ("none", "scale_shift"),
    eval_all_frames: bool = True,
    gt_traj_csv: Optional[str] = None,
    qual_dir: Optional[str] = None,
    n_qual: int = 4,
    max_frames: Optional[int] = 100,
    rgb_subdir: str = ADT_RGB_SUBDIR,
    depth_subdir: str = "depth_npy",
    rectify: bool = True,
    camera_preset: str = "aria-214-1",
    fisheye_k: str = "",
    fisheye_d: str = "",
) -> Dict[str, dict]:
    """Run depth evaluation against ADT with dense GT.

    Parameters
    ----------
    predict_fn    : callable(images [B,S,3,H,W] on device)
                    → (depth_np [B,S,H,W], pose_enc_np [B,S,9] or None).
                    Use make_vggt_predict() or make_dav2_predict() from run_eval.
    label         : display name, e.g. "VGGT pretrained" or "DAv2 finetuned".
    seq_dirs      : list of ADT sequence dirs containing <rgb_subdir>/ and depth_npy/.
    device        : torch device.
    seq_len       : number of frames per window (must match model training).
    image_resolution : target image resolution.
    batch_size    : windows per GPU batch.
    depth_scale   : multiplied into raw GT depth values (0.001 converts uint16 mm→m).
    depth_max_m   : max valid GT depth in metres.
    align_modes   : alignment modes to report.
                    VGGT (metric): include 'none'. DAv2 (relative): skip 'none'.
    eval_all_frames : if True, score every frame; if False, only the center frame.
    gt_traj_csv   : optional ADT groundtruth/aria_trajectory.csv for pose ATE
                    (only meaningful when predict_fn returns pose_enc, i.e. VGGT).
    rgb_subdir    : RGB source dir (default videos_synthetic, GT-aligned rendered RGB).
    rectify       : fisheye→pinhole rectify RGB+depth to match training (default True).
    camera_preset : rectification preset (default aria-214-1); with fisheye_k/d overrides.

    Returns
    -------
    dict mapping align_mode → aggregated metrics dict.
    Plus 'pose' key if gt_traj_csv is given and pose_enc is returned.
    """
    dataset = ADTWindowDataset(
        seq_dirs,
        seq_len=seq_len,
        window_stride=seq_len,  # non-overlapping: each frame scored once
        image_resolution=image_resolution,
        depth_scale=depth_scale,
        depth_max_m=depth_max_m,
        max_frames=max_frames,
        rgb_subdir=rgb_subdir,
        depth_subdir=depth_subdir,
        rectify=rectify,
        camera_preset=camera_preset,
        fisheye_k=fisheye_k,
        fisheye_d=fisheye_d,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    frame_metrics: Dict[str, List[dict]] = {m: [] for m in align_modes}
    pred_positions: List[np.ndarray] = []
    # Alignment mode used for qualitative visualizations (prefer scale_shift → first available)
    _vis_mode = "scale_shift" if "scale_shift" in align_modes else align_modes[0]
    n_saved = 0

    for batch in loader:
        images = batch["images"].to(device)     # [B,S,3,H,W]
        depths_gt = batch["depths"].numpy()      # [B,S,H,W]
        masks_gt = batch["valid_masks"].numpy()  # [B,S,H,W] bool

        depth_pred, pose_enc = predict_fn(images)   # both numpy [B,S,H,W] / [B,S,9]

        if gt_traj_csv is not None and pose_enc is not None:
            B, S = depth_pred.shape[:2]
            for b in range(B):
                for s in range(S):
                    pred_positions.append(pose_enc[b, s, :3])

        frame_indices = range(depth_pred.shape[1]) if eval_all_frames else [depth_pred.shape[1] // 2]
        images_np = images.float().cpu().numpy()   # [B,S,3,H,W]  kept for qual saves
        for b in range(depth_pred.shape[0]):
            # ── qualitative save (first n_qual windows) ───────────────────────
            if qual_dir is not None and n_saved < n_qual:
                s_c = depth_pred.shape[1] // 2   # centre frame of window
                pred_c = depth_pred[b, s_c]
                gt_c   = depths_gt[b, s_c]
                mask_c = masks_gt[b, s_c]
                if mask_c.sum() >= 10:
                    aligned_c = align_depth(pred_c, gt_c, mask_c, mode=_vis_mode)
                    # Build per-frame arrays for _save_qual_grid (expects [S,...])
                    _save_qual_grid(
                        save_path=os.path.join(
                            qual_dir,
                            f"{n_saved:04d}.png",
                        ),
                        images_np=images_np[b],           # [S,3,H,W]
                        pred_depth=np.stack(
                            [align_depth(depth_pred[b, s], depths_gt[b, s],
                                         masks_gt[b, s], mode=_vis_mode)
                             for s in range(depth_pred.shape[1])]
                        ),                                 # [S,H,W] aligned
                        gt_depth=depths_gt[b],             # [S,H,W]
                        valid_mask=masks_gt[b],            # [S,H,W]
                        label=label,
                    )
                    n_saved += 1

            for s in frame_indices:
                pred = depth_pred[b, s]
                gt = depths_gt[b, s]
                mask = masks_gt[b, s]
                if mask.sum() < 10:
                    continue
                for mode in align_modes:
                    aligned = align_depth(pred, gt, mask, mode=mode)
                    m = depth_metrics(aligned, gt, mask, max_depth=depth_max_m)
                    frame_metrics[mode].append(m)

    results: Dict[str, dict] = {}
    for mode in align_modes:
        results[mode] = aggregate_metrics(frame_metrics[mode])
        print_depth_summary(results[mode], label=f"{label} [ADT real]", align=mode)

    if gt_traj_csv is not None and pred_positions:
        gt_positions = _load_gt_trajectory(gt_traj_csv)
        n = min(len(pred_positions), len(gt_positions))
        results["pose"] = pose_ate_rpe(
            np.stack(pred_positions[:n]), gt_positions[:n], align_sim3=True
        )
        print_pose_summary(results["pose"], label=f"{label} [ADT]")

    return results
