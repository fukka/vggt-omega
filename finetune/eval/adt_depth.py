# Copyright (c) 2026.
"""ADT (Aria Digital Twin) dense-GT depth evaluation for VGGT-Omega.

ADT provides real Aria sensor frames with paired dense GT depth maps (from the
digital-twin simulation), giving absolute-accuracy eval on the same fisheye camera
that Ego-Exo4D uses.

Expected ADT sequence layout
-----------------------------
  <seq_dir>/
    videos_rgb/   *.jpg | *.png   (real Aria RGB frames; typically 1408×1408)
    depth_npy/    *.npy           (GT depth maps, matching stems; uint16 millimetres)
    groundtruth/
      aria_trajectory.csv         (GT camera pose: world-from-device, XYZW, us timestamps)

Important: real Aria frames are rotated 90° CW in storage; apply 270° CCW rotation
before use (same as ADT evaluation convention). Depth values are in millimetres —
pass depth_scale=0.001 to convert to metres (the default).

Dataset and eval runner
-----------------------
  ADTWindowDataset  -- consecutive windows of seq_len frames with GT depth
  run_adt_eval      -- runs VGGT-Omega, compares predicted vs GT depth (and poses)
"""
from __future__ import annotations

import glob
import os
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

def _collect_paired_frames(rgb_dir: str, depth_dir: str) -> List[Tuple[str, str]]:
    """Scan rgb_dir for images; match each with a depth .npy by stem."""
    pairs = []
    seen = set()
    all_rgb = sorted(
        glob.glob(os.path.join(rgb_dir, "*.png"))
        + glob.glob(os.path.join(rgb_dir, "*.jpg"))
        + glob.glob(os.path.join(rgb_dir, "*.jpeg"))
    )
    for rgb_path in all_rgb:
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        if stem in seen:
            continue
        depth_path = os.path.join(depth_dir, f"{stem}.npy")
        if os.path.exists(depth_path):
            seen.add(stem)
            pairs.append((rgb_path, depth_path))
    return pairs


def _gather_real_frames(seq_dir: str) -> List[Tuple[str, str]]:
    """Return paired (rgb, depth) frames from the real ADT sensor data.

    Layout: <seq_dir>/videos_rgb/*.jpg  +  <seq_dir>/depth_npy/*.npy
    Depth values are uint16 millimetres; caller applies depth_scale=0.001.
    """
    rgb_dir = os.path.join(seq_dir, "videos_rgb")
    depth_dir = os.path.join(seq_dir, "depth_npy")
    if not os.path.isdir(rgb_dir):
        print(f"  [ADT] WARN videos_rgb not found: {rgb_dir}")
        return []
    if not os.path.isdir(depth_dir):
        print(f"  [ADT] WARN depth_npy not found: {depth_dir}")
        return []
    pairs = _collect_paired_frames(rgb_dir, depth_dir)
    print(f"  [ADT] {len(pairs)} real frames — {seq_dir}")
    return pairs


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
    """Consecutive windows of seq_len frames from ADT real sensor data.

    Each item
    ---------
    "images"      : (S, 3, H, W) float32 in [0, 1]  (VGGT input format)
    "depths"      : (S, H, W) float32 metres (GT, resized to model resolution)
    "valid_masks" : (S, H, W) bool — True where GT depth is valid
    "rgb_paths"   : list[str] of length S (for debugging)

    Notes
    -----
    • Real Aria frames are stored 90° CW; a 270° CCW rotation is applied to both
      RGB and depth before resizing, matching the ADT evaluation convention.
    • Depth values in depth_npy/ are uint16 millimetres; depth_scale=0.001 converts
      them to metres (the default).
    • VGGT-Omega takes [0, 1] images (no ImageNet normalization).
    • Windows within one sequence are non-overlapping by default (stride=seq_len).
      Use window_stride=1 for maximum overlap, but then evaluate only the center
      frame to avoid redundant scoring.
    • depth_max_m controls the valid-pixel ceiling (ADT interiors: 10 m is safe).
    """

    def __init__(
        self,
        seq_dirs: List[str],
        seq_len: int = 8,
        window_stride: Optional[int] = None,
        image_resolution: int = 512,
        patch_size: int = 16,
        depth_scale: float = 0.001,   # uint16 mm → metres
        depth_max_m: float = 10.0,
        rotation: int = 270,          # CCW degrees; corrects Aria sensor orientation
    ) -> None:
        self.seq_len = seq_len
        self.resolution = image_resolution
        self.patch = patch_size
        self.depth_scale = depth_scale
        self.depth_max_m = depth_max_m
        self.rot_k = {0: 0, 90: 1, 180: 2, 270: 3}[rotation]
        # Default: non-overlapping windows so each frame is evaluated once
        self.window_stride = window_stride if window_stride is not None else seq_len

        self.windows: List[List[Tuple[str, str]]] = []
        for seq_dir in seq_dirs:
            pairs = _gather_real_frames(seq_dir)
            if not pairs:
                continue
            for start in range(0, len(pairs) - seq_len + 1, self.window_stride):
                self.windows.append(pairs[start : start + seq_len])

        if not self.windows:
            seq_list = "\n  ".join(seq_dirs)
            raise RuntimeError(
                f"No windows of length {seq_len} found in ADT seq dirs:\n  {seq_list}\n"
                "Ensure videos_rgb/ and depth_npy/ exist in each sequence dir."
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
            images.append(img_t)

            # ── Depth ─────────────────────────────────────────────────────────
            d = np.load(dep_p).astype(np.float32)
            if d.ndim == 3:
                d = d.squeeze(-1)
            if self.rot_k:
                d = np.rot90(d, k=self.rot_k).copy()
            d = d * self.depth_scale
            d_t = torch.from_numpy(d)
            d_t = F.interpolate(
                d_t.unsqueeze(0).unsqueeze(0), size=(th, tw), mode="nearest"
            ).squeeze(0).squeeze(0)
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
    vggt,
    seq_dirs: List[str],
    device: torch.device,
    seq_len: int = 8,
    image_resolution: int = 512,
    batch_size: int = 1,
    depth_scale: float = 0.001,   # uint16 mm → metres
    depth_max_m: float = 10.0,
    align_modes: Tuple[str, ...] = ("none", "scale_only", "scale_shift"),
    eval_all_frames: bool = True,
    gt_traj_csv: Optional[str] = None,
) -> Dict[str, dict]:
    """Run VGGT-Omega depth evaluation against ADT real sensor data with dense GT.

    Parameters
    ----------
    vggt          : VGGT-Omega model (nn.Module, already on device, in eval mode).
    seq_dirs      : list of ADT sequence dirs containing videos_rgb/ and depth_npy/.
    device        : torch device.
    seq_len       : number of frames per VGGT window (must match model training).
    image_resolution : target image resolution (must match model training).
    batch_size    : number of windows per GPU batch.
    depth_scale   : multiplied into raw GT depth values (0.001 converts uint16 mm→m).
    depth_max_m   : max valid GT depth in metres.
    align_modes   : which alignment modes to report (subset of 'none', 'scale_only',
                    'scale_shift', 'disparity_scale_shift').
    eval_all_frames : if True, evaluate every frame in each window; if False, only
                      the center frame (useful when windows are overlapping).
    gt_traj_csv   : optional path to ADT groundtruth/aria_trajectory.csv for pose ATE.

    Returns
    -------
    dict mapping align_mode → aggregated metrics dict (keys: AbsRel, SqRel, RMSE,
    RMSElog, delta1, delta2, delta3, scale_ratio, n_frames, n_valid_total).
    Plus 'pose' key if gt_traj_csv is given.
    """
    dataset = ADTWindowDataset(
        seq_dirs,
        seq_len=seq_len,
        window_stride=seq_len,  # non-overlapping
        image_resolution=image_resolution,
        depth_scale=depth_scale,
        depth_max_m=depth_max_m,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    vggt.eval()
    frame_metrics: Dict[str, List[dict]] = {m: [] for m in align_modes}
    pred_positions: List[np.ndarray] = []

    for batch in loader:
        images = batch["images"].to(device)          # [B,S,3,H,W]
        depths_gt = batch["depths"].numpy()           # [B,S,H,W]
        masks_gt = batch["valid_masks"].numpy()       # [B,S,H,W] bool

        B, S, _, H, W = images.shape
        preds = vggt(images)

        # depth: model returns [B,S,H,W,1] or [B,S,H,W]; normalise to [B,S,H,W]
        depth_pred = preds["depth"]
        if depth_pred.ndim == 5:
            depth_pred = depth_pred.squeeze(-1)
        depth_pred = depth_pred.float().cpu().numpy()  # [B,S,H,W]

        # Optionally collect predicted positions for ATE
        if gt_traj_csv is not None and "pose_enc" in preds:
            pose_enc = preds["pose_enc"].float().cpu().numpy()  # [B,S,9]
            for b in range(B):
                for s in range(S):
                    pred_positions.append(pose_enc[b, s, :3])

        # Evaluate each frame in the batch
        frame_indices = range(S) if eval_all_frames else [S // 2]
        for b in range(B):
            for s in frame_indices:
                pred = depth_pred[b, s]    # (H, W)
                gt = depths_gt[b, s]       # (H, W)
                mask = masks_gt[b, s]      # (H, W)
                if mask.sum() < 10:
                    continue
                for mode in align_modes:
                    aligned = align_depth(pred, gt, mask, mode=mode)
                    m = depth_metrics(aligned, gt, mask, max_depth=depth_max_m)
                    frame_metrics[mode].append(m)

    results: Dict[str, dict] = {}
    for mode in align_modes:
        results[mode] = aggregate_metrics(frame_metrics[mode])
        print_depth_summary(results[mode], label="VGGT-Omega [ADT real]", align=mode)

    # Pose ATE (optional, requires GT trajectory CSV)
    if gt_traj_csv is not None and pred_positions:
        gt_positions = _load_gt_trajectory(gt_traj_csv)
        # Subsample GT to match number of predicted positions if unequal
        n = min(len(pred_positions), len(gt_positions))
        pred_t = np.stack(pred_positions[:n])
        gt_t = gt_positions[:n]
        results["pose"] = pose_ate_rpe(pred_t, gt_t, align_sim3=True)
        print_pose_summary(results["pose"], label="VGGT-Omega [ADT]")

    return results
