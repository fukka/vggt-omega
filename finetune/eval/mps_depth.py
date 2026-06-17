# Copyright (c) 2026.
"""MPS (Machine Perception Services) sparse-depth evaluation for VGGT-Omega.

Aria MPS provides semi-dense SLAM point clouds and closed-loop trajectories
for Ego-Exo4D takes. Projecting these into the RGB image plane gives a sparse
depth map (~5-15% pixel coverage) that can serve as ground-truth for in-domain
depth evaluation where no depth sensor was present.

Limitations vs ADT
------------------
• Sparse (texture-biased, misses walls/ceilings and moving objects).
• No points on hands / dynamic foreground → metrics are optimistic.
• Metric accuracy depends on SLAM quality (quality_score column).
• Because of the above, MPS eval tells you "did I improve on Ego-Exo4D domain?"
  while ADT tells you absolute accuracy.

Key coordinate convention
-------------------------
  closed_loop_trajectory.csv stores `T_world_device`:
      (tx_world_device, ty_world_device, tz_world_device,
       qx_world_device, qy_world_device, qz_world_device, qw_world_device)
  Meaning: p_world = T_world_device * p_device  (world-from-device, XYZW quat)

  VGGT-Omega convention: camera-from-world (`pose_enc` slots [3:7] are XYZW).

  Projection pipeline:
      p_world → T_device_world = inv(T_world_device) → p_device
      p_device → T_camera_device (from calib JSONL) → p_camera
      p_camera → project with K → (u, v, d)

Usage
-----
  # Load bundle for one take
  bundle = MPSBundle(
      traj_csv="<take>/mps/slam/closed_loop_trajectory.csv",
      points_csv_gz="<take>/mps/slam/semidense_points.csv.gz",
      calib_jsonl="<take>/mps/online_calibration.jsonl",
  )
  # Get sparse GT depth for one frame
  depth_sparse, valid_mask = bundle.project_frame(
      timestamp_us=87552137422,
      K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]),
      T_cam_device=np.eye(4),  # 4x4 cam-from-device (from calib or known for Aria)
      H=512, W=512,
  )
  # Compare vs VGGT prediction at valid pixels
  from .metrics import align_depth, depth_metrics
  aligned = align_depth(pred, depth_sparse, valid_mask, mode='none')  # metric
  m = depth_metrics(aligned, depth_sparse, valid_mask)
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .metrics import aggregate_metrics, align_depth, depth_metrics, print_depth_summary


# --------------------------------------------------------------------------- #
# Quaternion / SE3 helpers (pure numpy, no torch dependency)
# --------------------------------------------------------------------------- #

def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """XYZW scalar-last quaternion → 3×3 rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z),     s * (x * y - z * w),     s * (x * z + y * w)],
        [    s * (x * y + z * w), 1 - s * (x * x + z * z),     s * (y * z - x * w)],
        [    s * (x * z - y * w),     s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def _make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -(R.T @ t)
    return Tinv


# --------------------------------------------------------------------------- #
# MPS Bundle — one take
# --------------------------------------------------------------------------- #

class MPSBundle:
    """Loads and indexes MPS data (trajectory + semi-dense points) for one take.

    Parameters
    ----------
    traj_csv      : path to closed_loop_trajectory.csv (world-from-device poses).
    points_csv_gz : path to semidense_points.csv.gz  (may be None — then project()
                    will raise; useful if you only need poses for ATE eval).
    calib_jsonl   : path to online_calibration.jsonl (may be None — then
                    project_frame() requires explicit K and T_cam_device).
    quality_min   : minimum quality_score for trajectory poses (0–1; 1.0 = best).
    inv_dist_std_thresh, dist_std_thresh : confidence filters for 3D points.
                    Tighter = fewer but more accurate projected depths.
    """

    def __init__(
        self,
        traj_csv: str,
        points_csv_gz: Optional[str] = None,
        calib_jsonl: Optional[str] = None,
        quality_min: float = 0.5,
        inv_dist_std_thresh: float = 1e-3,
        dist_std_thresh: float = 0.15,
    ) -> None:
        self.inv_dist_std_thresh = inv_dist_std_thresh
        self.dist_std_thresh = dist_std_thresh

        # ── Trajectory ───────────────────────────────────────────────────────
        print(f"  [MPS] loading trajectory: {traj_csv}")
        self._timestamps_us, self._T_world_device = self._load_traj(
            traj_csv, quality_min
        )
        print(f"  [MPS] {len(self._timestamps_us)} trajectory poses loaded")

        # ── 3D points ─────────────────────────────────────────────────────────
        self._points_world: Optional[np.ndarray] = None   # (N, 3)
        if points_csv_gz is not None:
            print(f"  [MPS] loading semidense points: {points_csv_gz}")
            self._points_world = self._load_points(
                points_csv_gz, inv_dist_std_thresh, dist_std_thresh
            )
            print(f"  [MPS] {len(self._points_world)} points after confidence filter")

        # ── Online calibration ────────────────────────────────────────────────
        self._calib: Optional[List[dict]] = None
        if calib_jsonl is not None:
            print(f"  [MPS] loading calibration: {calib_jsonl}")
            self._calib = self._load_calib(calib_jsonl)
            print(f"  [MPS] {len(self._calib)} calibration entries")

    # ── I/O ──────────────────────────────────────────────────────────────── #

    @staticmethod
    def _load_traj(
        csv_path: str, quality_min: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (timestamps_us [N], T_world_device [N, 4, 4])."""
        timestamps, poses = [], []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = float(row.get("quality_score", 1.0))
                if q < quality_min:
                    continue
                ts = int(row["tracking_timestamp_us"])
                tx = float(row["tx_world_device"])
                ty = float(row["ty_world_device"])
                tz = float(row["tz_world_device"])
                qx = float(row["qx_world_device"])
                qy = float(row["qy_world_device"])
                qz = float(row["qz_world_device"])
                qw = float(row["qw_world_device"])
                R = _quat_xyzw_to_mat(np.array([qx, qy, qz, qw]))
                T = _make_se3(R, np.array([tx, ty, tz]))
                timestamps.append(ts)
                poses.append(T)
        return np.array(timestamps, dtype=np.int64), np.stack(poses)

    @staticmethod
    def _load_points(
        gz_path: str,
        inv_dist_std_thresh: float,
        dist_std_thresh: float,
    ) -> np.ndarray:
        """Load semidense_points.csv.gz → (N, 3) world positions after filtering."""
        pts = []
        opener = gzip.open if gz_path.endswith(".gz") else open
        with opener(gz_path, "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    inv_std = float(row.get("inv_dist_std", 0.0))
                    d_std = float(row.get("dist_std", 999.0))
                except (ValueError, KeyError):
                    continue
                # Confidence filter: smaller std = more reliable
                if inv_std > inv_dist_std_thresh or d_std > dist_std_thresh:
                    continue
                pts.append([
                    float(row["px_world"]),
                    float(row["py_world"]),
                    float(row["pz_world"]),
                ])
        return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3))

    @staticmethod
    def _load_calib(jsonl_path: str) -> List[dict]:
        """Load online_calibration.jsonl → list of calib dicts keyed by timestamp."""
        entries = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    # ── Pose interpolation ───────────────────────────────────────────────── #

    def get_T_world_device(self, timestamp_us: int) -> np.ndarray:
        """Linearly interpolated T_world_device (4×4) at given timestamp."""
        ts = self._timestamps_us
        poses = self._T_world_device

        idx = np.searchsorted(ts, timestamp_us)
        if idx == 0:
            return poses[0]
        if idx >= len(ts):
            return poses[-1]

        t0, t1 = ts[idx - 1], ts[idx]
        alpha = float(timestamp_us - t0) / max(float(t1 - t0), 1e-6)
        # Linear interp on translation; SLERP would be more accurate for rotation
        # but linear is fine for the small inter-frame intervals (~1ms at 1kHz).
        P0, P1 = poses[idx - 1], poses[idx]
        T = P0 + alpha * (P1 - P0)
        # Re-orthogonalise R via QR to correct drift from linear interp
        Q, _ = np.linalg.qr(T[:3, :3])
        T[:3, :3] = Q
        return T

    # ── Calibration lookup ────────────────────────────────────────────────── #

    def get_rgb_calib_at(
        self, timestamp_us: int, label: str = "camera-rgb"
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Look up RGB camera intrinsics K (3×3) and T_cam_device (4×4) from calib JSONL.

        Returns None if no calibration entry is found near the given timestamp.
        The calibration changes slowly (< Hz) so we return the nearest entry.

        For Aria the label is usually 'camera-rgb'.
        T_Device_Camera in the JSONL is DEVICE-from-CAMERA; we return T_cam_device
        = inv(T_device_cam) = camera-from-device.
        """
        if self._calib is None:
            return None

        # Find nearest calibration entry by UTC timestamp (nanoseconds → microseconds)
        best = None
        best_dt = np.inf
        for entry in self._calib:
            utc_ns = entry.get("UTC_time_ns", None)
            if utc_ns is None:
                continue
            dt = abs(utc_ns / 1000 - timestamp_us)
            if dt < best_dt:
                best_dt = dt
                best = entry

        if best is None:
            return None

        for cam in best.get("CameraCalibrations", []):
            if cam.get("Label", "") != label:
                continue
            proj = cam.get("Projection", {})
            fx = float(proj.get("fx", proj.get("FocalLengthX", 0)))
            fy = float(proj.get("fy", proj.get("FocalLengthY", fx)))
            cx = float(proj.get("cx", proj.get("PrincipalPointX", 0)))
            cy = float(proj.get("cy", proj.get("PrincipalPointY", 0)))
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

            # T_Device_Camera = device-from-camera
            tdc = cam.get("T_Device_Camera", {})
            t_dc = np.array(tdc.get("Translation", [0, 0, 0]), dtype=np.float64)
            q_dc = tdc.get("Quaternion", [0, 0, 0, 1])
            R_dc = _quat_xyzw_to_mat(np.array(q_dc, dtype=np.float64))
            T_device_cam = _make_se3(R_dc, t_dc)
            T_cam_device = _invert_se3(T_device_cam)
            return K, T_cam_device

        return None

    # ── Projection ───────────────────────────────────────────────────────── #

    def project_frame(
        self,
        timestamp_us: int,
        K: np.ndarray,
        T_cam_device: np.ndarray,
        H: int,
        W: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project world SLAM points to one RGB frame → sparse GT depth.

        Parameters
        ----------
        timestamp_us  : frame timestamp in microseconds.
        K             : (3, 3) camera intrinsic matrix (pixels).
        T_cam_device  : (4, 4) camera-from-device transform (from calibration).
        H, W          : image dimensions (output depth map size).

        Returns
        -------
        depth_map  : float32 (H, W) — metric depth at projected pixels (0 elsewhere).
        valid_mask : bool   (H, W) — True at valid projected pixels.
        """
        if self._points_world is None or len(self._points_world) == 0:
            return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

        T_world_device = self.get_T_world_device(timestamp_us)
        T_device_world = _invert_se3(T_world_device)
        T_cam_world = T_cam_device @ T_device_world  # 4×4 cam-from-world

        # Transform world points to camera frame
        pts_w = self._points_world            # (N, 3)
        pts_h = np.hstack([pts_w, np.ones((len(pts_w), 1))])  # (N, 4)
        pts_c = (T_cam_world @ pts_h.T).T[:, :3]               # (N, 3)

        # Keep points in front of camera
        in_front = pts_c[:, 2] > 0.01
        pts_c = pts_c[in_front]
        if len(pts_c) == 0:
            return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

        # Project to image plane
        u = pts_c[:, 0] / pts_c[:, 2] * K[0, 0] + K[0, 2]
        v = pts_c[:, 1] / pts_c[:, 2] * K[1, 1] + K[1, 2]
        z = pts_c[:, 2]

        # Keep points within image bounds
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = u[in_img].astype(np.int32)
        vi = v[in_img].astype(np.int32)
        zi = z[in_img].astype(np.float32)

        # For pixels with multiple projections, keep the nearest (frontmost)
        depth_map = np.zeros((H, W), dtype=np.float32)
        order = np.argsort(zi)[::-1]  # farthest first so nearest overwrites
        depth_map[vi[order], ui[order]] = zi[order]

        valid_mask = depth_map > 0
        return depth_map, valid_mask

    # ── GT trajectory (world-space device positions) ─────────────────────── #

    def gt_positions(self) -> np.ndarray:
        """Return (N, 3) device positions in world frame for ATE evaluation."""
        return self._T_world_device[:, :3, 3]


# --------------------------------------------------------------------------- #
# Dataset for iterating MPS eval frames
# --------------------------------------------------------------------------- #

def _parse_timestamp_from_filename(fname: str) -> Optional[int]:
    """Try to parse a microsecond timestamp from a filename.

    Ego-Exo4D VRS-extracted frames are typically named one of:
      <timestamp_us>.jpg
      frame_<timestamp_us>.jpg
      <sequence>_<timestamp_us>.jpg
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    # Try last numeric run ≥ 10 digits (timestamps are ~17 digits in ns or ~11 in us)
    nums = re.findall(r"\d+", stem)
    for n in reversed(nums):
        if len(n) >= 10:
            ts = int(n)
            # Distinguish ns vs us: ns values are ~1e18 (year ~2020)
            if ts > 1e15:
                ts //= 1000  # ns → us
            return ts
    return None


class MPSEvalDataset(Dataset):
    """Iterates egocentric frames paired with MPS sparse GT depth.

    Parameters
    ----------
    frame_dir     : directory of extracted RGB frames (e.g. aria01_214-1/).
    bundle        : MPSBundle for this take.
    K             : (3, 3) intrinsic matrix for the RGB camera (pixels).
                    If None and bundle has calib JSONL, K is looked up per frame.
    T_cam_device  : (4, 4) camera-from-device transform.
                    If None and bundle has calib JSONL, looked up per frame.
    image_resolution : target image resolution.
    patch_size    : resolution rounding.
    timestamp_csv : optional CSV with columns 'filename,timestamp_us' to override
                    the auto-parse logic for frame timestamps.
    """

    _IMG_EXTS = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        frame_dir: str,
        bundle: MPSBundle,
        K: Optional[np.ndarray] = None,
        T_cam_device: Optional[np.ndarray] = None,
        image_resolution: int = 512,
        patch_size: int = 16,
        timestamp_csv: Optional[str] = None,
    ) -> None:
        self.bundle = bundle
        self.K = K
        self.T_cam_device = T_cam_device
        self.resolution = image_resolution
        self.patch = patch_size

        # ── Collect frames ─────────────────────────────────────────────────
        all_files = sorted([
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if f.lower().endswith(self._IMG_EXTS)
        ])

        # ── Timestamps ────────────────────────────────────────────────────
        ts_map: Dict[str, int] = {}
        if timestamp_csv is not None:
            with open(timestamp_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts_map[row["filename"]] = int(row["timestamp_us"])

        self.samples: List[Tuple[str, int]] = []
        skipped = 0
        for fpath in all_files:
            bname = os.path.basename(fpath)
            if bname in ts_map:
                ts = ts_map[bname]
            else:
                ts = _parse_timestamp_from_filename(fpath)
            if ts is None:
                skipped += 1
                continue
            self.samples.append((fpath, ts))

        print(
            f"  [MPS] {len(self.samples)} frames with timestamps "
            f"({skipped} skipped — filename timestamp parse failed)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _target_hw(w: int, h: int, res: int, patch: int) -> Tuple[int, int]:
        if h >= w:
            th = res
            tw = max(patch, int(round(res * w / max(h, 1) / patch)) * patch)
        else:
            tw = res
            th = max(patch, int(round(res * h / max(w, 1) / patch)) * patch)
        return th, tw

    def __getitem__(self, idx: int) -> dict:
        fpath, timestamp_us = self.samples[idx]

        with Image.open(fpath) as im:
            w0, h0 = im.size
            im = im.convert("RGB")
        th, tw = self._target_hw(w0, h0, self.resolution, self.patch)

        import torch.nn.functional as F
        img = np.array(im, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1)
        img_t = F.interpolate(
            img_t.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
        ).squeeze(0)

        # Resolve K and T_cam_device
        K = self.K
        T_cam_device = self.T_cam_device
        if K is None or T_cam_device is None:
            calib = self.bundle.get_rgb_calib_at(timestamp_us)
            if calib is not None:
                K_cal, T_cal = calib
                if K is None:
                    K = K_cal
                if T_cam_device is None:
                    T_cam_device = T_cal

        if K is None or T_cam_device is None:
            # Return zeros — caller must handle missing calibration
            depth_sparse = np.zeros((th, tw), dtype=np.float32)
            valid_mask = np.zeros((th, tw), dtype=bool)
        else:
            # Scale K to match the resized resolution
            sx = tw / w0
            sy = th / h0
            K_scaled = K.copy().astype(np.float64)
            K_scaled[0, 0] *= sx; K_scaled[0, 2] *= sx
            K_scaled[1, 1] *= sy; K_scaled[1, 2] *= sy

            depth_sparse, valid_mask = self.bundle.project_frame(
                timestamp_us, K_scaled, T_cam_device, H=th, W=tw
            )

        return {
            "image":        img_t,                         # [3,H,W]
            "depth_sparse": torch.from_numpy(depth_sparse),  # [H,W]
            "valid_mask":   torch.from_numpy(valid_mask),    # [H,W] bool
            "timestamp_us": timestamp_us,
            "path":         fpath,
        }


# --------------------------------------------------------------------------- #
# Eval runner
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_mps_eval(
    vggt,
    frame_dir: str,
    bundle: MPSBundle,
    device: torch.device,
    K: Optional[np.ndarray] = None,
    T_cam_device: Optional[np.ndarray] = None,
    seq_len: int = 8,
    image_resolution: int = 512,
    align_modes: Tuple[str, ...] = ("none", "scale_only"),
    min_valid_pixels: int = 20,
    gt_traj_for_ate: bool = True,
) -> Dict[str, dict]:
    """Run VGGT-Omega depth evaluation against MPS sparse GT.

    VGGT-Omega requires a WINDOW of frames as input (multi-frame model), so we
    collect a window of seq_len frames around each eval frame and take only the
    center frame's predicted depth for comparison.

    Parameters
    ----------
    vggt          : VGGT-Omega model (eval mode, on device).
    frame_dir     : extracted egocentric RGB frame directory.
    bundle        : MPSBundle for this take.
    device        : torch device.
    K             : (3,3) camera intrinsics. If None, read from bundle's calib JSONL.
    T_cam_device  : (4,4) camera-from-device. If None, read from bundle's calib JSONL.
    seq_len       : frames per VGGT window.
    image_resolution : target image resolution.
    align_modes   : alignment modes to evaluate.
    min_valid_pixels : skip frames with fewer valid MPS projected pixels.
    gt_traj_for_ate  : if True, also run ATE comparison vs MPS trajectory.

    Returns
    -------
    dict mapping align_mode → aggregated metrics dict; plus 'pose' if requested.
    """
    dataset = MPSEvalDataset(
        frame_dir, bundle, K=K, T_cam_device=T_cam_device,
        image_resolution=image_resolution,
    )
    if len(dataset) == 0:
        print("  [MPS] No frames found — skipping MPS eval")
        return {}

    vggt.eval()
    frame_metrics: Dict[str, List[dict]] = {m: [] for m in align_modes}
    pred_positions: List[np.ndarray] = []

    # We need windows around each frame: build index list and pad edges
    n = len(dataset)
    half = seq_len // 2

    for center_idx in range(n):
        # Collect seq_len frames centered on center_idx (clamp at edges)
        idxs = [min(max(center_idx - half + i, 0), n - 1) for i in range(seq_len)]
        items = [dataset[i] for i in idxs]

        images = torch.stack([it["image"] for it in items]).unsqueeze(0).to(device)
        # [1, S, 3, H, W]

        preds = vggt(images)
        depth_pred = preds["depth"]
        if depth_pred.ndim == 5:
            depth_pred = depth_pred.squeeze(-1)
        depth_pred = depth_pred[0, half].float().cpu().numpy()  # center frame (H, W)

        # Collect position from predicted pose_enc for ATE
        if gt_traj_for_ate and "pose_enc" in preds:
            pred_positions.append(preds["pose_enc"][0, half, :3].float().cpu().numpy())

        # GT sparse depth at center frame
        center_item = items[half]
        depth_sparse = center_item["depth_sparse"].numpy()
        valid_mask = center_item["valid_mask"].numpy()

        if valid_mask.sum() < min_valid_pixels:
            continue

        for mode in align_modes:
            aligned = align_depth(depth_pred, depth_sparse, valid_mask, mode=mode)
            m = depth_metrics(aligned, depth_sparse, valid_mask)
            frame_metrics[mode].append(m)

    results: Dict[str, dict] = {}
    for mode in align_modes:
        results[mode] = aggregate_metrics(frame_metrics[mode])
        print_depth_summary(results[mode], label="VGGT-Omega [MPS sparse]", align=mode)

    # Pose ATE vs MPS bundle trajectory
    if gt_traj_for_ate and pred_positions:
        gt_t = bundle.gt_positions()
        n_pred = len(pred_positions)
        # Subsample GT to match — every (n_gt / n_pred) steps
        if len(gt_t) >= n_pred:
            step = max(1, len(gt_t) // n_pred)
            gt_sub = gt_t[::step][:n_pred]
        else:
            gt_sub = gt_t[:n_pred]
        pred_t = np.stack(pred_positions[: len(gt_sub)])
        from .metrics import pose_ate_rpe, print_pose_summary
        results["pose"] = pose_ate_rpe(pred_t, gt_sub, align_sim3=True)
        print_pose_summary(results["pose"], label="VGGT-Omega [MPS trajectory]")

    return results
