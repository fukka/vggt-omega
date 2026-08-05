# Copyright (c) 2026.
"""ADT (Aria Digital Twin) two-view dataset for CAM3R.

ADT is the fisheye dataset CAM3R reports on -- 99.0 / 95.0 RRA@15 / RTA@15 in
Table 1 -- and the one this repo already works in.  Three things here are easy
to get wrong and are handled explicitly:

**Depth domain.**  ADT's ``depth_npy`` is *planar z* (measured, see
``VGGT-360-fisheye/checks/check_gt_depth_domain.py``).  CAM3R's Cross-view
Module regresses *radial distance* ``r`` with ``X = d * r``.  These differ by
``1/cos(theta)``, which reaches 2.15x at the Aria rim, and the difference is
radial so no affine depth alignment absorbs it.  GT is converted z -> range on
load.

**Sensor rotation.**  Aria stores frames 90 deg CW and the ADT convention
rotates them 270 deg CCW.  Rotating pixels without rotating the extrinsics
leaves every GT relative pose wrong by 90 deg about the optical axis, so the
device-from-camera transform is composed with :data:`IMAGE_ROT_TO_CAMERA`
(verified against the two ray fields in ``tests/test_adt.py``).

**Camera extrinsics.**  Relative *camera* pose is
``T_cam_device . T_d1_world . T_world_d2 . T_device_cam``; the device-from-camera
term does not cancel, and ignoring it costs up to ~8 deg of translation
direction at a 0.35 m baseline -- material at RTA@15.  :func:`resolve_extrinsics`
tries an explicit JSON, then MPS ``online_calibration.jsonl``, then
``projectaria_tools``, and only then falls back to the device frame.  The
fallback is recorded in ``extrinsics_source`` and ``extrinsics_exact`` so a run
that used it cannot be mistaken for a calibrated one.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cam3r.cameras import aria_214_1_kb4
from cam3r.geometry import geodesic_angle, relative_pose, se3_inverse

# Rotating the stored frame 270 deg CCW rotates ray directions by Rz(+90 deg).
IMAGE_ROT_TO_CAMERA = torch.tensor(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
)

DEFAULT_RGB_SUBDIR = "videos_rgb"
DEFAULT_DEPTH_SUBDIR = "depth_npy"


# --------------------------------------------------------------------------- #
# Discovery and trajectory
# --------------------------------------------------------------------------- #

def find_adt_sequences(
    root: str, rgb_subdir: str = DEFAULT_RGB_SUBDIR, depth_subdir: str = DEFAULT_DEPTH_SUBDIR
) -> List[str]:
    """Sequence directories under ``root`` that have both RGB and depth."""
    found = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        rgb, depth = os.path.join(d, rgb_subdir), os.path.join(d, depth_subdir)
        if os.path.isdir(rgb) and os.path.isdir(depth) and os.listdir(rgb) and os.listdir(depth):
            found.append(d)
    return found


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / max(np.linalg.norm(q), 1e-12)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_trajectory(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """ADT ``aria_trajectory.csv`` -> ``(timestamps_us (N,), T_world_device (N,4,4))``."""
    ts: List[float] = []
    mats: List[np.ndarray] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ts.append(float(row["tracking_timestamp_us"]))
            R = _quat_xyzw_to_matrix(
                np.array([float(row[f"q{a}_world_device"]) for a in "xyzw"], dtype=np.float64)
            )
            t = np.array([float(row[f"t{a}_world_device"]) for a in "xyz"], dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3], T[:3, 3] = R, t
            mats.append(T)

    if not ts:
        raise RuntimeError(f"no rows in {csv_path}")
    order = np.argsort(np.asarray(ts))
    return (
        torch.tensor(np.asarray(ts)[order], dtype=torch.float64),
        torch.tensor(np.stack(mats)[order], dtype=torch.float64),
    )


# --------------------------------------------------------------------------- #
# Extrinsics
# --------------------------------------------------------------------------- #

def _se3_from_translation_quat(t: Sequence[float], q_xyzw: Sequence[float]) -> torch.Tensor:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_xyzw_to_matrix(np.asarray(q_xyzw, dtype=np.float64))
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return torch.tensor(T, dtype=torch.float64)


def resolve_extrinsics(
    seq_dir: str, extrinsics_json: Optional[str] = None, label: str = "camera-rgb"
) -> Tuple[torch.Tensor, str, bool]:
    """Find ``T_device_camera``; returns ``(T, source, exact)``.

    ``exact=False`` means no calibration was found and the device frame is being
    used as a stand-in, which biases translation direction by the lever arm
    between the device origin and the RGB sensor.
    """
    if extrinsics_json:
        with open(extrinsics_json) as f:
            blob = json.load(f)
        node = blob.get("T_device_camera", blob)
        return (
            _se3_from_translation_quat(
                node.get("translation", node.get("Translation", [0, 0, 0])),
                node.get("quaternion_xyzw", node.get("Quaternion", [0, 0, 0, 1])),
            ),
            "json",
            True,
        )

    for rel in ("mps/online_calibration.jsonl", "mps/slam/online_calibration.jsonl"):
        path = os.path.join(seq_dir, rel)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for cam in entry.get("CameraCalibrations", []):
                    if cam.get("Label") != label:
                        continue
                    tdc = cam.get("T_Device_Camera", {})
                    return (
                        _se3_from_translation_quat(
                            tdc.get("Translation", [0, 0, 0]), tdc.get("Quaternion", [0, 0, 0, 1])
                        ),
                        "mps",
                        True,
                    )

    vrs = os.path.join(seq_dir, "main_recording.vrs")
    if os.path.exists(vrs):
        try:
            from projectaria_tools.core import data_provider  # type: ignore

            provider = data_provider.create_vrs_data_provider(vrs)
            calib = provider.get_device_calibration().get_camera_calib(label)
            T = calib.get_transform_device_camera().to_matrix()
            return torch.tensor(np.asarray(T), dtype=torch.float64), "projectaria_tools", True
        except Exception:
            pass

    return torch.eye(4, dtype=torch.float64), "device-frame-fallback", False


# --------------------------------------------------------------------------- #
# Pair selection
# --------------------------------------------------------------------------- #

def select_pairs(
    poses_cw: torch.Tensor,
    baseline_m: Tuple[float, float] = (0.35, 1.75),
    angle_deg: Tuple[float, float] = (25.0, 65.0),
    max_pairs: Optional[int] = None,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    """Pairs whose baseline and viewing-angle change fall in the paper's window.

    Paper Sec. D.3: ``0.35 m <= b <= 1.75 m`` and ``25 deg <= theta <= 65 deg``.
    Too close and the pair is degenerate; too far and there is no overlap.
    ``poses_cw`` is (N, 4, 4) camera-from-world.
    """
    n = poses_cw.shape[0]
    lo_b, hi_b = baseline_m
    lo_a, hi_a = angle_deg
    eye = torch.eye(3, dtype=poses_cw.dtype).unsqueeze(0)

    out: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            R_ij, t_ij = relative_pose(poses_cw[i].unsqueeze(0), poses_cw[j].unsqueeze(0))
            b = float(t_ij.norm())
            if not (lo_b <= b <= hi_b):
                continue
            a = float(torch.rad2deg(geodesic_angle(R_ij, eye)))
            if not (lo_a <= a <= hi_a):
                continue
            out.append((i, j))

    if max_pairs is not None and len(out) > max_pairs:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(out), generator=g)[:max_pairs].tolist()
        out = [out[k] for k in sorted(idx)]
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def _frame_timestamp_us(path: str) -> Optional[float]:
    """``frame_000400_13863283646787.jpg`` -> 13863283646.787 us (stem is ns)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return float(parts[2]) / 1000.0
    except ValueError:
        return None


class ADTPairDataset(Dataset):
    """Two-view ADT items with GT rays, radial pointmaps and relative pose.

    Each item:
        images  : list[2] of (3, H, W) float32 in [0, 1]
        rays    : list[2] of (3, H, W) unit GT ray directions (Aria KB4)
        points  : list[2] of (3, H, W) GT pointmaps, ``range * ray``
        valid   : list[2] of (H, W) bool -- inside the imaged cone and depth > 0
        R, t    : (3,3) / (3,) GT relative pose, mapping view 2 into view 1
    """

    def __init__(
        self,
        seq_dirs: Sequence[str],
        resolution: int = 512,
        patch_size: int = 16,
        max_frames: Optional[int] = 100,
        rgb_subdir: str = DEFAULT_RGB_SUBDIR,
        depth_subdir: str = DEFAULT_DEPTH_SUBDIR,
        depth_scale: float = 0.001,
        depth_max_m: float = 10.0,
        baseline_m: Tuple[float, float] = (0.35, 1.75),
        angle_deg: Tuple[float, float] = (25.0, 65.0),
        max_pairs: Optional[int] = None,
        extrinsics_json: Optional[str] = None,
        rotation: int = 270,
    ) -> None:
        if resolution % patch_size:
            raise ValueError(f"resolution {resolution} must be a multiple of patch {patch_size}")
        self.resolution = resolution
        self.depth_scale = depth_scale
        self.depth_max_m = depth_max_m
        self.rot_k = {0: 0, 90: 1, 180: 2, 270: 3}[rotation]

        self.rgb_paths: List[str] = []
        self.depth_paths: List[str] = []
        poses_cw: List[torch.Tensor] = []
        self.extrinsics_source = "unset"
        self.extrinsics_exact = False

        for seq_dir in seq_dirs:
            frames = self._gather_frames(seq_dir, rgb_subdir, depth_subdir, max_frames)
            if not frames:
                continue
            ts, T_wd = load_trajectory(os.path.join(seq_dir, "groundtruth", "aria_trajectory.csv"))
            T_dc, source, exact = resolve_extrinsics(seq_dir, extrinsics_json)
            # Compose the pixel rotation into the extrinsics.
            R_fix = torch.eye(4, dtype=torch.float64)
            R_fix[:3, :3] = IMAGE_ROT_TO_CAMERA
            T_dc_rot = T_dc @ se3_inverse(R_fix)
            self.extrinsics_source, self.extrinsics_exact = source, exact

            for rgb, depth in frames:
                stamp = _frame_timestamp_us(rgb)
                if stamp is None:
                    continue
                k = int(torch.argmin((ts - stamp).abs()))
                if float((ts[k] - stamp).abs()) > 50_000.0:   # >50 ms from any pose
                    continue
                T_wc = T_wd[k] @ T_dc_rot
                self.rgb_paths.append(rgb)
                self.depth_paths.append(depth)
                poses_cw.append(se3_inverse(T_wc.unsqueeze(0))[0])

        if not poses_cw:
            raise RuntimeError(f"no usable ADT frames found under {list(seq_dirs)}")

        self.poses_cw = torch.stack(poses_cw)
        self.pairs = select_pairs(self.poses_cw, baseline_m, angle_deg, max_pairs)
        if not self.pairs:
            raise RuntimeError(
                f"{len(self.poses_cw)} frames but no pair met baseline {baseline_m} m / "
                f"angle {angle_deg} deg -- widen the window or use more frames"
            )

        self._ray_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _gather_frames(seq_dir, rgb_subdir, depth_subdir, max_frames) -> List[Tuple[str, str]]:
        rgb_dir, depth_dir = os.path.join(seq_dir, rgb_subdir), os.path.join(seq_dir, depth_subdir)
        depth_by_id = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(depth_dir, "*.npy"))
        }
        out = []
        for rgb in sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) + glob.glob(os.path.join(rgb_dir, "*.png"))):
            key = os.path.splitext(os.path.basename(rgb))[0]
            if key in depth_by_id:
                out.append((rgb, depth_by_id[key]))
        return out[:max_frames] if max_frames else out

    def _rays(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if (H, W) not in self._ray_cache:
            self._ray_cache[(H, W)] = aria_214_1_kb4(H, W, rotated=True).ray_field(H, W)
        return self._ray_cache[(H, W)]

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_view(self, idx: int) -> Dict[str, torch.Tensor]:
        from PIL import Image

        res = self.resolution
        rgb = np.asarray(Image.open(self.rgb_paths[idx]).convert("RGB"), dtype=np.float32) / 255.0
        depth = np.load(self.depth_paths[idx]).astype(np.float32) * self.depth_scale

        if self.rot_k:
            rgb = np.rot90(rgb, k=self.rot_k, axes=(0, 1)).copy()
            depth = np.rot90(depth, k=self.rot_k, axes=(0, 1)).copy()

        img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=(res, res), mode="bilinear", align_corners=False)[0]
        d = torch.from_numpy(depth).view(1, 1, *depth.shape)
        d = torch.nn.functional.interpolate(d, size=(res, res), mode="nearest")[0, 0]

        rays, cone = self._rays(res, res)
        # ADT depth is planar z; CAM3R wants radial range = z / cos(theta).
        cos_theta = rays[..., 2].clamp(min=1e-3)
        rng = d / cos_theta
        valid = cone & (d > 0) & (d < self.depth_max_m) & torch.isfinite(rng)
        points = rays.permute(2, 0, 1) * rng.unsqueeze(0)

        return {
            "image": img.clamp(0.0, 1.0),
            "rays": rays.permute(2, 0, 1).contiguous(),
            "points": torch.where(valid.unsqueeze(0), points, torch.zeros_like(points)),
            "valid": valid,
        }

    def __getitem__(self, index: int) -> Dict:
        i, j = self.pairs[index]
        a, b = self._load_view(i), self._load_view(j)
        R, t = relative_pose(self.poses_cw[i].unsqueeze(0), self.poses_cw[j].unsqueeze(0))
        return {
            "images": [a["image"], b["image"]],
            "rays": [a["rays"], b["rays"]],
            "points": [a["points"], b["points"]],
            "valid": [a["valid"], b["valid"]],
            "R": R[0].float(),
            "t": t[0].float(),
            "frame_idx": (i, j),
        }
