"""ADT two-view dataset: pose bookkeeping, pair selection, depth domain.

Tests that need the real sequence skip cleanly when it is absent, so the suite
still runs on a machine without the ADT sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cam3r.adt import (
    IMAGE_ROT_TO_CAMERA,
    ADTPairDataset,
    find_adt_sequences,
    load_trajectory,
    select_pairs,
)
from cam3r.cameras import aria_214_1_kb4
from cam3r.geometry import geodesic_angle, make_se3, random_rotations, relative_pose

ADT_ROOT = Path("/Users/fengjiazhang/Documents/projectaria_tools_adt_data")


def _require_adt() -> Path:
    seqs = find_adt_sequences(str(ADT_ROOT)) if ADT_ROOT.exists() else []
    if not seqs:
        pytest.skip(f"no ADT sequence with rgb+depth under {ADT_ROOT}")
    return Path(seqs[0])


# ------------------------------------------------------------------ geometry

def test_image_rotation_maps_to_a_z_rotation_of_the_camera_frame():
    """Rotating the stored frame 270 deg CCW rotates the camera frame by Rz(+90).

    The ADT loader rotates pixels; the extrinsics must be rotated to match, or
    every GT relative pose is wrong by 90 deg about the optical axis.
    """
    H = W = 128
    rot, _ = aria_214_1_kb4(H, W, rotated=True).ray_field(H, W)
    raw, valid = aria_214_1_kb4(H, W, rotated=False).ray_field(H, W)
    raw_rot = torch.from_numpy(np.rot90(raw.numpy(), k=3, axes=(0, 1)).copy())
    valid_rot = torch.from_numpy(np.rot90(valid.numpy(), k=3, axes=(0, 1)).copy())

    predicted = raw_rot.double() @ IMAGE_ROT_TO_CAMERA.T
    rot = rot.double()
    err = torch.rad2deg(torch.arccos((predicted * rot).sum(-1).clamp(-1, 1)))[valid_rot]
    assert float(err.mean()) < 0.05, f"mean {float(err.mean()):.4f} deg"


def test_pair_selection_honours_the_papers_window():
    """0.35 <= baseline <= 1.75 m and 25 <= viewing angle <= 65 deg (Sec. D.3)."""
    torch.manual_seed(0)
    n = 60
    R = random_rotations(n, seed=1, dtype=torch.float64)
    t = torch.randn(n, 3, dtype=torch.float64) * 1.5
    T_cw = make_se3(R, t)                      # camera-from-world

    pairs = select_pairs(T_cw, baseline_m=(0.35, 1.75), angle_deg=(25.0, 65.0))
    assert pairs, "no pairs selected from a random layout"
    for i, j in pairs:
        R_ij, t_ij = relative_pose(T_cw[i].unsqueeze(0), T_cw[j].unsqueeze(0))
        b = float(t_ij.norm())
        a = float(torch.rad2deg(geodesic_angle(R_ij, torch.eye(3, dtype=torch.float64).unsqueeze(0))))
        assert 0.35 <= b <= 1.75, f"baseline {b:.3f} out of window"
        assert 25.0 <= a <= 65.0, f"angle {a:.2f} out of window"


def test_pair_selection_returns_nothing_for_a_static_camera():
    T = make_se3(torch.eye(3, dtype=torch.float64).repeat(10, 1, 1), torch.zeros(10, 3, dtype=torch.float64))
    assert select_pairs(T, baseline_m=(0.35, 1.75), angle_deg=(25.0, 65.0)) == []


def test_pair_selection_respects_max_pairs():
    torch.manual_seed(2)
    T = make_se3(random_rotations(80, seed=3, dtype=torch.float64),
                 torch.randn(80, 3, dtype=torch.float64) * 1.5)
    pairs = select_pairs(T, baseline_m=(0.0, 99.0), angle_deg=(0.0, 180.0), max_pairs=17)
    assert len(pairs) == 17


# ------------------------------------------------------------- real sequence

def test_trajectory_loads_with_valid_poses():
    seq = _require_adt()
    ts, T_wd = load_trajectory(str(seq / "groundtruth" / "aria_trajectory.csv"))
    assert len(ts) > 100 and T_wd.shape == (len(ts), 4, 4)
    R = T_wd[:, :3, :3]
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=R.dtype).expand_as(R), atol=1e-6)
    assert bool((ts[1:] >= ts[:-1]).all()), "timestamps must be sorted"


def test_dataset_builds_and_returns_consistent_tensors():
    seq = _require_adt()
    ds = ADTPairDataset([str(seq)], resolution=64, max_frames=28,
                        baseline_m=(0.0, 99.0), angle_deg=(0.0, 180.0))
    assert len(ds) > 0
    item = ds[0]

    for v in (0, 1):
        assert item["images"][v].shape == (3, 64, 64)
        assert item["rays"][v].shape == (3, 64, 64)
        assert item["points"][v].shape == (3, 64, 64)
        assert item["valid"][v].shape == (64, 64)
        assert torch.allclose(item["rays"][v].norm(dim=0), torch.ones(64, 64), atol=1e-4)
        assert float(item["images"][v].min()) >= 0.0 and float(item["images"][v].max()) <= 1.0

    R = item["R"]
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
    assert item["t"].shape == (3,)


def test_points_are_radial_distance_not_planar_z():
    """CAM3R regresses radial distance; ADT GT depth is planar z.

    Getting this backwards is a >2x error at the fisheye rim (CONTEXT.md), and
    an affine depth alignment cannot absorb it because it is radial.
    """
    seq = _require_adt()
    ds = ADTPairDataset([str(seq)], resolution=64, max_frames=28,
                        baseline_m=(0.0, 99.0), angle_deg=(0.0, 180.0))
    item = ds[0]
    rays, points, valid = item["rays"][0], item["points"][0], item["valid"][0]

    radial = points.norm(dim=0)
    planar_z = points[2]
    cos_theta = rays[2]
    # points = radial * ray, and radial = z / cos(theta) -> z component is the GT z.
    assert torch.allclose(points[:, valid], (rays * radial.unsqueeze(0))[:, valid], atol=1e-4)
    on_rim = valid & (cos_theta < 0.6)
    if bool(on_rim.any()):
        ratio = (radial[on_rim] / planar_z[on_rim].clamp(min=1e-6)).median()
        assert float(ratio) > 1.5, f"rim radial/z ratio {float(ratio):.3f}: looks like planar z"


def test_valid_mask_excludes_the_fisheye_corners():
    seq = _require_adt()
    ds = ADTPairDataset([str(seq)], resolution=64, max_frames=28,
                        baseline_m=(0.0, 99.0), angle_deg=(0.0, 180.0))
    valid = ds[0]["valid"][0]
    assert not bool(valid[0, 0]), "corner outside the imaged cone must be invalid"
    assert bool(valid[32, 32]), "centre must be valid"
    assert 0.3 < float(valid.float().mean()) < 0.95


def test_extrinsics_source_is_reported():
    """Whether the camera extrinsic is real or assumed must be visible."""
    seq = _require_adt()
    ds = ADTPairDataset([str(seq)], resolution=64, max_frames=28,
                        baseline_m=(0.0, 99.0), angle_deg=(0.0, 180.0))
    assert ds.extrinsics_source in {"json", "mps", "projectaria_tools", "device-frame-fallback"}
    assert isinstance(ds.extrinsics_exact, bool)


def test_relative_pose_matches_a_direct_computation():
    seq = _require_adt()
    ds = ADTPairDataset([str(seq)], resolution=64, max_frames=28,
                        baseline_m=(0.0, 99.0), angle_deg=(0.0, 180.0))
    i, j = ds.pairs[0]
    item = ds[0]
    R_gt, t_gt = relative_pose(ds.poses_cw[i].unsqueeze(0), ds.poses_cw[j].unsqueeze(0))
    assert torch.allclose(item["R"].double(), R_gt[0], atol=1e-6)
    assert torch.allclose(item["t"].double(), t_gt[0], atol=1e-6)
