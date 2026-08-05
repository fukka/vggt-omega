"""Pose metrics: RRA/RTA, mAA@30, ATE (paper Sec. 4.1-4.2)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cam3r.geometry import random_rotations
from cam3r.metrics import (
    ate_rmse,
    mean_average_accuracy,
    pose_accuracy,
    relative_pose_error,
    umeyama_alignment,
)


def _rot_z(deg: float) -> torch.Tensor:
    a = math.radians(deg)
    return torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                         [math.sin(a), math.cos(a), 0.0],
                         [0.0, 0.0, 1.0]]).unsqueeze(0)


def test_zero_error_on_identical_poses():
    R = random_rotations(8, seed=0)
    t = torch.randn(8, 3)
    rot, tra = relative_pose_error(R, t, R.clone(), t.clone())
    assert float(rot.max()) < 1e-4 and float(tra.max()) < 1e-4


def test_translation_error_is_an_angle_not_a_distance():
    """RTA grades direction only -- predictions are up to scale."""
    R = torch.eye(3).unsqueeze(0)
    t_gt = torch.tensor([[0.0, 0.0, 1.0]])
    _, tra = relative_pose_error(R, t_gt * 1000.0, R, t_gt)
    assert float(tra) < 1e-4


def test_known_rotation_error():
    R_gt = torch.eye(3).unsqueeze(0)
    t = torch.tensor([[0.0, 0.0, 1.0]])
    rot, _ = relative_pose_error(_rot_z(20.0), t, R_gt, t)
    assert abs(float(rot) - 20.0) < 1e-3


def test_pose_accuracy_thresholding():
    rot = torch.tensor([1.0, 10.0, 20.0, 40.0])
    tra = torch.tensor([1.0, 2.0, 3.0, 4.0])
    acc = pose_accuracy(rot, tra, thresholds=(15, 30))
    assert acc["RRA@15"] == 0.5           # 1 and 10 deg pass
    assert acc["RRA@30"] == 0.75
    assert acc["RTA@15"] == 1.0


def test_maa_is_between_the_loosest_and_tightest_accuracy():
    torch.manual_seed(1)
    rot = torch.rand(500) * 40
    tra = torch.rand(500) * 40
    maa = mean_average_accuracy(rot, tra, max_deg=30)
    strict = float(((torch.maximum(rot, tra)) < 1).float().mean())
    loose = float(((torch.maximum(rot, tra)) < 30).float().mean())
    assert strict <= maa <= loose


def test_maa_is_one_when_everything_is_exact():
    z = torch.zeros(10)
    assert abs(mean_average_accuracy(z, z, max_deg=30) - 1.0) < 1e-6


def test_umeyama_recovers_a_known_similarity():
    torch.manual_seed(2)
    src = torch.randn(50, 3)
    R = random_rotations(1, seed=3)[0].double()
    s, t = 2.5, torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    dst = s * (src.double() @ R.T) + t
    s_hat, R_hat, t_hat = umeyama_alignment(src.double(), dst)
    assert abs(s_hat - s) < 1e-6
    assert torch.allclose(R_hat, R, atol=1e-6)
    assert torch.allclose(t_hat, t, atol=1e-5)


def test_ate_is_zero_after_a_similarity_transform():
    """ATE is measured post-Umeyama, so a globally scaled/rotated path is exact."""
    torch.manual_seed(4)
    gt = torch.randn(30, 3).double()
    R = random_rotations(1, seed=5)[0].double()
    pred = 3.0 * (gt @ R.T) + torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)
    assert ate_rmse(pred, gt) < 1e-6


def test_ate_grows_with_a_real_deviation():
    torch.manual_seed(6)
    gt = torch.randn(30, 3).double()
    pred = gt.clone()
    pred[10] += torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert ate_rmse(pred, gt) > 0.1


def test_pose_error_batches_over_many_pairs():
    R_pred, R_gt = random_rotations(64, seed=7), random_rotations(64, seed=8)
    t_pred, t_gt = torch.randn(64, 3), torch.randn(64, 3)
    rot, tra = relative_pose_error(R_pred, t_pred, R_gt, t_gt)
    assert rot.shape == (64,) and tra.shape == (64,)
    assert float(rot.min()) >= 0 and float(rot.max()) <= 180.0
    assert float(tra.min()) >= 0 and float(tra.max()) <= 180.0
