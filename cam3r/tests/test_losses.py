"""CAM3R training objectives (Eq. 5-11)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cam3r.geometry import make_se3, random_rotations, relative_pose
from cam3r.losses import (
    angular_loss,
    asymmetric_angular_loss,
    cam3r_loss,
    local_regression_loss,
    relative_pose_loss,
)


# --------------------------------------------------------------- Eq. 5 / 6

def test_asymmetric_loss_is_zero_on_a_perfect_match():
    x = torch.randn(100)
    assert float(asymmetric_angular_loss(x, x.clone(), alpha=0.7)) == 0.0


def test_alpha_penalizes_under_prediction_more():
    """alpha=0.7 weights theta_hat < theta_star at 0.7 and the other side at 0.3.

    This asymmetry is the paper's stated mechanism against collapsing to a
    narrow pinhole FoV: predicting too small an incidence angle costs more.
    """
    gt = torch.zeros(1)
    under = asymmetric_angular_loss(torch.tensor([-1.0]), gt, alpha=0.7)
    over = asymmetric_angular_loss(torch.tensor([1.0]), gt, alpha=0.7)
    assert abs(float(under) - 0.7) < 1e-6
    assert abs(float(over) - 0.3) < 1e-6
    assert float(under) > float(over)


def test_alpha_half_is_symmetric_l1():
    pred, gt = torch.randn(64), torch.randn(64)
    got = asymmetric_angular_loss(pred, gt, alpha=0.5)
    assert torch.allclose(got, 0.5 * (pred - gt).abs().mean(), atol=1e-6)


def test_angular_loss_is_zero_on_identical_ray_fields():
    torch.manual_seed(0)
    rays = torch.nn.functional.normalize(torch.randn(2, 3, 8, 8), dim=1)
    assert float(angular_loss(rays, rays.clone())) < 1e-6


def test_angular_loss_wraps_azimuth_across_the_seam():
    """Two rays 0.02 rad apart straddling phi = +-pi must read as close."""
    a = torch.tensor([[math.cos(math.pi - 0.01), math.sin(math.pi - 0.01), 0.0]])
    b = torch.tensor([[math.cos(-math.pi + 0.01), math.sin(-math.pi + 0.01), 0.0]])
    a = a.T.reshape(1, 3, 1, 1)
    b = b.T.reshape(1, 3, 1, 1)
    assert float(angular_loss(a, b)) < 0.02, "azimuth difference did not wrap"


def test_angular_loss_respects_the_valid_mask():
    torch.manual_seed(1)
    rays = torch.nn.functional.normalize(torch.randn(1, 3, 8, 8), dim=1)
    other = torch.nn.functional.normalize(torch.randn(1, 3, 8, 8), dim=1)
    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[0, :4] = True
    mixed = rays.clone()
    mixed[:, :, 4:] = other[:, :, 4:]          # corrupt only the masked-out half
    assert float(angular_loss(mixed, rays, valid=mask)) < 1e-6


# ------------------------------------------------------------------- Eq. 7 / 8

def test_local_regression_is_zero_when_shapes_agree_up_to_scale():
    """Both sides are normalized by their own mean norm, so scale cancels."""
    torch.manual_seed(2)
    pts = torch.randn(2, 3, 16, 16).abs() + 1.0
    assert float(local_regression_loss(pts * 7.5, pts)) < 1e-6


def test_local_regression_penalizes_shape_error():
    torch.manual_seed(3)
    gt = torch.randn(1, 3, 8, 8).abs() + 1.0
    bad = gt.clone()
    bad[:, 0] *= 3.0                            # anisotropic: not absorbed by scale
    assert float(local_regression_loss(bad, gt)) > 0.1


def test_local_regression_ignores_invalid_pixels():
    torch.manual_seed(4)
    gt = torch.randn(1, 3, 8, 8).abs() + 1.0
    pred = gt.clone()
    pred[:, :, 4:] = 99.0
    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[0, :4] = True
    assert float(local_regression_loss(pred, gt, valid=mask)) < 1e-6


def test_confidence_weighting_rewards_being_confident_when_right():
    """DUSt3R-style conf term: a correct prediction should prefer high sigma."""
    torch.manual_seed(5)
    gt = torch.randn(1, 3, 8, 8).abs() + 1.0
    conf_hi = torch.full((1, 8, 8), 5.0)
    conf_lo = torch.full((1, 8, 8), 0.2)
    hi = local_regression_loss(gt.clone(), gt, conf=conf_hi, conf_mode="dust3r")
    lo = local_regression_loss(gt.clone(), gt, conf=conf_lo, conf_mode="dust3r")
    assert float(hi) < float(lo)


# ------------------------------------------------------------------ Eq. 9 / 10

def test_pose_loss_is_zero_at_ground_truth():
    R = random_rotations(4, seed=6)
    t = torch.randn(4, 3)
    s = t.norm(dim=-1)
    t_hat = torch.nn.functional.normalize(t, dim=-1)
    loss, logs = relative_pose_loss(R, t_hat, s, R.clone(), t.clone())
    assert float(loss) < 1e-6, logs


def test_pose_loss_rotation_term_is_the_geodesic_angle():
    a = math.radians(20.0)
    Rz = torch.tensor([[[math.cos(a), -math.sin(a), 0.0],
                        [math.sin(a), math.cos(a), 0.0],
                        [0.0, 0.0, 1.0]]])
    t = torch.tensor([[0.0, 0.0, 1.0]])
    _, logs = relative_pose_loss(Rz, t, torch.ones(1), torch.eye(3).unsqueeze(0), t.clone())
    assert abs(logs["pose/rot_rad"] - a) < 1e-5


def test_pose_loss_translation_term_ignores_predicted_magnitude():
    """The translation term grades direction only; magnitude is L_scale's job."""
    R = torch.eye(3).unsqueeze(0)
    t_gt = torch.tensor([[0.0, 0.0, 2.0]])
    d = torch.tensor([[0.0, 0.0, 1.0]])
    _, a = relative_pose_loss(R, d, torch.tensor([1.0]), R, t_gt)
    _, b = relative_pose_loss(R, d, torch.tensor([9.0]), R, t_gt)
    assert a["pose/trans_sq"] == b["pose/trans_sq"] == 0.0


def test_pose_loss_supervises_scale_against_the_metric_gt():
    R = torch.eye(3).unsqueeze(0)
    t_gt = torch.tensor([[0.0, 0.0, 2.0]])
    d = torch.tensor([[0.0, 0.0, 1.0]])
    _, exact = relative_pose_loss(R, d, torch.tensor([2.0]), R, t_gt)
    _, wrong = relative_pose_loss(R, d, torch.tensor([0.2]), R, t_gt)
    assert exact["pose/scale_log_sq"] < 1e-10
    assert wrong["pose/scale_log_sq"] > 1.0


def test_pose_loss_does_not_reward_shrinking_the_scale():
    """Regression: the naive Eq. 10 reading is minimized by driving s -> 0.

    With the prediction ``s*u_p`` and the target ``sg(s)*u_g``, the loss is
    ``s^2||u_p-u_g||^2`` and dL/ds = 2s(1-cos) > 0 for any direction error, so
    gradient descent collapses the scale instead of fixing the direction.
    Nothing else in the objective pins ``s`` down, so this has to be checked here.
    """
    R = torch.eye(3).unsqueeze(0)
    t_gt = torch.tensor([[0.0, 0.0, 1.0]])
    bad_dir = torch.tensor([[1.0, 0.0, 0.0]])          # 90 deg wrong
    scale = torch.tensor([1.0], requires_grad=True)

    loss, _ = relative_pose_loss(R, bad_dir, scale, R, t_gt, w_scale=0.0)
    loss.backward()
    grad = 0.0 if scale.grad is None else float(scale.grad)
    assert grad == 0.0, f"direction error still pushes on scale (grad {grad})"

    # And with scale supervision on, the pull is toward the GT magnitude.
    scale2 = torch.tensor([0.1], requires_grad=True)
    loss2, _ = relative_pose_loss(R, bad_dir, scale2, R, t_gt)
    loss2.backward()
    assert float(scale2.grad) < 0, "under-scaled prediction should be pushed up"


def test_scale_supervision_can_be_disabled_for_up_to_scale_corpora():
    R = torch.eye(3).unsqueeze(0)
    t_gt = torch.tensor([[0.0, 0.0, 5.0]])
    d = torch.tensor([[0.0, 0.0, 1.0]])
    loss, _ = relative_pose_loss(R, d, torch.tensor([0.01]), R, t_gt, w_scale=0.0)
    assert float(loss) < 1e-6


# ------------------------------------------------------------------- Eq. 11

def test_total_loss_runs_and_backpropagates():
    torch.manual_seed(7)
    B, H, W = 2, 8, 8
    preds = {
        "rays": [torch.nn.functional.normalize(torch.randn(B, 3, H, W), dim=1).requires_grad_() for _ in range(2)],
        "points": [torch.randn(B, 3, H, W).abs().add(1.0).requires_grad_() for _ in range(2)],
        "conf": [torch.rand(B, H, W).add(0.5).requires_grad_() for _ in range(2)],
        "R": random_rotations(B, seed=8).requires_grad_(),
        "t_dir": torch.nn.functional.normalize(torch.randn(B, 3), dim=-1).requires_grad_(),
        "scale": torch.rand(B).add(0.5).requires_grad_(),
    }
    T1 = make_se3(random_rotations(B, seed=9), torch.randn(B, 3))
    T2 = make_se3(random_rotations(B, seed=10), torch.randn(B, 3))
    R_gt, t_gt = relative_pose(T1, T2)
    targets = {
        "rays": [torch.nn.functional.normalize(torch.randn(B, 3, H, W), dim=1) for _ in range(2)],
        "points": [torch.randn(B, 3, H, W).abs().add(1.0) for _ in range(2)],
        "valid": [torch.ones(B, H, W, dtype=torch.bool) for _ in range(2)],
        "R": R_gt,
        "t": t_gt,
    }
    loss, logs = cam3r_loss(preds, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert preds["R"].grad is not None and torch.isfinite(preds["R"].grad).all()
    assert preds["rays"][0].grad.abs().sum() > 0
    for key in ("loss/angular", "loss/regr", "loss/pose", "loss/total"):
        assert key in logs
