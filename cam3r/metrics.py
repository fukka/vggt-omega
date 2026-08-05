# Copyright (c) 2026.
"""Pose metrics used by CAM3R's tables (paper Sec. 4.1-4.2).

* **RRA@t / RTA@t** -- fraction of pairs whose relative *rotation* / *translation
  direction* error is below ``t`` degrees.  Translation is graded as an angle
  because a two-view prediction is only defined up to scale.
* **mAA@30** -- mean over integer thresholds ``1..30`` of the fraction of pairs
  where *both* errors are under the threshold.  This is the RelPose/PoseDiffusion
  definition the multi-view tables use.
* **ATE RMSE** -- trajectory error after Umeyama sim(3) alignment, as stated in
  the paper ("post-Umeyama alignment").
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from cam3r.geometry import angle_between, geodesic_angle


def relative_pose_error(
    R_pred: torch.Tensor, t_pred: torch.Tensor, R_gt: torch.Tensor, t_gt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> ``(rotation_deg, translation_deg)``, both (B,).

    Translation error is the angle between predicted and GT direction, so it is
    invariant to the predicted magnitude.
    """
    rot = torch.rad2deg(geodesic_angle(R_pred, R_gt))
    tra = torch.rad2deg(angle_between(t_pred, t_gt))
    return rot, tra


def pose_accuracy(
    rot_deg: torch.Tensor, tra_deg: torch.Tensor, thresholds: Sequence[int] = (5, 15, 30)
) -> Dict[str, float]:
    """RRA@t and RTA@t for each threshold, as fractions in [0, 1]."""
    out: Dict[str, float] = {}
    for t in thresholds:
        out[f"RRA@{t}"] = float((rot_deg < t).float().mean())
        out[f"RTA@{t}"] = float((tra_deg < t).float().mean())
    return out


def mean_average_accuracy(
    rot_deg: torch.Tensor, tra_deg: torch.Tensor, max_deg: int = 30
) -> float:
    """mAA@``max_deg``: average over integer thresholds of joint rot+trans accuracy."""
    worst = torch.maximum(rot_deg, tra_deg)
    accs = [float((worst < t).float().mean()) for t in range(1, max_deg + 1)]
    return sum(accs) / len(accs)


def umeyama_alignment(
    src: torch.Tensor, dst: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Least-squares similarity ``dst ~ s R src + t`` (Umeyama 1991).

    ``src``/``dst`` are (N, 3); ``weights`` an optional (N,) of non-negative
    per-correspondence weights (the confidence sigma, when this is used to seed
    global alignment).  Returns ``(s, R, t)``.
    """
    if src.shape != dst.shape or src.dim() != 2 or src.shape[1] != 3:
        raise ValueError(f"expected matching (N, 3) inputs, got {tuple(src.shape)} and {tuple(dst.shape)}")
    src = src.to(torch.float64)
    dst = dst.to(torch.float64)

    if weights is None:
        w = torch.full((src.shape[0],), 1.0 / src.shape[0], dtype=torch.float64)
    else:
        w = weights.to(torch.float64).clamp(min=0.0)
        w = w / w.sum().clamp(min=1e-12)

    mu_s, mu_d = (w.unsqueeze(-1) * src).sum(0), (w.unsqueeze(-1) * dst).sum(0)
    s_c, d_c = src - mu_s, dst - mu_d

    cov = d_c.T @ (w.unsqueeze(-1) * s_c)
    U, D, Vt = torch.linalg.svd(cov)
    S = torch.eye(3, dtype=torch.float64)
    if torch.det(U) * torch.det(Vt) < 0:
        S[2, 2] = -1.0                      # keep the result a rotation, not a reflection
    R = U @ S @ Vt

    var_s = (w * (s_c**2).sum(-1)).sum()
    scale = float((D * torch.diagonal(S)).sum() / var_s.clamp(min=1e-12))
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def ate_rmse(pred_positions: torch.Tensor, gt_positions: torch.Tensor) -> float:
    """RMSE of camera positions after Umeyama sim(3) alignment."""
    scale, R, t = umeyama_alignment(pred_positions, gt_positions)
    aligned = scale * (pred_positions.to(torch.float64) @ R.T) + t
    return float(((aligned - gt_positions.to(torch.float64)) ** 2).sum(-1).mean().sqrt())
