"""H5 losses: compression-weighted depth, rim-feature preservation,
multi-frame rim consistency. Protocol: ../protocol.md.

Pure tensor functions — no backbone import — so the CPU test suite runs in
seconds and the GPU trainer imports the same, tested code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

RUN009 = (Path(__file__).resolve().parents[2]
          / "h2-center-safe-adapter" / "results" / "run_009.json")


def compression_weight_map(theta: torch.Tensor, gt_range: torch.Tensor,
                           alpha: float = 2.0,
                           bias_json: Path = RUN009) -> torch.Tensor:
    """`w = 1 + alpha*|b(theta, gt_range)|` from the measured bias map.

    The run_009 grid is 8 theta bins x 5 depth bands; interpolate bilinearly
    in (theta, log-depth) and clamp outside the measured support. NaN cells
    (never measured) contribute the neutral weight 1.
    """
    d = json.loads(Path(bias_json).read_text())
    bias = torch.tensor(d["bias_log"], dtype=torch.float32)      # (8, 5)
    bias = torch.nan_to_num(bias, nan=0.0)
    t_mid = torch.tensor(d["theta_bin_mid_deg"], dtype=torch.float32)
    edges = d["depth_edges_m"]
    d_mid = torch.tensor([0.5 * (edges[i] + edges[i + 1])
                          for i in range(len(edges) - 1)],
                         dtype=torch.float32).log()

    # normalized grid coords for grid_sample: x over depth bands, y over theta
    tq = torch.rad2deg(theta).clamp(float(t_mid[0]), float(t_mid[-1]))
    ty = (tq - t_mid[0]) / (t_mid[-1] - t_mid[0]) * 2 - 1
    dq = gt_range.clamp_min(1e-3).log().clamp(float(d_mid[0]), float(d_mid[-1]))
    dx = (dq - d_mid[0]) / (d_mid[-1] - d_mid[0]) * 2 - 1
    grid = torch.stack([dx, ty], dim=-1)[None]                   # (1, H, W, 2)
    b = F.grid_sample(bias[None, None], grid, mode="bilinear",
                      padding_mode="border", align_corners=True)[0, 0]
    return 1.0 + alpha * b.abs()


def depth_loss(pred_range: torch.Tensor, gt_range: torch.Tensor,
               valid: torch.Tensor, theta: torch.Tensor,
               alpha: float = 2.0) -> torch.Tensor:
    """Compression-weighted L1 on log-range (protocol loss 1)."""
    w = compression_weight_map(theta, gt_range, alpha=alpha).to(pred_range)
    r = (pred_range.clamp_min(1e-6).log() - gt_range.clamp_min(1e-6).log()).abs()
    m = valid.float()
    return (w * r * m).sum() / (w * m).sum().clamp_min(1.0)


def rim_feature_loss(student_tokens: torch.Tensor, teacher_tokens: torch.Tensor,
                     theta_patch: torch.Tensor,
                     rim_deg: float = 35.0) -> torch.Tensor:
    """L2 between student and (detached) teacher tokens on rim patches
    (protocol loss 2). Tokens: (N, C); theta_patch: (N,) radians."""
    rim = theta_patch > math.radians(rim_deg)
    if not bool(rim.any()):
        return student_tokens.sum() * 0.0
    diff = student_tokens[rim] - teacher_tokens[rim].detach()
    return diff.pow(2).mean()


def warp_i_to_j(range_i: torch.Tensor, camera, R_rel: torch.Tensor,
                t_rel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    """Lift frame-i range through the KB4 camera, transform with (R_rel,
    t_rel) (j-from-i), and return (uv_j, range_in_j, front_mask).

    range_i: (H, W) euclidean range. camera: raytun3r Camera (unproject /
    project / theta_max). Differentiable w.r.t. range_i.
    """
    h, w = range_i.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=range_i.dtype),
                            torch.arange(w, dtype=range_i.dtype), indexing="ij")
    uv = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
    rays = camera.unproject(uv).to(range_i.dtype)                # unit (N, 3)
    X_i = rays * range_i.reshape(-1, 1)
    X_j = X_i @ R_rel.to(range_i.dtype).T + t_rel.to(range_i.dtype)[None, :]
    r_j = X_j.norm(dim=-1)
    theta_j = torch.atan2(X_j[:, :2].norm(dim=-1), X_j[:, 2])
    uv_j = camera.project(X_j)
    front = ((X_j[:, 2] > 1e-3) & (theta_j <= camera.theta_max)
             & (uv_j[:, 0] >= 0) & (uv_j[:, 0] < w)
             & (uv_j[:, 1] >= 0) & (uv_j[:, 1] < h))
    return (uv_j.reshape(h, w, 2), r_j.reshape(h, w),
            front.reshape(h, w))


def multiframe_rim_loss(range_i: torch.Tensor, range_j: torch.Tensor,
                        camera, R_rel: torch.Tensor, t_rel: torch.Tensor,
                        theta: torch.Tensor,
                        rim_power: float = 2.0) -> torch.Tensor:
    """Warp i's prediction into j and penalize log-range disagreement with
    j's prediction, weighted toward the rim of the SOURCE pixel (protocol
    loss 3). Both ranges differentiable; weight `(theta/theta_max)^rim_power`.
    """
    h, w = range_i.shape
    uv_j, r_in_j, front = warp_i_to_j(range_i, camera, R_rel, t_rel)
    gx = 2 * uv_j[..., 0] / (w - 1) - 1
    gy = 2 * uv_j[..., 1] / (h - 1) - 1
    grid = torch.stack([gx, gy], dim=-1)[None]
    r_j_sampled = F.grid_sample(range_j[None, None], grid, mode="bilinear",
                                padding_mode="zeros",
                                align_corners=True)[0, 0]
    ok = front & (r_j_sampled > 1e-3)
    if not bool(ok.any()):
        return range_i.sum() * 0.0
    wgt = (theta / float(camera.theta_max)).clamp(0, 1) ** rim_power
    diff = (r_in_j.clamp_min(1e-6).log()
            - r_j_sampled.clamp_min(1e-6).log()).abs()
    m = ok.float() * wgt.to(range_i.dtype)
    return (diff * m).sum() / m.sum().clamp_min(1.0)
