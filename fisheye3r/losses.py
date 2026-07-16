"""Training objectives for the three learning schemes (paper Sec. 3.3, Eq. 11-13).

The paper reuses "the loss used by the original 3D reconstruction method".
For VGGT-Omega that is a confidence-weighted depth regression plus a camera
loss on the 9D pose encoding; we implement the standard VGGT forms:

  depth: conf-weighted L1 on depth + log-depth gradient matching
         L = mean( c * |d - d*| - alpha * log c ) + w_grad * L_grad
  pose:  smooth-L1 on the 9D encoding (translation, quaternion, FoV).

Supervision is always applied in the *perspective* domain: fisheye-frame
predictions are undistorted with T^-1 before the loss (Eq. 11-12) and masked
to pixels where the distortion round-trip is defined. Poses need no transform
(extrinsics are camera-type independent).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fisheye3r.distortion import KannalaBrandtCamera, undistort_dense


def confidence_depth_loss(
    pred: torch.Tensor,
    conf: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    alpha: float = 0.2,
    grad_weight: float = 0.5,
    grad_scales: int = 4,
) -> torch.Tensor:
    """pred/conf/target: (N, H, W); valid: (N, H, W) bool."""
    valid = valid & (target > 0) & torch.isfinite(target)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    err = (pred - target).abs()
    conf = conf.clamp(min=1.0 + 1e-6)
    nll = conf * err - alpha * torch.log(conf)
    loss = nll[valid].mean()

    if grad_weight > 0:
        log_p = torch.log(pred.clamp(min=1e-6)).unsqueeze(1)
        log_t = torch.log(target.clamp(min=1e-6)).unsqueeze(1)
        m = valid.unsqueeze(1).float()
        g = 0.0
        for s in range(grad_scales):
            if s > 0:
                log_p = F.avg_pool2d(log_p, 2)
                log_t = F.avg_pool2d(log_t, 2)
                m = (F.avg_pool2d(m, 2) > 0.999).float()
            diff = (log_p - log_t) * m
            gx = (diff[..., :, 1:] - diff[..., :, :-1]).abs() * m[..., :, 1:] * m[..., :, :-1]
            gy = (diff[..., 1:, :] - diff[..., :-1, :]).abs() * m[..., 1:, :] * m[..., :-1, :]
            denom = m[..., :, 1:].sum() + m[..., 1:, :].sum() + 1e-6
            g = g + (gx.sum() + gy.sum()) / denom
        loss = loss + grad_weight * g / grad_scales
    return loss


def pose_encoding_loss(
    pred_enc: torch.Tensor,
    target_enc: torch.Tensor,
    supervise_fov: torch.Tensor | None = None,
) -> torch.Tensor:
    """pred/target: (B, S, 9). supervise_fov: optional (B, S) bool - the FoV
    components of the encoding describe the *perspective* image, which is
    ill-defined when supervising directly on real fisheye GT (SL+); callers
    can switch those two dims off per frame."""
    l = F.smooth_l1_loss(pred_enc, target_enc, beta=0.1, reduction="none")
    if supervise_fov is not None:
        w = torch.ones_like(l)
        w[..., 7:] = supervise_fov.unsqueeze(-1).float()
        l = l * w
    return l.mean()


def _flatten_frames(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *x.shape[2:])


def undistort_predictions(
    depth: torch.Tensor,
    conf: torch.Tensor,
    cam: KannalaBrandtCamera,
    flags: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply T^-1 to the fisheye frames of a batch of dense predictions.

    depth/conf: (B, S, H, W); cam: batched over B*S (one entry per frame);
    flags: (B, S) bool. Perspective frames pass through with a full mask.
    Returns (depth_p, conf_p, valid) on the perspective grid.
    """
    B, S, H, W = depth.shape
    d = _flatten_frames(depth).unsqueeze(1)
    c = _flatten_frames(conf).unsqueeze(1)
    stacked, valid = undistort_dense(torch.cat([d, c], dim=1), cam)
    f = flags.reshape(-1, 1, 1)
    d_out = torch.where(f, stacked[:, 0], _flatten_frames(depth))
    c_out = torch.where(f, stacked[:, 1], _flatten_frames(conf))
    v_out = torch.where(f, valid, torch.ones_like(valid))
    return d_out.view(B, S, H, W), c_out.view(B, S, H, W), v_out.view(B, S, H, W)


def scheme_loss(
    student: dict[str, torch.Tensor],
    target_depth: torch.Tensor,
    target_pose_enc: torch.Tensor,
    cam: KannalaBrandtCamera | None,
    flags: torch.Tensor,
    target_valid: torch.Tensor | None = None,
    teacher_conf: torch.Tensor | None = None,
    teacher_conf_quantile: float = 0.2,
    pose_weight: float = 5.0,
    already_undistorted: bool = False,
    supervise_fov_on_fisheye: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Shared loss for SSL (Eq. 11), SL (Eq. 12) and SL+ (Eq. 13).

    SSL: target_depth/pose_enc = frozen-teacher predictions on the perspective
         frames, teacher_conf gates unreliable pseudo-labels (implementation
         choice; the paper does not detail pseudo-label filtering).
    SL:  target_depth/pose_enc = perspective ground truth.
    SL+: real fisheye GT; pass already_undistorted=True and cam=None to
         supervise in the fisheye domain directly (Eq. 13 applies no T^-1).
    """
    depth, conf = student["depth"], student["depth_conf"]
    if not already_undistorted:
        assert cam is not None
        depth, conf, valid = undistort_predictions(depth, conf, cam, flags)
    else:
        valid = torch.ones_like(depth, dtype=torch.bool)
    if target_valid is not None:
        valid = valid & target_valid

    if teacher_conf is not None:
        # Keep only pixels where the teacher is confident (per-frame quantile).
        q = torch.quantile(
            teacher_conf.flatten(2).float(), teacher_conf_quantile, dim=-1
        ).unsqueeze(-1).unsqueeze(-1)
        valid = valid & (teacher_conf >= q)

    l_depth = confidence_depth_loss(
        _flatten_frames(depth), _flatten_frames(conf), _flatten_frames(target_depth), _flatten_frames(valid)
    )
    fov_mask = None if supervise_fov_on_fisheye else ~flags
    l_pose = pose_encoding_loss(student["pose_enc"], target_pose_enc, supervise_fov=fov_mask)
    total = l_depth + pose_weight * l_pose
    return total, {"depth": float(l_depth.detach()), "pose": float(l_pose.detach())}
