# Copyright (c) 2026.
"""Unsupervised losses for DAC-style ERP finetuning of DAv2 (no depth GT).

The DAC novelty is to reason about depth in an ERP canonical space and to be
**equivariant** to the viewpoint/scale (pitch / roll / FOV) augmentations DAC
applies on the sphere. We instil that equivariance without labels, on EgoExo4D
fisheye, via three terms:

* :func:`erp_equivariance_loss` — apply a 2-D similarity ``T`` (in-plane roll +
  zoom/FOV + small shift) to the ERP patch; a depth model that reasons correctly
  in ERP space must satisfy ``DAv2(T·I) ≈ T·DAv2(I)`` up to scale & shift. Rotation
  is exactly depth-equivariant and SSI absorbs the zoom's global scale, so this is
  a well-posed consistency. The target ``T·DAv2(I)`` comes from a frozen / EMA
  teacher, making it a self-distillation that propagates the teacher's reliable
  (low-distortion, central) structure into the augmented / peripheral views.
* :func:`erp_anchor_loss` — a scale-shift-invariant anchor to the **frozen
  pretrained** DAv2, protecting its strong structure prior from drifting while the
  consistency term adapts it to the fisheye/ERP domain.
* edge-aware smoothness (reused from :mod:`finetune.losses.photometric`).

All terms operate inside the ERP ``active`` cone, so the fold-masked periphery
never supervises the model.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .distillation import ssi_loss


def random_similarity(
    batch: int,
    roll_deg: float,
    scale_lo: float,
    scale_hi: float,
    trans_frac: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Random 2-D similarity ``theta`` ``[B,2,3]`` for ``F.affine_grid``.

    Combines an in-plane rotation (``±roll_deg``), an isotropic scale
    (``[scale_lo, scale_hi]``, the ERP zoom ≈ FOV change) and a small normalized
    translation (``±trans_frac``). The same ``theta`` is applied to the input image
    and to the teacher depth, so the equivariance target is exact.
    """
    def _u(lo: float, hi: float) -> torch.Tensor:
        r = torch.rand(batch, device=device, generator=generator)
        return lo + (hi - lo) * r

    ang = _u(-math.radians(roll_deg), math.radians(roll_deg))
    s = _u(scale_lo, scale_hi)
    tx = _u(-trans_frac, trans_frac)
    ty = _u(-trans_frac, trans_frac)
    ca, sa = torch.cos(ang) * s, torch.sin(ang) * s
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = ca, -sa, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sa, ca, ty
    return theta


def warp_similarity(x: torch.Tensor, theta: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Apply ``theta`` to ``x`` ``[B,C,H,W]`` via ``affine_grid`` + ``grid_sample``
    (zeros padding, so out-of-frame samples are 0 = invalid)."""
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)


def _bhw(depth: torch.Tensor) -> torch.Tensor:
    """``[B,1,H,W]`` or ``[B,1,1,H,W]`` → ``[B,H,W]``."""
    while depth.dim() > 3:
        depth = depth[:, 0]
    return depth


def erp_equivariance_loss(
    student_depth_aug: torch.Tensor,   # DAv2(T·I)   [B,1,H,W] or [B,H,W]
    teacher_depth: torch.Tensor,       # DAv2(I)     (detached upstream)
    theta: torch.Tensor,               # the same T applied to the input
    active: torch.Tensor,              # [B,1,H,W] or [B,H,W] ERP cone mask of I
    min_valid: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SSI consistency ``DAv2(T·I) ≈ T·DAv2(I)`` on the warped active region.

    Returns ``(loss, valid_mask[B,H,W])``. ``loss`` is a scalar; if no batch item
    has enough overlap it is a (grad-free) zero.
    """
    s = _bhw(student_depth_aug)
    t = _bhw(teacher_depth)
    a = _bhw(active)
    B, H, W = s.shape
    # warp teacher depth + active by the SAME theta into the augmented frame.
    t_w = _bhw(warp_similarity(t.unsqueeze(1), theta))
    a_w = _bhw(warp_similarity(a.unsqueeze(1), theta, mode="nearest"))
    valid = (a_w > 0.5) & (a > 0.5)            # in-FOV in both I and T·I frames
    if valid.reshape(B, -1).sum(1).min() < min_valid:
        # drop items with too little overlap to fit a stable scale/shift
        keep = valid.reshape(B, -1).sum(1) >= min_valid
        if keep.sum() == 0:
            return s.new_zeros(()), valid
        s, t_w, valid = s[keep], t_w[keep], valid[keep]
    loss = ssi_loss(s, t_w.detach(), mask=valid.float())
    return loss, valid


def erp_anchor_loss(
    student_depth: torch.Tensor,   # DAv2(I)            [B,1,H,W] or [B,H,W]
    ref_depth: torch.Tensor,       # frozen pretrained  (detached upstream)
    active: torch.Tensor,
) -> torch.Tensor:
    """Scale-shift-invariant anchor to the frozen pretrained DAv2 on the active cone."""
    s = _bhw(student_depth)
    r = _bhw(ref_depth)
    a = _bhw(active)
    return ssi_loss(s, r.detach(), mask=(a > 0.5).float())
