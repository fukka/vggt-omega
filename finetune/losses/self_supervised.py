# Copyright (c) 2026.
"""Label-free self-supervision over a window of frames.

Implements, in one pass over temporal neighbour pairs:

* **photometric view-synthesis** with min-reprojection over neighbours and
  static-pixel auto-masking;
* **geometric cross-view consistency** via the scale-normalized depth residual
  ``|d_proj - d_sampled| / (d_proj + d_sampled)``;
* **edge-aware smoothness**;

and returns the *free dynamic/occlusion mask* ``M = 1 - r`` (detached) that the
geometric residual produces, which higher-level code feeds back as a weight or
routes to dynamic-object handling.

Everything is computed from the model's own predictions (depth + pose); raw
frames enter only to establish photometric correspondences. No depth GT.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from ..geometry import (
    EPS,
    decode_pose_encoding,
    masked_weighted_mean,
    relative_pose,
    sample,
    shifted_indices,
    warp,
)
from .photometric import edge_aware_smoothness, photometric_error


def _geometric_residual(
    depth: torch.Tensor,
    extr: torch.Tensor,
    K: torch.Tensor,
    offset: int,
    images: torch.Tensor | None = None,
    alpha: float = 0.85,
):
    """Per-offset warp. Returns (photo_err|None, residual_r, valid, warped|None).

    Shapes are flattened to ``N = B * S`` over the leading dims.
    """
    B, S = depth.shape[:2]
    H, W = depth.shape[-2:]
    device = depth.device
    idx = shifted_indices(S, offset, device)
    self_pair = idx == torch.arange(S, device=device)  # clamped ends -> src == ref

    ref_d = depth.reshape(B * S, H, W)
    ref_E = extr.reshape(B * S, 4, 4)
    ref_K = K.reshape(B * S, 3, 3)
    src_d = depth[:, idx].reshape(B * S, H, W)
    src_E = extr[:, idx].reshape(B * S, 4, 4)
    src_K = K[:, idx].reshape(B * S, 3, 3)

    T = relative_pose(ref_E, src_E)

    if images is not None:
        src_img = images[:, idx].reshape(B * S, 3, H, W)
        warped, proj_d, pix, valid = warp(src_img, ref_d, ref_K, src_K, T)
    else:
        from ..geometry import backproject, project, transform_points

        pts = transform_points(backproject(ref_d, ref_K), T)
        pix, proj_d = project(pts, src_K)
        warped, valid = None, None
        _, valid = sample(ref_d.unsqueeze(1), pix)  # bounds validity only

    sampled_src_d, valid_d = sample(src_d.unsqueeze(1), pix)
    sampled_src_d = sampled_src_d[:, 0]
    valid_all = valid_d & (proj_d > 0) & (sampled_src_d > 0)
    if valid is not None:
        valid_all = valid_all & valid
    # drop degenerate self-pairs created by clamping at the sequence ends
    not_self = (~self_pair).reshape(1, S).expand(B, S).reshape(B * S)[:, None, None]
    valid_all = valid_all & not_self

    r = (proj_d - sampled_src_d).abs() / (proj_d + sampled_src_d).clamp_min(EPS)

    pe = None
    if images is not None:
        pe = photometric_error(warped, images.reshape(B * S, 3, H, W), alpha=alpha)
        pe_identity = photometric_error(
            images[:, idx].reshape(B * S, 3, H, W),
            images.reshape(B * S, 3, H, W),
            alpha=alpha,
        )
        # auto-mask: keep a neighbour only where warping *improves* over the
        # unwarped frame (suppresses static / textureless / ego-stationary pixels).
        pe = torch.where(pe < pe_identity, pe, torch.full_like(pe, float("inf")))
        pe = pe.masked_fill(~valid_all, float("inf"))

    return pe, r, valid_all, warped


def compute_self_supervised_losses(
    images: torch.Tensor,
    depth: torch.Tensor,
    pose_enc: torch.Tensor,
    conf: torch.Tensor | None = None,
    offsets: Sequence[int] = (-1, 1),
    alpha: float = 0.85,
    extra_static_mask: torch.Tensor | None = None,
    dynamic_mask: bool = False,
    dynamic_pow: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Compute the self-supervised loss dict and the dynamic mask.

    Args:
        images: ``[B,S,3,H,W]`` in ``[0,1]``.
        depth: ``[B,S,H,W]`` metric, strictly positive (squeeze the trailing
            singleton from VGGT-Omega's ``[B,S,H,W,1]`` before calling).
        pose_enc: ``[B,S,9]`` VGGT-Omega pose encoding.
        conf: optional ``[B,S,H,W]`` depth confidence (>= 1); used, detached, as
            a per-pixel weight.
        offsets: temporal neighbour offsets for pairing.
        extra_static_mask: optional ``[B,S,H,W]`` in ``[0,1]`` extra weight
            (e.g. ``1 - hand_mask``) multiplied into the photometric/geometric weight.

    Returns:
        ``(losses, dyn_mask)`` where ``dyn_mask`` is ``[B,S,H,W]`` in ``[0,1]``,
        detached, with 1 = static / geometrically-consistent.
    """
    B, S, _, H, W = images.shape
    extr, K = decode_pose_encoding(pose_enc.float(), (H, W))

    pe_list, r_list, valid_list = [], [], []
    for off in offsets:
        pe, r, valid, _ = _geometric_residual(depth, extr, K, off, images=images, alpha=alpha)
        pe_list.append(pe)
        r_list.append(r)
        valid_list.append(valid)

    pe_stack = torch.stack(pe_list, dim=1)      # [N,O,H,W]
    r_stack = torch.stack(r_list, dim=1)
    valid_stack = torch.stack(valid_list, dim=1)

    # min-reprojection over neighbours
    pe_min, _ = pe_stack.min(dim=1)             # [N,H,W]
    finite = torch.isfinite(pe_min)
    pe_min = pe_min.masked_fill(~finite, 0.0)

    # geometric residual averaged over valid neighbours
    cnt = valid_stack.float().sum(1).clamp_min(1.0)
    r_mean = (r_stack * valid_stack).sum(1) / cnt
    any_valid = valid_stack.any(1)

    # per-pixel weight (detached): depth confidence * optional static mask
    weight = torch.ones_like(r_mean)
    if conf is not None:
        weight = conf.reshape(B * S, H, W).detach().clamp(1.0, 50.0).log() + EPS
    if extra_static_mask is not None:
        weight = weight * extra_static_mask.reshape(B * S, H, W).clamp(0, 1)

    # robust dynamic/occlusion reweighting: down-weight pixels whose cross-view
    # depth is inconsistent (moving / occluded). Detached so it acts as an
    # IRLS-style robust weight (the model can't game it by predicting bad depth).
    if dynamic_mask:
        dyn_w = (1.0 - r_mean).clamp(0.0, 1.0).detach()
        if dynamic_pow != 1.0:
            dyn_w = dyn_w.clamp_min(EPS) ** dynamic_pow
        weight = weight * dyn_w

    photo = masked_weighted_mean(pe_min, weight, finite & any_valid)
    geo = masked_weighted_mean(r_mean, weight, any_valid)

    smooth = depth.new_zeros(())
    for s in range(S):
        smooth = smooth + edge_aware_smoothness(depth[:, s], images[:, s])
    smooth = smooth / S

    losses = {"photometric": photo, "geometric": geo, "smoothness": smooth}

    # free dynamic/occlusion mask: high residual -> likely moving / occluded
    dyn = (1.0 - r_mean).clamp(0, 1)
    dyn = dyn.masked_fill(~any_valid, 1.0)
    dyn_mask = dyn.reshape(B, S, H, W).detach()
    return losses, dyn_mask
