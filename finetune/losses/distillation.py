# Copyright (c) 2026.
"""Cross-model distillation terms for the alternating DAv2 <-> VGGT-Omega loop.

Direction A (DAv2 <- VGGT): ``affine_invariant_l1`` aligns DAv2's affine-invariant
depth to VGGT-Omega's multi-view-consistent (metric) depth and matches it, plus
``multiview_consistency_loss`` injects multi-view coherence into DAv2 using
VGGT-Omega's predicted poses.

Direction B (VGGT <- DAv2): ``ssi_loss`` + ``gradient_matching_loss`` transfer
DAv2's sharp per-frame structure into VGGT-Omega **without** overwriting its
metric scale (scale-shift invariant), supplying a finite-depth prior that kills
the infinite-depth holes on hands / carried objects.
"""
from __future__ import annotations

from typing import Sequence

import torch

from ..geometry import EPS, decode_pose_encoding, masked_mean
from .self_supervised import _geometric_residual


def to_disparity(depth: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Metric depth -> disparity (inverse depth), the natural space for relative
    monocular models and for structure comparison."""
    return 1.0 / depth.clamp_min(eps)


def _affine_fit(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Per-sample least-squares ``s, o`` minimizing ``||s*pred + o - target||^2``.

    ``pred, target, mask`` are ``[N, K]`` (flattened per sample). Returns
    ``(s [N,1], o [N,1])``.
    """
    w = mask.to(pred.dtype)
    sw = w.sum(1).clamp_min(1.0)
    mp = (pred * w).sum(1) / sw
    mt = (target * w).sum(1) / sw
    pc = pred - mp[:, None]
    tc = target - mt[:, None]
    cov = (pc * tc * w).sum(1)
    var = (pc * pc * w).sum(1).clamp_min(1e-8)
    s = cov / var
    o = mt - s * mp
    return s[:, None], o[:, None]


def affine_invariant_l1(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Align ``pred`` to ``target`` per-sample (scale + shift), then masked L1.

    Use in *disparity* space. ``pred, target [N,H,W]``.
    """
    N = pred.shape[0]
    p = pred.reshape(N, -1)
    t = target.reshape(N, -1)
    m = torch.ones_like(p) if mask is None else mask.reshape(N, -1).to(p.dtype)
    s, o = _affine_fit(p, t, m)
    return masked_mean((s * p + o - t).abs(), m)


def _ssi_normalize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Scale-shift normalization by weighted mean + mean-absolute-deviation."""
    N = x.shape[0]
    xf = x.reshape(N, -1)
    m = mask.reshape(N, -1).to(x.dtype)
    sw = m.sum(1).clamp_min(1.0)
    mean = (xf * m).sum(1) / sw
    dev = ((xf - mean[:, None]).abs() * m).sum(1) / sw
    return (xf - mean[:, None]) / dev[:, None].clamp_min(1e-6)


def ssi_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Scale-shift-invariant L1 between two depth/disparity maps ``[N,H,W]``."""
    N = pred.shape[0]
    m = torch.ones_like(pred.reshape(N, -1)) if mask is None else mask.reshape(N, -1)
    pn = _ssi_normalize(pred, m)
    tn = _ssi_normalize(target, m)
    return masked_mean((pn - tn).abs(), m)


def gradient_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    scales: int = 4,
) -> torch.Tensor:
    """Multi-scale gradient matching on the aligned residual (transfers edges).

    ``pred, target [N,H,W]`` in disparity space.
    """
    N = pred.shape[0]
    p = pred.reshape(N, -1)
    t = target.reshape(N, -1)
    m = torch.ones_like(p) if mask is None else mask.reshape(N, -1).to(p.dtype)
    s, o = _affine_fit(p, t, m)
    diff = (s * p + o - t).reshape(N, pred.shape[1], pred.shape[2])

    total = pred.new_zeros(())
    for k in range(scales):
        step = 2 ** k
        d = diff[:, ::step, ::step]
        gx = (d[:, :, :-1] - d[:, :, 1:]).abs().mean()
        gy = (d[:, :-1, :] - d[:, 1:, :]).abs().mean()
        total = total + gx + gy
    return total / scales


def multiview_consistency_loss(
    depth: torch.Tensor,
    pose_enc: torch.Tensor,
    offsets: Sequence[int] = (-1, 1),
) -> torch.Tensor:
    """Geometric cross-view consistency of ``depth`` under the given poses.

    Used in Phase A to make DAv2's per-frame depth multi-view consistent using
    VGGT-Omega's predicted cameras (no images, no GT). ``depth [B,S,H,W]``.
    """
    B, S = depth.shape[:2]
    H, W = depth.shape[-2:]
    extr, K = decode_pose_encoding(pose_enc.float(), (H, W))
    r_list, v_list = [], []
    for off in offsets:
        _, r, valid, _ = _geometric_residual(depth, extr, K, off, images=None)
        r_list.append(r)
        v_list.append(valid)
    r_stack = torch.stack(r_list, 1)
    v_stack = torch.stack(v_list, 1)
    cnt = v_stack.float().sum(1).clamp_min(1.0)
    r_mean = (r_stack * v_stack).sum(1) / cnt
    return masked_mean(r_mean, v_stack.any(1))
