# Copyright (c) 2026.
"""Photometric reconstruction terms (appearance-grounded supervision)."""
from __future__ import annotations

import torch

from .ssim import SSIM

_SSIM_CACHE: dict = {}


def _ssim_for(device: torch.device) -> SSIM:
    key = str(device)
    if key not in _SSIM_CACHE:
        _SSIM_CACHE[key] = SSIM().to(device)
    return _SSIM_CACHE[key]


def photometric_error(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
    """``pred, target [N,3,H,W]`` in ``[0,1]`` -> per-pixel error ``[N,H,W]``.

    Convex combination of structural dissimilarity and L1, the standard
    self-supervised photometric metric. ``alpha`` weights SSIM.
    """
    l1 = (pred - target).abs().mean(1)
    ssim = _ssim_for(pred.device)(pred, target).mean(1)
    return alpha * ssim + (1.0 - alpha) * l1


def edge_aware_smoothness(depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Edge-aware first-order smoothness on mean-normalized disparity.

    ``depth [N,H,W]``, ``image [N,3,H,W]``. Regularizes the (otherwise
    under-constrained) depth in textureless regions while respecting image edges.
    """
    disp = 1.0 / depth.clamp_min(1e-3)
    disp = disp / (disp.mean(dim=(1, 2), keepdim=True) + 1e-7)

    d_dx = (disp[:, :, :-1] - disp[:, :, 1:]).abs()
    d_dy = (disp[:, :-1, :] - disp[:, 1:, :]).abs()

    gray = image.mean(1)
    i_dx = (gray[:, :, :-1] - gray[:, :, 1:]).abs()
    i_dy = (gray[:, :-1, :] - gray[:, 1:, :]).abs()

    d_dx = d_dx * torch.exp(-i_dx)
    d_dy = d_dy * torch.exp(-i_dy)
    return d_dx.mean() + d_dy.mean()
