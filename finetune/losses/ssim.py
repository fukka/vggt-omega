# Copyright (c) 2026.
"""Structural dissimilarity used inside the photometric loss (Monodepth2 style)."""
from __future__ import annotations

import torch
import torch.nn as nn


class SSIM(nn.Module):
    """Per-pixel ``(1 - SSIM) / 2`` in ``[0, 1]`` (lower = more similar)."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(3, 1)
        self.pad = nn.ReflectionPad2d(1)
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        y = self.pad(y)
        mu_x = self.pool(x)
        mu_y = self.pool(y)
        sigma_x = self.pool(x * x) - mu_x ** 2
        sigma_y = self.pool(y * y) - mu_y ** 2
        sigma_xy = self.pool(x * y) - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x + sigma_y + self.c2)
        return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)
