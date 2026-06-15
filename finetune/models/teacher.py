# Copyright (c) 2026.
"""Exponential-moving-average teacher for stabilizing self-distillation."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EmaTeacher:
    """Maintains an EMA copy of a student model (frozen, eval mode)."""

    def __init__(self, student: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.model = copy.deepcopy(student)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(self.model.parameters(), student.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for ema_b, b in zip(self.model.buffers(), student.buffers()):
            ema_b.copy_(b)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
