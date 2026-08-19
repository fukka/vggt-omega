"""Minimal LoRA for the H5 finetune — no external dependency.

Wraps chosen nn.Linear layers with a zero-initialized low-rank residual.
`enabled` can be toggled to recover the pristine base path exactly (that IS
the teacher — no second copy of the backbone in memory).
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.enabled:
            y = y + (x @ self.A.T @ self.B.T) * self.scale
        return y


def inject(model: nn.Module, name_patterns: List[str], r: int = 8,
           alpha: float = 16.0) -> List[Tuple[str, LoRALinear]]:
    """Replace every nn.Linear whose qualified name matches any pattern."""
    pats = [re.compile(p) for p in name_patterns]
    hits: List[Tuple[str, LoRALinear]] = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(p.search(full) for p in pats):
                wrapped = LoRALinear(child, r=r, alpha=alpha)
                setattr(mod, child_name, wrapped)
                hits.append((full, wrapped))
    return hits


def lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            yield mod.A
            yield mod.B


@contextmanager
def lora_disabled(model: nn.Module):
    """The pristine-base (teacher) path: base weights only, bit-identical to
    the un-injected model."""
    loras = [m for m in model.modules() if isinstance(m, LoRALinear)]
    prev = [m.enabled for m in loras]
    for m in loras:
        m.enabled = False
    try:
        yield
    finally:
        for m, e in zip(loras, prev):
            m.enabled = e
