# Copyright (c) 2026.
"""Minimal LoRA for ``nn.Linear`` + freeze utilities.

Finetuning VGGT-Omega fully would risk catastrophic forgetting of its broad 3D
prior (and is memory-heavy). We instead inject low-rank adapters into the
attention / MLP linear layers and train only those plus the dense + camera heads.
"""
from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # start as a no-op (delta = 0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __getattr__(self, name: str):
        # Proxy nn.Linear attributes (in_features, out_features, weight, bias …)
        # that callers may read directly on the wrapped module.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + self.scaling * delta


def apply_lora(
    model: nn.Module,
    target_substrings: Iterable[str] = ("qkv", "proj", "fc1", "fc2", "q", "k", "v"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> int:
    """Replace matching ``nn.Linear`` modules with ``LoRALinear`` in-place.

    Returns the number of layers adapted. Matching is by the leaf attribute name
    containing any of ``target_substrings``.
    """
    targets = tuple(target_substrings)
    count = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and any(t in name for t in targets):
                setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                count += 1
    return count


def mark_trainable(model: nn.Module, train_head_substrings: Iterable[str] = ("head",)) -> None:
    """Freeze everything except LoRA adapters and the named prediction heads.

    A subtlety: a head module may itself contain LoRA-wrapped linears (e.g. the
    camera head's attention trunk). Those wrapped weights live under ``.base.``
    and must STAY frozen — only their ``lora_A``/``lora_B`` train. Re-enabling
    them by name would silently turn LoRA on that layer into a redundant full
    finetune. We therefore skip ``.base.`` params when un-freezing heads.
    """
    heads = tuple(train_head_substrings)
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
    for name, p in model.named_parameters():
        if any(h in name for h in heads) and ".base." not in name:
            p.requires_grad_(True)


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
