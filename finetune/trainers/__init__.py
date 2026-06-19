# Copyright (c) 2026.
"""Training strategies, selected by name from a run's YAML (``trainer: ...``).

Importing this package registers every strategy in
``finetune.registry.TRAINER_REGISTRY``. Use :func:`build_trainer` to construct
the one a config asks for.
"""
from __future__ import annotations

import torch.nn as nn

from ..config import FinetuneConfig
from ..registry import TRAINER_REGISTRY
from .base import BaseAlternatingTrainer
from .dav2 import DAv2Trainer  # noqa: F401       (registers "dav2")
from .metric_anchor import MetricAnchorTrainer  # noqa: F401  (registers "metric_anchor")
from .ssi import SSITrainer  # noqa: F401        (registers "ssi")

# Backward-compatible default (smoke_test / older imports expect this name).
AlternatingTrainer = SSITrainer


def build_trainer(
    cfg: FinetuneConfig, vggt: nn.Module, dav2: nn.Module, device="cpu"
) -> BaseAlternatingTrainer:
    """Instantiate the trainer class named by ``cfg.trainer``."""
    cls = TRAINER_REGISTRY.get(cfg.trainer)
    return cls(vggt, dav2, cfg, device=device)


__all__ = [
    "BaseAlternatingTrainer",
    "SSITrainer",
    "MetricAnchorTrainer",
    "DAv2Trainer",
    "AlternatingTrainer",
    "build_trainer",
    "TRAINER_REGISTRY",
]
