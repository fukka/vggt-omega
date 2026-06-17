# Copyright (c) 2026.
"""Self-supervised egocentric finetuning for VGGT-Omega.

Alternating co-distillation between Depth-Anything-V2 and VGGT-Omega on real
egocentric video, with **no depth ground truth**. See ``finetune/README.md``.
"""
from __future__ import annotations

__version__ = "0.2.0"

from .config import FinetuneConfig
from .registry import TRAINER_REGISTRY

__all__ = ["FinetuneConfig", "TRAINER_REGISTRY"]
