# Copyright (c) 2026.
"""Backward-compat shim.

The alternating trainer was refactored into ``finetune.trainers`` (a base class
plus registered strategy subclasses). ``AlternatingTrainer`` here is an alias for
the default (SSI) strategy so older imports keep working.
"""
from __future__ import annotations

from ..trainers import AlternatingTrainer, BaseAlternatingTrainer, build_trainer

__all__ = ["AlternatingTrainer", "BaseAlternatingTrainer", "build_trainer"]
