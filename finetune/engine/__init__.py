# Copyright (c) 2026.
"""Engine utilities. The trainer classes now live in ``finetune.trainers``;
``AlternatingTrainer`` is re-exported here (lazily, to avoid an import cycle)
for backward compatibility."""
from __future__ import annotations

from .logger import TrainLogger

__all__ = ["TrainLogger", "AlternatingTrainer"]


def __getattr__(name: str):  # PEP 562: lazy attribute to dodge the import cycle
    if name == "AlternatingTrainer":
        from ..trainers import AlternatingTrainer

        return AlternatingTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
