# Copyright (c) 2026.
"""Tiny name->class registry (BasicSR style).

A ``Registry`` lets training strategies register themselves by name so they can
be selected from a config file (``trainer: metric_anchor``) without the launcher
importing each class explicitly. This is the mechanism that makes "run variant X
vs variant Y in parallel" a one-line YAML change.

Example
-------
    TRAINER_REGISTRY = Registry("trainer")

    @TRAINER_REGISTRY.register()
    class SSITrainer(BaseAlternatingTrainer):
        ...

    cls = TRAINER_REGISTRY.get("ssi")        # -> SSITrainer
"""
from __future__ import annotations

from typing import Callable, Dict, Optional


class Registry:
    """Maps a string key to a registered object (class or callable)."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        if name in self._obj_map and self._obj_map[name] is not obj:
            raise KeyError(f"'{name}' already registered in '{self._name}' registry")
        self._obj_map[name] = obj

    def register(self, obj: Optional[object] = None, name: Optional[str] = None):
        """Use as ``@reg.register()``/``@reg.register(name="x")`` or ``reg.register(obj)``."""
        if obj is None:  # used as a decorator with optional name=
            def deco(o):
                key = name or _default_key(o)
                self._do_register(key, o)
                return o

            return deco
        self._do_register(name or _default_key(obj), obj)
        return obj

    def get(self, name: str):
        if name not in self._obj_map:
            raise KeyError(
                f"'{name}' not found in '{self._name}' registry. "
                f"Available: {sorted(self._obj_map)}"
            )
        return self._obj_map[name]

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def keys(self):
        return sorted(self._obj_map)


def _default_key(obj) -> str:
    """Class ``FooBarTrainer`` -> ``foo_bar``; otherwise the object's name."""
    name = getattr(obj, "__name__", str(obj))
    if name.endswith("Trainer"):
        name = name[: -len("Trainer")]
    # CamelCase -> snake_case
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


# The trainer registry is instantiated here and populated by ``finetune.trainers``.
TRAINER_REGISTRY = Registry("trainer")
