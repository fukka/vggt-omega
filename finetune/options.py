# Copyright (c) 2026.
"""Config loading + run-directory management (the experiment-tracking layer).

Responsibilities
----------------
* Merge a YAML config onto the :class:`FinetuneConfig` defaults. YAML keys are
  the dataclass field names; they may be grouped under arbitrary section headers
  (``data:``, ``losses:`` ...) purely for readability — the loader flattens them.
  A ``__base__: other.yaml`` key pulls in a base config first (single-level
  inheritance), so variant files stay tiny.
* Apply ``key=value`` CLI overrides on top (highest priority), type-coerced
  against the dataclass field types.
* Resolve ``out_dir = <runs_root>/<name>`` and create it, refusing to clobber an
  existing run unless ``resume``/``overwrite`` is set.
* Dump the fully-resolved config to ``<out_dir>/config.yaml`` and a
  ``<out_dir>/provenance.json`` (git SHA, argv, time, host), and append a row to
  ``<runs_root>/index.csv`` — the human-readable record of "what has been trained".

This keeps the trainer/loss code unaware of files: it only ever sees a populated
``FinetuneConfig``.
"""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import FinetuneConfig


# --------------------------------------------------------------------------- #
# YAML <-> FinetuneConfig
# --------------------------------------------------------------------------- #
def _require_yaml():
    try:
        import yaml  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "Reading run configs needs PyYAML. `pip install pyyaml` "
            "(or pass --dummy with no --config for the CPU dry run)."
        ) from exc
    import yaml
    return yaml


_VALID_FIELDS = {f.name for f in dataclasses.fields(FinetuneConfig)}


def _flatten(d: Dict[str, Any], out: Dict[str, Any]) -> None:
    """Collapse the (optionally sectioned) YAML dict to flat ``field: value``.

    Section dicts whose key is NOT a dataclass field are descended into; a key
    that IS a field is treated as a leaf even if its value is a dict (none are).
    Unknown leaf keys raise, which catches typos early.
    """
    for k, v in d.items():
        if k == "__base__":
            continue
        if isinstance(v, dict) and k not in _VALID_FIELDS:
            _flatten(v, out)
        else:
            if k not in _VALID_FIELDS:
                raise KeyError(
                    f"Unknown config key {k!r}. Valid fields: {sorted(_VALID_FIELDS)}"
                )
            out[k] = v


def _load_yaml_with_base(path: str) -> Dict[str, Any]:
    yaml = _require_yaml()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    merged: Dict[str, Any] = {}
    base = raw.get("__base__")
    if base:
        bases = [base] if isinstance(base, str) else list(base)
        for b in bases:
            b_path = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
            merged.update(_resolve_yaml(b_path))
    flat: Dict[str, Any] = {}
    _flatten(raw, flat)
    merged.update(flat)
    return merged


def _resolve_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config not found: {path}")
    return _load_yaml_with_base(path)


def _coerce(field_type, value):
    """Coerce a string/scalar to the dataclass field's type (for CLI overrides).

    ``from __future__ import annotations`` makes the dataclass ``.type`` a string
    (e.g. ``"float"``, ``"Tuple[int, ...]"``), so match by name.
    """
    tname = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", str(field_type))
    # Tuples (e.g. offsets) accept "a,b,c" or a list.
    if "Tuple" in tname or "tuple" in tname:
        if isinstance(value, str):
            value = [v for v in value.replace("(", "").replace(")", "").split(",") if v != ""]
        return tuple(int(v) for v in value)
    if tname == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if tname == "int":
        return int(float(value))   # tolerate "2e3"
    if tname == "float":
        return float(value)
    return str(value)


def apply_overrides(cfg: FinetuneConfig, overrides: List[str]) -> None:
    """Apply ``field=value`` strings (CLI ``--set``) onto ``cfg`` in place."""
    types = {f.name: f.type for f in dataclasses.fields(FinetuneConfig)}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects field=value, got {item!r}")
        key, val = item.split("=", 1)
        key = key.strip()
        if key not in _VALID_FIELDS:
            raise KeyError(f"--set unknown field {key!r}")
        setattr(cfg, key, _coerce(types[key], val.strip()))


def build_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> FinetuneConfig:
    """Defaults -> YAML (with __base__) -> CLI overrides -> name. Returns a config."""
    cfg = FinetuneConfig()
    types = {f.name: f.type for f in dataclasses.fields(FinetuneConfig)}
    if config_path:
        for k, v in _resolve_yaml(config_path).items():
            # YAML already types ints/floats/bools/lists; only strings need
            # coercion (and PyYAML 1.1 parses "2e-4" as a string).
            setattr(cfg, k, _coerce(types[k], v) if isinstance(v, str) else v)
    apply_overrides(cfg, overrides or [])
    if name:
        cfg.name = name
    return cfg


# --------------------------------------------------------------------------- #
# Run directory + provenance
# --------------------------------------------------------------------------- #
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return False


def _run_has_outputs(out_dir: str) -> bool:
    if not os.path.isdir(out_dir):
        return False
    sentinels = ("metrics.jsonl", "checkpoint_last.pt", "checkpoint_final.pt", "config.yaml")
    return any(os.path.exists(os.path.join(out_dir, s)) for s in sentinels)


def setup_run_dir(cfg: FinetuneConfig, resume: bool = False, overwrite: bool = False) -> str:
    """Resolve + create ``<runs_root>/<name>``, guard against clobber, dump provenance.

    Returns the absolute run directory. Safe to call on rank 0 only; other ranks
    should just use ``cfg.out_dir`` (set here).
    """
    if not cfg.out_dir:
        cfg.out_dir = os.path.join(cfg.runs_root, cfg.name)
    out_dir = cfg.out_dir

    if _run_has_outputs(out_dir) and not (resume or overwrite):
        raise FileExistsError(
            f"Run dir {out_dir!r} already has outputs. Use a different --name, "
            f"or pass --resume / --overwrite. (This guard prevents two experiments "
            f"silently overwriting each other.)"
        )
    os.makedirs(out_dir, exist_ok=True)

    _dump_config_yaml(cfg, os.path.join(out_dir, "config.yaml"))
    _dump_provenance(cfg, os.path.join(out_dir, "provenance.json"))
    _append_index(cfg)
    return out_dir


def _dump_config_yaml(cfg: FinetuneConfig, path: str) -> None:
    yaml = _require_yaml()
    data = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(cfg).items()}
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True, default_flow_style=False)


def _dump_provenance(cfg: FinetuneConfig, path: str) -> None:
    prov = {
        "name": cfg.name,
        "trainer": cfg.trainer,
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "host": socket.gethostname(),
        "argv": sys.argv,
        "python": sys.version.split()[0],
    }
    with open(path, "w") as f:
        json.dump(prov, f, indent=2)


def _append_index(cfg: FinetuneConfig) -> None:
    """One row per run in ``<runs_root>/index.csv`` — the at-a-glance training log."""
    import csv

    os.makedirs(cfg.runs_root, exist_ok=True)
    index = os.path.join(cfg.runs_root, "index.csv")
    cols = ["time_utc", "name", "trainer", "git_sha", "rounds", "steps_per_phase",
            "lr_vggt_lora", "w_metric_anchor", "w_a_photometric", "lora_rank", "notes"]
    row = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": cfg.name, "trainer": cfg.trainer, "git_sha": _git_sha()[:10],
        "rounds": cfg.rounds, "steps_per_phase": cfg.steps_per_phase,
        "lr_vggt_lora": cfg.lr_vggt_lora, "w_metric_anchor": cfg.w_metric_anchor,
        "w_a_photometric": cfg.w_a_photometric, "lora_rank": cfg.lora_rank,
        "notes": cfg.notes,
    }
    new = not os.path.exists(index)
    with open(index, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)
