# Copyright (c) 2026.
"""Load the initializations CAM3R prescribes (paper Sec. 4.2 / Table S3).

    Ray Module ViT      <- UniK3D angular module
    Cross-view Module   <- DUSt3R
    everything else     <- scratch

Neither checkpoint ships with this repo and neither is needed to construct or
train the model -- these loaders are an upgrade path, not a dependency.  Both
report exactly how many tensors transferred, because a rename upstream would
otherwise show up as "initialized from DUSt3R" while silently loading nothing.

DUSt3R packs self-attention as one fused ``qkv`` projection and cross-attention
as separate ``projq``/``projk``/``projv``; :class:`cam3r.model.Attention` keeps
q/k/v separate throughout, so the fused tensors are split on the way in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LoadReport:
    """What actually transferred."""

    loaded: List[str] = field(default_factory=list)
    skipped_shape: List[Tuple[str, str]] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    source: str = ""

    @property
    def n_loaded(self) -> int:
        return len(self.loaded)

    def summary(self) -> str:
        return (
            f"[{self.source}] loaded {self.n_loaded} tensors, "
            f"{len(self.skipped_shape)} shape-mismatched, {len(self.missing)} left at init"
        )

    def raise_if_empty(self) -> None:
        if self.n_loaded == 0:
            raise RuntimeError(
                f"{self.source}: no tensors matched. The checkpoint layout is probably "
                f"different from what this loader expects -- inspect its keys rather than "
                f"proceeding with a silently random initialization."
            )


def _read_state_dict(path: str) -> Dict[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu")
    for key in ("model", "state_dict", "weights"):
        if isinstance(blob, dict) and key in blob and isinstance(blob[key], dict):
            blob = blob[key]
            break
    if not isinstance(blob, dict):
        raise RuntimeError(f"{path}: expected a state dict, got {type(blob).__name__}")
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in blob.items()}


def _assign(target: nn.Module, name: str, tensor: torch.Tensor, report: LoadReport) -> None:
    own = dict(target.named_parameters())
    own.update(dict(target.named_buffers()))
    if name not in own:
        return
    if own[name].shape != tensor.shape:
        report.skipped_shape.append((name, f"{tuple(own[name].shape)} != {tuple(tensor.shape)}"))
        return
    with torch.no_grad():
        own[name].copy_(tensor)
    report.loaded.append(name)


def _split_fused_qkv(prefix: str, weight: torch.Tensor, bias: Optional[torch.Tensor]):
    """``attn.qkv`` (3D, D) -> three (D, D) blocks keyed for our q/k/v Linears."""
    dim = weight.shape[0] // 3
    out: Dict[str, torch.Tensor] = {}
    for i, part in enumerate("qkv"):
        out[f"{prefix}.{part}.weight"] = weight[i * dim : (i + 1) * dim]
        if bias is not None:
            out[f"{prefix}.{part}.bias"] = bias[i * dim : (i + 1) * dim]
    return out


def load_dust3r_into_cross_view(
    cross_view: nn.Module, checkpoint: str, strict_nonempty: bool = True
) -> LoadReport:
    """Initialize a :class:`cam3r.model.CrossViewModule` from a DUSt3R checkpoint."""
    src = _read_state_dict(checkpoint)
    report = LoadReport(source="DUSt3R")

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in src.items():
        # Fused self-attention -> separate q/k/v.
        if key.endswith("attn.qkv.weight"):
            prefix = key[: -len(".qkv.weight")]
            remapped.update(_split_fused_qkv(prefix, value, src.get(prefix + ".qkv.bias")))
            continue
        if key.endswith("attn.qkv.bias"):
            continue                                    # consumed above
        # CroCo cross-attention uses projq/projk/projv.
        for a, b in (("projq", "q"), ("projk", "k"), ("projv", "v")):
            if f".{a}." in key:
                key = key.replace(f".{a}.", f".{b}.")
                break
        remapped[key] = value

    for name, tensor in remapped.items():
        _assign(cross_view, name, tensor, report)

    own = set(dict(cross_view.named_parameters()))
    report.missing = sorted(own - set(report.loaded))
    if strict_nonempty:
        report.raise_if_empty()
    return report


def load_unik3d_into_ray_module(
    ray_module: nn.Module, checkpoint: str, strict_nonempty: bool = True
) -> LoadReport:
    """Initialize a :class:`cam3r.model.RayModule` from a UniK3D checkpoint.

    Only the angular module transfers cleanly: UniK3D's encoder is a DINOv2
    backbone with a different block layout, so its trunk is left at init and the
    SH read-out head (``aggregate1/2``, ``latents_pos``, ``project_*``,
    ``out_*``) is what carries over -- which is precisely the "angular module"
    the paper names.
    """
    src = _read_state_dict(checkpoint)
    report = LoadReport(source="UniK3D-angular")

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in src.items():
        if "angular_module." not in key:
            continue
        tail = key.split("angular_module.", 1)[1]
        # UniK3D names the per-degree projections/read-outs individually.
        for i, deg in enumerate((1, 2, 3)):
            tail = tail.replace(f"project_deg{deg}.", f"projects.{i}.")
            tail = tail.replace(f"out_deg{deg}.", f"outs.{i}.")
        remapped[f"head.{tail}"] = value

    for name, tensor in remapped.items():
        _assign(ray_module, name, tensor, report)

    own = set(dict(ray_module.named_parameters()))
    report.missing = sorted(own - set(report.loaded))
    if strict_nonempty:
        report.raise_if_empty()
    return report


def initialize_cam3r(
    model: nn.Module, dust3r: Optional[str] = None, unik3d: Optional[str] = None, verbose: bool = True
) -> List[LoadReport]:
    """Apply whichever of the two prescribed initializations are available."""
    reports: List[LoadReport] = []
    if dust3r:
        reports.append(load_dust3r_into_cross_view(model.cross_view, dust3r))
    if unik3d:
        reports.append(load_unik3d_into_ray_module(model.ray_module, unik3d))
    if verbose:
        for r in reports:
            print("  " + r.summary())
        if not reports:
            print("  [init] no pretrained weights given -- training from scratch")
    return reports
