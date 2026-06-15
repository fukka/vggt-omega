# Copyright (c) 2026.
"""Depth-Anything-V2 wrapper (the monocular partner in the alternating loop).

The real model is loaded lazily from HuggingFace ``transformers`` so the rest of
the package imports without that dependency. A tiny ``DummyDepthModel`` provides
the same interface for offline / smoke-test runs (no network, no weights).

Both return a per-frame, affine-invariant depth map ``[B,S,H,W]`` (strictly
positive). The distillation losses treat it in disparity space and never assume
its absolute scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DepthAnythingV2(nn.Module):
    """Thin differentiable wrapper around ``transformers`` Depth-Anything-V2."""

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf") -> None:
        super().__init__()
        try:
            from transformers import AutoModelForDepthEstimation
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "DepthAnythingV2 requires `transformers`. Install it or use "
                "DummyDepthModel / pass --dav2-dummy for offline runs."
            ) from exc
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _round14(self, x: int) -> int:
        return max(14, int(round(x / 14)) * 14)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images [B,S,3,H,W]`` in ``[0,1]`` -> depth ``[B,S,H,W]`` (>0)."""
        B, S, _, H, W = images.shape
        x = images.reshape(B * S, 3, H, W)
        x = (x - self.mean) / self.std
        h14, w14 = self._round14(H), self._round14(W)
        if (h14, w14) != (H, W):
            x = F.interpolate(x, size=(h14, w14), mode="bilinear", align_corners=False)
        out = self.model(pixel_values=x).predicted_depth  # [B*S, h, w], inverse-depth-like
        if out.dim() == 4:
            out = out.squeeze(1)
        out = out.unsqueeze(1)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        # DAv2 relative output is disparity-like; convert to a positive "depth" so
        # the rest of the pipeline is uniform. We keep it affine-invariant.
        depth = 1.0 / (F.softplus(out) + 1e-3)
        return depth.reshape(B, S, H, W)


class DummyDepthModel(nn.Module):
    """Tiny trainable per-frame depth net for offline / smoke-test use."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, S, _, H, W = images.shape
        x = images.reshape(B * S, 3, H, W)
        logits = self.net(x)[:, 0]
        depth = torch.exp(0.5 * torch.tanh(logits))  # positive, O(1)
        return depth.reshape(B, S, H, W)


def build_depth_anything(use_dummy: bool = False, model_name: str | None = None) -> nn.Module:
    if use_dummy:
        return DummyDepthModel()
    if model_name:
        return DepthAnythingV2(model_name=model_name)
    return DepthAnythingV2()
