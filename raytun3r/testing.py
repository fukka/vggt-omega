"""Tiny randomly-initialised backbones, so every code path runs on CPU.

Both in-tree models allocate parameters with ``torch.empty`` and only initialise
some of them, so a freshly constructed model emits NaN. :func:`init_random`
gives every leaf a value; without it the smoke test would "fail" on the model's
uninitialised memory rather than on anything in this package.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from .backbones import VGGTBackbone, VGGTOmegaBackbone
from .cameras import KannalaBrandt

__all__ = ["init_random", "tiny_vggt", "tiny_vggt_omega", "toy_camera"]

_ROOT = Path(__file__).resolve().parents[1]


def init_random(model: nn.Module, gain: float = 0.3, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=gain)
            else:
                p.normal_(0.0, 0.02)
        for b in model.buffers():
            if b.is_floating_point() and not torch.isfinite(b).all():
                b.normal_(0.0, 1.0)
    return model


def toy_camera(height: int, width: int, fov_deg: float = 180.0) -> KannalaBrandt:
    """A KB4 lens whose imaged cone is about ``fov_deg`` wide at this size."""
    f = (min(height, width) / 2.0) / math.radians(fov_deg / 2.0)
    return KannalaBrandt(fx=f, fy=f, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
                         width=width, height=height,
                         k=(-0.02, 0.003, -0.0005, 0.00002))


def tiny_vggt(embed_dim: int = 384, depth: int = 4, img_size: int = 70,
              seed: int = 0) -> VGGTBackbone:
    """A 4-block VGGT with a DINOv2-small patch embed: abs PE *and* RoPE."""
    if str(_ROOT / "VGGT-360-fisheye") not in sys.path:
        sys.path.insert(0, str(_ROOT / "VGGT-360-fisheye"))
    from vggt_visfeat.heads.camera_head import CameraHead
    from vggt_visfeat.heads.dpt_head import DPTHead
    from vggt_visfeat.models.aggregator import Aggregator
    from vggt_visfeat.models.vggt import VGGT

    model = VGGT.__new__(VGGT)
    nn.Module.__init__(model)
    model.aggregator = Aggregator(img_size=img_size, patch_size=14, embed_dim=embed_dim,
                                  depth=depth, num_heads=6, patch_embed="dinov2_vits14_reg")
    model.camera_head = CameraHead(dim_in=2 * embed_dim)
    model.point_head = None
    model.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp",
                               conf_activation="expp1",
                               intermediate_layer_idx=list(range(depth)))
    init_random(model, seed=seed)
    bb = VGGTBackbone(model)
    bb.embed_dim = embed_dim
    return bb


def tiny_vggt_omega(embed_dim: int = 64, seed: int = 0) -> VGGTOmegaBackbone:
    """A small VGGT-Omega: RoPE only, no absolute PE table."""
    from vggt_omega.models.vggt_omega import VGGTOmega

    model = init_random(VGGTOmega(patch_size=16, embed_dim=embed_dim), seed=seed)
    bb = VGGTOmegaBackbone(model)
    bb.embed_dim = embed_dim
    return bb
