# Copyright (c) 2026.
"""CAM3R's two-view network -- paper Sec. 3.1-3.2.

The architecture's organizing idea is that a pixel's 3D position factorizes as
``X(u) = d(u) * r(u)`` (Eq. 1): a **direction**, which is a property of the lens
alone, times a **radial distance**, which is a property of the scene.  Two
separate branches predict them:

* :class:`RayModule` sees one image at a time and emits the spherical-harmonic
  ray field of :mod:`cam3r.rays`.  Because camera geometry cannot depend on what
  a *different* camera saw, this branch is deliberately per-view -- a property
  ``test_model.py`` pins down.
* :class:`CrossViewModule` is a Siamese encoder plus a two-branch decoder whose
  blocks attend across views (Eq. 3), feeding a DPT head for radial distance and
  confidence, and a pose head for ``R_2->1``, ``t_hat_2->1`` and scale ``s``.

Sizes follow Table S3 (both ViT-L, 24 layers; patch 14 for the Ray Module and 16
for the Cross-view encoder) but every dimension is a config field, so tests run
a 32-wide, 2-layer version on CPU.

The paper initializes the Ray Module from UniK3D's angular module and the
Cross-view Module from DUSt3R.  Neither checkpoint is required to construct the
model; see :mod:`cam3r.pretrained` for the loaders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cam3r.rays import decode_rays


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class CAM3RConfig:
    img_size: int = 512
    patch_size: int = 16          # Cross-view encoder (Table S3)
    ray_patch_size: int = 14      # Ray Module (Table S3)

    # Ray Module trunk = DINOv2 ViT-L/14, the backbone UniK3D initializes.
    ray_embed_dim: int = 1024     # ViT-L
    ray_depth: int = 24
    ray_heads: int = 16
    ray_register_tokens: int = 1  # DINOv2-with-registers
    ray_pos_grid: int = 37        # DINOv2 pos-embed grid (518 / 14), interpolated
    # Table S3: "Linear projection (1024 -> 512)" sits between the trunk and the
    # angular module, which therefore runs at 512 -- the width of UniK3D's
    # ``angular_module``, whose weights initialize it.  One projection per
    # read-out depth, as in UniK3D's ``camera_token_adapter``.
    ray_angular_dim: int = 512
    ray_angular_heads: int = 8
    ray_layer_scale: float = 1e-4   # UniK3D config: layer_scale = 0.0001
    # 1-based trunk layers whose class token feeds the angular module.  ``None``
    # spreads them evenly, which reproduces the paper's {6, 12, 18, 24} at
    # ray_depth 24 and still works for the small configs the tests build.
    ray_readout_layers: Optional[Tuple[int, ...]] = None

    cv_embed_dim: int = 1024      # ViT-L encoder
    cv_enc_depth: int = 24
    cv_dec_embed_dim: int = 768   # decoder runs narrower than the encoder, as in DUSt3R
    cv_dec_depth: int = 12
    cv_heads: int = 16
    cv_dec_heads: int = 12        # DUSt3R's BaseDecoder: 768 wide, 12 heads

    sh_degree: int = 3
    mlp_ratio: float = 4.0
    dpt_features: int = 256       # Table S3: D_feat = 256
    # Table S3: "Multi-scale fusion {0, 6, 9, 12}" -- DUSt3R's hook set, where 0
    # is the encoder output and 6/9/12 are 1-based decoder layers.
    dpt_hooks: Tuple[int, ...] = (0, 6, 9, 12)


# --------------------------------------------------------------------------- #
# Transformer pieces
# --------------------------------------------------------------------------- #

class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head attention; self-attention when ``context`` is None."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"embed dim {dim} not divisible by {heads} heads")
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        return x.view(B, N, self.heads, C // self.heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx = x if context is None else context
        q, k, v = self._split(self.q(x)), self._split(self.k(ctx)), self._split(self.v(ctx))
        att = F.scaled_dot_product_attention(q, k, v)
        B, H, N, D = att.shape
        return self.proj(att.transpose(1, 2).reshape(B, N, H * D))


class LayerScale(nn.Module):
    """DINOv2's per-channel residual gate.

    Present in UniK3D's encoder blocks as ``ls1``/``ls2``.  Loading those blocks
    into a trunk that lacks it would silently drop the learned residual scaling,
    so the module is part of the block rather than an option.
    """

    def __init__(self, dim: int, init: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    """Pre-norm self-attention block.

    ``layer_scale`` selects the DINOv2 layout used by the Ray Module trunk, whose
    weights come from UniK3D.  The Cross-view Module is CroCo/DUSt3R, which has
    no LayerScale, so it leaves the flag off and keeps the plain residual.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, layer_scale: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)
        self.ls1 = LayerScale(dim) if layer_scale else nn.Identity()
        self.ls2 = LayerScale(dim) if layer_scale else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DecoderBlock(nn.Module):
    """Eq. 3: self-attention on one view, then cross-attention to the other."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = Attention(dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), self.norm_ctx(other))
        return x + self.mlp(self.norm3(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch: int, dim: int, in_chans: int = 3) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, _, H, W = x.shape
        if H % self.patch or W % self.patch:
            raise ValueError(
                f"input {H}x{W} is not a multiple of the patch size {self.patch}"
            )
        y = self.proj(x)
        return y.flatten(2).transpose(1, 2), y.shape[2], y.shape[3]


def sincos_pos_embed(dim: int, h: int, w: int, device, dtype) -> torch.Tensor:
    """2D sine-cosine positional embedding; resolution-agnostic, so a model
    trained at one size still runs at another."""
    if dim % 4:
        raise ValueError("pos-embed dim must be divisible by 4")
    quarter = dim // 4
    omega = torch.exp(torch.arange(quarter, device=device, dtype=torch.float32) * (-math.log(10000.0) / quarter))
    gy, gx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    out = []
    for grid in (gy, gx):
        ang = grid.reshape(-1, 1) * omega.reshape(1, -1)
        out += [torch.sin(ang), torch.cos(ang)]
    return torch.cat(out, dim=1).unsqueeze(0).to(dtype)


# --------------------------------------------------------------------------- #
# Ray Module
# --------------------------------------------------------------------------- #

class UniK3DMlp(nn.Module):
    """UniK3D's ``MLP``: LayerNorm, expand, GELU, contract.

    Differs from :class:`Mlp` in carrying its own input norm, which is why the
    block around it has no separate ``norm2``.  Name-compatible with UniK3D so
    the angular module's weights load.
    """

    def __init__(self, dim: int, expansion: float = 4.0, output_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = nn.LayerNorm(dim)
        self.proj1 = nn.Linear(dim, hidden)
        self.proj2 = nn.Linear(hidden, output_dim or dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.act(self.proj1(self.norm(x))))


class AngularAttentionBlock(nn.Module):
    """UniK3D's ``AttentionBlock``, self-attention mode.

    Two details separate it from :class:`Block`, and both matter for loading
    ``angular_module`` weights: the query and key/value paths take *different*
    LayerNorms of the same input, and the latent positions are added to the
    per-head **queries** after projection rather than to the token embeddings.
    Projections are bias-free.
    """

    def __init__(self, dim: int, heads: int, expansion: float = 4.0, layer_scale: float = 1e-4) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"angular dim {dim} not divisible by {heads} heads")
        self.heads = heads
        self.mlp = UniK3DMlp(dim, expansion)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ls1_1 = LayerScale(dim, layer_scale)
        self.ls2 = LayerScale(dim, layer_scale)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        h, d = self.heads, C // self.heads
        # UniK3D packs kv as (kv h d), so k is the first half of the output.
        kv = self.kv(self.norm_attnctx(x)).view(B, N, 2, h, d)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        q = self.q(self.norm_attnx(x)).view(B, N, h, d).transpose(1, 2)
        q = q + pos_embed.reshape(B, N, h, d).transpose(1, 2)

        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        x = self.ls1_1(self.out(a)) + x
        return self.ls2(self.mlp(x)) + x


class AngularHead(nn.Module):
    """UniK3D's ``AngularModule``: class tokens -> base-grid intrinsics + SH coefficients.

    One group of class tokens per read-out depth is expanded into 3/3/5/7 latent
    slots (pinhole parameters plus SH degrees 1-3), refined by two attention
    blocks, then read out.  Submodule names, widths and biases match UniK3D's so
    the checkpoint the paper initializes from maps across.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        degree: int = 3,
        expansion: float = 4.0,
        layer_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        self.pin_params = 3
        self.deg_params = (3, 5, 7)[:degree]
        self.degree = degree
        self.num_params = self.pin_params + sum(self.deg_params)

        self.aggregate1 = AngularAttentionBlock(dim, heads, expansion, layer_scale)
        self.aggregate2 = AngularAttentionBlock(dim, heads, expansion, layer_scale)
        self.latents_pos = nn.Parameter(torch.randn(1, self.num_params, dim))

        self.project_pin = nn.Linear(dim, self.pin_params * dim, bias=False)
        self.projects = nn.ModuleList(
            [nn.Linear(dim, n * dim, bias=False) for n in self.deg_params]
        )
        # UniK3D reads out through a two-layer MLP, not a bare Linear.
        self.out_pinhole = UniK3DMlp(dim, expansion=1.0, output_dim=1)
        self.outs = nn.ModuleList(
            [UniK3DMlp(dim, expansion=1.0, output_dim=3) for _ in self.deg_params]
        )

    def forward(self, cls_tokens: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, n_groups, dim = cls_tokens.shape
        groups = cls_tokens.chunk(1 + len(self.deg_params), dim=1)

        toks = [self.project_pin(groups[0]).reshape(B, -1, dim)]
        for g, proj in zip(groups[1:], self.projects):
            toks.append(proj(g).reshape(B, -1, dim))
        x = torch.cat(toks, dim=1)

        pos = self.latents_pos.expand(B, -1, -1)
        x = self.aggregate2(self.aggregate1(x, pos), pos)
        splits = torch.split(x, [self.pin_params] + list(self.deg_params), dim=1)

        intr = self._fill_intrinsics(self.out_pinhole(splits[0]).squeeze(-1), H, W)
        coeffs = torch.cat([out(s) for out, s in zip(self.outs, splits[1:])], dim=1)
        return intr, coeffs

    @staticmethod
    def _fill_intrinsics(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """3 logits -> ``(hfov, vfov, cx, cy)``, bounded by construction.

        The ``-1.1`` offset is UniK3D's: it makes a zero logit decode to
        ``hfov = pi/2``, so an untrained head starts at a plausible camera
        rather than a degenerate one.
        """
        hfov_logit, cx_logit, cy_logit = x.unbind(dim=-1)
        hfov = torch.sigmoid(hfov_logit - 1.1) * 2 * math.pi
        vfov = hfov * (H / W)
        cx = torch.sigmoid(cx_logit) * W
        cy = torch.sigmoid(cy_logit) * H
        return torch.stack([hfov, vfov, cx, cy], dim=-1)


def default_readout_layers(depth: int, n_groups: int) -> Tuple[int, ...]:
    """Evenly spaced 1-based trunk layers, one per angular-module token group.

    ``(24, 4)`` gives ``(6, 12, 18, 24)`` -- the paper's set, which is in turn
    UniK3D's ``pixel_encoder.depths``.  A trunk shallower than ``n_groups``
    (the tests build 2-block ones) repeats layers rather than failing; the
    groups stay distinct because each has its own projection.
    """
    return tuple(
        min(depth, max(1, int(round((i + 1) * depth / n_groups)))) for i in range(n_groups)
    )


class RayModule(nn.Module):
    """Per-image branch predicting the SH ray field (Eq. 2).

    DINOv2 ViT-L/14 trunk with the class token read at
    :func:`default_readout_layers` -- the paper's {6, 12, 18, 24} -- each depth
    projected 1024 -> 512 by its own linear (Table S3, UniK3D's
    ``camera_token_adapter``) before the angular module consumes the stack.  The
    layout is UniK3D's throughout, because that is the checkpoint the paper
    initializes this backbone from.
    """

    def __init__(self, cfg: CAM3RConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.ray_embed_dim
        self.n_groups = 1 + len((3, 5, 7)[: cfg.sh_degree])
        self.readout_layers = tuple(
            cfg.ray_readout_layers or default_readout_layers(cfg.ray_depth, self.n_groups)
        )
        if len(self.readout_layers) != self.n_groups:
            raise ValueError(
                f"the angular module wants {self.n_groups} token groups but "
                f"ray_readout_layers names {len(self.readout_layers)} layers"
            )
        if max(self.readout_layers) > cfg.ray_depth or min(self.readout_layers) < 1:
            raise ValueError(f"ray_readout_layers {self.readout_layers} outside 1..{cfg.ray_depth}")

        self.patch_embed = PatchEmbed(cfg.ray_patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.register_tokens = (
            nn.Parameter(torch.randn(1, cfg.ray_register_tokens, dim) * 0.02)
            if cfg.ray_register_tokens
            else None
        )
        g = cfg.ray_pos_grid
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + g * g, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(dim, cfg.ray_heads, cfg.mlp_ratio, layer_scale=True) for _ in range(cfg.ray_depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.camera_token_adapter = nn.ModuleList(
            [nn.Linear(dim, cfg.ray_angular_dim) for _ in range(self.n_groups)]
        )
        self.head = AngularHead(
            cfg.ray_angular_dim,
            cfg.ray_angular_heads,
            cfg.sh_degree,
            cfg.mlp_ratio,
            cfg.ray_layer_scale,
        )

    def _positions(self, gh: int, gw: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Class and patch position embeddings, bicubically resized to the grid."""
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        g = int(round(math.sqrt(patch_pos.shape[1])))
        if (gh, gw) != (g, g):
            p = patch_pos.reshape(1, g, g, -1).permute(0, 3, 1, 2)
            p = F.interpolate(p, size=(gh, gw), mode="bicubic", align_corners=False)
            patch_pos = p.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
        return cls_pos.to(dtype), patch_pos.to(dtype)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> ``(rays (B,3,H,W), intrinsics (B,4), sh_coeffs (B,C,3))``."""
        B, _, H, W = img.shape
        # The ray patch size (14) need not divide the image size (a multiple of
        # 16); resize to the nearest multiple rather than refusing the input.
        ps = self.cfg.ray_patch_size
        rh, rw = max(ps, round(H / ps) * ps), max(ps, round(W / ps) * ps)
        x = img if (rh, rw) == (H, W) else F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)

        patches, gh, gw = self.patch_embed(x)
        cls_pos, patch_pos = self._positions(gh, gw, patches.dtype)
        parts = [self.cls_token.expand(B, -1, -1) + cls_pos]
        if self.register_tokens is not None:
            parts.append(self.register_tokens.expand(B, -1, -1))
        parts.append(patches + patch_pos)
        tokens = torch.cat(parts, dim=1)

        # DINOv2 applies the trunk norm to every intermediate it hands out, and
        # the class token is index 0 (registers sit between it and the patches).
        # Capture by depth, then index in request order -- a shallow trunk may
        # name the same layer for more than one group.
        wanted = set(self.readout_layers)
        captured: Dict[int, torch.Tensor] = {}
        for depth, blk in enumerate(self.blocks, start=1):
            tokens = blk(tokens)
            if depth in wanted:
                captured[depth] = self.norm(tokens)[:, 0]

        cam_tokens = torch.cat(
            [
                adapt(captured[depth]).unsqueeze(1)
                for adapt, depth in zip(self.camera_token_adapter, self.readout_layers)
            ],
            dim=1,
        )
        intr, coeffs = self.head(cam_tokens, H, W)
        return decode_rays(coeffs, intr, H, W, degree=self.cfg.sh_degree), intr, coeffs


# --------------------------------------------------------------------------- #
# Cross-view Module
# --------------------------------------------------------------------------- #

class DPTHead(nn.Module):
    """Dense head fusing several decoder stages back to full resolution.

    A compact reassemble/fuse DPT: each selected stage is projected to a common
    width, upsampled, and added into a running feature map, which a small conv
    stack reads out.  ``out_channels`` covers radial distance and confidence in
    one pass.
    """

    def __init__(self, dim: int, features: int, out_channels: int, n_stages: int = 4) -> None:
        super().__init__()
        self.projects = nn.ModuleList([nn.Linear(dim, features) for _ in range(n_stages)])
        self.fuse = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
                           nn.Conv2d(features, features, 3, padding=1))
             for _ in range(n_stages)]
        )
        self.out = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, out_channels, 1),
        )

    def forward(self, stages: List[torch.Tensor], gh: int, gw: int, H: int, W: int) -> torch.Tensor:
        acc: Optional[torch.Tensor] = None
        for feat, proj, fuse in zip(stages, self.projects, self.fuse):
            B, N, _ = feat.shape
            x = proj(feat).transpose(1, 2).reshape(B, -1, gh, gw)
            x = fuse(x) + x
            acc = x if acc is None else F.interpolate(acc, size=x.shape[-2:], mode="bilinear",
                                                      align_corners=False) + x
        assert acc is not None
        acc = F.interpolate(acc, size=(H, W), mode="bilinear", align_corners=False)
        return self.out(acc)


class PoseHead(nn.Module):
    """Pooled cross-view tokens -> ``R_2->1`` (6D) and ``t_hat_2->1`` on S^2.

    The paper's head regresses only these two; the magnitude of ``t = s t_hat``
    comes from the predicted pointmap, not from a learned scalar (Sec. 3.2.3 --
    "the scale factor is derived from the ratio of the predicted pointmap
    magnitudes").  A scale head here would take no gradient under Eq. 10.

    The paper does not state a parameterization; 6D rotation is used for its
    singularity-free gradients and translation is L2-normalized onto S^2.
    """

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )
        self.to_rot = nn.Linear(hidden, 6)
        self.to_trans = nn.Linear(hidden, 3)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        from cam3r.geometry import rot6d_to_matrix

        h = self.mlp(torch.cat([f1.mean(dim=1), f2.mean(dim=1)], dim=-1))
        R = rot6d_to_matrix(self.to_rot(h))
        t_dir = F.normalize(self.to_trans(h), dim=-1, eps=1e-8)
        return R, t_dir


class CrossViewModule(nn.Module):
    """Siamese encoder + cross-attending decoder + DPT and pose heads."""

    def __init__(self, cfg: CAM3RConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.cv_embed_dim
        self.patch_embed = PatchEmbed(cfg.patch_size, dim)
        self.enc_blocks = nn.ModuleList([Block(dim, cfg.cv_heads, cfg.mlp_ratio) for _ in range(cfg.cv_enc_depth)])
        self.enc_norm = nn.LayerNorm(dim)

        # The decoder is narrower than the encoder (DUSt3R: 1024 -> 768), so a
        # projection sits between them.
        ddim = cfg.cv_dec_embed_dim
        self.decoder_embed = nn.Linear(dim, ddim)
        self.dec_blocks = nn.ModuleList(
            [DecoderBlock(ddim, cfg.cv_dec_heads, cfg.mlp_ratio) for _ in range(cfg.cv_dec_depth)]
        )
        self.dec_blocks2 = nn.ModuleList(
            [DecoderBlock(ddim, cfg.cv_dec_heads, cfg.mlp_ratio) for _ in range(cfg.cv_dec_depth)]
        )
        self.dec_norm = nn.LayerNorm(ddim)

        # Table S3: "Multi-scale fusion {0, 6, 9, 12}" -- DUSt3R's hook set,
        # where 0 is the encoder output (before any decoder block) and 6/9/12
        # are 1-based decoder layers.  Clamped so the small configs the tests
        # build still name distinct, in-range stages.
        self.dpt_hooks = sorted({min(h, cfg.cv_dec_depth) for h in cfg.dpt_hooks})
        n_stages = len(self.dpt_hooks)
        self.head1 = DPTHead(ddim, cfg.dpt_features, 2, n_stages)
        self.head2 = DPTHead(ddim, cfg.dpt_features, 2, n_stages)
        self.pose_head = PoseHead(ddim)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        tokens, gh, gw = self.patch_embed(img)
        tokens = tokens + sincos_pos_embed(tokens.shape[-1], gh, gw, tokens.device, tokens.dtype)
        for blk in self.enc_blocks:
            tokens = blk(tokens)
        return self.enc_norm(tokens)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict:
        B, _, H, W = img1.shape
        gh, gw = H // self.cfg.patch_size, W // self.cfg.patch_size

        f1, f2 = self.encode(img1), self.encode(img2)
        g1, g2 = self.decoder_embed(f1), self.decoder_embed(f2)
        # Hook 0 is the encoder output, before any decoder block has run.
        stages1: List[torch.Tensor] = [g1] if 0 in self.dpt_hooks else []
        stages2: List[torch.Tensor] = [g2] if 0 in self.dpt_hooks else []
        for depth, (b1, b2) in enumerate(zip(self.dec_blocks, self.dec_blocks2), start=1):
            # Both branches read the *previous* layer's other-view state (Eq. 3),
            # so the update is symmetric rather than order-dependent.
            n1, n2 = b1(g1, g2), b2(g2, g1)
            g1, g2 = n1, n2
            if depth in self.dpt_hooks:
                stages1.append(g1)
                stages2.append(g2)

        g1, g2 = self.dec_norm(g1), self.dec_norm(g2)
        o1 = self.head1(stages1, gh, gw, H, W)
        o2 = self.head2(stages2, gh, gw, H, W)
        R, t_dir = self.pose_head(g1, g2)

        return {
            # softplus keeps distance and confidence strictly positive without a
            # hard clamp that would zero the gradient.
            "radial": [F.softplus(o1[:, 0]) + 1e-4, F.softplus(o2[:, 0]) + 1e-4],
            "conf": [F.softplus(o1[:, 1]) + 1e-4, F.softplus(o2[:, 1]) + 1e-4],
            "R": R,
            "t_dir": t_dir,
        }


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

def pointmap_scale(
    points: torch.Tensor, valid: Optional[torch.Tensor] = None, eps: float = 1e-6
) -> torch.Tensor:
    """Mean point norm ``eta`` of a (B, 3, H, W) pointmap -- Eq. 7's scale.

    One definition serves both the normalization in
    :func:`cam3r.losses.local_regression_loss` and the ``s`` of Eq. 10, so the
    two cannot drift apart.
    """
    norms = points.norm(dim=1).flatten(1)
    if valid is None:
        eta = norms.mean(dim=1)
    else:
        m = valid.flatten(1).to(norms.dtype)
        eta = (norms * m).sum(1) / m.sum(1).clamp(min=1.0)
    return eta.clamp(min=eps)


class CAM3R(nn.Module):
    """Ray Module + Cross-view Module, combined per Eq. 1."""

    def __init__(self, cfg: Optional[CAM3RConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or CAM3RConfig()
        self.ray_module = RayModule(self.cfg)
        self.cross_view = CrossViewModule(self.cfg)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict:
        if img1.shape != img2.shape:
            raise ValueError(f"the two views must match in shape: {tuple(img1.shape)} vs {tuple(img2.shape)}")

        rays1, intr1, c1 = self.ray_module(img1)
        rays2, intr2, c2 = self.ray_module(img2)
        cv = self.cross_view(img1, img2)

        rays = [rays1, rays2]
        points = [rays[v] * cv["radial"][v].unsqueeze(1) for v in range(2)]
        return {
            "rays": rays,
            "intrinsics": [intr1, intr2],
            "sh_coeffs": [c1, c2],
            "radial": cv["radial"],
            "conf": cv["conf"],
            "points": points,
            "R": cv["R"],
            "t_dir": cv["t_dir"],
            # Sec. 3.2.3: t = s * t_hat, with s the scale the pointmap itself
            # implies.  Reference view is view 1, the frame the pose maps into.
            "scale": pointmap_scale(points[0]),
        }
