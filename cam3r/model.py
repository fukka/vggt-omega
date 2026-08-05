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

    ray_embed_dim: int = 1024     # ViT-L
    ray_depth: int = 24
    ray_heads: int = 16

    cv_embed_dim: int = 1024      # ViT-L encoder
    cv_enc_depth: int = 24
    cv_dec_embed_dim: int = 768   # decoder runs narrower than the encoder, as in DUSt3R
    cv_dec_depth: int = 12
    cv_heads: int = 16
    cv_dec_heads: int = 12

    sh_degree: int = 3
    mlp_ratio: float = 4.0
    dpt_features: int = 128


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


class Block(nn.Module):
    """Pre-norm self-attention block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


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

class AngularHead(nn.Module):
    """Class tokens -> base-grid intrinsics + SH coefficients.

    Mirrors UniK3D's ``AngularModule``: four groups of class tokens are expanded
    into 3/3/5/7 latent slots (pinhole parameters plus SH degrees 1-3), refined
    by two attention blocks, then read out.  Submodule names match UniK3D's so a
    checkpoint maps across.
    """

    def __init__(self, dim: int, heads: int, degree: int = 3) -> None:
        super().__init__()
        self.pin_params = 3
        self.deg_params = (3, 5, 7)[:degree]
        self.degree = degree
        self.num_params = self.pin_params + sum(self.deg_params)

        self.aggregate1 = Block(dim, heads)
        self.aggregate2 = Block(dim, heads)
        self.latents_pos = nn.Parameter(torch.randn(1, self.num_params, dim) * 0.02)

        self.project_pin = nn.Linear(dim, self.pin_params * dim, bias=False)
        self.projects = nn.ModuleList(
            [nn.Linear(dim, n * dim, bias=False) for n in self.deg_params]
        )
        self.out_pinhole = nn.Linear(dim, 1)
        self.outs = nn.ModuleList([nn.Linear(dim, 3) for _ in self.deg_params])

    def forward(self, cls_tokens: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, n_groups, dim = cls_tokens.shape
        groups = cls_tokens.chunk(1 + len(self.deg_params), dim=1)

        toks = [self.project_pin(groups[0]).reshape(B, -1, dim)]
        for g, proj in zip(groups[1:], self.projects):
            toks.append(proj(g).reshape(B, -1, dim))
        x = torch.cat(toks, dim=1) + self.latents_pos

        x = self.aggregate2(self.aggregate1(x))
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


class RayModule(nn.Module):
    """Per-image branch predicting the SH ray field (Eq. 2)."""

    def __init__(self, cfg: CAM3RConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.ray_embed_dim
        self.n_groups = 1 + len(( 3, 5, 7)[: cfg.sh_degree])
        self.patch_embed = PatchEmbed(cfg.ray_patch_size, dim)
        self.cls_tokens = nn.Parameter(torch.randn(1, self.n_groups, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, cfg.ray_heads, cfg.mlp_ratio) for _ in range(cfg.ray_depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = AngularHead(dim, cfg.ray_heads, cfg.sh_degree)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> ``(rays (B,3,H,W), intrinsics (B,4), sh_coeffs (B,C,3))``."""
        B, _, H, W = img.shape
        # The ray patch size (14) need not divide the image size (a multiple of
        # 16); resize to the nearest multiple rather than refusing the input.
        ps = self.cfg.ray_patch_size
        rh, rw = max(ps, round(H / ps) * ps), max(ps, round(W / ps) * ps)
        x = img if (rh, rw) == (H, W) else F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)

        tokens, gh, gw = self.patch_embed(x)
        tokens = tokens + sincos_pos_embed(tokens.shape[-1], gh, gw, tokens.device, tokens.dtype)
        tokens = torch.cat([self.cls_tokens.expand(B, -1, -1), tokens], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        intr, coeffs = self.head(tokens[:, : self.n_groups], H, W)
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
    """Pooled cross-view tokens -> ``R_2->1`` (6D), ``t_hat_2->1``, scale ``s``.

    The paper does not state a parameterization; 6D rotation is used here for
    its singularity-free gradients, translation is emitted as a direction and
    L2-normalized onto S^2, and scale is exponentiated so it stays positive.
    """

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )
        self.to_rot = nn.Linear(hidden, 6)
        self.to_trans = nn.Linear(hidden, 3)
        self.to_scale = nn.Linear(hidden, 1)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from cam3r.geometry import rot6d_to_matrix

        h = self.mlp(torch.cat([f1.mean(dim=1), f2.mean(dim=1)], dim=-1))
        R = rot6d_to_matrix(self.to_rot(h))
        t_dir = F.normalize(self.to_trans(h), dim=-1, eps=1e-8)
        scale = torch.exp(self.to_scale(h).squeeze(-1).clamp(-15.0, 15.0))
        return R, t_dir, scale


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

        n_stages = min(4, cfg.cv_dec_depth)
        self.stage_idx = [
            int(round(i * (cfg.cv_dec_depth - 1) / max(n_stages - 1, 1))) for i in range(n_stages)
        ]
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
        stages1: List[torch.Tensor] = []
        stages2: List[torch.Tensor] = []
        for idx, (b1, b2) in enumerate(zip(self.dec_blocks, self.dec_blocks2)):
            # Both branches read the *previous* layer's other-view state (Eq. 3),
            # so the update is symmetric rather than order-dependent.
            n1, n2 = b1(g1, g2), b2(g2, g1)
            g1, g2 = n1, n2
            if idx in self.stage_idx:
                stages1.append(g1)
                stages2.append(g2)

        g1, g2 = self.dec_norm(g1), self.dec_norm(g2)
        o1 = self.head1(stages1, gh, gw, H, W)
        o2 = self.head2(stages2, gh, gw, H, W)
        R, t_dir, scale = self.pose_head(g1, g2)

        return {
            # softplus keeps distance and confidence strictly positive without a
            # hard clamp that would zero the gradient.
            "radial": [F.softplus(o1[:, 0]) + 1e-4, F.softplus(o2[:, 0]) + 1e-4],
            "conf": [F.softplus(o1[:, 1]) + 1e-4, F.softplus(o2[:, 1]) + 1e-4],
            "R": R,
            "t_dir": t_dir,
            "scale": scale,
        }


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

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
            "scale": cv["scale"],
        }
