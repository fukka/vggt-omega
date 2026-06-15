# Copyright (c) 2026.
"""A tiny stand-in for VGGT-Omega that mimics its output dict.

Used for the CPU smoke test and ``--vggt-dummy`` runs so the full training loop
can be exercised without the gated checkpoint or a GPU. Output contract matches
``vggt_omega.models.VGGTOmega.forward``:

* ``depth``: ``[B,S,H,W,1]`` strictly positive
* ``depth_conf``: ``[B,S,H,W]`` >= 1
* ``pose_enc``: ``[B,S,9]`` = (translation, quaternion XYZW, fov_h, fov_w)
* ``camera_and_register_tokens``: ``[B,S,1+R,C]``
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyVGGT(nn.Module):
    def __init__(self, embed: int = 32, n_register: int = 4) -> None:
        super().__init__()
        self.embed = embed
        self.n_register = n_register
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
        )
        self.depth_head = nn.Conv2d(16, 1, 3, padding=1)
        self.conf_head = nn.Conv2d(16, 1, 3, padding=1)
        self.pose_head = nn.Linear(16, 9)
        self.token_proj = nn.Linear(16, embed)

    def forward(self, images: torch.Tensor) -> dict:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        B, S, _, H, W = images.shape
        x = images.reshape(B * S, 3, H, W)
        f = self.enc(x)

        depth = torch.exp(0.3 * torch.tanh(self.depth_head(f)[:, 0]))  # positive, ~O(1)
        conf = 1.0 + F.softplus(self.conf_head(f)[:, 0])

        pooled = f.mean(dim=(2, 3))  # [B*S,16]
        raw = self.pose_head(pooled)
        t = 0.1 * torch.tanh(raw[:, :3])
        q = raw[:, 3:7] + raw.new_tensor([0.0, 0.0, 0.0, 1.0])  # bias toward identity
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        fov = 0.3 + 1.2 * torch.sigmoid(raw[:, 7:9])  # radians, in (0.3, 1.5)
        pose_enc = torch.cat([t, q, fov], dim=-1).reshape(B, S, 9)

        tokens = self.token_proj(pooled).reshape(B, S, 1, self.embed)
        tokens = tokens.expand(B, S, 1 + self.n_register, self.embed).contiguous()

        out = {
            "depth": depth.reshape(B, S, H, W, 1),
            "depth_conf": conf.reshape(B, S, H, W),
            "pose_enc": pose_enc,
            "camera_and_register_tokens": tokens,
        }
        if not self.training:
            out["images"] = images
        return out
