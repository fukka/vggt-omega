"""H6: peripheral cross-frame attention module. Protocol: ../protocol.md.

Rim tokens of frame t query all tokens of frame t-1; a zero-initialized
output gate makes the module identity at init. The update is written into a
COPY of the final feats level consumed only by the depth head — the camera
path reads the original feats (structural pose safety; premise verified in
the protocol, 2026-08-24).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


class PeripheralCrossFrameAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Linear(dim, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, rim_tokens: torch.Tensor,
                prev_tokens: torch.Tensor) -> torch.Tensor:
        """rim_tokens (B, R, C) += gated attention over prev_tokens (B, P, C)."""
        q = self.norm_q(rim_tokens)
        kv = self.norm_kv(prev_tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return rim_tokens + self.gate(out)


def apply_to_final_level(module: PeripheralCrossFrameAttention,
                         feats_t: Sequence, feats_prev: Sequence,
                         rim_mask: torch.Tensor) -> List:
    """Return a depth-head-only copy of ``feats_t`` whose FINAL level's rim
    patch tokens are updated by cross-frame attention into ``feats_prev``'s
    final-level tokens. All other levels and the cls tokens are shared
    references (no copy, no compute)."""
    patches_t, cls_t = feats_t[-1]
    patches_p, _ = feats_prev[-1]
    b, n, p, c = patches_t.shape
    assert n == 1, "one view per frame in this integration"
    rim_idx = torch.nonzero(rim_mask, as_tuple=False).squeeze(-1)
    rim = patches_t[:, 0, rim_idx, :]                       # (B, R, C)
    prev = patches_p[:, 0, :, :]                            # (B, P, C)
    updated = module(rim, prev)
    new_patches = patches_t.clone()
    new_patches[:, 0, rim_idx, :] = updated
    out = list(feats_t)
    out[-1] = (new_patches, cls_t)
    return out


def rim_mask_for(theta_patch: torch.Tensor, rim_deg: float = 35.0,
                 theta_max: float | None = None) -> torch.Tensor:
    """Rim = inside the imaged cone AND beyond rim_deg. Without the cone
    intersection the square grid's dead corners (theta > theta_max) became
    queries — caught 2026-08-26 by the efficiency measurement (73% of the
    grid instead of the intended rim band)."""
    m = theta_patch > math.radians(rim_deg)
    if theta_max is not None:
        m &= theta_patch <= theta_max
    return m
