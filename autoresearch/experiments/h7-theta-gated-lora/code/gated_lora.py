"""H7: theta-gated LoRA. Protocol: ../protocol.md.

GatedLoRALinear subclasses the H5 LoRALinear so `lora_disabled` (the
bit-identical teacher path) and isinstance-based tooling keep working. The
gate is rank-wise and token-wise: g = 1 + tanh(MLP(theta_feat)), W2
zero-init => g == 1 at init == exactly uniform LoRA.

Token layout is set once per process via `set_theta(theta_patch, n_total)`:
patch tokens get their theta feature, the leading (n_total - n_patch)
special tokens get a zeroed tanh (g = 1 always).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h5-rim-finetune" / "code"))
import lora as h5_lora  # noqa: E402

_CTX = {"feat": None, "special": None}


def set_theta(theta_patch: torch.Tensor, n_total: int,
              theta_max: float) -> None:
    """theta_patch: (n_patch,) radians for the patch tokens, in token order."""
    n_patch = theta_patch.numel()
    t = (theta_patch / theta_max).clamp(0, 1.5)
    feat = torch.stack([t, t * t], -1)                      # (n_patch, 2)
    pad = torch.zeros(n_total - n_patch, 2)
    _CTX["feat"] = torch.cat([pad, feat], 0)                # specials lead
    _CTX["special"] = torch.cat([torch.ones(n_total - n_patch),
                                 torch.zeros(n_patch)]).bool()


class GatedLoRALinear(h5_lora.LoRALinear):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__(base, r=r, alpha=alpha)
        self.g1 = nn.Linear(2, 16)
        self.g2 = nn.Linear(16, r)
        nn.init.zeros_(self.g2.weight)
        nn.init.zeros_(self.g2.bias)
        self.g1.to(base.weight.device)
        self.g2.to(base.weight.device)

    def gate(self, device, dtype) -> torch.Tensor:
        feat = _CTX["feat"].to(device=device, dtype=dtype)
        g = 1 + torch.tanh(self.g2(torch.relu(self.g1(feat))))   # (N, r)
        return torch.where(_CTX["special"].to(device)[:, None],
                           torch.ones_like(g), g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.enabled:
            low = x @ self.A.T                               # (..., N, r)
            if _CTX["feat"] is not None and low.shape[-2] == _CTX["feat"].shape[0]:
                low = low * self.gate(x.device, x.dtype)
            y = y + (low @ self.B.T) * self.scale
        return y


def inject(model: nn.Module, name_patterns: List[str], r: int = 8,
           alpha: float = 16.0) -> List[Tuple[str, GatedLoRALinear]]:
    pats = [re.compile(p) for p in name_patterns]
    hits: List[Tuple[str, GatedLoRALinear]] = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(p.search(full)
                                                    for p in pats):
                wrapped = GatedLoRALinear(child, r=r, alpha=alpha)
                setattr(mod, child_name, wrapped)
                hits.append((full, wrapped))
    return hits


def parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for mod in model.modules():
        if isinstance(mod, GatedLoRALinear):
            yield from (mod.A, mod.B, mod.g1.weight, mod.g1.bias,
                        mod.g2.weight, mod.g2.bias)


def state_of(hits) -> dict:
    return {name: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu(),
                   "g1w": m.g1.weight.detach().cpu(),
                   "g1b": m.g1.bias.detach().cpu(),
                   "g2w": m.g2.weight.detach().cpu(),
                   "g2b": m.g2.bias.detach().cpu()}
            for name, m in hits}


def load_into(net: nn.Module, ck: dict) -> int:
    hits = inject(net, ck["patterns"], r=ck["config"]["lora_r"],
                  alpha=2 * ck["config"]["lora_r"])
    by_name = dict(hits)
    n = 0
    for name, st in ck["lora"].items():
        m = by_name[name]
        with torch.no_grad():
            m.A.copy_(st["A"]); m.B.copy_(st["B"])
            m.g1.weight.copy_(st["g1w"]); m.g1.bias.copy_(st["g1b"])
            m.g2.weight.copy_(st["g2w"]); m.g2.bias.copy_(st["g2b"])
        n += 1
    assert n == len(hits)
    return n


def gate_curve(hits, theta_max: float, n: int = 50) -> dict:
    """Mechanism readout for P2: mean |g-1| per theta, averaged over layers."""
    t = torch.linspace(0, theta_max, n)
    feat = torch.stack([(t / theta_max), (t / theta_max) ** 2], -1)
    curves = []
    with torch.no_grad():
        for _, m in hits:
            g = 1 + torch.tanh(m.g2(torch.relu(m.g1(feat))))
            curves.append((g - 1).abs().mean(-1))
    return {"theta_deg": [math.degrees(x) for x in t.tolist()],
            "mean_abs_gate_minus_1":
                torch.stack(curves).mean(0).tolist()}
