"""Per-token FiLM conditioning driven by the lens-Jacobian field.

The injection point is the last ViT block's token output, modulated as
``tok <- tok * (1 + gamma) + beta`` with ``(gamma, beta)`` produced per token by
a small MLP over that token's lens geometry. Only patch tokens are touched;
prefix tokens (CLS/register) are left exactly as they were, because they have no
position on the image plane and inventing one for them would be the same class
of error as inventing depth where the mesh has a hole.

`gamma`/`beta` are zero-initialised through a zeroed output layer, so at step 0
the conditioned network is BIT-IDENTICAL to the unconditioned one. That is what
makes "the arms differ only in what the field says" checkable rather than
asserted -- see `test_film.py::test_at_init_the_module_is_an_exact_identity`.

THE ARMS
--------
``jac``      the real field: (log_area, log_aniso, theta/theta_max) per token
``shuffled`` the SAME values under a fixed permutation of token positions --
             identical distribution, identical parameter count, no spatial
             correspondence. This is the control that decides the experiment.
``theta``    theta/theta_max only, zero-padded to the same width. Tests whether
             the Jacobian carries anything beyond the angle, which
             test_jacobian.py argues on the geometry side.

A win over ``theta`` without a win over ``shuffled`` is not a result: it would
only say the extra channels added capacity.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FiLMConditioner", "make_arm_field"]


class FiLMConditioner(nn.Module):
    def __init__(self, in_ch: int, dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.GELU(), nn.Linear(hidden, 2 * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.dim = dim

    def forward(self, tok: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """``tok`` (..., N, dim) with N >= field.shape[0]; field (P, in_ch).

        The last ``P`` tokens are treated as the patch tokens, matching how
        h5's trainer slices them. Any prefix tokens pass through untouched.
        """
        p = field.shape[0]
        gb = self.net(field.to(tok.dtype))
        gamma, beta = gb[:, :self.dim], gb[:, self.dim:]
        # Concatenate rather than assign into a slice: an in-place write to a
        # view of `tok` bumps its autograd version counter and backward dies
        # with "a variable needed for gradient computation has been modified by
        # an inplace operation". Hit on the first box smoke, 2026-08-22.
        patch = tok[..., -p:, :] * (1.0 + gamma) + beta
        if p >= tok.shape[-2]:
            return patch
        return torch.cat([tok[..., :-p, :], patch], dim=-2)


def make_arm_field(field_jac: torch.Tensor, arm: str,
                   generator: torch.Generator) -> torch.Tensor:
    """Build the per-token field an arm sees. ``field_jac`` is (P, 3)."""
    if arm == "jac":
        return field_jac
    if arm == "theta":
        out = torch.zeros_like(field_jac)
        out[:, 2] = field_jac[:, 2]
        return out
    if arm == "shuffled":
        # One fixed permutation for the whole run, drawn from the run's own
        # seeded generator. Re-drawing it per step would average the shuffle
        # away and quietly turn the control into "no conditioning at all".
        perm = torch.randperm(field_jac.shape[0], generator=generator)
        return field_jac[perm].contiguous()
    raise ValueError(f"unknown arm {arm!r}")
