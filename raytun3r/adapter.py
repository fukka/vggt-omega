"""The learned part of RayTun3R: residual corrections to positional encodings.

Paper Sec. 4.2, Eq. 5 and 6. These are the *only* trainable parameters in the
method -- every backbone weight, attention block, MLP and prediction head stays
frozen.

    P'(u,v) = P_A(u,v) + t_r(rho) + rho * delta_theta(theta)          (Eq. 5)
    omega'(u,v) = omega(u,v) + Delta_r(rho)                            (Eq. 6)

``rho`` is the patch's radius from the calibrated principal point, normalised to
[0, 1] by the token-grid boundary; ``theta`` is its polar angle. Both lookup
tables are evaluated at continuous ``rho``/``theta`` by linear interpolation over
their bins, and both are zero-initialised so training starts exactly at the
pretrained encoding.

Parameter count with the paper's defaults (N_r = 20, N_theta = 8) on a width-C
backbone: ``20 C`` radial + ``8 C`` angular + ``20`` RoPE. For DA3-Small
(C = 384) the two PE tables are 7680 + 3072 = 10,752 -- the figure quoted in the
abstract -- and the radial RoPE table brings the full adapter to 10,772. See the
README: "10,752" and "the adapter" are not quite the same object.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .cameras import Camera

__all__ = ["RadialAngularPE", "RadialRoPE", "RayTun3RAdapter", "patch_polar_coords"]


def patch_polar_coords(camera: Camera, grid_h: int, grid_w: int, patch: int,
                       *, device=None, dtype=torch.float32) -> Tuple[Tensor, Tensor]:
    """Polar coordinates of each patch centre about the principal point.

    Returns ``(rho, theta)``, each ``(grid_h, grid_w)``. ``rho`` is normalised to
    [0, 1] by the grid boundary -- the paper computes it "as a patch radius,
    normalized to [0,1] by the grid boundary", so the normaliser is the distance
    from the principal point to the furthest grid corner. That keeps rho == 1
    reachable regardless of where the principal point sits, which matters for
    off-centre fisheye calibrations.
    """
    # Patch centres in image pixels.
    py = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) * patch - 0.5
    px = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) * patch - 0.5
    vv, uu = torch.meshgrid(py, px, indexing="ij")

    du = uu - camera.cx
    dv = vv - camera.cy
    radius = torch.sqrt(du * du + dv * dv)

    corners = torch.tensor(
        [[0.0, 0.0], [camera.width - 1.0, 0.0],
         [0.0, camera.height - 1.0], [camera.width - 1.0, camera.height - 1.0]],
        device=device, dtype=dtype,
    )
    far = torch.sqrt(((corners[:, 0] - camera.cx) ** 2
                      + (corners[:, 1] - camera.cy) ** 2)).max().clamp_min(1e-6)

    rho = (radius / far).clamp(0.0, 1.0)
    theta = torch.atan2(dv, du) % (2 * math.pi)
    return rho, theta


def _interp_radial(table: Tensor, rho: Tensor) -> Tensor:
    """Linear interpolation of a ``(N_r, C)`` table at ``rho`` in [0, 1].

    Bin centres are placed at the ends of the interval (``rho=0`` hits bin 0,
    ``rho=1`` hits bin N_r-1) so the table spans the full radius without
    extrapolation.
    """
    n = table.shape[0]
    if n == 0:
        return rho.new_zeros(*rho.shape, table.shape[-1])
    if n == 1:
        return table[0].expand(*rho.shape, -1)
    x = rho.clamp(0.0, 1.0) * (n - 1)
    lo = x.floor().clamp(0, n - 2).long()
    w = (x - lo.to(x.dtype)).unsqueeze(-1)
    return table[lo] * (1.0 - w) + table[lo + 1] * w


def _interp_angular(table: Tensor, theta: Tensor) -> Tensor:
    """Linear interpolation of a ``(N_theta, C)`` table over a periodic angle."""
    n = table.shape[0]
    if n == 0:
        return theta.new_zeros(*theta.shape, table.shape[-1])
    if n == 1:
        return table[0].expand(*theta.shape, -1)
    x = (theta % (2 * math.pi)) / (2 * math.pi) * n
    lo = x.floor().long() % n
    hi = (lo + 1) % n
    w = (x - x.floor()).unsqueeze(-1)
    return table[lo] * (1.0 - w) + table[hi] * w


class RadialAngularPE(nn.Module):
    """Eq. 5 -- residual on the absolute positional embedding table.

    ``forward`` returns the *residual* only; the caller adds it to ``P_A``. That
    keeps the pretrained table untouched and makes the L2 regulariser (Eq. 11)
    exactly the squared norm of this residual.
    """

    def __init__(self, dim: int, n_radial: int = 20, n_angular: int = 8):
        super().__init__()
        self.dim = dim
        self.n_radial = n_radial
        self.n_angular = n_angular
        # Zero init: "all residual adapter parameters are initialized to zero".
        self.t_r = nn.Parameter(torch.zeros(n_radial, dim))
        self.delta_theta = nn.Parameter(torch.zeros(max(n_angular, 0), dim))

    def forward(self, rho: Tensor, theta: Tensor) -> Tensor:
        """``(..., )`` polar coords -> ``(..., C)`` residual."""
        out = _interp_radial(self.t_r, rho)
        if self.n_angular > 0:
            # The rho factor suppresses the angular term at the centre, where the
            # angle is ill-defined, and lets it grow toward the periphery.
            out = out + rho.unsqueeze(-1) * _interp_angular(self.delta_theta, theta)
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_radial={self.n_radial}, n_angular={self.n_angular}"


class RadialRoPE(nn.Module):
    """Eq. 6 -- radial correction to the rotary angle, shared across frequencies.

    One scalar per radial bin, added to the rotary angle of every frequency and
    both axes. Applying it to the returned ``(sin, cos)`` of a frozen RoPE module
    uses the angle-addition identities, so no re-derivation of the backbone's
    angle schedule is needed:

        sin(w + d) = sin w cos d + cos w sin d
        cos(w + d) = cos w cos d - sin w sin d
    """

    def __init__(self, n_radial: int = 20):
        super().__init__()
        self.n_radial = n_radial
        self.delta_r = nn.Parameter(torch.zeros(n_radial, 1))

    def forward(self, rho: Tensor) -> Tensor:
        """``(...,)`` radius -> ``(...,)`` angle offset in radians."""
        return _interp_radial(self.delta_r, rho).squeeze(-1)

    def rotate(self, sin: Tensor, cos: Tensor, rho: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply the offset to a precomputed ``(sin, cos)`` pair.

        ``sin``/``cos`` are ``(N, D)`` over N tokens; ``rho`` is ``(N,)``. Used by
        backbones whose RoPE module *returns* the pair (DINOv3-style).
        """
        d = self.forward(rho).unsqueeze(-1).to(sin.dtype)
        sd, cd = torch.sin(d), torch.cos(d)
        return sin * cd + cos * sd, cos * cd - sin * sd

    def rotate_tokens(self, tokens: Tensor, rho: Tensor, n_blocks: int = 2) -> Tensor:
        """Apply the offset to already-rotated tokens (VGGT/DINOv2-style RoPE).

        VGGT's ``RotaryPositionEmbedding2D`` consumes integer grid positions and
        returns rotated tokens, so there is no ``(sin, cos)`` to intercept.
        Composing rotations is still exact: rotating the output by ``Delta`` is
        the same as having rotated by ``omega + Delta`` in the first place.

        ``tokens`` is ``(..., N, D)`` where the last dimension holds ``n_blocks``
        independently-rotated axial halves (2 for 2D axial RoPE: one for y, one
        for x). The offset is applied *within* each half, so the y and x
        subspaces never mix -- matching Eq. 6's "shared across RoPE frequencies".
        """
        lead = tokens.shape[:-2]
        n, dim = tokens.shape[-2], tokens.shape[-1]
        if dim % (2 * n_blocks):
            raise ValueError(f"feature dim {dim} is not divisible by 2*{n_blocks}")

        d = self.forward(rho.to(tokens.device)).to(tokens.dtype)
        cos = torch.cos(d).reshape(*([1] * len(lead)), n, 1, 1)
        sin = torch.sin(d).reshape(*([1] * len(lead)), n, 1, 1)

        t = tokens.reshape(*lead, n, n_blocks, dim // n_blocks)
        half = (dim // n_blocks) // 2
        t1, t2 = t[..., :half], t[..., half:]
        rot = torch.cat((-t2, t1), dim=-1)
        return (t * cos + rot * sin).reshape(*lead, n, dim)

    def extra_repr(self) -> str:
        return f"n_radial={self.n_radial}"


class RayTun3RAdapter(nn.Module):
    """Bundles the learned tables and evaluates them on a fixed token grid.

    The polar coordinates depend only on the camera and the token grid, so they
    are computed once in :meth:`bind` and cached as buffers. A single adapter
    instance therefore corresponds to one camera at one input resolution, which
    matches the paper's setting (limitation (i): a different fisheye camera
    requires a new adaptation).
    """

    def __init__(self, dim: int, *, n_radial: int = 20, n_angular: int = 8,
                 n_rope_radial: int = 20, use_abs_pe: bool = True, use_rope: bool = True):
        super().__init__()
        self.dim = dim
        self.pe = RadialAngularPE(dim, n_radial, n_angular) if use_abs_pe else None
        self.rope = RadialRoPE(n_rope_radial) if use_rope else None
        self.register_buffer("rho", torch.zeros(0), persistent=False)
        self.register_buffer("theta", torch.zeros(0), persistent=False)
        self._grid: Optional[Tuple[int, int]] = None

    # -- binding ----------------------------------------------------------

    def bind(self, camera: Camera, grid_h: int, grid_w: int, patch: int) -> "RayTun3RAdapter":
        """Precompute the patch polar coordinates for this camera and grid."""
        device = next(self.parameters()).device if any(True for _ in self.parameters()) else None
        rho, theta = patch_polar_coords(camera, grid_h, grid_w, patch, device=device)
        self.rho = rho
        self.theta = theta
        self._grid = (grid_h, grid_w)
        return self

    @property
    def is_bound(self) -> bool:
        return self._grid is not None and self.rho.numel() > 0

    def _require_bound(self) -> None:
        if not self.is_bound:
            raise RuntimeError(
                "RayTun3RAdapter used before bind(camera, grid_h, grid_w, patch); "
                "the backbone wrapper normally calls this on the first forward."
            )

    # -- evaluation -------------------------------------------------------

    def pe_residual(self, flatten: bool = True) -> Tensor:
        """Eq. 5 residual on the token grid: ``(H*W, C)`` or ``(H, W, C)``."""
        self._require_bound()
        if self.pe is None:
            gh, gw = self._grid
            z = self.rho.new_zeros(gh, gw, self.dim)
            return z.reshape(-1, self.dim) if flatten else z
        out = self.pe(self.rho, self.theta)
        return out.reshape(-1, self.dim) if flatten else out

    def rope_sincos(self, sin: Tensor, cos: Tensor) -> Tuple[Tensor, Tensor]:
        """Eq. 6 applied to a frozen RoPE module's ``(sin, cos)`` output."""
        if self.rope is None:
            return sin, cos
        self._require_bound()
        return self.rope.rotate(sin, cos, self.rho.reshape(-1).to(sin.device))

    def rope_tokens(self, tokens: Tensor, n_blocks: int = 2) -> Tensor:
        """Eq. 6 applied to a frozen RoPE module's rotated-token output."""
        if self.rope is None:
            return tokens
        self._require_bound()
        return self.rope.rotate_tokens(tokens, self.rho.reshape(-1), n_blocks=n_blocks)

    # -- bookkeeping ------------------------------------------------------

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> dict:
        out = {}
        if self.pe is not None:
            out["pe_radial"] = self.pe.t_r.numel()
            out["pe_angular"] = self.pe.delta_theta.numel()
        if self.rope is not None:
            out["rope_radial"] = self.rope.delta_r.numel()
        out["total"] = sum(out.values())
        return out
