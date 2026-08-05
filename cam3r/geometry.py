# Copyright (c) 2026.
"""SO(3) / SE(3) helpers and the ray <-> spherical-angle conversions.

Conventions (shared with the rest of this repo -- see CONTEXT.md and
``VGGT-360-fisheye/utils/fisheye_cam.py``):

* Camera frame is **x right, y down, z forward**; z is the optical axis.
* ``theta`` is the *incidence* angle from the optical axis, ``arccos(z)``, and
  ``phi`` the image-plane azimuth ``atan2(y, x)``.  These are the ``(theta, phi)``
  the asymmetric angular loss (CAM3R Eq. 5-6) regresses.
* A pose ``T`` is 4x4 **camera-from-world**, matching
  ``finetune/eval/mps_depth.py``'s ``T_cam_device``.  ``relative_pose(T1, T2)``
  returns ``(R_2->1, t_2->1)`` such that ``x_1 = R_2->1 x_2 + t_2->1``, which is
  the direction CAM3R Eq. 4 uses.

Everything is batched over a leading ``(B,)`` dimension and differentiable.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# Rotations
# --------------------------------------------------------------------------- #

def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 6D continuous rotation -> (..., 3, 3) via Gram-Schmidt.

    The paper does not state the pose head's rotation parameterization; 6D is
    used here because it is the standard singularity-free choice for regression.
    """
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6): the first two *columns*, inverse of the above."""
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


def geodesic_angle(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angle in radians of ``R1 R2^T`` -- CAM3R Eq. 9's rotation error.

    Computed as ``atan2(sin, cos)`` of the axis-angle decomposition rather than
    the literal ``arccos((tr - 1)/2)`` of Eq. 9.  The two agree analytically,
    but arccos loses precision near 0 and pi (where its argument is flat) and
    its gradient diverges there -- this form stays exact and finite at both
    ends, which matters because the same function serves as both the rotation
    loss and the RRA metric.
    """
    rel = R1 @ R2.transpose(-1, -2)
    cos = (rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2] - 1.0) / 2.0
    axis = torch.stack(
        [
            rel[..., 2, 1] - rel[..., 1, 2],
            rel[..., 0, 2] - rel[..., 2, 0],
            rel[..., 1, 0] - rel[..., 0, 1],
        ],
        dim=-1,
    )
    sin = axis.norm(dim=-1) / 2.0
    return torch.atan2(sin, cos)


def random_rotations(n: int, seed: Optional[int] = None, dtype=torch.float32) -> torch.Tensor:
    """(n, 3, 3) uniformly-random rotations (QR of a Gaussian, det fixed to +1)."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    flip = torch.linalg.det(q) < 0
    q[flip, :, 0] = -q[flip, :, 0]
    return q.to(dtype)


def angle_between(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Unsigned angle in radians between two (..., 3) vectors, ignoring magnitude."""
    v1 = torch.nn.functional.normalize(v1, dim=-1, eps=eps)
    v2 = torch.nn.functional.normalize(v2, dim=-1, eps=eps)
    # atan2 form: numerically stable at both 0 and pi, unlike arccos of a dot.
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1), (v1 * v2).sum(-1))


# --------------------------------------------------------------------------- #
# SE(3)
# --------------------------------------------------------------------------- #

def make_se3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) + (..., 3) -> (..., 4, 4) homogeneous transform."""
    shape = R.shape[:-2]
    T = torch.eye(4, dtype=R.dtype, device=R.device).expand(*shape, 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    R, t = T[..., :3, :3], T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    return make_se3(Rt, -(Rt @ t.unsqueeze(-1)).squeeze(-1))


def se3_compose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A @ B


def transform_points(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply ``T`` (..., 4, 4) to points (..., N, 3)."""
    R, t = T[..., :3, :3], T[..., :3, 3]
    return pts @ R.transpose(-1, -2) + t.unsqueeze(-2)


def relative_pose(T1: torch.Tensor, T2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Camera-from-world poses -> ``(R_2->1, t_2->1)`` with ``x_1 = R x_2 + t``."""
    T = se3_compose(T1, se3_inverse(T2))
    return T[..., :3, :3], T[..., :3, 3]


# --------------------------------------------------------------------------- #
# Rays <-> spherical angles
# --------------------------------------------------------------------------- #

def rays_to_spherical(d: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit rays (..., 3) -> ``(theta, phi)``, both (...,), in radians.

    ``theta = arccos(z)`` in ``[0, pi]``; ``phi = atan2(y, x)`` in ``(-pi, pi]``.
    ``phi`` is degenerate on the optical axis (``theta = 0``); callers that
    difference it should weight by ``sin(theta)`` or accept the noise there.

    Both poles are guarded, because on a real camera they are *hit exactly*:
    the on-axis pixel has ``z == 1`` and ``x == y == 0``.  ``arccos`` has an
    infinite derivative at ``|z| = 1``, so an accurate ray field -- which is
    what a UniK3D-initialized Ray Module produces on its very first step --
    NaNs the whole backward pass through it.  Clamping the *value* would not
    help; the derivative is what diverges, and a clamp wide enough to tame it
    costs ~0.08 deg on axis.

    ``theta = atan2(hypot(x, y), z)`` is exact for a unit vector and, because
    ``rho^2 + z^2 = 1`` there, has ``dtheta/drho = z`` and ``dtheta/dz = -rho``
    -- both bounded by 1 everywhere on the sphere.  The ``1e-20`` under the
    square root keeps ``drho/dx = x/rho`` finite at the axis without moving
    ``rho`` by more than 1e-10.  ``phi`` takes zero gradient where it carries no
    information.
    """
    d = torch.nn.functional.normalize(d, dim=-1, eps=eps)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]

    rho = torch.sqrt(x * x + y * y + 1e-20)
    theta = torch.atan2(rho, z)

    on_axis = (x * x + y * y) < eps
    phi = torch.atan2(
        torch.where(on_axis, torch.zeros_like(y), y),
        torch.where(on_axis, torch.ones_like(x), x),
    )
    return theta, phi


def spherical_to_rays(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """``(theta, phi)`` -> unit rays (..., 3). Inverse of :func:`rays_to_spherical`."""
    st = torch.sin(theta)
    return torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)], dim=-1)


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap radians into ``(-pi, pi]`` so azimuth differences cross the seam correctly."""
    return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi
