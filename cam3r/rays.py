# Copyright (c) 2026.
"""Ray field parameterization -- CAM3R Eq. 2.

The Ray Module never regresses per-pixel directions directly.  It emits a small
set of spherical-harmonic coefficients ``c in R^{15x3}`` plus four base-grid
parameters, and the dense field is reconstructed analytically:

    d_i(u) = normalize( sum_{l=1..3} sum_m c_{l,m} Y_l^m( psi(u) ) )

``psi(u)`` maps pixel coordinates onto the sphere.  Its exact form is inherited
from UniK3D's angular module (an FoV-parameterized longitude/latitude map) so
UniK3D weights remain a valid initialization, which is what the paper
prescribes.  Note that ``psi`` is only a *domain* for the basis -- the mapping
from it to real ray directions lives entirely in the learned coefficients, so
``psi``'s own axis convention need not match the camera frame.  It follows that
an untrained Ray Module emits meaningless (if well-formed) rays.

The base-grid parameters are ``(hfov, vfov, cx, cy)``: angular extents in
radians and the principal point in pixels.

:func:`fit_sh_coeffs` solves the linear least-squares problem in the other
direction -- given a known ray field, what coefficients reproduce it?  That is
how ``cam3r/tests/test_rays.py`` checks that a degree-3 expansion can actually
express the Aria KB4 fisheye, and how :mod:`cam3r.adt` can hand the model an
analytic target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from cam3r.sht import SH_DEGREE_WIDTHS, sh_basis

RAY_INTRINSIC_DIM = 4  # (hfov, vfov, cx, cy)


def n_sh_coeffs(degree: int = 3) -> int:
    return sum(SH_DEGREE_WIDTHS[:degree])


def default_intrinsics(H: int, W: int, hfov: float = math.pi / 2) -> torch.Tensor:
    """A sane ``(hfov, vfov, cx, cy)`` starting point: centred, vfov scaled by aspect."""
    return torch.tensor([hfov, hfov * H / W, W / 2.0, H / 2.0], dtype=torch.float32)


def sphere_grid(intrinsics: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """``psi(u)``: (B, 4) intrinsics -> (B, H, W, 3) unit vectors.

    Replicates UniK3D ``Decoder.run_camera``: pixel offsets from the principal
    point are scaled by the FoV into longitude/latitude, then mapped to the
    sphere.  Kept identical so pretrained SH coefficients stay meaningful.
    """
    B = intrinsics.shape[0]
    device, dtype = intrinsics.device, intrinsics.dtype
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    hfov = intrinsics[:, 0].view(B, 1, 1)
    vfov = intrinsics[:, 1].view(B, 1, 1)
    cx = intrinsics[:, 2].view(B, 1, 1)
    cy = intrinsics[:, 3].view(B, 1, 1)

    longitude = (xs.unsqueeze(0) - cx) / W * hfov
    latitude = (ys.unsqueeze(0) - cy) / H * vfov
    cos_lat = torch.cos(latitude)
    grid = torch.stack(
        [cos_lat * torch.sin(longitude), -torch.sin(latitude), cos_lat * torch.cos(longitude)], dim=-1
    )
    return torch.nn.functional.normalize(grid, dim=-1, eps=1e-8)


def sh_ray_field(
    sh_coeffs: torch.Tensor, intrinsics: torch.Tensor, H: int, W: int, degree: int = 3
) -> torch.Tensor:
    """The raw SH expansion, before normalization: (B, 3, H, W).

    Exposed separately because it is *linear* in the coefficients, which is what
    :func:`fit_sh_coeffs` inverts.
    """
    want = n_sh_coeffs(degree)
    if sh_coeffs.shape[1] != want:
        raise ValueError(f"degree {degree} needs {want} coefficients, got {sh_coeffs.shape[1]}")
    basis = sh_basis(sphere_grid(intrinsics, H, W), degree=degree)   # (B, H, W, C)
    return torch.einsum("bhwc,bcd->bhwd", basis, sh_coeffs).permute(0, 3, 1, 2)


def decode_rays(
    sh_coeffs: torch.Tensor, intrinsics: torch.Tensor, H: int, W: int, degree: int = 3
) -> torch.Tensor:
    """(B, C, 3) coefficients + (B, 4) intrinsics -> (B, 3, H, W) unit rays."""
    raw = sh_ray_field(sh_coeffs, intrinsics, H, W, degree=degree)
    return torch.nn.functional.normalize(raw, dim=1, eps=1e-5)


def fit_sh_coeffs(
    rays: torch.Tensor,
    intrinsics: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    degree: int = 3,
) -> torch.Tensor:
    """Least-squares fit of coefficients to a known ray field.

    ``rays`` is (B, 3, H, W), ``valid`` an optional (B, H, W) mask.  Solves
    ``min_c || B c - d ||^2`` per batch element on the valid pixels only, in
    float64 -- the normal equations on a near-degenerate basis are ill-
    conditioned in float32.
    """
    B, _, H, W = rays.shape
    basis = sh_basis(sphere_grid(intrinsics, H, W), degree=degree).to(torch.float64)
    target = rays.permute(0, 2, 3, 1).to(torch.float64)

    out = []
    for b in range(B):
        A = basis[b].reshape(-1, basis.shape[-1])
        y = target[b].reshape(-1, 3)
        if valid is not None:
            m = valid[b].reshape(-1)
            A, y = A[m], y[m]
        out.append(torch.linalg.lstsq(A, y).solution)
    return torch.stack(out).to(rays.dtype)


@dataclass
class RayFitResult:
    """Outcome of :func:`fit_ray_field`."""

    coeffs: torch.Tensor      # (1, C, 3)
    intrinsics: torch.Tensor  # (1, 4)
    mean_deg: float
    max_deg: float

    @property
    def hfov(self) -> float:
        return float(self.intrinsics[0, 0])


def fit_ray_field(
    rays: torch.Tensor,
    valid: torch.Tensor,
    H: int,
    W: int,
    degree: int = 3,
    hfov_candidates: Optional[Sequence[float]] = None,
) -> RayFitResult:
    """Best SH fit to a ray field, searching over the base grid's FoV.

    The network predicts the base-grid FoV jointly with the coefficients; this
    mimics that by sweeping ``hfov`` and keeping the fit with the lowest mean
    angular error.  Inputs are a single camera's ``(H, W, 3)`` rays and
    ``(H, W)`` validity.
    """
    if rays.dim() == 3:
        rays = rays.unsqueeze(0).permute(0, 3, 1, 2)
    if valid.dim() == 2:
        valid = valid.unsqueeze(0)
    if hfov_candidates is None:
        hfov_candidates = [math.radians(d) for d in range(30, 361, 10)]

    best: Optional[RayFitResult] = None
    for hfov in hfov_candidates:
        intr = torch.tensor([[hfov, hfov * H / W, W / 2.0, H / 2.0]], dtype=torch.float32)
        coeffs = fit_sh_coeffs(rays, intr, valid, degree=degree)
        recon = decode_rays(coeffs, intr, H, W, degree=degree)
        dot = (recon * rays).sum(1).clamp(-1.0, 1.0)
        err = torch.rad2deg(torch.arccos(dot))[valid]
        cand = RayFitResult(coeffs, intr, float(err.mean()), float(err.max()))
        if best is None or cand.mean_deg < best.mean_deg:
            best = cand
    assert best is not None
    return best
