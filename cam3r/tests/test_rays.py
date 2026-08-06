"""The Ray Module's SH ray parameterization (CAM3R Eq. 2).

The load-bearing question these tests answer is *representational*: can a
degree-<=3 SH vector field actually express the lens geometries CAM3R claims to
cover -- in particular the Aria KB4 fisheye that ADT is shot on?  If it cannot,
no amount of training fixes it, so the fit error is asserted here rather than
being left to emerge as bad numbers later.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cam3r.cameras import Equirect, Pinhole, aria_214_1_kb4
from cam3r.rays import (
    RAY_INTRINSIC_DIM,
    decode_rays,
    default_intrinsics,
    fit_ray_field,
    fit_sh_coeffs,
    sh_ray_field,
    sphere_grid,
)


def _mean_max_deg(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor):
    dot = (a * b).sum(-1).clamp(-1.0, 1.0)
    err = torch.rad2deg(torch.arccos(dot))[valid]
    return float(err.mean()), float(err.max())


def test_sphere_grid_is_unit_and_shaped():
    intr = default_intrinsics(32, 64).unsqueeze(0)
    assert intr.shape == (1, RAY_INTRINSIC_DIM)
    g = sphere_grid(intr, 32, 64)
    assert g.shape == (1, 32, 64, 3)
    assert torch.allclose(g.norm(dim=-1), torch.ones(1, 32, 64), atol=1e-5)


def test_decoded_rays_are_unit_vectors():
    torch.manual_seed(0)
    coeffs = torch.randn(2, 15, 3)
    intr = default_intrinsics(24, 32).unsqueeze(0).repeat(2, 1)
    rays = decode_rays(coeffs, intr, 24, 32)
    assert rays.shape == (2, 3, 24, 32)
    assert torch.allclose(rays.norm(dim=1), torch.ones(2, 24, 32), atol=1e-5)


def test_fit_recovers_coefficients_of_a_synthetic_sh_field():
    """Fitting the raw (pre-normalization) field is an exact linear inverse."""
    torch.manual_seed(1)
    H, W = 24, 32
    intr = default_intrinsics(H, W).unsqueeze(0)
    true_c = torch.randn(1, 15, 3)
    raw = sh_ray_field(true_c, intr, H, W)
    assert torch.allclose(fit_sh_coeffs(raw, intr), true_c, atol=1e-4)


def test_fit_is_insensitive_to_the_base_grid_fov():
    """Fit quality is flat across a wide band of base-grid FoV.

    The SH coefficients absorb the base grid's angular scale, so ``hfov`` is
    weakly identified: every value from 10 to 60 deg fits the Aria lens to well
    under a tenth of a degree.  Worth pinning down -- it means a network that
    predicts a "wrong" base FoV is not thereby broken.
    """
    H = W = 96
    rays, valid = aria_214_1_kb4(H, W).ray_field(H, W)
    means = []
    for deg in (10.0, 20.0, 40.0, 60.0):
        hf = math.radians(deg)
        intr = torch.tensor([[hf, hf * H / W, W / 2.0, H / 2.0]])
        coeffs = fit_sh_coeffs(rays.unsqueeze(0).permute(0, 3, 1, 2), intr, valid.unsqueeze(0))
        recon = decode_rays(coeffs, intr, H, W)
        mean, _ = _mean_max_deg(recon.permute(0, 2, 3, 1), rays.unsqueeze(0), valid.unsqueeze(0))
        means.append(mean)
    assert max(means) < 0.15, f"means {means}"


def test_equirect_is_exactly_representable_at_degree_1():
    """The l=1 block is linear in (x, y, z), so an ERP ray field is an exact fit.

    Exact only when the base grid shares the pixel-centre convention of paper
    Sec. D.3, which it does through the principal point: ``cx = (W - 1) / 2``
    makes ``sphere_grid`` and ``Equirect.ray_field`` differ by a sign flip on
    ``y``, and a sign flip is linear.  This is the regression test for that.
    """
    H, W = 32, 64
    rays, valid = Equirect().ray_field(H, W)
    intr = default_intrinsics(H, W, hfov=2 * math.pi).unsqueeze(0)
    coeffs = fit_sh_coeffs(rays.unsqueeze(0).permute(0, 3, 1, 2), intr, degree=1)
    recon = decode_rays(coeffs, intr, H, W, degree=1)
    mean, mx = _mean_max_deg(recon.permute(0, 2, 3, 1), rays.unsqueeze(0), valid.unsqueeze(0))
    assert mx < 0.05, f"ERP fit max {mx:.4f} deg (mean {mean:.4f})"


def test_a_half_pixel_base_grid_offset_costs_a_constant_180_over_w():
    """Why the principal point is ``(W - 1) / 2``: the old ``W / 2`` is half a pixel out.

    The residual is not noise -- it is a constant angular bias of ``180 / W``
    degrees, which no amount of SH degree removes because it is a latitude
    shift, and a latitude shift is not a rotation.
    """
    H, W = 32, 64
    rays, valid = Equirect().ray_field(H, W)
    off = torch.tensor([[2 * math.pi, math.pi, W / 2.0, H / 2.0]])
    coeffs = fit_sh_coeffs(rays.unsqueeze(0).permute(0, 3, 1, 2), off, degree=1)
    recon = decode_rays(coeffs, off, H, W, degree=1)
    mean, mx = _mean_max_deg(recon.permute(0, 2, 3, 1), rays.unsqueeze(0), valid.unsqueeze(0))
    assert abs(mean - 180.0 / W) < 0.05, f"expected ~{180.0 / W:.3f} deg, got {mean:.3f}"
    assert mx - mean < 0.05, "the offset shows up as a constant bias, not a tail"


def test_pinhole_ray_field_is_well_approximated():
    H = W = 64
    cam = Pinhole.from_fov(H, W, 90.0)
    rays, valid = cam.ray_field(H, W)
    res = fit_ray_field(rays, valid, H, W)
    assert res.max_deg < 0.5, f"pinhole fit mean {res.mean_deg:.4f} max {res.max_deg:.4f} deg"


def test_aria_kb4_ray_field_is_well_approximated():
    """The lens ADT is actually shot on -- the fit that matters for this repo.

    Measured at H=W=96 over the true imaged cone (54.83 deg): **mean 0.055 deg,
    max 0.32 deg**.  An earlier version of this test masked to the fold-back
    turnover (62.33 deg) instead and reported 0.155 / 2.82 -- the whole tail was
    dead vignette ring the lens never images.  So a degree-3 expansion expresses
    this lens comfortably; the bounds below come from that measurement.
    """
    H = W = 96
    rays, valid = aria_214_1_kb4(H, W).ray_field(H, W)
    res = fit_ray_field(rays, valid, H, W)
    print(f"\n  Aria KB4 SH(deg<=3) fit: mean {res.mean_deg:.4f} deg, "
          f"max {res.max_deg:.4f} deg, hfov* {math.degrees(res.hfov):.1f} deg")
    assert res.mean_deg < 0.10, f"mean {res.mean_deg:.4f} deg"
    assert res.max_deg < 0.6, f"max {res.max_deg:.4f} deg"


def test_kb4_fit_error_is_confined_to_the_rim():
    """Error still grows toward the rim, just an order of magnitude smaller."""
    H = W = 96
    cam = aria_214_1_kb4(H, W)
    rays, valid = cam.ray_field(H, W)
    res = fit_ray_field(rays, valid, H, W)
    recon = decode_rays(res.coeffs, res.intrinsics, H, W)
    err = torch.rad2deg(torch.arccos((recon * rays.unsqueeze(0).permute(0, 3, 1, 2)).sum(1).clamp(-1, 1)))[0]

    theta = torch.arccos(rays[..., 2].clamp(-1, 1))
    inner = valid & (theta < 0.9 * cam.theta_max(H, W))
    assert float(err[inner].max()) < 0.3, f"inner-cone max {float(err[inner].max()):.4f} deg"
    assert float(err[valid].quantile(0.99)) < 0.5


def test_fit_ignores_invalid_pixels():
    """Masked-out corners must not drag the fit; KB4 corners hold garbage rays."""
    H = W = 64
    rays, valid = aria_214_1_kb4(H, W).ray_field(H, W)
    poisoned = rays.clone()
    poisoned[~valid] = torch.tensor([0.0, 0.0, -1.0])   # nonsense outside the cone
    a = fit_ray_field(rays, valid, H, W)
    b = fit_ray_field(poisoned, valid, H, W)
    assert abs(a.mean_deg - b.mean_deg) < 1e-3


def test_degree_3_beats_degree_1_on_fisheye():
    """Higher SH degrees buy accuracy on a distorted lens -- why L=3, not L=1."""
    H = W = 96
    rays, valid = aria_214_1_kb4(H, W).ray_field(H, W)
    d1 = fit_ray_field(rays, valid, H, W, degree=1)
    d3 = fit_ray_field(rays, valid, H, W, degree=3)
    assert d3.mean_deg < d1.mean_deg
