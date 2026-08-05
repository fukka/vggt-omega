"""Camera models that generate ground-truth ray fields.

The Aria KB4 numbers here are cross-checked against the repo's existing,
independently-validated implementation in ``VGGT-360-fisheye/utils/fisheye_cam.py``
(calibration validated to <0.22 deg against projectaria_tools).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cam3r.cameras import Equirect, Pinhole, aria_214_1_kb4, aria_valid_theta_max, build_camera


def test_pinhole_centre_ray_is_the_optical_axis():
    # Integer principal point so one pixel sits exactly on the axis.
    cam = Pinhole(fx=100.0, fy=100.0, cx=32.0, cy=32.0)
    rays, valid = cam.ray_field(64, 64)
    assert rays.shape == (64, 64, 3) and valid.all()
    assert torch.allclose(rays[32, 32], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)
    # Half-pixel offsets are real, not rounded away.
    assert float(rays[32, 33, 0]) > 0 and float(rays[33, 32, 1]) > 0


def test_ray_fields_are_unit_norm():
    for cam in (Pinhole(80.0, 80.0, 31.5, 31.5), aria_214_1_kb4(64, 64), Equirect()):
        rays, _ = cam.ray_field(64, 32)
        assert torch.allclose(rays.norm(dim=-1), torch.ones(64, 32), atol=1e-5)


def test_pinhole_fov_is_recovered_from_the_corner_ray():
    """A 90 deg horizontal FoV pinhole puts its edge ray at 45 deg incidence."""
    W = H = 64
    f = (W / 2.0) / math.tan(math.radians(45.0))
    cam = Pinhole(f, f, (W - 1) / 2.0, (H - 1) / 2.0)
    rays, _ = cam.ray_field(H, W)
    edge = rays[H // 2, -1]
    assert abs(math.degrees(math.acos(float(edge[2]))) - 44.55) < 0.5   # (W-1)/2 pixels out


def test_equirect_spans_the_full_sphere():
    rays, valid = Equirect().ray_field(64, 128)
    assert valid.all()
    assert float(rays[..., 2].max()) > 0.999 and float(rays[..., 2].min()) < -0.999
    # Row 0 is the top of the panorama -> y = -1 (y points down).
    assert float(rays[0, :, 1].mean()) < -0.99


def test_kb4_matches_the_repo_reference_implementation():
    """Agreement with VGGT-360-fisheye's KB4, which was validated against projectaria_tools."""
    ref_dir = Path(__file__).resolve().parents[2] / "VGGT-360-fisheye"
    if not (ref_dir / "utils" / "fisheye_cam.py").exists():
        pytest.skip("VGGT-360-fisheye not present")
    sys.path.insert(0, str(ref_dir))
    from utils.fisheye_cam import aria_intrinsics, fisheye_ray_lut

    H = W = 96
    ref_cam = aria_intrinsics(H, W, rotated=True)
    ref_rays, ref_valid = fisheye_ray_lut(ref_cam)

    rays, valid = aria_214_1_kb4(H, W).ray_field(H, W)
    # Exact mask agreement, not an intersection -- intersecting would hide
    # precisely the failure mode this test exists to catch (a cone defined by
    # the wrong limit).
    assert np.array_equal(valid.numpy(), np.asarray(ref_valid)), (
        f"validity masks differ on {int((valid.numpy() != np.asarray(ref_valid)).sum())} pixels"
    )
    dot = (rays.numpy()[valid.numpy()] * np.asarray(ref_rays)[valid.numpy()]).sum(-1).clip(-1, 1)
    max_deg = float(np.degrees(np.arccos(dot)).max())
    assert max_deg < 0.05, f"max ray disagreement {max_deg:.4f} deg"


def test_imaged_cone_is_not_the_fold_back_turnover():
    """CONTEXT.md: the imaged cone is 54.83 deg; the turnover is 62.33 deg.

    Using the turnover as the cone admits a ring of dead vignette pixels that
    the lens never illuminates.  theta_max must be the min of the two.
    """
    cam = aria_214_1_kb4(128, 128)
    assert abs(math.degrees(cam.turnover_theta()) - 62.33) < 0.1
    assert abs(math.degrees(aria_valid_theta_max()) - 54.83) < 0.05
    assert abs(math.degrees(cam.theta_max(128, 128)) - 54.83) < 0.05


def test_kb4_masks_rays_outside_the_imaged_cone():
    cam = aria_214_1_kb4(128, 128)
    rays, valid = cam.ray_field(128, 128)
    theta = torch.arccos(rays[..., 2].clamp(-1, 1))
    assert math.degrees(float(theta[valid].max())) < 55.0
    assert 0.3 < float(valid.float().mean()) < 0.95, "expected a circular valid cone"


def test_imaged_cone_is_resolution_invariant():
    """The cone is a property of the lens, not of the grid a frame was resized to."""
    fracs = [float(aria_214_1_kb4(n, n).ray_field(n, n)[1].float().mean()) for n in (64, 128, 256)]
    assert max(fracs) - min(fracs) < 0.01, fracs


def test_build_camera_round_trips_through_its_spec():
    cam = aria_214_1_kb4(64, 64)
    rebuilt = build_camera(cam.spec())
    a, va = cam.ray_field(48, 48)
    b, vb = rebuilt.ray_field(48, 48)
    assert torch.allclose(a, b, atol=1e-6)
    # The imaged cone must survive the round-trip, not silently widen.
    assert torch.equal(va, vb)


def test_kb4_projection_inverts_unprojection():
    cam = aria_214_1_kb4(64, 64)
    rays, valid = cam.ray_field(64, 64)
    uv = cam.project(rays)
    ys, xs = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    grid = torch.stack([xs, ys], dim=-1)
    err = (uv - grid).norm(dim=-1)[valid]
    assert float(err.max()) < 1e-2, f"max reprojection error {float(err.max()):.4f} px"
