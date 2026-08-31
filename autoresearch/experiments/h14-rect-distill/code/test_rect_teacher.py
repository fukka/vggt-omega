"""CPU tests for the fisheye <-> co-axial pinhole transport. No weights, no data.

Run on the box (this Mac has no torch -- POLICY.md, 2026-08-22 note):
    python -m pytest autoresearch/experiments/h14-rect-distill/code/ -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from raytun3r.cameras import from_aria, pixel_grid  # noqa: E402
import rect_teacher as RT  # noqa: E402

def aria(size: int = 504):
    """The calibration of record, from the repo's own constructor.

    Not a hard-coded copy: `from_aria` needs no data files, and a duplicated
    (fx, k, theta_max) here would be free to drift from the lens every other
    experiment in this workspace is using. `rotated=False` matches
    `AriaLocalPairs`, which uses frames in native sensor orientation.
    """
    return from_aria(size, size, rotated=False)


def smooth_range_field(rays: torch.Tensor) -> torch.Tensor:
    """A smooth positive function of DIRECTION only.

    Direction-only is the whole point: two cameras sharing an optical centre
    see the same world point along the same ray, so any such field is
    identically observed by both. If the transport were wrong the two would
    disagree, and no smoothness could hide it.
    """
    return 2.0 + 0.7 * rays[..., 0] + 0.4 * rays[..., 1] + 0.3 * rays[..., 2]


# --------------------------------------------------------------- construction

def test_a_110_deg_pinhole_covers_the_whole_aria_cone():
    cam = aria()
    pin = RT.virtual_pinhole(cam, fov_deg=110.0, size=504)
    assert RT.coverage(cam, pin) > 0.999


def test_the_fovbench_85_deg_rectification_does_not_cover_the_cone():
    """The measurement that licenses this experiment used a ~85 deg pinhole.

    `fovbench/geometry.py` renders at focal 0.55*max(H, W) -- 84.6 deg -- whose
    frame reaches 42.3 deg on axis and 52.1 deg only in the corners. So 024A's
    "the rect rim penalty collapses to ~1.0" is a statement about a pinhole
    that does not image the whole rim, and a teacher built at that FOV would
    inherit the gap. This test exists so that number can never be quietly
    reused as if it covered the cone.
    """
    cam = aria()
    with pytest.raises(ValueError, match="does not image the whole cone"):
        RT.virtual_pinhole(cam, fov_deg=84.6, size=504)


def test_virtual_pinhole_refuses_a_non_patch_multiple_size():
    with pytest.raises(ValueError, match="multiple of patch"):
        RT.virtual_pinhole(aria(), fov_deg=110.0, size=505)


def test_the_real_aria_camera_passes_the_shared_axis_check():
    """Regression: Aria's principal point is ~4.5 px off the frame centre.

    An earlier draft of `assert_shared_axis` tested "principal point at the
    frame centre" and would have rejected the real camera -- the transport
    would then have been unreachable on the only data this experiment runs on.
    Co-axiality is about the AXIS, not about where the axis lands in the frame.
    """
    cam = aria(504)
    pin = RT.virtual_pinhole(cam, fov_deg=110.0, size=700)
    assert abs(cam.cx - (cam.width - 1) / 2.0) > 2.0, (
        "fixture no longer has the off-centre principal point this pins")
    RT.assert_shared_axis(cam, pin)          # must not raise


def test_size_700_restores_centre_sampling_parity():
    """The docstring's arithmetic, pinned.

    At 504 the fisheye's on-axis focal is ~245 px. A 110 deg pinhole has focal
    0.350*size, so 504 gives 176 px (0.72x -- the centre is BLURRED) and 700
    gives 245 px (parity). The centre is the region every method in this
    project has had to protect, so a teacher that blurs it is a trap, and the
    number that avoids it belongs in a test rather than in a comment.
    """
    cam = aria(504)
    f_fish = cam.fx
    f_504 = RT.virtual_pinhole(cam, 110.0, 504).fx
    f_700 = RT.virtual_pinhole(cam, 110.0, 700).fx
    assert f_504 / f_fish == pytest.approx(0.72, abs=0.02)
    assert f_700 / f_fish == pytest.approx(1.00, abs=0.03)


# ------------------------------------------------------------------ transport

def test_range_and_z_are_both_invariant_under_the_transport():
    """The claim that licenses converting nothing, checked numerically.

    A direction-only field is rendered independently in each camera, then the
    pinhole one is transported onto the fisheye grid. Agreement to
    interpolation error means the address math is right AND that neither
    convention needs a cos(theta) anywhere -- the error class that invalidated
    #38 v1 and cost a four-row re-run.
    """
    cam = aria(504)
    pin = RT.virtual_pinhole(cam, fov_deg=110.0, size=700)
    grid, covered = RT.grid_pinhole_to_fisheye(cam, pin)

    rays_f = cam.ray_grid(cam.height, cam.width)
    rays_p = pin.ray_grid(pin.height, pin.width)
    cos_f = rays_f[..., 2].clamp_min(1e-6)
    cos_p = rays_p[..., 2].clamp_min(1e-6)

    for name, make in (("range", lambda r, c: smooth_range_field(r)),
                       ("z", lambda r, c: smooth_range_field(r) * c)):
        on_fisheye = make(rays_f, cos_f)
        on_pinhole = make(rays_p, cos_p)
        moved = RT.warp(on_pinhole, grid)
        # Ignore a 2 px band at the cone edge: there the pinhole neighbourhood
        # straddles the frame border and bilinear sampling mixes in a zero.
        theta = torch.acos(rays_f[..., 2].clamp(-1, 1))
        inner = covered & (theta < cam.theta_max - math.radians(1.0))
        err = ((moved - on_fisheye).abs() / on_fisheye.abs())[inner]
        assert float(err.max()) < 2e-3, f"{name}: max rel err {float(err.max()):.2e}"


def test_out_of_cone_pixels_are_zero_not_replicated():
    """Zero-filled, never border-replicated: a replicated pixel is invented."""
    cam = aria(252)
    pin = RT.virtual_pinhole(cam, fov_deg=110.0, size=280)
    grid, valid = RT.grid_fisheye_to_pinhole(cam, pin)
    img = torch.ones(3, cam.height, cam.width)
    out = RT.warp(img, grid)
    assert not valid.all(), "a square pinhole must have corners outside the cone"
    assert float(out[:, ~valid].abs().max()) == 0.0
    assert float(out[:, valid].min()) > 0.99


def test_the_round_trip_returns_a_smooth_map_inside_the_cone():
    """fisheye -> pinhole -> fisheye is the CONTROL arm's whole pipeline.

    It must be close to the identity, because the control's claim is "the same
    resampling budget, without the change of image formation". If the round
    trip destroyed the map, the control would be handicapped rather than
    matched and any win for the treatment would be uninterpretable.
    """
    cam = aria(504)
    pin = RT.virtual_pinhole(cam, fov_deg=110.0, size=700)
    g_in, _ = RT.grid_fisheye_to_pinhole(cam, pin)
    g_out, covered = RT.grid_pinhole_to_fisheye(cam, pin)

    rays_f = cam.ray_grid(cam.height, cam.width)
    field = smooth_range_field(rays_f)
    there_and_back = RT.warp(RT.warp(field, g_in), g_out)

    theta = torch.acos(rays_f[..., 2].clamp(-1, 1))
    inner = covered & (theta < cam.theta_max - math.radians(1.0))
    err = ((there_and_back - field).abs() / field)[inner]
    assert float(err.max()) < 3e-3


def test_the_two_grids_are_inverses_of_each_other_as_addresses():
    """Address-level check, independent of any sampled values."""
    cam = aria(252)
    pin = RT.virtual_pinhole(cam, fov_deg=110.0, size=280)
    rays = cam.ray_grid(cam.height, cam.width)
    uv_pin = pin.project(rays)
    back = cam.project(pin.unproject(uv_pin))
    uv_fish = pixel_grid(cam.height, cam.width, dtype=torch.float32)
    cone = torch.acos(rays[..., 2].clamp(-1, 1)) <= cam.theta_max
    assert float((back - uv_fish).abs()[cone].max()) < 0.02


def test_the_cone_is_the_lens_cone_not_the_kb4_turnover():
    """`theta_max` must stay the usable cone, not the polynomial's turnover.

    `jacobian.py` and `scannetpp-camera-reference.md` both warn that a KB4 fit
    read past the field it was fitted on stops describing the physical lens.
    The pinhole is sized from `theta_max`, so if that ever became the turnover
    (62.33 deg for Aria) the teacher would be asked to image rays the lens
    never captured.
    """
    cam = aria(504)
    assert 50.0 < math.degrees(cam.theta_max) < 58.0
