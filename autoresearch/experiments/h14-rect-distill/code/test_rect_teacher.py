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
    # ...and the diagnostic sweep can still build one, deliberately.
    pin = RT.virtual_pinhole(cam, fov_deg=84.6, size=504, require_full_cone=False)
    assert RT.coverage(cam, pin) < 0.9


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
    pin = RT.virtual_pinhole(cam, fov_deg=110.0, size=630)
    assert abs(cam.cx - (cam.width - 1) / 2.0) > 2.0, (
        "fixture no longer has the off-centre principal point this pins")
    RT.assert_shared_axis(cam, pin)          # must not raise


def test_630_restores_centre_sampling_parity_and_504_does_not():
    """The teacher size, pinned against the camera rather than against a comment.

    Aria's measured focal at 504 px is 218.69. A 110 deg pinhole has focal
    0.3501*size, so 504 gives 176.45 (0.807x -- the centre is BLURRED) and 630
    gives 220.57 (1.009x, parity). The centre is the region every method in
    this project has had to protect, so a teacher that blurs it is a trap, and
    the number that avoids it belongs in a test. An earlier draft asserted
    0.72x and 700 px from a guessed focal, and this test is what caught it.
    """
    cam = aria(504)
    f_fish = cam.fx
    assert f_fish == pytest.approx(218.69, abs=0.05)
    assert RT.virtual_pinhole(cam, 110.0, 504).fx / f_fish == pytest.approx(0.807, abs=0.005)
    assert RT.virtual_pinhole(cam, 110.0, 630).fx / f_fish == pytest.approx(1.009, abs=0.005)


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
    pin = RT.virtual_pinhole(cam, fov_deg=110.0, size=630)
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
    pin = RT.virtual_pinhole(cam, fov_deg=110.0, size=630)
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


# ------------------------------------------------------------- the view rig

def test_rotation_is_a_rotation_and_points_where_it_says():
    for tilt, az in ((0.0, 0.0), (36.0, 0.0), (36.0, 120.0), (55.0, 300.0)):
        R = RT.ViewSpec(50.0, 280, 154, tilt_deg=tilt, azimuth_deg=az).rotation()
        assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-5)
        assert float(torch.det(R)) == pytest.approx(1.0, abs=1e-5)
        axis = R[:, 2]
        assert math.degrees(math.acos(float(axis[2]))) == pytest.approx(tilt, abs=1e-4)
        if tilt > 0:
            got = math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 360.0
            assert got == pytest.approx(az % 360.0, abs=1e-3)


def test_tilted_SQUARE_views_cannot_be_filled_and_the_geometry_says_why():
    """The impossibility that turned out to be about squares, with the number.

    A square view tilted by ``t`` with half-FOV ``h`` has corner half-angle
    ``atan(sqrt(2) tan h)``, and the sqrt(2) is the diagonal OF A SQUARE. On
    Aria's 54.7 deg cone that makes a 5x90 deg rig fill only ~66% of its
    frames -- worse than the 110 deg single view that the sweep already
    measured inverting the teacher. Kept as a test because the NEXT test is
    the point: drop the square and the impossibility goes away.
    """
    cam = aria()
    sq5 = RT.Rig.square_multi(cam, fov_deg=90.0, size=630, n=5, tilt_deg=40.0)
    one110 = RT.Rig.single(cam, fov_deg=110.0, size=630)
    assert sq5.coverage > 0.999
    assert sq5.fill_fraction == pytest.approx(0.658, abs=0.02)
    assert sq5.fill_fraction < one110.fill_fraction


def test_the_ring_fills_its_frames_AND_covers_the_cone():
    """Both at once -- which the square rig could not do.

    A tangentially elongated view sweeps ALONG the annulus rather than reaching
    across it, so widening its long axis costs nothing in radial reach. That is
    the whole idea, and these three numbers are it.
    """
    cam = aria()
    rig = RT.Rig.ring(cam)
    theta = torch.acos(cam.ray_grid(cam.height, cam.width)[..., 2].clamp(-1, 1))
    assert rig.coverage > 0.999, f"coverage {rig.coverage:.4f}"
    assert rig.fill_fraction > 0.98, f"mean fill {rig.fill_fraction:.4f}"
    # the band the whole project is about, which the 95 deg single view could
    # only answer for ~70% of
    assert rig.zone_coverage(theta, 38.0, 54.9) > 0.995


def test_the_ring_answers_for_the_rim_band_that_the_single_view_cannot():
    cam = aria()
    theta = torch.acos(cam.ray_grid(cam.height, cam.width)[..., 2].clamp(-1, 1))
    single = RT.Rig.single(cam, fov_deg=95.0, size=630)
    ring = RT.Rig.ring(cam)
    assert single.zone_coverage(theta, 38.0, 54.9) < 0.85
    assert ring.zone_coverage(theta, 38.0, 54.9) > 0.995
    assert ring.fill_fraction > single.fill_fraction - 0.02


def test_the_rig_reports_its_distinct_frame_sizes():
    """The caller re-installs the backbone per size; it must know how many."""
    assert RT.Rig.single(aria()).sizes == [(630, 630)]
    assert RT.Rig.ring(aria()).sizes == [(630, 630), (154, 280)]


def _analytic_forward(rig, scale=None):
    """A teacher that answers exactly, from geometry, in each view's own frame."""
    def forward_z(_img, view):
        uv = pixel_grid(view.spec.height, view.spec.width, dtype=torch.float32)
        ray_view = view.pin.unproject(uv)
        world = ray_view @ view.R_vc.transpose(0, 1)
        z = smooth_range_field(world) * ray_view[..., 2]
        if scale is not None:
            k = rig.views.index(view)
            z = z * scale[k]
        return z
    return forward_z


@pytest.mark.parametrize("layout", ["single", "ring"])
def test_the_rig_transports_a_direction_only_field_exactly(layout):
    """The whole depth bookkeeping -- rotations, addresses, the cos division.

    Every view is handed the analytic planar z of one direction-only range
    field IN ITS OWN FRAME. If the rotations, the sampling addresses or the
    `cos_local` division were wrong for any view, the fused map would disagree
    with the same field evaluated directly on the fisheye grid, and it would
    disagree in a radially smooth way -- the error class no scale alignment can
    absorb and that this project has already paid for once (#38 v1).
    """
    cam = aria(252)
    rig = (RT.Rig.single(cam, fov_deg=95.0, size=280) if layout == "single"
           else RT.Rig.ring(cam, centre_size=280, ring_width=140, ring_height=84))
    fused, info = rig.teach(_analytic_forward(rig), torch.zeros(3, cam.height, cam.width),
                            align=False)
    assert len(info["log_scale"]) == len(rig.views)

    direct = smooth_range_field(cam.ray_grid(cam.height, cam.width))
    theta = torch.acos(cam.ray_grid(cam.height, cam.width)[..., 2].clamp(-1, 1))
    inner = rig.covered & (theta < cam.theta_max - math.radians(1.5))
    err = ((fused - direct).abs() / direct.abs())[inner]
    assert float(err.max()) < 6e-3, f"{layout}: max rel err {float(err.max()):.2e}"


def test_alignment_removes_a_seam_that_unaligned_fusion_would_stitch_in():
    """Views are run independently, so their scales are their own.

    Fusing without aligning them writes a step discontinuity into the target
    along every view boundary, and a student trained on it would learn the
    step. The scales here are deliberately gross so the test fails loudly if
    alignment is ever dropped.
    """
    cam = aria(252)
    rig = RT.Rig.ring(cam, centre_size=280, ring_width=140, ring_height=84)
    scales = [1.0, 1.35, 0.74, 1.20, 0.83, 1.28, 0.79]
    img = torch.zeros(3, cam.height, cam.width)
    raw, _ = rig.teach(_analytic_forward(rig, scales), img, align=False)
    fixed, info = rig.teach(_analytic_forward(rig, scales), img, align=True)

    direct = smooth_range_field(cam.ray_grid(cam.height, cam.width))
    theta = torch.acos(cam.ray_grid(cam.height, cam.width)[..., 2].clamp(-1, 1))
    inner = rig.covered & (theta < cam.theta_max - math.radians(1.5))
    err_raw = float(((raw - direct).abs() / direct)[inner].max())
    err_fix = float(((fixed - direct).abs() / direct)[inner].max())
    assert err_raw > 0.15, f"the unaligned seam should be gross, got {err_raw:.3f}"
    assert err_fix < 6e-3, f"alignment left {err_fix:.3f}"
    for k, sc in enumerate(scales):
        assert info["log_scale"][k] == pytest.approx(-math.log(sc), abs=0.03)


def test_the_rig_roundtrip_control_is_close_to_the_identity():
    cam = aria(252)
    rig = RT.Rig.ring(cam, centre_size=280, ring_width=140, ring_height=84)
    field = smooth_range_field(cam.ray_grid(cam.height, cam.width))
    back, _ = rig.roundtrip(field)
    theta = torch.acos(cam.ray_grid(cam.height, cam.width)[..., 2].clamp(-1, 1))
    inner = rig.covered & (theta < cam.theta_max - math.radians(1.5))
    assert float(((back - field).abs() / field)[inner].max()) < 9e-3
