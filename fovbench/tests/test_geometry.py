# Copyright (c) 2026.
"""Geometry invariants for the ADT-FOV test.

Every assertion here is checked against a quantity derived *independently* of
the implementation — an analytic scene whose depth is known in closed form, or
a scalar predicted from the view geometry by hand. That is deliberate: the
failure this module exists to prevent (comparing planar z against euclidean
range, or against planar z along a *different* axis) is invisible in the output
— both sides are plain positive depth maps of the same shape — and on Aria
geometry it is worth up to 2.15x, an order of magnitude more than the effect
the benchmark measures.

CPU-only: no weights, no ADT, no GPU.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from fovbench import geometry as G  # noqa: E402
from fovbench.geometry import aria_cam  # noqa: E402

# Small frame: every helper is resolution-invariant, and 176 = 1408/8 keeps the
# KB4 principal-point offset proportionally identical to the real frame.
FRAME = 176
OUT = 64


def _cam():
    return aria_cam(FRAME, FRAME)


def _plane_scene(cam, z_m=2.0):
    """GT for a fronto-parallel plane at ``z = z_m``: planar z is constant."""
    gt = np.full((cam.H, cam.W), z_m, np.float32)
    _, cone = G.fisheye_rays(cam)
    return gt * cone, cone


def _sphere_scene(cam, r_m=3.0):
    """GT for a sphere of radius ``r_m`` about the optical centre.

    Every ray has euclidean range ``r_m``, so ADT-style planar z is
    ``r_m * cos(theta)`` — a *radial* GT, which is exactly the pattern a
    range/z confusion produces. If a warp gets the convention wrong this scene
    is where it shows.
    """
    rays, cone = G.fisheye_rays(cam)
    return (r_m * rays[..., 2]).astype(np.float32) * cone, cone


def _rgb(cam):
    """A textured frame — content is irrelevant to geometry, but the renderers
    take an image, and a constant one would hide a transposed sampling map."""
    ys, xs = np.mgrid[0:cam.H, 0:cam.W]
    g = ((xs * 3 + ys * 7) % 256).astype(np.uint8)
    return np.stack([g, np.roll(g, 5, 1), np.roll(g, 9, 0)], -1)


# --------------------------------------------------------------------------- #
# theta maps — the binning axis
# --------------------------------------------------------------------------- #

def test_fisheye_theta_map_agrees_with_the_ray_lut():
    cam = _cam()
    rays, cone = G.fisheye_rays(cam)
    theta = G.theta_map_fisheye(cam)
    expect = np.degrees(np.arccos(np.clip(rays[..., 2], -1, 1)))
    assert np.allclose(theta[cone], expect[cone], atol=1e-4)


def test_fisheye_theta_map_is_zero_on_axis_and_reaches_the_usable_rim():
    cam = _cam()
    theta = G.theta_map_fisheye(cam)
    on_axis = theta[int(round(cam.cy)), int(round(cam.cx))]
    assert on_axis < 0.5
    # The usable cone is ~54.83 deg (fisheye_cam.aria_valid_theta_max).
    assert 54.0 < math.degrees(cam.theta_max()) < 55.5
    _, cone = G.fisheye_rays(cam)
    assert theta[cone].max() <= math.degrees(cam.theta_max()) + 1e-3


def test_pinhole_theta_map_matches_the_rectifier_focal():
    """The rectified frame shares the fisheye's optical axis, so its theta is
    the same physical incidence angle — that is what makes the rectified and
    raw arms of the benchmark binnable on one axis."""
    theta = G.theta_map_pinhole(OUT, OUT, focal_frac=0.55)
    assert theta[OUT // 2, OUT // 2] < 2.0
    edge = math.degrees(math.atan(0.5 / 0.55))          # 42.27 deg
    assert abs(theta[OUT // 2, -1] - edge) < 1.5
    corner = math.degrees(math.atan(math.sqrt(0.5) / 0.55))
    assert abs(theta[0, 0] - corner) < 1.5


# --------------------------------------------------------------------------- #
# Window GT warp — the depth-convention seam
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", ["rect", "fisheye"])
def test_centre_window_of_a_plane_keeps_gt_unchanged(kind):
    """tilt=0 shares the camera axis, so planar z must survive the warp as-is."""
    cam = _cam()
    gt, cone = _plane_scene(cam, z_m=2.0)
    w = G.render_window(_rgb(cam), gt, cone, cam, azimuth=0.0, tilt=0.0,
                        fov=40.0, out_size=OUT, kind=kind)
    assert w.valid.mean() > 0.9
    assert np.allclose(w.gt_z[w.valid], 2.0, atol=2e-2)


@pytest.mark.parametrize("kind", ["rect", "fisheye"])
@pytest.mark.parametrize("tilt", [15.0, 30.0, 40.0])
def test_tilted_window_of_a_plane_matches_the_hand_computed_centre(kind, tilt):
    """At the centre pixel the view ray IS the window axis, so a plane at z=c
    sits at range c/cos(tilt) — and planar z along the window axis equals that
    range. Predicted by hand from the view geometry, not from the code."""
    cam = _cam()
    gt, cone = _plane_scene(cam, z_m=2.0)
    w = G.render_window(_rgb(cam), gt, cone, cam, azimuth=37.0, tilt=tilt,
                        fov=40.0, out_size=OUT, kind=kind)
    c = OUT // 2
    assert w.valid[c, c]
    assert w.gt_z[c, c] == pytest.approx(2.0 / math.cos(math.radians(tilt)),
                                         rel=2e-2)


@pytest.mark.parametrize("kind", ["rect", "fisheye"])
def test_sphere_window_gt_is_range_times_cos_from_the_window_axis(kind):
    """A constant-range scene must come out as ``R * cos(angle from the window
    axis)`` — zero at nothing, R at the centre, and falling toward the corners.
    Getting this wrong (leaving GT as range, or as z about the *camera* axis)
    is the error the whole convention machinery guards against."""
    cam = _cam()
    gt, cone = _sphere_scene(cam, r_m=3.0)
    w = G.render_window(_rgb(cam), gt, cone, cam, azimuth=90.0, tilt=20.0,
                        fov=50.0, out_size=OUT, kind=kind)
    assert np.allclose(w.gt_z[w.valid], 3.0 * w.cos_view[w.valid], atol=3e-2)
    c = OUT // 2
    assert w.gt_z[c, c] == pytest.approx(3.0, rel=2e-2)


def test_tangent_grid_cos_view_is_the_angle_from_the_window_axis():
    """``cos_view`` is read off before the rotation; it must equal the dot
    product with the rotated axis, or every window's GT is warped about the
    wrong direction."""
    d_cam, cos_view = G.tangent_rays(azimuth=110.0, tilt=25.0, fov_deg=45.0,
                                     out_size=OUT)
    axis = G.view_center_dir(110.0, 25.0)
    assert np.allclose(d_cam @ axis, cos_view, atol=1e-9)


def test_project_dirs_agrees_with_the_rendered_maps():
    """``project_dirs`` re-implements the KB4 projection that
    ``fisheye_to_persp`` performs inside its remap (which does not expose it).
    Hold the two copies together."""
    cam = _cam()
    d_cam, _ = G.tangent_rays(30.0, 20.0, 45.0, OUT)
    u, v = G.project_dirs(cam, d_cam)
    from utils.fisheye_views import fisheye_to_persp
    _, valid, mapx, mapy = fisheye_to_persp(
        np.zeros((cam.H, cam.W), np.float32), cam, 30.0, 20.0, 45.0,
        height=OUT, width=OUT, return_maps=True)
    m = valid > 0.5
    assert np.abs(u[m] - mapx[m]).max() < 1e-3
    assert np.abs(v[m] - mapy[m]).max() < 1e-3


def test_raw_fisheye_window_covers_the_same_directions():
    """The two arms must cover the same directions: the raw-fisheye window is
    the square containing the rectified window's own source footprint, so its
    angular radius is at least the rectified one's and its axis is the same."""
    cam = _cam()
    gt, cone = _plane_scene(cam)
    kw = dict(azimuth=45.0, tilt=25.0, fov=40.0, out_size=OUT)
    a = G.render_window(_rgb(cam), gt, cone, cam, kind="rect", **kw)
    b = G.render_window(_rgb(cam), gt, cone, cam, kind="fisheye", **kw)
    # same axis
    assert np.allclose(a.axis, b.axis, atol=1e-9)
    # the raw crop covers at least the rectified window's angular extent
    ang_a = np.degrees(np.arccos(np.clip(a.cos_view[a.valid], -1, 1)))
    ang_b = np.degrees(np.arccos(np.clip(b.cos_view[b.valid], -1, 1)))
    assert ang_b.max() >= ang_a.max() - 1.0


def test_window_reports_the_dead_pixel_fraction():
    """The confound this benchmark must not repeat: a window pushed past the
    imaged cone is mostly black, and that has to be visible in the output
    rather than absorbed into the score."""
    cam = _cam()
    gt, cone = _plane_scene(cam)
    inside = G.render_window(_rgb(cam), gt, cone, cam, azimuth=0.0, tilt=0.0,
                             fov=40.0, out_size=OUT, kind="rect")
    outside = G.render_window(_rgb(cam), gt, cone, cam, azimuth=0.0, tilt=50.0,
                              fov=40.0, out_size=OUT, kind="rect")
    assert inside.in_cone_frac > 0.99
    assert outside.in_cone_frac < 0.85
    assert outside.in_cone_frac == pytest.approx(float(outside.valid.mean()),
                                                 abs=0.05)


def test_window_eccentricity_is_the_camera_axis_angle_not_the_window_axis():
    """``theta`` must stay measured from the CAMERA's optical axis: it is the
    distortion coordinate, and a window-relative angle would make every window
    look identical."""
    cam = _cam()
    gt, cone = _plane_scene(cam)
    w = G.render_window(_rgb(cam), gt, cone, cam, azimuth=0.0, tilt=30.0,
                        fov=30.0, out_size=OUT, kind="rect")
    c = OUT // 2
    assert w.theta[c, c] == pytest.approx(30.0, abs=1.0)
    assert w.theta[w.valid].min() > 5.0


# --------------------------------------------------------------------------- #
# Full-frame views
# --------------------------------------------------------------------------- #

def test_full_fisheye_frame_leaves_gt_in_its_stored_convention():
    cam = _cam()
    gt, cone = _sphere_scene(cam, r_m=3.0)
    v = G.full_frame_view(_rgb(cam), gt, cone, cam, out_size=OUT, kind="fisheye")
    # planar z of a sphere = R cos(theta), and theta is the camera-axis angle.
    # 2 cm at a 3 m radius is the residual of nearest-sampling a 2.75x
    # downsample; on cv2's floored INTER_NEAREST the same check reads 4.5 cm.
    expect = 3.0 * np.cos(np.radians(v.theta))
    assert np.allclose(v.gt_z[v.valid], expect[v.valid], atol=2e-2)


def test_rectified_full_frame_preserves_depth_values():
    """Rectification shares the camera centre and axis, so planar z is
    unchanged — only the pixel grid moves. A plane must stay constant."""
    cam = _cam()
    gt, cone = _plane_scene(cam, z_m=2.5)
    v = G.full_frame_view(_rgb(cam), gt, cone, cam, out_size=OUT, kind="rect")
    assert v.valid.mean() > 0.5
    assert np.allclose(v.gt_z[v.valid], 2.5, atol=1e-3)


def test_rectified_full_frame_cannot_reach_the_fisheye_rim():
    """A reportable limitation, not a bug: the ~85 deg rectified pinhole has no
    pixels past 42.3 deg off-axis except in its corners (52.6 deg at most),
    while the raw frame reaches the lens' full 54.8 deg cone. This is why the
    outermost bins of the rectified arm are thin or empty."""
    cam = _cam()
    gt, cone = _plane_scene(cam)
    rect = G.full_frame_view(_rgb(cam), gt, cone, cam, out_size=OUT, kind="rect")
    fish = G.full_frame_view(_rgb(cam), gt, cone, cam, out_size=OUT, kind="fisheye")
    corner = math.degrees(math.atan(math.sqrt(0.5) / 0.55))     # 52.55 deg
    assert rect.theta[rect.valid].max() <= corner + 1e-3
    assert fish.theta[fish.valid].max() > corner
    # and the outer annulus is only reachable in the rectified corners
    assert (rect.theta[rect.valid] > 50).mean() < 0.05
    assert (fish.theta[fish.valid] > 50).mean() > 0.05


# --------------------------------------------------------------------------- #
# Radial profile — one alignment fit, many bins
# --------------------------------------------------------------------------- #

def _profile_inputs(bias=None):
    """A 64x64 patch of GT with a theta coordinate running 0..55 deg."""
    ys, xs = np.mgrid[0:64, 0:64]
    theta = np.hypot(xs - 31.5, ys - 31.5) / 31.5 * 55.0
    gt = (2.0 + 0.01 * xs + 0.02 * ys).astype(np.float32)
    pred = gt.copy()
    if bias is not None:
        pred = pred * bias(theta)
    return pred.astype(np.float32), gt, np.ones_like(gt, bool), theta.astype(np.float32)


def test_radial_profile_scores_a_perfect_prediction_flat():
    pred, gt, mask, theta = _profile_inputs()
    out = G.radial_profile(pred, gt, mask, theta, G.THETA_EDGES, "scale_shift")
    assert out["overall"]["AbsRel"] < 1e-4
    for b in out["bins"]:
        if b["n_valid"]:
            assert b["AbsRel"] < 1e-3


def test_unaligned_radial_curve_rises_monotonically_with_eccentricity():
    """With no alignment in the way, an error that grows with eccentricity must
    read as a monotonically rising per-bin AbsRel. This is the benchmark's core
    claim on its own metric."""
    pred, gt, mask, theta = _profile_inputs(
        bias=lambda t: 1.0 + 0.6 * (np.radians(t) ** 2))
    out = G.radial_profile(pred, gt, mask, theta, G.THETA_EDGES, "none")
    curve = [b["AbsRel"] for b in out["bins"] if b["n_valid"] > 32]
    assert len(curve) >= 4
    assert curve[-1] > 10 * curve[0]
    assert all(x <= y + 1e-6 for x, y in zip(curve, curve[1:]))


def test_shared_affine_alignment_turns_a_radial_bias_into_a_bowl():
    """How to read an aligned curve — and why a *rising* one is the wrong thing
    to look for.

    A global affine fit is free to choose the radius at which it is right, and
    least squares puts that near the pixel-weighted middle of the frame. A
    monotone radial bias therefore comes out U-shaped: worst at the centre and
    at the rim, best in between. The signal is that the curve is not FLAT —
    ``scale_ratio`` is the column that keeps the monotone reading.
    """
    pred, gt, mask, theta = _profile_inputs(
        bias=lambda t: 1.0 + 0.6 * (np.radians(t) ** 2))
    out = G.radial_profile(pred, gt, mask, theta, G.THETA_EDGES, "scale_shift")
    curve = [b["AbsRel"] for b in out["bins"] if b["n_valid"] > 32]
    assert max(curve) > 1.5 * min(curve)
    assert curve[0] > min(curve) and curve[-1] > min(curve)


def test_radial_profile_scale_ratio_tracks_the_radial_drift():
    """``scale_ratio`` = median(gt/pred) per bin, measured before alignment —
    it reads out the radial scale drift directly, independent of AbsRel."""
    pred, gt, mask, theta = _profile_inputs(
        bias=lambda t: 1.0 + 0.6 * (np.radians(t) ** 2))
    out = G.radial_profile(pred, gt, mask, theta, G.THETA_EDGES, "none")
    ratios = [b["scale_ratio"] for b in out["bins"] if b["n_valid"] > 32]
    assert ratios[0] > ratios[-1]                     # pred grows with theta
    assert ratios[0] / ratios[-1] > 1.3


def test_per_bin_alignment_would_erase_the_effect():
    """Characterises the design decision. Fitting scale per bin absorbs the
    radial drift and reports a uniformly healthy curve for a model that is in
    fact bending depth with eccentricity — which is why ``radial_profile`` fits
    once over the whole frame. Under the shared fit EVERY bin scores worse than
    the WORST bin does under per-bin fitting."""
    pred, gt, mask, theta = _profile_inputs(
        bias=lambda t: 1.0 + 0.6 * (np.radians(t) ** 2))
    shared = G.radial_profile(pred, gt, mask, theta, G.THETA_EDGES, "scale_shift")
    per_bin = []
    for lo, hi in zip(G.THETA_EDGES[:-1], G.THETA_EDGES[1:]):
        m = mask & (theta >= lo) & (theta < hi)
        if m.sum() > 32:
            per_bin.append(G.radial_profile(pred, gt, m, theta,
                                            (lo, hi), "scale_shift")["overall"]["AbsRel"])
    shared_curve = [b["AbsRel"] for b in shared["bins"] if b["n_valid"] > 32]
    assert min(shared_curve) > max(per_bin)


def test_bin_by_scores_two_axes_off_one_shared_fit():
    """The rule the whole benchmark rests on: the scale is fitted once over the
    whole valid frame and frozen, and an axis is only a set of masks over it.
    Two axes over the same frame must therefore agree wherever they select the
    same pixels — here a single-bin radius axis, which selects everything."""
    pred, gt, mask, theta = _profile_inputs(
        bias=lambda t: 1.0 + 0.6 * (np.radians(t) ** 2))
    radius = theta / 55.0            # same coordinate, different units
    # One bin wide enough to hold the corners too, so it selects every pixel.
    out = G.bin_by(pred, gt, mask, "scale_shift",
                   {"theta": (theta, G.THETA_EDGES), "radius": (radius, (0.0, 9.0))})
    whole = out["radius"][0]
    assert whole["n_bin"] == int(mask.sum())
    assert whole["AbsRel"] == pytest.approx(out["overall"]["AbsRel"], rel=1e-9)
    # The theta axis stops at 55 deg, so it partitions the pixels inside its
    # own range — off the SAME frozen prediction, never a fit of its own.
    in_theta = int((mask & (theta < G.THETA_EDGES[-1])).sum())
    assert sum(b["n_bin"] for b in out["theta"]) == in_theta


def test_bin_by_reports_each_bin_s_own_gt_depth():
    pred, gt, mask, theta = _profile_inputs()
    out = G.bin_by(pred, gt, mask, "none", {"theta": (theta, G.THETA_EDGES)})
    for b, (lo, hi) in zip(out["theta"], zip(G.THETA_EDGES[:-1], G.THETA_EDGES[1:])):
        if b["n_bin"] == 0:
            continue
        m = mask & (theta >= lo) & (theta < hi)
        # float32 GT, medianed in float64 — agreement is to float32, not to bit.
        assert b["gt_median"] == pytest.approx(float(np.median(gt[m])), rel=1e-6)


def test_gt_median_exposes_the_depth_confound_in_absrel():
    """Why ``gt_median`` is reported at all.

    A model with a *constant absolute* error and no radial behaviour whatever
    scores a rising AbsRel curve wherever the scene gets nearer with
    eccentricity — AbsRel is the absolute error over the depth. Reading that
    rise as "the periphery is worse" would repeat the mistake the withdrawn
    drift column made. ``gt_median`` is what makes it visible: it falls exactly
    where AbsRel rises.
    """
    ys, xs = np.mgrid[0:64, 0:64]
    theta = (np.hypot(xs - 31.5, ys - 31.5) / 31.5 * 55.0).astype(np.float32)
    gt = (5.0 - 0.05 * theta).astype(np.float32)   # nearer toward the rim
    pred = gt + 0.10                                # 10 cm off everywhere
    mask = np.ones_like(gt, bool)
    out = G.bin_by(pred, gt, mask, "none", {"theta": (theta, G.THETA_EDGES)})
    bins = [b for b in out["theta"] if b["n_bin"] > 32]
    absrel = [b["AbsRel"] for b in bins]
    depth = [b["gt_median"] for b in bins]
    assert absrel[-1] > 1.4 * absrel[0]            # looks like a rim penalty
    assert depth[-1] < 0.75 * depth[0]             # and is entirely this
    # AbsRel x depth recovers the one constant error, to within the slack of
    # comparing a mean of 1/gt against the reciprocal of a median.
    assert all(a * d == pytest.approx(0.10, rel=0.02)
               for a, d in zip(absrel, depth))


def _depth_falls_outward(const_err=0.10, n=160, seed=0):
    """A scene whose radial AbsRel penalty is ENTIRELY depth: a flat absolute
    error, and a depth that falls with eccentricity. The depth spread is
    multiplicative and identical everywhere, so the bins overlap in depth and
    the question "what would this bin score at the frame's depth mix" has an
    answer. ``mask`` stops at 55 deg so the frame and the bins cover the same
    directions, as the fisheye cone does in the real pipeline."""
    rng = np.random.default_rng(seed)
    c = (n - 1) / 2
    ys, xs = np.mgrid[0:n, 0:n]
    theta = (np.hypot(xs - c, ys - c) / c * 55.0).astype(np.float32)
    gt = ((4.0 - 0.025 * theta)
          * rng.uniform(0.4, 2.5, theta.shape)).astype(np.float32)
    return (gt + const_err).astype(np.float32), gt, theta < 55.0, theta


def _pen(bins, key):
    v = [b[key] for b in bins]
    return v[-1] / v[0]


def test_radial_profile_bins_partition_the_valid_pixels():
    pred, gt, mask, theta = _profile_inputs()
    out = G.radial_profile(pred, gt, mask, theta, G.THETA_EDGES, "none")
    binned = sum(b["n_bin"] for b in out["bins"])
    in_range = int((mask & (theta >= G.THETA_EDGES[0])
                    & (theta < G.THETA_EDGES[-1])).sum())
    assert binned == in_range


def test_radial_profile_reports_empty_bins_without_crashing():
    pred, gt, mask, theta = _profile_inputs()
    out = G.radial_profile(pred, gt, mask, np.zeros_like(theta),
                           G.THETA_EDGES, "scale_shift")
    assert out["bins"][0]["n_valid"] > 0
    assert all(b["n_valid"] == 0 for b in out["bins"][1:])
    assert math.isnan(out["bins"][-1]["AbsRel"])


@pytest.mark.parametrize("kind", ["rect", "fisheye"])
def test_a_window_aimed_off_axis_is_sampled_at_least_as_densely_as_the_centre(kind):
    """The window arm's resolution confound, and its direction.

    The natural worry is that a window aimed at the rim is built from fewer raw
    pixels — a fisheye compresses the periphery — so a rising error across aims
    would be blur rather than geometry. On Aria's KB4 the opposite holds: a
    40 deg window on axis covers a small central disc and is *upsampled* to the
    model's grid (0.73 source px per output px), while the same window at 40 deg
    covers a wider annulus and is sampled at 1.03 (rect) / 1.19 (fisheye).

    So resolution *improves* toward the rim. Any error that still rises with aim
    is not the sampling — which makes the window result stronger, not weaker.
    Pinned here because the sign is counter-intuitive and was assumed backwards
    when the first run was read.
    """
    cam = _cam()
    gt, cone = _plane_scene(cam)
    d = [G.render_window(_rgb(cam), gt, cone, cam, 0.0, t, 40.0, OUT,
                         kind).src_px_per_out_px for t in (0.0, 40.0)]
    assert d[1] >= d[0]
    assert all(x > 0 for x in d)


# --------------------------------------------------------------------------- #
# Continuous profiles
# --------------------------------------------------------------------------- #

def test_a_pooled_profile_reproduces_a_coarse_bin_on_one_frame():
    """The curve and the table must be the same measurement read at two
    resolutions. On a single frame the two estimators coincide (there is
    nothing to weight differently), so re-aggregating the fine bins that fall
    inside a coarse bin has to return that coarse bin's AbsRel exactly."""
    pred, gt, mask, theta = _profile_inputs(
        bias=lambda t: 1.0 + 0.6 * (np.radians(t) ** 2))
    fine = tuple(float(x) for x in range(0, 56))
    out = G.bin_by(pred, gt, mask, "scale_shift", {"theta": (theta, G.THETA_EDGES)},
                   profile_edges={"theta": fine})
    prof = out["profiles"]["theta"]
    for b, (lo, hi) in zip(out["theta"], zip(G.THETA_EDGES[:-1], G.THETA_EDGES[1:])):
        if b["n_valid"] < 64:
            continue
        sel = [i for i, e in enumerate(fine[:-1]) if lo <= e < hi]
        n = sum(prof["n"][i] for i in sel)
        s = sum(prof["sum_absrel"][i] for i in sel)
        assert n == b["n_valid"]
        assert s / n == pytest.approx(b["AbsRel"], rel=1e-9)


def test_the_profile_resolves_a_knee_the_coarse_bins_cannot():
    """Why the fine axis exists. A model that is exact out to 30 deg and then
    bends reads, on six bins, as a gentle ramp starting somewhere in 20-30; the
    profile puts the corner in the right 1 deg bin."""
    ys, xs = np.mgrid[0:256, 0:256]
    theta = (np.hypot(xs - 127.5, ys - 127.5) / 127.5 * 55.0).astype(np.float32)
    gt = (2.0 + 0.01 * xs).astype(np.float32)
    pred = (gt * (1.0 + 0.02 * np.clip(theta - 30.0, 0, None))).astype(np.float32)
    mask = theta < 55.0
    fine = tuple(float(x) for x in range(0, 56))
    out = G.bin_by(pred, gt, mask, "none", {"theta": (theta, G.THETA_EDGES)},
                   profile_edges={"theta": fine})
    pooled = G.pool_profiles([out["profiles"]["theta"]])
    a = np.asarray(pooled["AbsRel"], float)
    n = np.asarray(pooled["n"])
    flat = a[(n > 64) & (np.arange(55) < 28)]
    assert np.nanmax(flat) < 1e-6                    # exact below the knee
    rise = np.where((n > 64) & (a > 1e-4))[0]
    assert 29 <= rise[0] <= 32                       # and the knee is located


def test_pooling_is_pixel_weighted_not_frame_weighted():
    """The documented difference from the coarse tables. Two frames, one with a
    hundred times the pixels of the other in the same bin: the pooled value must
    follow the big frame, which a mean of per-frame means would not."""
    big = {"edges": [0.0, 1.0], "n": [1000], "sum_absrel": [100.0],
           "sum_delta1": [1000.0], "sum_gt": [2000.0], "sum_gt2": [4000.0]}
    small = {"edges": [0.0, 1.0], "n": [10], "sum_absrel": [5.0],
             "sum_delta1": [0.0], "sum_gt": [20.0], "sum_gt2": [40.0]}
    pooled = G.pool_profiles([big, small])
    assert pooled["AbsRel"][0] == pytest.approx(105.0 / 1010.0)   # not (0.1+0.5)/2
    assert pooled["n"] == [1010]
    assert pooled["n_frames"] == 2


def test_an_empty_profile_bin_is_nan_not_zero():
    pred, gt, mask, theta = _profile_inputs()
    fine = (0.0, 1.0, 2.0, 3.0)
    out = G.bin_by(pred, gt, mask, "none", {"theta": (theta, G.THETA_EDGES)},
                   profile_edges={"theta": fine})
    pooled = G.pool_profiles([out["profiles"]["theta"]])
    for n, a in zip(pooled["n"], pooled["AbsRel"]):
        assert (n == 0) == math.isnan(a)


def test_profiles_are_absent_unless_asked_for():
    pred, gt, mask, theta = _profile_inputs()
    out = G.bin_by(pred, gt, mask, "none", {"theta": (theta, G.THETA_EDGES)})
    assert "profiles" not in out


def test_pooling_refuses_to_add_profiles_of_different_axes():
    a = {"edges": [0.0, 1.0], "n": [4], "sum_absrel": [1.0],
         "sum_delta1": [4.0], "sum_gt": [8.0], "sum_gt2": [16.0]}
    b = dict(a, edges=[0.0, 2.0])
    with pytest.raises(ValueError):
        G.pool_profiles([a, b])


# --------------------------------------------------------------------------- #
# What an EMPTY ROOM alone produces — the depth confound, sized
# --------------------------------------------------------------------------- #

def _empty_room(L=5.0, W=4.0, H=2.6, h=1.5, pitch_deg=0.0, out=259):
    """Ray-box intersection for a camera inside an empty rectangular room,
    through the real Aria fisheye ray field.

    No renderer, no data, no model: closed-form geometry, so the numbers below
    are a property of the room and the lens and of nothing else. Camera frame is
    +x right, +y down, +z along the optical axis; the camera sits at plan centre
    at height ``h`` and may be pitched nose-down.

    Returns ``(theta_deg, euclidean_range, planar_z)`` over the imaged cone.
    """
    cam = G.scaled_cam(aria_cam(1408, 1408), out)
    rays, cone = G.fisheye_rays(cam)
    d = rays[cone]
    theta = np.degrees(np.arccos(np.clip(d[:, 2], -1.0, 1.0)))
    p = math.radians(pitch_deg)
    rot = np.array([[1, 0, 0],
                    [0, math.cos(p), -math.sin(p)],
                    [0, math.sin(p), math.cos(p)]])
    w = d @ rot.T
    lo = np.array([-W / 2.0, -(H - h), -L / 2.0])
    hi = np.array([W / 2.0, h, L / 2.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.minimum(np.maximum(lo / w, hi / w), 1e9).min(axis=1)
    return theta, t, t * d[:, 2]


def _band(theta, v, lo, hi):
    m = (theta >= lo) & (theta < hi)
    return float(np.median(v[m])) if m.sum() > 50 else float("nan")


def test_an_empty_room_leaves_euclidean_range_almost_flat():
    """The intuition that says depth should NOT fall toward the rim is correct
    about *range*: an enclosing room is roughly equidistant, and range even
    rises in the middle of the field, where the diagonal to the facing wall is
    longer than the perpendicular."""
    theta, rng, _ = _empty_room()
    centre = _band(theta, rng, 0, 5)
    rim = _band(theta, rng, 50, 55)
    assert 0.85 < rim / centre < 1.05
    assert _band(theta, rng, 30, 35) > centre        # it rises before it falls


def test_but_planar_z_falls_by_a_factor_of_two_on_the_same_room():
    """ADT GT is planar z about the optical axis, and that is a different
    quantity. Out to ~35 deg the field is still on the facing wall, which is
    fronto-parallel, so z is *exactly* constant. Past that the field leaves the
    wall and catches floor, ceiling and side walls — surfaces the optical axis
    is parallel to, where z goes as the perpendicular distance over tan(theta) —
    and z collapses. Nothing about "the periphery contains clutter" is needed.
    """
    theta, _, z = _empty_room()
    centre = _band(theta, z, 0, 5)
    assert _band(theta, z, 30, 35) == pytest.approx(centre, rel=1e-3)
    assert _band(theta, z, 50, 55) / centre < 0.65


def test_the_room_alone_produces_the_penalty_the_run_reported():
    """The reason this matters, and the sharpest statement of the confound.

    A model with a CONSTANT absolute error and NO radial behaviour whatever,
    scored on this empty room, already produces `pen` 1.18 to 1.94 — the two
    ends being an error fixed in euclidean range and one fixed in planar z, the
    bracketing cases for a real model. Sweeping plausible apartment rooms (3x3
    to 10x8 m, camera 1.2-1.5 m, pitch 0-20 deg) widens that to **0.81 - 2.95**.

    The 200-frame run reported fisheye synthetic `pen` of 1.18, 1.79, 1.83 and
    1.97. **Every one of those sits inside the envelope the geometry alone
    spans.** So "error nearly doubles toward the rim" is not yet a field-position
    result; it is consistent with an empty room and a model with no radial
    behaviour at all. Only the measured per-bin depth, reported beside the
    curves, lets a reader weigh the two.

    AbsRel is invariant to the convention — numerator and denominator both carry
    the same 1/cos(theta), pinned below — so this is not an argument for scoring
    range instead: the confound is in how deep each bin's content is, which is
    why that depth is measured and reported rather than corrected away.
    """
    theta, rng, z = _empty_room()
    inner, outer = (theta < 5), (theta >= 50) & (theta < 55)
    pens = []
    for gt, pred in ((z, z + 0.10), (rng, rng + 0.10)):
        a, b = [float(np.mean(np.abs(pred[m] - gt[m]) / gt[m]))
                for m in (inner, outer)]
        pens.append(b / a)
    assert 1.10 < min(pens) < 1.30            # error fixed in range
    assert 1.85 < max(pens) < 2.05            # error fixed in planar z
    # the run's own vggt_1b and vggt_omega land inside this one room's envelope
    assert min(pens) < 1.79 < max(pens)
    assert min(pens) < 1.83 < max(pens)


def test_absrel_does_not_care_which_depth_convention_it_is_scored_in():
    """Numerator and denominator both scale by 1/cos(theta), so it cancels
    exactly. Worth pinning: it is the reason the fix for the confound is
    reporting the per-bin depth and not a change of convention."""
    theta, rng, z = _empty_room()
    cos = z / rng
    m = (theta >= 40) & (theta < 55)
    for gt_z, pred_z in ((z, z + 0.10), (z, z * 1.05)):
        as_z = np.mean(np.abs(pred_z[m] - gt_z[m]) / gt_z[m])
        as_r = np.mean(np.abs(pred_z[m] / cos[m] - gt_z[m] / cos[m])
                       / (gt_z[m] / cos[m]))
        assert as_z == pytest.approx(as_r, rel=1e-9)


# --------------------------------------------------------------------------- #
# The rectified arm's theta axis is cut short, and the tables do not say so
# --------------------------------------------------------------------------- #

def test_the_raw_fisheye_images_whole_rings_and_the_rectified_one_does_not():
    """The confound behind every rect-vs-fisheye `pen`: a theta bin on the raw
    lens is a full 360-deg annulus, but on the rectified pinhole it is whatever
    part of that annulus fell inside a square. Past the inscribed circle the
    rect arm is four corner wedges — fewer pixels AND a biased set of
    directions."""
    fish = G.full_ring_limit(518, "fisheye")
    rect = G.full_ring_limit(518, "rect")
    # the fisheye stays whole right out to its cone (the 0.2 deg is the grid)
    assert fish > 54.5
    # the rectified view stops being a ring at ~42 deg, well inside the cone
    assert 41.0 < rect < 43.0
    # ...and still reaches ~52 deg in the corners, which is what makes the
    # outer bins look populated rather than absent.
    assert G.reach_by_azimuth(518, "rect").max() > 51.0


def test_the_rect_outer_bins_are_corners_not_rings():
    cov = {c["bin_lo"]: c for c in G.ring_coverage(518, "rect", G.THETA_EDGES)}
    assert cov[0.0]["complete_frac"] == pytest.approx(1.0, abs=1e-3)
    assert cov[30.0]["complete_frac"] == pytest.approx(1.0, abs=1e-3)
    assert 0.4 < cov[40.0]["complete_frac"] < 0.7      # half a ring
    assert cov[50.0]["complete_frac"] < 0.05           # four slivers
    for c in G.ring_coverage(518, "fisheye", G.THETA_EDGES):
        assert c["complete_frac"] > 0.95


def test_the_same_bin_label_means_a_different_angle_in_the_two_views():
    """`pen` divides the outermost bin by the innermost, and the report compares
    that ratio across views. The inner bins agree to a fifth of a degree, but
    the rect 40-50 bin averages ~1.4 deg closer to the axis than the fisheye one
    and the 50-55 bin ~1.7 deg, because the missing azimuths are the ones that
    reach furthest. Rect is therefore flattered by the comparison."""
    f = {c["bin_lo"]: c["mean_theta"]
         for c in G.ring_coverage(518, "fisheye", G.THETA_EDGES)}
    r = {c["bin_lo"]: c["mean_theta"]
         for c in G.ring_coverage(518, "rect", G.THETA_EDGES)}
    for lo in (0.0, 10.0, 20.0, 30.0):
        assert abs(f[lo] - r[lo]) < 0.25                  # agree where whole
    assert f[40.0] - r[40.0] > 1.0                        # rect sits inward
    assert f[50.0] - r[50.0] > 1.0


def test_the_radius_axis_compared_across_views_is_not_merely_unfair_it_inverts():
    """Rect's own radius runs to sqrt(2) in its corners while the fisheye stops
    at 1.0, so drawing both raw on one axis says the fisheye sees less field.
    It sees more. Converted to where the ray lands on the sensor, rect's
    furthest corner is INSIDE the fisheye's last ring."""
    r = G.raw_sensor_radius(518, "rect", [0.0, 1.0, 1.411])
    assert r[0] == pytest.approx(0.0, abs=1e-6)
    # the rectified inscribed circle (42.2 deg) is only ~0.73 of the sensor
    assert 0.70 < r[1] < 0.76
    # and its extreme corner still lands inside the fisheye's reach
    fish_reach = G.coverage_span(518, "fisheye", "radius")[1]
    assert r[2] < fish_reach
    assert 0.90 < r[2] < 0.95
    # the fisheye's own radius already IS the sensor radius
    assert np.allclose(G.raw_sensor_radius(518, "fisheye", [0.3, 0.9]), [0.3, 0.9])
    # monotone, so it is a reparametrisation and cannot reorder any curve
    v = np.linspace(0, 1.4, 60)
    assert np.all(np.diff(G.raw_sensor_radius(518, "rect", v)) > 0)


def test_on_the_sensor_axis_the_rectified_view_is_the_one_that_runs_out_first():
    """The whole point of the conversion: on both physical axes the rectified
    arm must be the shorter one, because it is."""
    for coord in ("theta", "radius_raw"):
        _, fish = G.coverage_span(518, "fisheye", coord)
        _, rect = G.coverage_span(518, "rect", coord)
        assert rect < fish, coord
    # ...whereas in each view's own frame it looks the other way round
    assert (G.coverage_span(518, "rect", "radius")[1]
            > G.coverage_span(518, "fisheye", "radius")[1])


# --------------------------------------------------------------------------- #
# The joint table — the depth-controlled read of the same frozen prediction
# --------------------------------------------------------------------------- #

def _room_frame(out=259, **kw):
    """The empty room as a 2-D frame: (theta_deg, planar-z GT, cone), all HxW.

    Same closed-form geometry as ``_empty_room``, kept in image shape because
    ``bin_by`` bins maps rather than lists. This is the real confound: z falls
    from ~2.6 m on axis to ~1.4 m at the rim with nothing in the scene but four
    walls, a floor and a ceiling.
    """
    theta, _, z = _empty_room(out=out, **kw)
    cam = G.scaled_cam(aria_cam(1408, 1408), out)
    _, cone = G.fisheye_rays(cam)
    th2 = np.zeros(cone.shape, np.float32)
    z2 = np.zeros(cone.shape, np.float32)
    th2[cone] = theta
    z2[cone] = z
    return th2, z2, cone


def test_the_joint_table_dissolves_a_gradient_that_is_only_the_depth():
    """The reason this table exists, stated as a test it can fail.

    The model here has NO radial behaviour at all: its error is a pure function
    of the GT depth (a fixed 12 cm absolute offset, so AbsRel = 0.12/z). On the
    real room geometry that alone produces a rising theta curve, because the rim
    is nearer. The 1-D table must show that false gradient; every row of the
    joint table — same depth band, moving outward — must be flat, because within
    a band nothing about the model changes.
    """
    theta, z, cone = _room_frame()
    gt = z.astype(np.float32)
    pred = (gt + 0.12).astype(np.float32)            # depth-only error
    out = G.bin_by(pred, gt, cone, "none", {"theta": (theta, G.THETA_EDGES)},
                   joint_depth_edges={"theta": G.DEPTH_EDGES})
    live = [b for b in out["theta"] if b["n_bin"] > 0]
    span = live[-1]["AbsRel"] / live[0]["AbsRel"]
    assert span > 1.9                                # the false gradient, 1.94x

    j = G.pool_joint([out["joint"]["theta"]])
    a = np.asarray(j["AbsRel"], float)
    n = np.asarray(j["n"])
    gm = np.asarray(j["gt_mean"], float)
    rows = 0
    for k in range(a.shape[1]):                      # one depth band at a time
        idx = np.where(n[:, k] >= G.MIN_JOINT_CELL_PX)[0]
        if idx.size < 3:
            continue
        rows += 1
        # A metre-wide band does not abolish the confound, it shrinks it: the
        # rim cells still sit nearer WITHIN the band. 1.94x collapses to <1.2x.
        assert a[idx[-1], k] / a[idx[0], k] < 0.65 * span
        # ...and what survives is that residual and nothing else. The model's
        # error is a fixed 0.12 m, so AbsRel x the cell's own mean depth must
        # come back to 0.12 everywhere it is populated.
        prod = a[idx, k] * gm[idx, k]
        assert np.allclose(prod, 0.12, rtol=0.05), prod
    assert rows >= 2, "the room must populate at least two bands widely enough"


def test_the_joint_table_keeps_a_gradient_that_is_really_the_field():
    """The converse, so the test above cannot pass by flattening everything.

    Same room, but the error now scales with theta and not with depth. The rise
    must survive inside every depth band.
    """
    theta, z, cone = _room_frame()
    gt = z.astype(np.float32)
    pred = (gt * (1.0 + 0.004 * theta)).astype(np.float32)
    out = G.bin_by(pred, gt, cone, "none", {"theta": (theta, G.THETA_EDGES)},
                   joint_depth_edges={"theta": G.DEPTH_EDGES})
    j = G.pool_joint([out["joint"]["theta"]])
    a = np.asarray(j["AbsRel"], float)
    n = np.asarray(j["n"])
    checked = 0
    for k in range(a.shape[1]):
        idx = np.where(n[:, k] >= G.MIN_JOINT_CELL_PX)[0]
        if idx.size < 3:
            continue
        checked += 1
        assert np.all(np.diff(a[idx, k]) > 0)        # monotone within the band
        # AbsRel here is 0.004 * theta by construction, so inverting a cell's
        # value must land back inside that cell's own angular bin — a tighter
        # statement than "it rises", and one no flattening bug can satisfy.
        for i in idx:
            assert G.THETA_EDGES[i] <= a[i, k] / 0.004 < G.THETA_EDGES[i + 1]
    assert checked >= 2


def test_the_joint_rows_add_back_up_to_the_pooled_profile():
    """The joint table is the theta axis subdivided, so summing a column's
    cells must return the theta bin itself. Same frozen prediction, same
    estimator — if these disagree the two tables are measuring different fits.
    """
    theta, z, cone = _room_frame()
    gt = z.astype(np.float32)
    pred = (gt * 1.05 + 0.05).astype(np.float32)
    out = G.bin_by(pred, gt, cone, "none", {"theta": (theta, G.THETA_EDGES)},
                   profile_edges={"theta": G.THETA_EDGES},
                   joint_depth_edges={"theta": (0.0, 100.0)})
    prof = G.pool_profiles([out["profiles"]["theta"]])
    j = G.pool_joint([out["joint"]["theta"]])
    for i, (pn, pa) in enumerate(zip(prof["n"], prof["AbsRel"])):
        assert j["n"][i][0] == pn
        if pn:
            assert j["AbsRel"][i][0] == pytest.approx(pa, rel=1e-9)


def test_a_depth_outside_the_edges_is_dropped_not_folded_into_the_end_band():
    """Clipping would widen the top band by everything above it. 0-2 m of edges
    over a frame that is half at 9 m: the far half must vanish, not pile up."""
    gt = np.where(np.mgrid[0:40, 0:40][1] < 20, 1.5, 9.0).astype(np.float32)
    pred = (gt + 0.1).astype(np.float32)
    theta = np.zeros_like(gt)
    mask = np.ones_like(gt, bool)
    g = G.joint_grid(pred, gt, mask, theta, (0.0, 1.0), (0.0, 1.0, 2.0))
    assert g["n"] == [[0, 800]]                      # 800 near, 800 far dropped


def test_an_unpopulated_joint_cell_is_nan_not_zero():
    theta, z, cone = _room_frame()
    gt = z.astype(np.float32)
    out = G.bin_by(gt.copy(), gt, cone, "none", {"theta": (theta, G.THETA_EDGES)},
                   joint_depth_edges={"theta": G.DEPTH_EDGES})
    j = G.pool_joint([out["joint"]["theta"]])
    empty = [(i, k) for i, row in enumerate(j["n"])
             for k, n in enumerate(row) if n == 0]
    assert empty, "the room must leave some corner of the grid unpopulated"
    for i, k in empty:
        assert math.isnan(j["AbsRel"][i][k])


def test_the_joint_table_is_absent_unless_asked_for():
    pred, gt, mask, theta = _profile_inputs()
    out = G.bin_by(pred, gt, mask, "none", {"theta": (theta, G.THETA_EDGES)})
    assert "joint" not in out


def test_pooling_refuses_to_add_joint_grids_of_different_partitions():
    a = {"coord_edges": [0.0, 1.0], "depth_edges": [0.0, 2.0], "n": [[4]],
         "sum_absrel": [[1.0]], "sum_delta1": [[4.0]], "sum_gt": [[8.0]]}
    b = dict(a, depth_edges=[0.0, 3.0])
    with pytest.raises(ValueError):
        G.pool_joint([a, b])
