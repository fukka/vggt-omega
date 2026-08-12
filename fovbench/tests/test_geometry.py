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
