# Copyright (c) 2026.
"""The FOV protocol, and the two ways it can report a finding that is not there.

Most of this file is about those two. Both are *silent*: neither produces an
error, an empty table or an implausible number — each produces a clean curve
with the wrong shape, which is the only kind of bug that survives review.

    per-bin alignment   erases a real field effect and reports a flat curve
    the depth confound  invents a field effect where the model has none

They fail in opposite directions, so a harness that only guarded one of them
would still be wrong half the time. The third block is the window arm's own
version: a tilted pinhole is not co-axial, and skipping the depth conversion
bends the window curve in the same shape as the effect it is measuring.

CPU-only, no data, no weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from slambench import fov as F
from slambench import data as D
from slambench.tests.test_baselines import cam896, slanted_plane

EDGES = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0]


def synth(n=20000, seed=0, bias_per_deg=0.0, depth_slope=0.0, strata=5):
    """A frame of points with a known field effect and a known depth gradient.

    ``bias_per_deg``   multiplicative error growing with eccentricity — a real
                       field effect, the thing the benchmark exists to find.
    ``depth_slope``    how much nearer the rim is than the centre — the
                       confound, present in the ground truth and in no model.

    Returns ``(pred, gt, theta, depth_edges)``.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 55.0, n)
    # Depth falls with eccentricity when depth_slope > 0, as it does on the
    # real release (2.57 m on axis to 0.71 m at 57 deg).
    centre = 3.0 * np.exp(-depth_slope * theta / 55.0)
    gt = centre * np.exp(rng.normal(0.0, 0.45, n))
    pred = gt * (1.0 + bias_per_deg * theta) / 3.0     # up-to-scale, as a model is
    q = np.percentile(gt, np.linspace(0, 100, strata + 1))
    q[0], q[-1] = 0.0, float("inf")
    return pred, gt, theta, [float(x) for x in q]


def table_of(pred, gt, theta, depth_edges, edges=EDGES, align="scale_shift"):
    t = F.Table(edges, depth_edges, "theta")
    t.add(F.frame_cells(pred, gt, theta, align, edges, depth_edges,
                        min_points=64))
    return t


# --------------------------------------------------------------------------- #
# 1. The protocol's load-bearing rule
# --------------------------------------------------------------------------- #

def test_per_bin_alignment_would_erase_the_effect():
    """One frozen affine, never one per bin — measured, not asserted.

    An up-to-scale model whose depth bends with eccentricity has no single scale
    that fits every radius. Fitting each bin its own hands it exactly the freedom
    it is being tested for, and the curve comes back flat and healthy. This is
    the local copy of ``fovbench``'s measurement of the same thing, and the
    numbers are the reason the rule is not negotiable.
    """
    pred, gt, theta, de = synth(bias_per_deg=0.006)      # +33 % at 55 deg

    shared = [table_of(pred, gt, theta, de).pos_row(t)["AbsRel"] for t in range(6)]

    # ...and the same data with the fit repeated inside every bin.
    per_bin = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (theta >= lo) & (theta < hi)
        per_bin.append(table_of(pred[m], gt[m], theta[m], de,
                                edges=[lo, hi]).pos_row(0)["AbsRel"])

    assert max(shared) > 0.10, f"the injected effect should be visible: {shared}"
    assert max(shared) / min(shared) > 3.0, f"shared fit should spread: {shared}"
    assert max(per_bin) < 0.03, f"per-bin fit should have erased it: {per_bin}"


def test_a_global_affine_makes_a_radial_bias_look_U_shaped_not_monotone():
    """How to read the radial table, pinned so it cannot quietly stop being true.

    Under one frozen affine a *monotone* radial bias does **not** come back as a
    monotone curve. Least squares puts the fitted scale in the middle of the
    bias range, so the on-axis points are over-corrected and the rim points
    under-corrected and the residual rises at **both** ends:

        0.105  0.056  0.019  0.044  0.094  0.131      (bias +0.6 %/deg)

    That is correct behaviour and it is a trap: the centre-to-rim ratio is 1.24
    on a bias that is 1.33 end to end, so an endpoint ratio badly understates
    what is there. The spread across bins is the statistic that does not, which
    is what the report prints for this protocol. The window protocol is not
    affected — each window gets its own fit — and there the endpoint ratio means
    what it looks like.
    """
    pred, gt, theta, de = synth(bias_per_deg=0.006)
    c = [table_of(pred, gt, theta, de).pos_row(t)["AbsRel"] for t in range(6)]
    assert c.index(min(c)) not in (0, len(c) - 1), (
        f"the minimum should sit in the middle of the field, not at an end: {c}")
    assert max(c) / min(c) > 3.0 > (c[-1] / c[0]), (
        f"spread should be much larger than the endpoint ratio: {c}")


def test_the_binning_never_reaches_back_into_the_alignment():
    """Changing the bin edges must not move a point's error.

    The guarantee that makes the tables and any later re-binning two readings of
    one measurement rather than two measurements.
    """
    pred, gt, theta, de = synth(bias_per_deg=0.004)
    coarse = table_of(pred, gt, theta, de, edges=[0.0, 27.5, 55.0])
    fine = table_of(pred, gt, theta, de, edges=EDGES)
    # The whole-frame totals must agree exactly: same points, same fit.
    a = sum(c["sum_absrel"] for c in coarse.cells.values())
    b = sum(c["sum_absrel"] for c in fine.cells.values())
    assert abs(a - b) < 1e-9 * max(1.0, abs(a)), (a, b)


# --------------------------------------------------------------------------- #
# 2. The confound, and the control that removes it
# --------------------------------------------------------------------------- #

def test_a_depth_gradient_alone_invents_a_field_effect():
    """The failure this module's two-way table exists to prevent.

    The model here has **no** field dependence at all — its error is a constant
    relative bias at every eccentricity. All that changes across the field is
    how far away the ground truth is, exactly as it does on the real release.
    The pooled curve must nonetheless slope, or this data does not reproduce the
    confound and the next test proves nothing.
    """
    pred, gt, theta, de = synth(n=200000, bias_per_deg=0.0, depth_slope=1.4)
    # Give the prediction an error that grows with distance and not with field:
    # a fixed error in metres, which is what a shift-free scale fit leaves on a
    # scene with depth structure.
    rng = np.random.default_rng(1)
    pred = pred + rng.normal(0.0, 0.25, pred.size) / 3.0
    pooled = [table_of(pred, gt, theta, de).pos_row(t)["AbsRel"] for t in range(6)]
    live = [v for v in pooled if np.isfinite(v)]
    assert len(live) >= 4
    assert live[-1] / live[0] > 1.5, (
        f"the depth confound did not show up, so the control below is not "
        f"being tested against anything: {pooled}")


def test_holding_distance_fixed_removes_what_distance_alone_put_there():
    """The control, on the same data that just faked a result.

    Standardising over distance strata shared by every bin must flatten the
    curve the previous test found, because there was never a field effect in it.
    """
    pred, gt, theta, de = synth(n=200000, bias_per_deg=0.0, depth_slope=1.4)
    rng = np.random.default_rng(1)
    pred = pred + rng.normal(0.0, 0.25, pred.size) / 3.0
    tab = table_of(pred, gt, theta, de)
    js = tab.to_json()
    pooled = [r["AbsRel"] for r in js["pos_rows"]]
    ctl = F.controlled(js["pos_rows"], js["cells"], "AbsRel")
    live_p = [v for v in pooled if np.isfinite(v)]
    live_c = [v for v in ctl["curve"] if np.isfinite(v)]
    assert len(live_c) >= 3, ctl
    rise_pooled = live_p[-1] / live_p[0]
    rise_ctl = live_c[-1] / live_c[0]
    assert rise_ctl < rise_pooled, (rise_pooled, rise_ctl)
    assert abs(rise_ctl - 1.0) < 0.35, (
        f"distance was held fixed and the curve still slopes by {rise_ctl:.2f}x, "
        f"so the standardisation is not removing what it claims to")


def test_the_control_keeps_a_real_field_effect():
    """The other half: it must not flatten an effect that IS about the field.

    A control that removes everything is not a control. Same standardisation,
    on a model whose error genuinely grows with eccentricity at every distance.
    """
    pred, gt, theta, de = synth(n=200000, bias_per_deg=0.006, depth_slope=1.4)
    tab = table_of(pred, gt, theta, de)
    js = tab.to_json()
    ctl = F.controlled(js["pos_rows"], js["cells"], "AbsRel")
    live = [v for v in ctl["curve"] if np.isfinite(v)]
    assert len(live) >= 3, ctl
    # Spread, not the endpoint ratio: under one frozen affine a monotone radial
    # bias comes back U-shaped, so the two ends are both high and their ratio is
    # near 1 however strong the effect is. See
    # ``test_a_global_affine_makes_a_radial_bias_look_U_shaped_not_monotone``.
    assert max(live) / min(live) > 2.0, (
        f"a real field effect survived standardisation as {live}; the control "
        f"is removing the signal along with the confound")


def test_the_control_reports_how_much_of_each_bin_it_speaks_for():
    """A standardised number over a tenth of a bin is a different claim.

    Centre and rim barely overlap in distance on this data, so the shared strata
    can cover very little of a bin. That has to be visible in the output or the
    control quietly narrows the question it is answering.
    """
    pred, gt, theta, de = synth(n=200000, bias_per_deg=0.0, depth_slope=2.5)
    js = table_of(pred, gt, theta, de).to_json()
    ctl = F.controlled(js["pos_rows"], js["cells"], "AbsRel")
    share = [s for s in ctl["share"] if np.isfinite(s)]
    assert share and all(0.0 <= s <= 1.0 + 1e-9 for s in share), share
    assert min(share) < 0.9, (
        f"a 2.5x depth gradient should leave some bin poorly covered by the "
        f"shared strata; got {share}")


def test_a_thin_cell_is_empty_rather_than_a_number():
    """Below the floor a cell reports nothing. A mean over eleven points is not
    a measurement, and one printed beside a mean over eleven thousand invites
    being read as one."""
    pred, gt, theta, de = synth(n=600)
    tab = table_of(pred, gt, theta, de)
    thin = [tab.cell(t, s) for t in range(6) for s in range(5)
            if 0 < tab.cells.get((t, s), {}).get("n", 0) < F.MIN_CELL_POINTS]
    assert thin, "this split should have produced at least one thin cell"
    assert all(not np.isfinite(c["AbsRel"]) for c in thin)
    assert all(c["n"] > 0 for c in thin), "the count must survive; only the score goes"


# --------------------------------------------------------------------------- #
# 3. The window arm: a tilted pinhole is not co-axial
# --------------------------------------------------------------------------- #

def test_a_window_at_zero_tilt_is_the_co_axial_pinhole():
    """The sweep's first point must be the arm the rest of the package already
    has, or the window curve has no anchor that anything else can check."""
    w = F.Window(256, 110.0, tilt_deg=0.0, azimuth_deg=0.0)
    assert np.allclose(w.R, np.eye(3), atol=1e-12)
    from slambench.baselines import Pinhole
    pin = Pinhole(256, 110.0)
    d = cam896().unproject(np.array([448.0, 500.0]), np.array([448.0, 402.0]))
    assert np.allclose(np.stack(w.project(d)), np.stack(pin.project(d)))
    assert np.allclose(w.z_to_camera(d), 1.0, atol=1e-12), (
        "at zero tilt the depth conversion must be the identity")


@pytest.mark.parametrize("tilt", [10.0, 25.0, 40.0])
def test_the_window_depth_conversion_against_a_closed_form(tilt):
    """A tilted window predicts planar z about ITS axis, and the ground truth is
    planar z about the camera's. The factor between them is derived in
    ``fov.py``; here it is checked against a scene where both are known.

    Forgetting it is worth up to 1.31x at 40 deg, and — the part that matters —
    it varies across the window, so the per-frame affine cannot stand in for it.
    A missing conversion would not scale the window arm, it would bend it, in
    the same shape as the effect the arm is measuring.
    """
    w = F.Window(96, 40.0, tilt_deg=tilt, azimuth_deg=30.0)
    dirs = w.rays().reshape(-1, 3)
    # A slanted plane: on a fronto-parallel one every mapping error still
    # returns the right depth and the test passes with the geometry removed.
    z_cam = slanted_plane(dirs)                       # planar z about the camera
    along = dirs @ w.axis
    z_win = z_cam / (dirs[:, 2] / along)              # ...about the window
    assert np.allclose(z_win * w.z_to_camera(dirs), z_cam, rtol=1e-12)
    # And it is a real correction, not a rounding one.
    assert abs(w.z_to_camera(w.axis[None, :])[0] - np.cos(np.radians(tilt))) < 1e-12
    assert w.z_to_camera(dirs).ptp() > 0.05, (
        "the conversion must VARY across the window — a constant one would be "
        "absorbed by the alignment and would not matter")


def test_a_window_reads_only_pixels_the_lens_actually_filled():
    """Bilinear over the edge of the imaged cone mixes real depth with the
    model's answer to black padding, and nothing downstream can tell. The
    co-axial arm never needs this check; a tilted window is where it starts to.
    """
    cam = cam896()
    w = F.Window(128, 40.0, tilt_deg=55.0, azimuth_deg=0.0)   # hangs off the lens
    view = F.WindowView(cam, w)
    assert not view.in_cone.all(), "this window should overhang the imaged cone"
    # Every point of a dense grid; those the sampler answers for must have a
    # fully-backed stencil.
    j = np.linspace(30.0, 865.0, 60)
    uu, vv = np.meshgrid(j, j)
    pts = D.FramePoints(u=uu.ravel().astype(np.float32),
                        v=vv.ravel().astype(np.float32),
                        d=np.ones(uu.size, np.float32),
                        inv_dist_std=np.zeros(uu.size, np.float32),
                        dist_std=np.zeros(uu.size, np.float32))
    got = view.sample(np.ones((128, 128), np.float32), pts)
    live = np.isfinite(got)
    assert live.any(), "the window should still see something"
    d = cam.unproject(pts.u[live].astype(np.float64), pts.v[live].astype(np.float64))
    u, v = w.project(d)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    x1, y1 = np.minimum(x0 + 1, 127), np.minimum(y0 + 1, 127)
    assert (view.in_cone[y0, x0] & view.in_cone[y0, x1]
            & view.in_cone[y1, x0] & view.in_cone[y1, x1]).all()


def test_a_window_that_is_mostly_black_is_refused_by_its_own_number():
    """The dead-pixel fraction is measured per window and is what the driver
    gates on. Held here so the gate cannot be relaxed by accident: an earlier
    sweep in this repository varied window width and aim together, so the dead
    fraction moved with the variable under test and the result was partly a
    measurement of black."""
    cam = cam896()
    wide = F.WindowView(cam, F.Window(96, 40.0, tilt_deg=75.0))
    narrow = F.WindowView(cam, F.Window(96, 40.0, tilt_deg=0.0))
    assert narrow.in_cone_frac == 1.0
    assert wide.in_cone_frac < F.MIN_IN_CONE_FRAC, wide.in_cone_frac


def test_the_four_azimuths_see_the_same_field_on_a_symmetric_lens():
    """At one tilt, the four azimuths differ only by a rotation about the optic
    axis, so their windows must cover the same *amount* of the lens. Where they
    do not, it is the thin-prism terms talking and not the sweep — which is what
    makes the four a control on each other rather than four experiments."""
    cam = cam896()
    fracs = [F.WindowView(cam, F.Window(96, 40.0, 40.0, az)).in_cone_frac
             for az in (0.0, 90.0, 180.0, 270.0)]
    assert max(fracs) - min(fracs) < 0.05, fracs


# --------------------------------------------------------------------------- #
# 4. Plumbing that would be silent if wrong
# --------------------------------------------------------------------------- #

def test_frames_pool_by_addition_not_by_averaging_bin_means():
    """Two frames pooled must equal the one frame holding both frames' points.

    A bin is 5 % of one frame and 24 % of another, so averaging per-frame bin
    means would let a frame that put nine points in a bin count as much as one
    that put nine hundred there.
    """
    pred, gt, theta, de = synth(n=8000, bias_per_deg=0.004)
    half = pred.size // 2
    both = table_of(pred, gt, theta, de)
    split = F.Table(EDGES, de, "theta")
    for sl in (slice(0, half), slice(half, None)):
        split.add(F.frame_cells(pred[sl], gt[sl], theta[sl], "none", EDGES, de,
                                min_points=64))
    ref = F.Table(EDGES, de, "theta")
    ref.add(F.frame_cells(pred, gt, theta, "none", EDGES, de, min_points=64))
    for t in range(6):
        assert abs(split.pos_row(t)["AbsRel"] - ref.pos_row(t)["AbsRel"]) < 1e-9
    assert split.n_frames == 2 and ref.n_frames == 1
    assert both.n_frames == 1


def test_a_frame_too_thin_to_fit_is_not_measured_rather_than_scored_badly():
    pred, gt, theta, de = synth(n=100)
    assert F.frame_cells(pred, gt, theta, "scale_shift", EDGES, de,
                         min_points=256) is None


def test_the_serialised_standardisation_matches_the_accumulator():
    """``fov.controlled`` reads results.json and ``Table.standardised`` reads the
    accumulator. Two implementations of one control is one place for it to
    drift, so they are pinned to each other."""
    pred, gt, theta, de = synth(bias_per_deg=0.005, depth_slope=1.0)
    tab = table_of(pred, gt, theta, de)
    js = tab.to_json()
    direct, strata = tab.standardised("AbsRel")
    via = F.controlled(js["pos_rows"], js["cells"], "AbsRel")
    assert strata == via["strata"]
    for a, b in zip(direct, via["curve"]):
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12


def test_depth_strata_are_equal_population_so_the_weights_are_uniform():
    """The standardisation is a plain mean over strata only because the strata
    are equal-population quantiles. If that stopped being true the mean would
    quietly become the wrong weighting."""
    rng = np.random.default_rng(3)
    g = np.exp(rng.normal(0.0, 1.0, 50000))
    edges = F.depth_edges_from([g], 5)
    counts = [int(((g >= lo) & (g < hi)).sum())
              for lo, hi in zip(edges[:-1], edges[1:])]
    assert max(counts) - min(counts) < 0.02 * g.size, counts
    assert edges[0] == 0.0 and edges[-1] == float("inf")
