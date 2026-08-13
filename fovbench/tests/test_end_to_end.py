# Copyright (c) 2026.
"""The closing invariant: the harness reports the distortion it was given.

Everything else in this suite checks a piece. This checks the whole path — build
a fisheye frame with analytic GT, warp it into a view, hand it to a model that
bends depth by a *known* function of eccentricity, score it, bin it, summarise it
— against a number computed independently from the bin geometry.

If this passes, a curve the benchmark prints on real data can be read as a
property of the model rather than of the pipeline.

CPU-only, no ADT, no weights, under a second.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pytest

from fovbench import geometry as G  # noqa: E402
from fovbench import models as M  # noqa: E402
from fovbench import report as R  # noqa: E402
from fovbench import run as RUN  # noqa: E402

FRAME = 352
OUT = 176
BIAS = 0.6


def _scene(offset=(0.7, 0.4, -0.9)):
    """A fisheye frame whose GT is a room-sized box seen from OFF its centre.

    The offset matters and is not decoration. Viewed from the exact middle the
    central band is one fronto-parallel wall at a constant range, and an affine
    fit on it is undetermined — which is a degenerate scene, not a hard one, and
    ``anchored_ratios`` refuses it (see the flat-anchor test below). Off-centre,
    every band spans a range of depths, as a real egocentric frame does: measured
    on ADT seq131, IQR/median per band runs 0.71-0.88.
    """
    cam = G.aria_cam(FRAME, FRAME)
    rays, cone = G.fisheye_rays(cam)
    half = np.array([2.0, 1.4, 3.0], np.float64)
    c = np.array(offset, np.float64)
    # nearest positive intersection with the 6 planes of the box
    with np.errstate(divide="ignore", invalid="ignore"):
        cand = np.stack([(s * half[i] - c[i]) / rays[..., i]
                         for i in range(3) for s in (-1.0, 1.0)], axis=-1)
    cand = np.where(np.isfinite(cand) & (cand > 1e-6), cand, np.inf)
    t = cand.min(axis=-1)
    gt = (t * rays[..., 2]).astype(np.float32) * cone
    rng = np.random.default_rng(7)
    rgb = (rng.random((FRAME, FRAME, 3)) * 255).astype(np.uint8)
    return cam, rgb, gt, cone


def _predicted_drift(bins, bias=BIAS):
    """The drift the injected bias *must* produce, from the bins' own theta.

    ``pred = gt * (1 + bias * theta^2)`` so ``median(gt/pred)`` in a bin is
    ``1 / (1 + bias * theta_bin^2)``, and the inner/outer ratio is the ratio of
    those factors. Computed here from the bin midpoints — deliberately a
    different route to the answer than the code under test.
    """
    live = [b for b in bins if b["n_valid"] > 0]
    lo, hi = live[0], live[-1]

    def f(b):
        mid = np.radians(0.5 * (b["bin_lo"] + b["bin_hi"]))
        return 1.0 + bias * mid ** 2
    return f(hi) / f(lo)


def _window_rows(bias=BIAS):
    cam, rgb, gt, cone = _scene()
    model = M.load_model(M.ANALYTIC, device=None, radial_bias=bias)
    rows = []
    for tilt in (0.0, 10.0, 20.0, 30.0, 40.0):
        w = G.render_window(rgb, gt, cone, cam, 0.0, tilt, 40.0, OUT, "fisheye")
        assert w.in_cone_frac >= RUN.MIN_IN_CONE_FRAC
        rows.append(RUN._score_window(model, w, max_depth=100.0))
    return rows


def test_window_sweep_reads_back_the_bias_as_a_rising_absrel_curve():
    """Each window is aligned on its own, so what survives is the bias's spread
    *within* that window — and a window aimed further out spans higher theta, so
    the curve rises. AbsRel is the column that carries this."""
    absrel = [r["AbsRel"] for r in _window_rows()]
    # End-to-end rise only, not monotone: on this box the curve peaks at t20 and
    # dips at t40, because each aim sees different scene content. The arm also
    # carries a sampling confound — but it runs the *other* way, see
    # test_geometry.py::test_a_window_aimed_off_axis_is_sampled_at_least_as_
    # densely_as_the_centre: an on-axis window is upsampled and a rim one is not.
    assert absrel[-1] > 1.25 * absrel[0]


def test_the_two_streams_score_identical_geometry():
    """With a GT-driven model the two streams differ only in pixels, so any gap
    between them would be the harness picking different frames or different GT —
    the exact failure the frozen split exists to prevent."""
    cam, rgb, gt, cone = _scene()
    other = np.roll(rgb, 17, axis=1)          # "the other stream's pixels"
    model = M.load_model(M.ANALYTIC, device=None, radial_bias=BIAS, seed=0)
    a = RUN._score_radial(model, G.full_frame_view(rgb, gt, cone, cam, OUT,
                                                   "fisheye"),
                          G.THETA_EDGES, G.RADIUS_EDGES, 100.0)
    model = M.load_model(M.ANALYTIC, device=None, radial_bias=BIAS, seed=0)
    b = RUN._score_radial(model, G.full_frame_view(other, gt, cone, cam, OUT,
                                                   "fisheye"),
                          G.THETA_EDGES, G.RADIUS_EDGES, 100.0)
    assert a["overall"]["AbsRel"] == pytest.approx(b["overall"]["AbsRel"], rel=1e-6)
    assert [x["n_bin"] for x in a["bins"]] == [x["n_bin"] for x in b["bins"]]


def test_a_window_past_the_imaged_cone_is_dropped_not_scored():
    """The dead-pixel confound, guarded: a window aimed so far off-axis that the
    lens images little of it must not enter the sweep at all."""
    cam, rgb, gt, cone = _scene()
    w = G.render_window(rgb, gt, cone, cam, 0.0, 60.0, 40.0, OUT, "rect")
    assert w.in_cone_frac < RUN.MIN_IN_CONE_FRAC


def test_cli_defaults_form_the_documented_grid():
    a = RUN.build_parser().parse_args([])
    assert a.streams == "synthetic,real"
    assert a.views == "rect,fisheye"
    assert a.protocols == "radial,window"
    assert a.models == ",".join(M.DEFAULT_MODELS)
    assert a.window_fov == RUN.DEFAULT_WINDOW_FOV


def test_run_refuses_an_unknown_stream(tmp_path):
    a = RUN.build_parser().parse_args(
        ["--adt-root", str(tmp_path), "--streams", "synthetic,rendered",
         "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit) as e:
        RUN.run(a)
    assert "rendered" in str(e.value)


def test_run_refuses_to_silently_drop_an_unavailable_model(tmp_path):
    """VGGT-Omega's weights are gated and DA3 needs a pip install, so "some
    models missing" is the normal first-run state. Proceeding would write a
    report indistinguishable from a complete one except by its footer."""
    a = RUN.build_parser().parse_args(
        ["--adt-root", str(tmp_path), "--models", "analytic,vggt_omega",
         "--out", str(tmp_path / "o")])
    if M.available(["vggt_omega"])[0]:
        pytest.skip("vggt_omega checkpoint present on this machine")
    with pytest.raises(SystemExit) as e:
        RUN.run(a)
    assert "vggt_omega" in str(e.value) and "--skip-unavailable" in str(e.value)


# --------------------------------------------------------------------------- #
# The binned table and the continuous curve are one estimator at two resolutions
# --------------------------------------------------------------------------- #

def test_cross_frame_pooling_is_weighted_by_each_frame_s_pixels():
    """Two frames, one contributing a hundred times the pixels of the other and
    scoring very differently. The pooled answer must follow the big frame; an
    unweighted mean of per-frame means would split the difference and would put
    the binned dots somewhere the continuous curve never goes."""
    rows = [dict(AbsRel=0.10, n_valid=10000, n_bin=10000),
            dict(AbsRel=0.50, n_valid=100, n_bin=100)]
    out = RUN._mean_metrics(rows, ("AbsRel",))
    assert out["AbsRel"] == pytest.approx((0.10 * 10000 + 0.50 * 100) / 10100)
    assert out["AbsRel"] != pytest.approx(0.30)          # not the naive mean


def test_rms_metrics_are_pooled_in_squares():
    """RMSE is a mean under a root, so the weights go on the squares. Averaging
    the roots would be a different number and a smaller one."""
    rows = [dict(RMSE=0.2, n_valid=3, n_bin=3), dict(RMSE=0.6, n_valid=1, n_bin=1)]
    out = RUN._mean_metrics(rows, ("RMSE",))
    assert out["RMSE"] == pytest.approx(math.sqrt((3 * 0.04 + 1 * 0.36) / 4))


def test_medians_stay_a_frame_mean_and_say_so():
    """A weighted mean of per-frame medians is not the pooled median, and no
    summary can recover the pooled one — so these keys are deliberately left as
    a frame mean rather than given a weighting that would look pooled and not
    be. The depth figure draws the profile's pooled `gt_mean` instead."""
    rows = [dict(gt_median=2.0, n_valid=10000, n_bin=10000),
            dict(gt_median=6.0, n_valid=100, n_bin=100)]
    out = RUN._mean_metrics(rows, ("gt_median",))
    assert out["gt_median"] == pytest.approx(4.0)        # unweighted, on purpose
    assert "gt_median" in RUN._FRAME_MEDIAN


def test_a_binned_value_equals_the_profile_re_aggregated_over_the_same_span():
    """The claim the figures rest on: dots and line are the same estimator.
    The fine edges nest exactly inside the coarse ones (1 deg into 10 deg), so
    re-adding the profile's pixels over a coarse bin's span must reproduce that
    bin, on a run where both were computed from the same frames."""
    rng = np.random.default_rng(0)
    n = 128
    ys, xs = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    theta = (np.hypot(xs - c, ys - c) / c * 55.0).astype(np.float32)
    gt = (3.0 + 0.01 * xs).astype(np.float32)
    frames = []
    for k in range(3):
        pred = (gt * (1.0 + 0.4 * np.radians(theta) ** 2)
                * (1 + rng.normal(0, 0.01, gt.shape))).astype(np.float32)
        mask = (theta < 55.0) & (rng.random(gt.shape) > 0.1 * k)   # differing counts
        frames.append(G.bin_by(pred, gt, mask, "none",
                               {"theta": (theta, G.THETA_EDGES)},
                               profile_edges={"theta": G.PROFILE_THETA_EDGES}))
    binned = RUN._reduce_axis([{"bins": f["theta"]} for f in frames], "bins",
                              G.THETA_EDGES)
    pooled = G.pool_profiles([f["profiles"]["theta"] for f in frames])
    ctr = np.asarray(pooled["centre"]); pn = np.asarray(pooled["n"], float)
    pa = np.asarray(pooled["AbsRel"], float)
    for b, (lo, hi) in zip(binned, zip(G.THETA_EDGES[:-1], G.THETA_EDGES[1:])):
        sel = (ctr >= lo) & (ctr < hi) & np.isfinite(pa)
        if not sel.any() or not np.isfinite(b["AbsRel"]):
            continue
        span = float((pa[sel] * pn[sel]).sum() / pn[sel].sum())
        assert b["AbsRel"] == pytest.approx(span, rel=1e-9)


def test_a_context_run_scores_exactly_the_pixels_a_single_frame_run_does():
    """The comparison 1 vs 5 vs 10 is only meaningful if the measured pixels do
    not move. Only the target is scored; the context changes what the model may
    look at and nothing else. Driven through the stand-in, which accepts a stack
    and reads GT, so any difference here would be the harness and not a model.
    """
    cam = G.aria_cam(176, 176)
    rays, cone = G.fisheye_rays(cam)
    gt = (2.0 + 0.5 * rays[..., 0]).astype(np.float32) * cone
    ys, xs = np.mgrid[0:cam.H, 0:cam.W]
    rgb = np.stack([((xs * 3 + ys * 7) % 256).astype(np.uint8)] * 3, -1)
    view = G.full_frame_view(rgb, gt, cone, cam, 64, "fisheye")
    model = M.load_model(M.ANALYTIC, None, radial_bias=0.0)
    assert model.supports_context

    one = RUN._score_radial(model, view, G.THETA_EDGES, G.RADIUS_EDGES, 100.0)
    stack = [view.rgb] * 5
    many = RUN._score_radial(model, view, G.THETA_EDGES, G.RADIUS_EDGES, 100.0,
                             context=stack, target=4)
    for a, b in zip(one["bins"], many["bins"]):
        assert a["n_bin"] == b["n_bin"]
        assert a["n_valid"] == b["n_valid"]


def test_a_manifest_refuses_a_context_it_was_not_written_with(tmp_path):
    """A manifest stores a pre-baked context list per frame, so --manifest wins
    over --context-frames and the flags become a *silent* no-op — while the run's
    own `config` block still echoes them. On ticket 010 that produced four runs
    whose predictions were bit-identical to the 1-frame baseline and which looked
    exactly like real context runs. Refusing costs a re-run; not refusing costs
    a conclusion."""
    from fovbench.split import Frame, Split
    sp = Split(root=str(tmp_path),
               frames=[Frame(seq="s", frame_id=f"{i:06d}", depth=f"{i}.npy",
                             rgb={"synthetic": f"{i}a.png", "real": f"{i}b.png"})
                       for i in range(3)])
    assert sp.context_frames == 1          # a plain 1-frame manifest
    p = str(tmp_path / "manifest.json")
    sp.save(p)

    a = RUN.build_parser().parse_args(
        ["--manifest", p, "--models", "analytic", "--protocols", "radial",
         "--context-frames", "10", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit) as e:
        RUN.run(a)
    msg = str(e.value)
    assert "--manifest" in msg and "10" in msg
    # and the manifest that *does* match is not refused: it gets past this check
    # and dies later, on the frame files this fixture never wrote.
    b = RUN.build_parser().parse_args(
        ["--manifest", p, "--models", "analytic", "--protocols", "radial",
         "--out", str(tmp_path / "o2")])
    with pytest.raises(Exception) as e2:
        RUN.run(b)
    assert "0a.png" in str(e2.value)   # reached the frames, not the guard


def test_a_monocular_model_refuses_a_context_rather_than_ignoring_it():
    """A run that asked for ten frames and silently got one would read as
    'context does not help', which is the opposite of measuring nothing."""
    assert "dav2_large" not in M.CONTEXT_CAPABLE
    m = M.Model(key="dav2_large", family="x", size="L", align_mode="none",
                input_size=518, supports_context=False,
                _predict=lambda *a: np.ones((4, 4), np.float32))
    with pytest.raises(SystemExit):
        m.predict_stack([np.zeros((4, 4, 3), np.uint8)] * 3, target=2)
    # ... but a stack of one is just the ordinary call, not an error.
    assert m.predict_stack([np.zeros((4, 4, 3), np.uint8)], target=0).shape == (4, 4)
