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


def _scene():
    """A fisheye frame whose GT is a room-sized box seen from the middle.

    Depth varies with direction, so the alignment has something to fit and the
    metrics are not degenerate — unlike a constant plane, where every model
    scores perfectly after a shift.
    """
    cam = G.aria_cam(FRAME, FRAME)
    rays, cone = G.fisheye_rays(cam)
    # distance to the box wall along each ray, then planar z about the axis
    half = np.array([2.0, 1.4, 3.0], np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.min(np.abs(half) / np.maximum(np.abs(rays), 1e-6), axis=-1)
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
        mid = np.radians(0.5 * (b["theta_lo"] + b["theta_hi"]))
        return 1.0 + bias * mid ** 2
    return f(hi) / f(lo)


@pytest.mark.parametrize("kind", ["rect", "fisheye"])
def test_radial_profile_reads_back_the_injected_radial_bias(kind):
    cam, rgb, gt, cone = _scene()
    view = G.full_frame_view(rgb, gt, cone, cam, OUT, kind)
    model = M.load_model(M.ANALYTIC, device=None, radial_bias=BIAS)

    prof = RUN._score_radial(model, view, G.THETA_EDGES, max_depth=100.0)
    run = dict(protocol="radial", bins=[
        dict(b, n_frames=1, n_px_mean=float(b["n_bin"])) for b in prof["bins"]])
    got = R.summarise(run)["drift"]
    assert got == pytest.approx(_predicted_drift(prof["bins"]), rel=0.10)
    assert got > 1.3                       # and it is a large, visible effect


@pytest.mark.parametrize("kind", ["rect", "fisheye"])
def test_an_unbiased_model_reads_back_as_no_distortion(kind):
    """The other half of the claim: with nothing injected, the harness must not
    manufacture a trend out of the view construction, the GT warp or the bins."""
    cam, rgb, gt, cone = _scene()
    view = G.full_frame_view(rgb, gt, cone, cam, OUT, kind)
    model = M.load_model(M.ANALYTIC, device=None, radial_bias=0.0)

    prof = RUN._score_radial(model, view, G.THETA_EDGES, max_depth=100.0)
    run = dict(protocol="radial", bins=[
        dict(b, n_frames=1, n_px_mean=float(b["n_bin"])) for b in prof["bins"]])
    s = R.summarise(run)
    assert s["drift"] == pytest.approx(1.0, abs=0.02)
    for b in prof["bins"]:
        if b["n_valid"] > 64:
            assert b["AbsRel"] < 0.01


def test_window_sweep_reads_back_the_bias_as_a_rising_curve():
    """In a window the theta span is narrow, so a global fit cannot bowl the
    curve — the aim's own eccentricity sets the error, and the sweep rises."""
    cam, rgb, gt, cone = _scene()
    model = M.load_model(M.ANALYTIC, device=None, radial_bias=BIAS)
    rows = []
    for tilt in (0.0, 10.0, 20.0, 30.0, 40.0):
        w = G.render_window(rgb, gt, cone, cam, 0.0, tilt, 40.0, OUT, "fisheye")
        assert w.in_cone_frac >= RUN.MIN_IN_CONE_FRAC
        rows.append(RUN._score_window(model, w, max_depth=100.0))
    absrel = [r["AbsRel"] for r in rows]
    drift = rows[0]["raw_scale_ratio"] / rows[-1]["raw_scale_ratio"]
    assert absrel[-1] > 2 * absrel[0]
    assert drift > 1.1


def test_the_two_streams_score_identical_geometry():
    """With a GT-driven model the two streams differ only in pixels, so any gap
    between them would be the harness picking different frames or different GT —
    the exact failure the frozen split exists to prevent."""
    cam, rgb, gt, cone = _scene()
    other = np.roll(rgb, 17, axis=1)          # "the other stream's pixels"
    model = M.load_model(M.ANALYTIC, device=None, radial_bias=BIAS, seed=0)
    a = RUN._score_radial(model, G.full_frame_view(rgb, gt, cone, cam, OUT,
                                                   "fisheye"),
                          G.THETA_EDGES, 100.0)
    model = M.load_model(M.ANALYTIC, device=None, radial_bias=BIAS, seed=0)
    b = RUN._score_radial(model, G.full_frame_view(other, gt, cone, cam, OUT,
                                                   "fisheye"),
                          G.THETA_EDGES, 100.0)
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
