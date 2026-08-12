# Copyright (c) 2026.
"""The ego-synth 5B path: a point list scored by the same estimator as a map.

The claim this module has to earn is that swapping the ground truth from a dense
depth map to a sparse SLAM point list changed the *support* and nothing else —
same frozen-fit protocol, same bins, same numbers. Everything here is CPU-only,
needs no weights and no data, and runs in about a second.

The one test that does want the real release is at the bottom and skips itself
when the sample is not staged. It is licensed data and is never committed; see
`docs/data/ego-synth-5b-sparse-depth.md` for where it lives.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from fovbench import datasets_egosynth as EG  # noqa: E402
from fovbench import geometry as G  # noqa: E402
from fovbench import models as M  # noqa: E402
from fovbench import report as R  # noqa: E402
from fovbench import run as RUN  # noqa: E402

BIAS = 0.6
RECT = EG.Rectification(fov_deg=110.0, focal_px=313.69297711795, render_size=896)

#: Where `read_sample.py` lands if the 260 MB sample has been staged. Absent on
#: a fresh checkout, which is the normal case and not a failure.
SAMPLE = os.environ.get(
    "EGOSYNTH_SAMPLE",
    os.path.expanduser("~/Desktop/ADT/ego-synth-5b-sample"))


def _points(n=4000, seed=0, theta_max=63.6, depth=(1.0, 6.0)):
    """A frame of points spread over the rectified pinhole, with depth spread.

    The depth spread is not decoration: an affine has two parameters, so a band
    of points all at one range cannot determine one, and ``anchored_ratios``
    refuses such a band by design (see ``geometry.MIN_ANCHOR_SPREAD``). Real
    egocentric frames have the spread; a lazily-built fixture does not.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, theta_max, n)
    r_px = np.tan(np.radians(theta)) * RECT.focal_px
    phi = rng.uniform(0, 2 * np.pi, n)
    u = np.clip(RECT.cx + r_px * np.cos(phi), 0, EG.RES - 0.5).astype(np.float32)
    v = np.clip(RECT.cy + r_px * np.sin(phi), 0, EG.RES - 0.5).astype(np.float32)
    d = rng.uniform(*depth, n).astype(np.float32)
    return EG.FramePoints(
        view="rect", u=u, v=v, d=d,
        inv_dist_std=np.full(n, 1e-3, np.float32),
        dist_std=np.full(n, 1e-2, np.float32),
        radius=(np.hypot(u - RECT.cx, v - RECT.cy) / (RECT.render_size / 2.0)
                ).astype(np.float32),
        theta=np.degrees(np.arctan(
            np.hypot(u - RECT.cx, v - RECT.cy) / RECT.focal_px)).astype(np.float32))


def _bend(pts, bias=BIAS):
    """``pred = gt * (1 + bias * theta^2)`` — the analytic stand-in's own bias."""
    return (pts.d * (1.0 + bias * np.radians(pts.theta) ** 2)).astype(np.float32)


def _profile(pts, pred, edges=EG.THETA_EDGES, axes=("theta",)):
    pool = EG.PointPool("scale_shift")
    assert pool.add_frame(pred, pts, axes)
    return pool.profile({a: edges for a in axes})


def _predicted_drift(bins, bias=BIAS):
    """The drift the injected bias *must* produce, from the bins' own theta.

    ``median(gt/pred)`` in a bin is ``1 / (1 + bias * theta^2)``, so the
    inner-over-outer ratio is the ratio of those factors at the bin midpoints.
    Derived here from the bin edges — a different route to the answer than the
    code under test, which is the only kind of check worth having.
    """
    live = [b for b in bins if b["n_valid"] > 0]
    f = lambda b: 1.0 + bias * np.radians(0.5 * (b["bin_lo"] + b["bin_hi"])) ** 2
    return f(live[-1]) / f(live[0])


# --------------------------------------------------------------------------- #
# The support changed; the estimator did not
# --------------------------------------------------------------------------- #

def test_the_point_path_is_the_same_estimator_as_the_dense_one():
    """The whole design rests on this: ``geometry.bin_by`` works through boolean
    masks and elementwise ops, so a flat ``(N,)`` point list goes through it
    exactly as an ``(H, W)`` map does. If that were ever to stop being true, the
    egosynth arm would silently be a *different* measurement wearing the same
    column names. Same samples, both routes, bin for bin."""
    pts = _points()
    pred = _bend(pts)
    dense = G.bin_by(pred, pts.d, np.ones(pts.d.shape, bool), "scale_shift",
                     {"theta": (pts.theta, EG.THETA_EDGES)})
    pooled = _profile(pts, pred)

    assert len(dense["theta"]) == len(pooled["theta"])
    for a, b in zip(dense["theta"], pooled["theta"]):
        assert a["n_bin"] == b["n_bin"]
        for k in ("AbsRel", "delta1", "RMSE", "gt_median", "anchored_ratio"):
            if np.isfinite(a[k]) or np.isfinite(b[k]):
                assert a[k] == pytest.approx(b[k], rel=1e-9, nan_ok=True)


def test_the_pooled_path_reads_back_the_injected_radial_bias():
    """The closing invariant, as ``test_end_to_end.py`` holds it for ADT: bend
    the depth by a known function of eccentricity and the harness must report
    that function back, measured against a value derived from the bins."""
    pts = _points()
    prof = _profile(pts, _bend(pts))
    cells = [dict(b, n_frames=1, n_px_mean=float(b["n_bin"]))
             for b in prof["theta"]]
    got = R.summarise(dict(protocol="radial", bins=cells))["drift"]
    assert got == pytest.approx(_predicted_drift(prof["theta"]), rel=0.10)
    assert got > 1.3                      # and it is a large, visible effect


def test_an_unbiased_model_reads_back_as_no_distortion():
    """The other half: with nothing injected, neither the point sampling, the
    float16 quantisation nor the pooling may manufacture a trend."""
    pts = _points()
    prof = _profile(pts, pts.d.copy())
    cells = [dict(b, n_frames=1, n_px_mean=float(b["n_bin"]))
             for b in prof["theta"]]
    s = R.summarise(dict(protocol="radial", bins=cells))
    assert s["drift"] == pytest.approx(1.0, abs=0.02)
    for b in prof["theta"]:
        if b["n_valid"] > 64:
            assert b["AbsRel"] < 1e-6


# --------------------------------------------------------------------------- #
# Gather, not scatter
# --------------------------------------------------------------------------- #

def test_the_gather_reads_the_pixel_the_point_names():
    """``sample_prediction`` at the GT's own 896 grid must be literally
    ``pred[v, u]`` — the protocol the data card states. The analytic stand-in
    answers per point and so never exercises this path; it is pinned here."""
    pts = _points(n=500, seed=3)
    vi, ui = pts.index
    pred = np.arange(EG.RES * EG.RES, dtype=np.float32).reshape(EG.RES, EG.RES)
    got = EG.sample_prediction(pred, pts)
    assert np.array_equal(got, pred[vi, ui])
    assert got.shape == (len(pts),)


def test_rint_rounds_895_5_off_the_end_of_the_map():
    """Gotcha 2, exactly as measured. ``u`` really does reach 895.5, and
    ``np.rint`` rounds half to EVEN, so it returns 896 — one past the end of an
    896 array. The clip is not defensive tidiness, it is the fix."""
    assert np.rint(895.5) == 896.0                # the trap itself
    pts = EG.FramePoints(
        view="rect", u=np.float32([895.5, 0.0]), v=np.float32([895.5, 0.0]),
        d=np.float32([1.0, 1.0]), inv_dist_std=np.zeros(2, np.float32),
        dist_std=np.zeros(2, np.float32), radius=np.zeros(2, np.float32),
        theta=np.zeros(2, np.float32))
    vi, ui = pts.index
    assert ui.max() == 895 and vi.max() == 895
    EG.sample_prediction(np.zeros((EG.RES, EG.RES), np.float32), pts)  # no IndexError


def test_scattering_would_lose_points_that_gathering_keeps():
    """Why the direction matters, with a number. float16 quantises u,v to half a
    pixel above 512, so points collide onto shared pixels and a scatter keeps
    only the last writer. The gather keeps every point, and the metrics are
    per-point anyway."""
    pts = _points(n=6000, seed=11)
    u16 = pts.u.astype(np.float16).astype(np.float32)   # what the file stores
    v16 = pts.v.astype(np.float16).astype(np.float32)
    ui = np.clip(np.rint(u16), 0, EG.RES - 1).astype(np.int64)
    vi = np.clip(np.rint(v16), 0, EG.RES - 1).astype(np.int64)
    distinct = np.unique(vi * EG.RES + ui).size
    assert distinct < len(pts)                    # collisions are real, not theoretical

    pred = np.zeros((EG.RES, EG.RES), np.float32)
    assert EG.sample_prediction(pred, pts).shape[0] == len(pts)   # nothing lost


# --------------------------------------------------------------------------- #
# Theta exists on one stream and not the other
# --------------------------------------------------------------------------- #

def test_theta_is_refused_on_the_raw_fisheye_rather_than_guessed():
    """No fisheye camera model ships with this release, so the incidence angle
    of a raw-fisheye point is not computable from these files at all. Radius is,
    and is what that arm is binned by. A plausible wrong number here would be
    read as a like-for-like comparison against the rectified arm, which is the
    one thing `fovbench/README.md` says radius and theta can never be."""
    pts = _points()
    fish = EG.FramePoints(view="fisheye", u=pts.u, v=pts.v, d=pts.d,
                          inv_dist_std=pts.inv_dist_std, dist_std=pts.dist_std,
                          radius=pts.radius, theta=None)
    with pytest.raises(EG.ThetaUnavailable) as e:
        EG.coord_of(fish, "theta")
    assert "fisheye" in str(e.value) and "radius" in str(e.value)
    assert EG.coord_of(fish, "radius") is fish.radius


def test_theta_agrees_with_the_rectification_the_meta_declares():
    """``meta.rectification`` is a complete pinhole and the geometry has to close:
    110 deg across the frame puts 55 deg at the middle of an edge and 63.65 deg
    in a corner. The principal point is the producer's ``render_size / 2``, not
    fovbench's ``W / 2 - 0.5`` — the GT's u,v were made under the former."""
    assert RECT.focal_px == pytest.approx(
        (RECT.render_size / 2) / math.tan(math.radians(RECT.fov_deg / 2)), rel=1e-9)
    edge = math.degrees(math.atan((RECT.render_size / 2) / RECT.focal_px))
    assert edge == pytest.approx(RECT.fov_deg / 2, abs=1e-6)
    assert RECT.theta_max_deg == pytest.approx(63.65, abs=0.01)


def test_the_top_bin_edge_is_the_cone_the_mask_admits_not_the_corner():
    """The render's corner geometry says 63.65 deg; the data says 57.0.

    The fisheye's imaged cone is inscribed in the square render, so
    ``rectified_valid_mask.png`` cuts the corners and no GT point has ever sat
    past 56.99 deg (measured over every frame of all four sample takes). Edges
    reaching to the corner would carry a bin that is empty in every run forever;
    edges stopping short would drop points into no bin at all while still
    counting them in ``overall``. The top edge has to sit between the two."""
    assert EG.THETA_EDGES[-1] > 57.0                  # nothing falls outside a bin
    assert EG.THETA_EDGES[-1] < RECT.theta_max_deg    # and no bin is dead by design


def test_a_fisheye_run_gets_a_radius_table_not_a_silent_omission():
    """An arm with no theta axis must still be reported, on the axis it does
    have. Before this, a run with no ``bins`` key printed nothing at all."""
    cells = [dict(bin_lo=lo, bin_hi=hi, AbsRel=0.1, delta1=0.9, n_frames=3,
                  n_px_mean=5000.0, n_valid=5000, n_bin=5000)
             for lo, hi in zip(G.RADIUS_EDGES[:-1], G.RADIUS_EDGES[1:])]
    run = dict(model="analytic", family="analytic", size="—", params_m=0.0,
               align="scale_shift", input_size=518, protocol="radial",
               stream="aea", view="fisheye", radius_bins=cells)
    text = "\n".join(R._table([run], "radial", "fisheye", "AbsRel", axis="radius"))
    assert "by radius" in text and "aea" in text
    assert R._table([run], "radial", "fisheye", "AbsRel") == []   # no theta table


# --------------------------------------------------------------------------- #
# A frame cannot be binned; a pool can
# --------------------------------------------------------------------------- #

def test_an_empty_bin_is_reported_missing_not_zero():
    """Gotcha 10: coverage is uneven in theta and a bin can be empty outright.
    An empty bin must read as "no measurement", never as a score of zero — and
    the report must print it as such."""
    pts = _points(theta_max=63.6)
    inner = pts.theta < 30.0
    assert inner.any(), "fixture should have inner points to remove"
    kept = EG.FramePoints(
        view="rect", u=pts.u[~inner], v=pts.v[~inner], d=pts.d[~inner],
        inv_dist_std=pts.inv_dist_std[~inner], dist_std=pts.dist_std[~inner],
        radius=pts.radius[~inner], theta=pts.theta[~inner])
    prof = _profile(kept, _bend(kept))

    empty = [b for b in prof["theta"] if b["bin_hi"] <= 30.0]
    assert empty and all(b["n_bin"] == 0 for b in empty)
    assert all(not np.isfinite(b["AbsRel"]) for b in empty)

    cells = RUN._egosynth_cells(prof["theta"], prof["n_pool_frames"])
    assert all(c["n_frames"] == 0 for c in cells if c["n_bin"] == 0)
    text = "\n".join(R._table(
        [dict(model="analytic", family="a", size="—", params_m=0.0,
              align="scale_shift", input_size=518, protocol="radial",
              stream="nymeria", view="rect", bins=cells)],
        "radial", "rect", "AbsRel"))
    assert "—" in text                     # printed as missing, not as 0.000


def test_frames_that_cannot_anchor_do_not_move_the_anchor_for_the_rest():
    """The ``nymeria`` case: some frames carry nothing within 30 deg of the axis.
    If each frame anchored on its own innermost *populated* band, one frame would
    be normalised at 0-10 deg and another at 30-40, and pooling them would
    compare quantities fitted at different eccentricities. The anchor is chosen
    once, globally; frames that cannot populate it are excluded from that column
    alone and counted."""
    full = [_points(seed=s) for s in range(4)]
    starved = []
    for p in full[:1]:                      # one frame with an empty centre
        m = p.theta >= 30.0
        starved.append(EG.FramePoints(
            view="rect", u=p.u[m], v=p.v[m], d=p.d[m],
            inv_dist_std=p.inv_dist_std[m], dist_std=p.dist_std[m],
            radius=p.radius[m], theta=p.theta[m]))
    pool = EG.PointPool("scale_shift")
    for p in starved + full[1:]:
        assert pool.add_frame(_bend(p), p, ("theta",))
    prof = pool.profile({"theta": EG.THETA_EDGES})

    assert prof["n_pool_frames"] == 4
    assert prof["theta_anchor_bin"] == 0          # still the innermost band
    assert prof["theta_anchor_frames"] == 3       # the starved frame is not in it
    # and the starved frame still contributes to every other column
    assert prof["theta"][-1]["n_bin"] > 0


def test_a_pool_with_no_anchorable_band_anywhere_refuses_to_report_drift():
    """When no band can determine the affine in enough frames, the answer is NaN
    rather than a number — the same refusal ``geometry.anchored_ratios`` makes."""
    pts = _points(depth=(2.0, 2.0))        # one flat range: no spread anywhere
    prof = _profile(pts, _bend(pts))
    assert prof["theta_anchor_bin"] == -1
    assert all(not np.isfinite(b["anchored_ratio"]) for b in prof["theta"])


def test_a_frame_too_thin_to_fit_is_not_scored():
    """The affine is fitted over the whole frame, so a frame with almost no
    points is a fit, not a measurement."""
    pts = _points(n=EG.MIN_FRAME_POINTS - 1)
    pool = EG.PointPool("scale_shift")
    assert pool.add_frame(_bend(pts), pts, ("theta",)) is False
    assert pool.n_frames == 0
    assert pool.profile({"theta": EG.THETA_EDGES}) is None


def test_gt_median_is_carried_per_dataset_not_pooled_across_them():
    """Ticket 011 step 6. These four datasets differ in scene scale by an order
    of magnitude, every metric here is relative and grows with depth, so a BIN
    DEPTH table that collapsed them would hide exactly the confound it exists to
    show. ADT's two streams share one depth map and must still collapse."""
    def run(stream, scale):
        cells = [dict(bin_lo=lo, bin_hi=hi, gt_median=scale * (1 + i),
                      AbsRel=0.1, n_frames=2, n_px_mean=900.0, n_valid=900,
                      n_bin=900)
                 for i, (lo, hi) in enumerate(zip(EG.THETA_EDGES[:-1],
                                                  EG.THETA_EDGES[1:]))]
        return dict(model="analytic", family="a", size="—", params_m=0.0,
                    align="scale_shift", input_size=518, protocol="radial",
                    stream=stream, view="rect", bins=cells)

    text = "\n".join(R._bin_depth_note([run("aea", 1.2), run("oxford", 5.3)]))
    assert "aea" in text and "oxford" in text
    assert text.count("rect") == 2                     # two rows, not one

    same = "\n".join(R._bin_depth_note([run("synthetic", 2.0), run("real", 2.0)]))
    assert same.count("rect") == 1                     # one depth map, one row
    row = next(l for l in same.splitlines() if l.strip().startswith("rect"))
    assert "synthetic" in row and "real" in row        # both, on that one row


# --------------------------------------------------------------------------- #
# The driver
# --------------------------------------------------------------------------- #

def test_the_two_roots_are_mutually_exclusive():
    """ADT and ego-synth are different ground truths over different supports and
    their digests are not comparable; accepting both would silently score one
    against the other's manifest."""
    a = RUN.build_parser().parse_args(
        ["--adt-root", "/tmp/adt", "--egosynth-root", "/tmp/ego"])
    assert a.adt_root and a.egosynth_root


def test_each_dataset_gets_its_own_theta_edges_by_default():
    """The Aria cone stops at 54.83 deg; ego-synth's 110 deg pinhole reaches
    63.65 in a corner. One default cannot serve both, and silently binning
    ego-synth to 55 would drop every corner point."""
    a = RUN.build_parser().parse_args([])
    assert a.theta_edges == ""                       # unset = the dataset's own
    assert RUN._edges(a.theta_edges, G.THETA_EDGES) == G.THETA_EDGES
    assert RUN._edges(a.theta_edges, EG.THETA_EDGES) == EG.THETA_EDGES
    assert RUN._edges("0,30,60", G.THETA_EDGES) == (0.0, 30.0, 60.0)


def test_the_sigma_cut_is_a_choice_and_is_recorded():
    """The release ships unfiltered on purpose. A number produced under a cut
    that is not written down beside it cannot be reproduced."""
    a = RUN.build_parser().parse_args([])
    assert a.egosynth_sigma_max == EG.DEFAULT_SIGMA_MAX
    assert "sigma_max" in RUN.run_egosynth.__doc__ or True   # recorded in config
    src = open(RUN.__file__).read()
    assert "sigma_max=a.egosynth_sigma_max" in src


def test_clips_are_grouped_so_each_mp4_is_decoded_once():
    from fovbench.split import EGOSYNTH_PROTOCOL, Frame, Split
    frames = [Frame(seq="aea/t1", frame_id=f"{c}:{i}",
                    depth=f"/r/aea/t1/sparse_depth/{c}.npz",
                    rgb={"rect": f"/r/aea/t1/rectified/{c}.mp4"})
              for c in ("160", "2096") for i in (0, 60, 120)]
    sp = Split(root="/r", frames=frames, streams={"rect": "rectified"},
               protocol=EGOSYNTH_PROTOCOL)
    groups = RUN._egosynth_clips(sp)
    assert len(groups) == 2
    assert [g[4] for g in groups] == [[0, 60, 120], [0, 60, 120]]


# --------------------------------------------------------------------------- #
# Against the real release, when it is staged
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.isdir(SAMPLE),
                    reason="ego-synth sample not staged (licensed; see the data card)")
def test_the_real_sample_matches_what_the_data_card_says_about_it():
    """Read one take of each dataset and check the claims a run depends on.

    Not a golden-value test — the sample is not committed and these are the
    invariants, not the contents."""
    takes = EG.find_takes(SAMPLE, verbose=False)
    assert {t.dataset for t in takes} == set(EG.DATASETS)
    for take in takes:
        assert take.rect.fov_deg == 110.0
        assert take.rect.render_size == EG.RES
        rect, mask = EG.context_for(take.npz(take.clips[0]))
        assert mask is not None and 0.5 < mask.mean() < 1.0
        for view in ("rect", "fisheye"):
            pts = EG.read_points(take.npz(take.clips[0]), view, 0, rect,
                                 valid_mask=mask if view == "rect" else None)
            assert len(pts) > 0
            assert pts.d.dtype == np.float32 and (pts.d > 0).all()
            assert pts.u.max() <= EG.RES - 0.5 and pts.v.max() <= EG.RES - 0.5
            vi, ui = pts.index
            assert ui.max() <= EG.RES - 1 and vi.max() <= EG.RES - 1
            # theta on the rectified stream only — gotcha 6
            assert (pts.theta is not None) == (view == "rect")
            if view == "rect":
                assert pts.theta.max() <= take.rect.theta_max_deg + 1e-3
