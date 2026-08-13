# Copyright (c) 2026.
"""The scoring protocol: what is fitted, over what, and in whose space.

The closing invariant of this package. If a model's error is known exactly, the
harness must report that error and not some artefact of the alignment — and if a
model is perfect up to the transform it is *allowed* to be wrong by, the harness
must report zero.

CPU-only, no data, no weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from slambench import metrics as MT
from slambench import models as M


def gt(n=4000, seed=0, lo=0.8, hi=8.0):
    """Depths with real spread — an affine has two parameters, and on a scene at
    one distance they cannot be separated."""
    return np.random.default_rng(seed).uniform(lo, hi, n)


def test_an_up_to_scale_model_that_is_exactly_right_scores_zero():
    """The oracle is handed ground truth divided by a scale and offset by a
    shift, i.e. exactly the freedom an up-to-scale model has. After the protocol
    fits that back out there is nothing left to report."""
    g = gt()
    model = M.load_model(M.ORACLE, device=None, scale=3.0, shift=0.4)
    pred = model.predict(None, gt=g)
    met = MT.score_frame(pred, g, "scale_shift")
    assert met["AbsRel"] < 1e-6, met["AbsRel"]
    assert met["delta1"] == pytest.approx(1.0)


def test_a_known_multiplicative_error_is_read_back_as_that_error():
    """AbsRel of a uniformly inflated prediction is the inflation — but only if
    the alignment does not quietly absorb it. A pure scale IS absorbable, so the
    error injected here is one the affine cannot remove: a depth-dependent term."""
    g = gt()
    model = M.load_model(M.ORACLE, device=None, scale=1.0, shift=0.0)
    pred = np.asarray(model.predict(None, gt=g), np.float64)
    # 20% error on the far half only: no single affine fixes both halves.
    far = g > np.median(g)
    pred = pred * np.where(far, 1.20, 1.0)
    met = MT.score_frame(pred, g, "scale_shift")
    assert 0.02 < met["AbsRel"] < 0.20, met["AbsRel"]
    assert met["delta1"] < 1.0


def test_a_disparity_model_needs_a_disparity_space_fit():
    """The protocol claim, measured rather than asserted.

    A model that is affine-invariant in *disparity* is exactly recoverable under
    a disparity-space fit and not under a depth-space one. Scoring DAv2 in depth
    space would report a large error that is the protocol's, not the model's.
    """
    g = gt()
    disp = 1.0 / g
    pred_disp = 0.37 * disp + 0.11          # an affine in disparity
    pred = 1.0 / pred_disp                  # the pipeline convention is depth
    right = MT.score_frame(pred, g, "disparity_scale_shift")
    wrong = MT.score_frame(pred, g, "scale_shift")
    assert right["AbsRel"] < 1e-6, right["AbsRel"]
    assert wrong["AbsRel"] > 20 * max(right["AbsRel"], 1e-9)


def test_check_protocol_refuses_a_disparity_model_aligned_in_depth():
    """The guard that would have caught the above before a run rather than
    after. A mis-aligned model just looks bad; nothing in the output says why."""
    with pytest.raises(SystemExit) as e:
        MT.check_protocol("dav2_large", "scale_shift")
    assert "disparity" in str(e.value)
    with pytest.raises(SystemExit):
        MT.check_protocol("vggt_1b", "disparity_scale_shift")
    MT.check_protocol("dav2_large", "disparity_scale_shift")   # the right pair
    MT.check_protocol("da3_large", "scale_shift")


def test_a_frame_too_thin_to_fit_is_not_scored_rather_than_scored_badly():
    """The affine is fitted over the whole frame, so a frame with almost no
    points has no fit. That is a frame which was not measured."""
    g = gt(n=10)
    assert MT.score_frame(g, g, "scale_shift", min_points=256) is None


def test_frames_combine_unweighted():
    """Point counts vary four-fold across frames, and the dense ones are the
    textured ones — not a random sample. Weighting by them would let them set
    the score."""
    a = {"AbsRel": 0.10, "delta1": 1.0, "n_points": 5000, "gt_median": 2.0}
    b = {"AbsRel": 0.20, "delta1": 1.0, "n_points": 100, "gt_median": 2.0}
    out = MT.aggregate([a, b])
    assert out["AbsRel"] == pytest.approx(0.15)     # not 0.102, the weighted one
    assert out["n_frames"] == 2
    assert out["n_points_total"] == 5100


def test_common_support_is_the_intersection_of_what_both_arms_predicted():
    """Two baselines do not see the same points: a pinhole cannot cover the whole
    fisheye cone. Scoring them over different sets compares the sets."""
    a = [np.array([1.0, 2.0, np.nan, 4.0])]
    b = [np.array([1.0, np.nan, 3.0, 4.0])]
    m = MT.common_support(a, b)[0]
    assert m.tolist() == [True, False, False, True]


def test_gt_median_is_carried_because_every_metric_here_is_relative():
    """Oxford is outdoors at ~5.3 m median and AEA indoors at ~1.2 m. A relative
    error grows with depth, so 'worse on Oxford' is not yet a statement about
    the model."""
    g = gt(lo=4.0, hi=6.0)
    met = MT.score_frame(g, g, "scale_shift")
    assert 4.0 < met["gt_median"] < 6.0


# --------------------------------------------------------------------------- #
# The shared module's own contract
# --------------------------------------------------------------------------- #
# ``finetune/eval/metrics.py`` owns ``align_depth`` and ``depth_metrics`` for the
# whole repository and belongs to neither experiment, so it has no suite of its
# own. These two live here because this is the suite that runs, and because this
# package's own docstring names that module as the authority — a contract nobody
# executes is a comment.


def test_scale_ratio_is_a_post_alignment_residual_not_a_metric_grade():
    """It is ``median(gt/pred)`` over the prediction *as passed*, and every caller
    passes an aligned map.

    ``depth_metrics`` never sees an unaligned prediction, so it cannot report a
    pre-alignment scale however it is documented — and it was documented as
    "median(gt/pred) before alignment, 1.0 = perfect metric scale" for long
    enough to reach three printed tables. Under ``scale_only`` the value is 1.0
    because that mode fits this very median; reading it as "the model is
    metrically correct" grades the alignment.
    """
    from finetune.eval import metrics as FM

    g = gt()
    mask = np.ones_like(g, bool)
    pred = g / 4.0                       # a model 4x out on metric scale

    raw = FM.depth_metrics(pred, g, mask)["scale_ratio"]
    assert raw == pytest.approx(4.0, rel=1e-3), "unaligned, this IS the scale error"

    for mode in ("scale_only", "scale_shift"):
        aligned = FM.align_depth(pred, g, mask, mode=mode)
        got = FM.depth_metrics(aligned, g, mask)["scale_ratio"]
        assert got == pytest.approx(1.0, rel=1e-3), (
            f"{mode}: {got} — the alignment set this, so it cannot also grade it")


def test_the_printed_label_says_which_of_those_two_it_is():
    """The number is right in both cases; only its caption could mislead, and a
    caption is what a reader of ``report.txt`` actually has."""
    from finetune.eval.metrics import scale_ratio_note

    assert "metrically correct" in scale_ratio_note("none")
    assert "metrically correct" in scale_ratio_note("")
    for mode in ("scale_only", "scale_shift", "disparity_scale_shift"):
        note = scale_ratio_note(mode)
        assert "residual" in note and "metrically correct" not in note, note
