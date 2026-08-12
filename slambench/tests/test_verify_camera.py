# Copyright (c) 2026.
"""The camera acceptance test's own logic.

``verify_camera`` is what stands between an unproven calibration and a published
``rect_derect`` number, so its verdict has to be trustworthy in both directions:
it must pass a camera that is right, and it must refuse one that is merely
close. Both are checked here against synthetic clouds where the answer is known.

CPU-only, no data, no weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from slambench import camera as C
from slambench import verify_camera as V
from slambench.tests.test_camera import PARAMS


def _clouds(offset=(0.0, 0.0), n=3000, seed=0, res=896):
    """A predicted cloud and an actual cloud, displaced by a known offset."""
    rng = np.random.default_rng(seed)
    actual = rng.uniform(80, res - 80, size=(n, 2))
    pred = actual + np.asarray(offset, float)
    return pred, actual


def test_a_perfect_prediction_reads_as_perfect():
    pred, actual = _clouds()
    r = V.cloud_distance(pred, actual)
    assert r.median_px < 1e-9
    assert r.within_half == pytest.approx(1.0)


def test_a_displaced_cloud_is_caught_even_though_neighbours_exist():
    """The statistic's whole job. A 4 px displacement leaves every predicted
    point with *some* neighbour nearby — a 3 000-point cloud in 896 has ~16 px
    mean spacing — so a pairing-based measure could still find matches. The
    nearest-neighbour distance cannot be fooled that way."""
    pred, actual = _clouds(offset=(4.0, 0.0))
    r = V.cloud_distance(pred, actual)
    assert r.median_px > 2.0
    assert r.within_1 < 0.2


def test_the_chance_rate_is_far_below_the_bar():
    """The null: an unrelated cloud. If this were anywhere near MIN_WITHIN_1PX
    the test would be able to pass a wrong camera."""
    rng = np.random.default_rng(1)
    actual = rng.uniform(80, 816, size=(3000, 2))
    pred = rng.uniform(80, 816, size=(3000, 2))
    r = V.cloud_distance(pred, actual)
    assert r.within_1 < 0.10, r.within_1
    assert r.within_1 < V.MIN_WITHIN_1PX / 4


def _result(rot, within_1, median=0.2):
    return V.RotationResult(rot, median, within_1 * 0.9, within_1,
                            min(1.0, within_1 * 1.05), 1000)


def test_decide_accepts_a_clear_winner():
    res = {0: _result(0, 0.02), 1: _result(1, 0.95), 2: _result(2, 0.02),
           3: _result(3, 0.03)}
    win, why = V.decide(res)
    assert win == 1 and why == "ok"


def test_decide_refuses_when_nothing_clears_the_bar():
    """The state this is in as of writing: everything is bad, so the answer is
    that the model or the convention is wrong — not 'whichever was least bad'."""
    res = {k: _result(k, 0.05 + 0.01 * k) for k in range(4)}
    win, why = V.decide(res)
    assert win is None and "within 1 px" in why


def test_decide_refuses_when_two_rotations_are_not_separated():
    """If two quarter turns both look good the statistic is not discriminating
    on this take, and reporting the better one would be reporting noise."""
    res = {0: _result(0, 0.90), 1: _result(1, 0.88), 2: _result(2, 0.02),
           3: _result(3, 0.02)}
    win, why = V.decide(res)
    assert win is None and "not separated" in why


def test_decide_refuses_a_winner_whose_median_is_above_the_floor():
    """Clearing the within-1px share is not enough on its own: the floor is the
    float16 quantisation of the stored coordinates, and a median above it means
    the model is systematically off even where it mostly lands."""
    res = {0: _result(0, 0.92, median=1.4), 1: _result(1, 0.02),
           2: _result(2, 0.02), 3: _result(3, 0.02)}
    win, why = V.decide(res)
    assert win is None and "median" in why


def test_predicted_pixels_round_trips_through_a_known_camera():
    """The rectified-point-to-fisheye-pixel path, on a camera whose answer is
    its own: project a set of rays out through the fisheye, feed the matching
    pinhole coordinates back in, and the predictions must return to where they
    started."""
    cam = C.Fisheye624(PARAMS, 2880, 2880).rescale(896)
    focal, centre = 313.69297711795, 448.0
    rng = np.random.default_rng(3)
    u = rng.uniform(150, 745, 200)
    v = rng.uniform(150, 745, 200)
    d = rng.uniform(1.0, 6.0, 200)
    rect = np.stack([u, v, d], axis=1).astype(np.float32)
    # the ray those rectified pixels encode, projected through the fisheye
    pred = V.predicted_pixels(rect, focal, centre, cam)
    x = (u - centre) / focal * d
    y = (v - centre) / focal * d
    want_u, want_v = cam.project(np.stack([x, y, d], axis=1))
    assert np.allclose(pred[:, 0], want_u, equal_nan=True)
    assert np.allclose(pred[:, 1], want_v, equal_nan=True)
