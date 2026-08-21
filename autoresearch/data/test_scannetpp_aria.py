"""Tests for the ScanNet++ -> Aria remap. cv2/torch required; skipped without."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from raytun3r.cameras import KannalaBrandt, from_aria  # noqa: E402
from autoresearch.data.scannetpp_aria import AriaRemap  # noqa: E402

# Real numbers from scene 00777c41d4's transforms.json.
SPP = dict(fx=617.0, fy=617.3, cx=876.0, cy=584.0,
           k=(0.0589, 0.0067, -0.0001, -0.0002), width=1752, height=1168)
SIDE = 256


def _pair():
    src = KannalaBrandt(**SPP)
    dst = from_aria(SIDE, SIDE)
    return AriaRemap.build(src, dst, (SIDE, SIDE))


def test_the_void_is_real_and_sits_in_the_rim_band():
    """The decision this module encodes only makes sense if the void exists and
    is where we said. If a calibration change ever makes coverage total, the
    masking is harmless but the ANALYSIS text needs re-reading."""
    r = _pair()
    st = r.stats()
    assert 0.005 < st["void_frac_of_disc"] < 0.10, st
    dst = from_aria(SIDE, SIDE)
    th = dst.incidence_grid(SIDE, SIDE).numpy()
    tmax = float(dst.theta_max)
    assert th[r.void].min() > 0.80 * tmax, "void should be a rim crescent"


def test_void_is_masked_and_zeroed_never_filled():
    """The whole point: a missing pixel comes back as 0 AND excluded, so no
    downstream average can quietly include it."""
    r = _pair()
    img = np.full((SPP["height"], SPP["width"], 3), 200, np.uint8)
    out, valid = r.image(img)
    assert not valid[r.void].any()
    assert (out[r.void] == 0).all()
    assert not valid[~r.in_cone].any(), "corners are outside the imaged disc"


def test_depth_uses_nearest_so_silhouettes_are_not_interpolated():
    """A bilinear resample of a step edge produces intermediate depths that
    exist nowhere in the source. Assert the output values are a SUBSET of the
    input values."""
    r = _pair()
    d = np.zeros((SPP["height"], SPP["width"]), np.uint16)
    d[:, :SPP["width"] // 2] = 1000
    d[:, SPP["width"] // 2:] = 5000
    out, valid = r.depth(d, hole_dilate=0)
    vals = np.unique(out[valid])
    assert set(vals.tolist()) <= {1000.0, 5000.0}, vals[:10]


def test_planar_z_is_invariant_under_this_remap():
    """No rotation, same optical axis -> a constant-z plane stays that constant.
    If this ever fails, someone has introduced a rotation and the depth needs
    converting both ways rather than resampling."""
    r = _pair()
    d = np.full((SPP["height"], SPP["width"]), 2500, np.uint16)
    out, valid = r.depth(d, hole_dilate=0)
    assert valid.any()
    assert np.allclose(out[valid], 2500.0)


def test_source_holes_are_dilated_away_not_carried_as_ribbons():
    """97.8% of white render_rgb is a zero-depth hole, but 0.31% of the frame is
    white with nonzero depth -- antialiased boundaries. Dilating the hole mask
    must strictly shrink the valid set."""
    r = _pair()
    d = np.full((SPP["height"], SPP["width"]), 3000, np.uint16)
    d[500:600, 800:900] = 0
    _, v0 = r.depth(d, hole_dilate=0)
    _, v1 = r.depth(d, hole_dilate=1)
    assert v1.sum() < v0.sum()
    assert not (v1 & ~v0).any(), "dilation must only ever remove"


def test_stats_are_self_consistent():
    r = _pair()
    st = r.stats()
    assert abs(st["covered_frac_of_disc"] + st["void_frac_of_disc"] - 1.0) < 1e-9
    assert 0.6 < st["disc_frac_of_square"] < 0.85
