# Copyright (c) 2026.
"""The Aria FISHEYE624 model: its inverse, its resize, and its rotation.

Every ``rect_derect`` number goes through all three, and a mistake in any of them
does not degrade a score — it moves the sample point, so the model gets graded on
a different part of the image. That is invisible in the output, which is why it
is pinned here.

CPU-only, no data, no weights.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from slambench import camera as C

#: A plausible Aria RGB calibration: the right magnitudes for a 2880 sensor, not
#: a copy of any device's. Tests must not carry data out of a licensed release.
PARAMS = (1200.0, 1440.0, 1440.0,
          0.40, -0.50, 0.24, 0.99, -1.57, 0.61,      # k1..k6
          -3.7e-4, -7.4e-4,                          # p1, p2
          -5.8e-4, -2.2e-4, 1.9e-4, -2.0e-4)         # s1..s4


def cam(size=2880):
    return C.Fisheye624(PARAMS, size, size, dataset="test", take="t")


def _pixels(c, n=40, margin=0.30):
    """A grid inside the imaged disc, avoiding the extreme rim.

    The polynomial is only invertible below its turnover, and the corners of a
    square frame are outside the lens entirely, so a grid over the whole frame
    would be testing points the camera never sees.
    """
    r = margin * c.width
    t = np.linspace(-r, r, n)
    xv, yv = np.meshgrid(t, t)
    keep = np.hypot(xv, yv) <= r
    return (xv[keep] + c.cx).ravel(), (yv[keep] + c.cy).ravel()


def test_unproject_inverts_project():
    """The inverse has no closed form and is solved by two nested fixed points,
    so its convergence is a claim about this lens, not a theorem."""
    c = cam()
    u, v = _pixels(c)
    d = c.unproject(u, v)
    u2, v2 = c.project(d)
    err = np.hypot(u2 - u, v2 - v)
    assert np.nanmax(err) < 1e-3, f"round trip max {np.nanmax(err):.2e} px"


def test_unprojected_rays_are_unit_and_forward():
    c = cam()
    u, v = _pixels(c)
    d = c.unproject(u, v)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-9)
    assert (d[:, 2] > 0).all(), "every imaged ray is in front of the camera"


def test_incidence_angle_grows_with_radius_and_reaches_a_wide_field():
    """A fisheye of this class images past 50 deg; a model that did not would be
    describing some other lens."""
    c = cam()
    r = np.linspace(1.0, 0.45 * c.width, 40)
    th = c.theta_of(c.cx + r, np.full_like(r, c.cy))
    assert np.all(np.diff(th) > 0), "theta must be monotone in radius"
    assert th[-1] > 50.0, f"only reaches {th[-1]:.1f} deg"


def test_rescale_moves_the_principal_point_on_the_pixel_centre_convention():
    """``c' = (c + 0.5) * s - 0.5``, not ``c * s``. At 2880 -> 896 the two differ
    by 0.35 px, which is the same order as the per-device differences this
    calibration exists to capture."""
    c = cam()
    small = c.rescale(896)
    s = 896 / 2880.0
    assert small.f == pytest.approx(c.f * s)
    assert small.cx == pytest.approx((c.cx + 0.5) * s - 0.5)
    assert abs(small.cx - c.cx * s) > 0.3          # and it is not the naive one
    assert small.k.tolist() == c.k.tolist(), "distortion is scale-invariant"
    assert small.s.tolist() == c.s.tolist()


def test_rescale_preserves_the_direction_a_pixel_sees():
    """The point of a resize: the same physical ray, relabelled. If this drifts,
    every rescaled camera samples the prediction from the wrong place."""
    c = cam()
    small = c.rescale(896)
    s = 896 / 2880.0
    u, v = _pixels(c, n=12, margin=0.25)
    u2 = (u + 0.5) * s - 0.5
    v2 = (v + 0.5) * s - 0.5
    a = c.theta_of(u, v)
    b = small.theta_of(u2, v2)
    assert np.nanmax(np.abs(a - b)) < 1e-6


def test_rotate90_sends_the_principal_point_where_the_pixels_go():
    """One CCW quarter turn maps ``(x, y) -> (N - y, x)``; the calibration has to
    follow the pixels or every ray is a quarter turn out."""
    c = cam()
    n = c.width - 1
    r = c.rotate90(1)
    assert r.cx == pytest.approx(n - c.cy)
    assert r.cy == pytest.approx(c.cx)
    assert r.f == c.f


def test_four_quarter_turns_are_the_identity():
    c = cam()
    r = c.rotate90(4)
    assert r.cx == pytest.approx(c.cx) and r.cy == pytest.approx(c.cy)


def test_native_resolution_is_inferred_from_the_principal_point():
    """MPS does not store the sensor size, so it is read off the calibration. The
    two Aria RGB sizes are far enough apart that the inference cannot land
    between them — and a file that is not this kind of calibration must be
    refused rather than guessed at."""
    assert C._infer_native(1440.0, 1440.0) == 2880
    assert C._infer_native(704.0, 704.0) == 1408
    with pytest.raises(C.CalibrationUnavailable):
        C._infer_native(120.0, 120.0)


def test_an_unverified_orientation_is_refused_not_warned():
    """A quarter-turn error does not make the score worse, it makes it about a
    different part of the image. That is worth a hard stop."""
    c = cam().rescale(896)
    c = C.Fisheye624(c.params, c.width, c.height, dataset="nymeria")
    assert "nymeria" not in C.VERIFIED_ROTATION, (
        "this test encodes the state at the time of writing; if the rotation has "
        "since been verified, point it at a dataset that has not been")
    with pytest.raises(C.OrientationUnverified):
        C.require_verified(c)
