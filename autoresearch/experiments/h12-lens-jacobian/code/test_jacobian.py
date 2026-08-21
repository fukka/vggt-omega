"""Tests for the lens-Jacobian conditioning field.

Pure numpy, no torch, no weights, no data -- runs anywhere in well under a
second. The point of each test is named in its docstring; a test whose failure
would not tell you something is not worth the line.
"""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import jacobian as J  # noqa: E402

ARIA_K = (0.3852, -0.4442, 0.5591, -0.3254)
ARIA_THETA_MAX = np.deg2rad(54.83)


def test_on_axis_is_exactly_zero_for_every_lens():
    """The field is defined as a departure from the on-axis case, so on axis it
    must be 0 -- not 1e-9, and above all not NaN, since d/sin(theta) is 0/0
    there and that pixel is definitionally the reference."""
    for d, dp in ((J.pinhole_d, J.pinhole_dprime),
                  (J.equidistant_d, J.equidistant_dprime),
                  (lambda t: J.kb4_d(t, ARIA_K), lambda t: J.kb4_dprime(t, ARIA_K))):
        la, ln = J.log_area_aniso(np.array([0.0]), d, dp)
        assert la[0] == 0.0 and ln[0] == 0.0


def test_pinhole_matches_its_closed_form():
    """A pinhole is not the identity of this field -- perspective stretches too.
    area = sec^3, aniso = sec. If this drifts, the 'departure from pinhole'
    reading of the channels is wrong."""
    th = np.linspace(0.0, 1.0, 41)
    la, ln = J.log_area_aniso(th, J.pinhole_d, J.pinhole_dprime)
    np.testing.assert_allclose(la, np.log(1 / np.cos(th) ** 3), atol=1e-10)
    np.testing.assert_allclose(ln, np.log(1 / np.cos(th)), atol=1e-10)


def test_equidistant_matches_its_closed_form():
    """area = theta/sin(theta), aniso = sin(theta)/theta -- exact negatives,
    because d' == 1 makes the two magnifications reciprocal."""
    th = np.linspace(1e-6, 1.0, 41)
    la, ln = J.log_area_aniso(th, J.equidistant_d, J.equidistant_dprime)
    np.testing.assert_allclose(la, np.log(th / np.sin(th)), atol=1e-10)
    np.testing.assert_allclose(ln, np.log(np.sin(th) / th), atol=1e-10)
    np.testing.assert_allclose(la, -ln, atol=1e-12)


def test_analytic_derivative_agrees_with_finite_difference():
    """kb4_dprime is hand-differentiated; a sign or coefficient slip there would
    corrupt every log_area silently, since nothing else recomputes it."""
    th = np.linspace(0.05, ARIA_THETA_MAX, 200)
    h = 1e-6
    fd = (J.kb4_d(th + h, ARIA_K) - J.kb4_d(th - h, ARIA_K)) / (2 * h)
    np.testing.assert_allclose(J.kb4_dprime(th, ARIA_K), fd, rtol=1e-6)


def test_focal_length_cancels_entirely():
    """The whole transfer claim rests on this: the field describes the LENS
    SHAPE, not the sensor. Scaling f must not move a single value."""
    a = J.jacobian_field(64, 64, 300.0, 300.0, 31.5, 31.5,
                         lambda t: J.kb4_d(t, ARIA_K),
                         lambda t: J.kb4_dprime(t, ARIA_K), ARIA_THETA_MAX)
    b = J.jacobian_field(64, 64, 900.0, 900.0, 31.5, 31.5,
                         lambda t: J.kb4_d(t, ARIA_K),
                         lambda t: J.kb4_dprime(t, ARIA_K), ARIA_THETA_MAX)
    # same theta grid is NOT expected (f changes which rays land on which
    # pixel); what must hold is that the VALUE attached to a given theta is
    # identical, so compare the field as a function of its own third channel.
    for f in (a, b):
        la, ln = J.log_area_aniso(f[..., 2] * ARIA_THETA_MAX,
                                  lambda t: J.kb4_d(t, ARIA_K),
                                  lambda t: J.kb4_dprime(t, ARIA_K))
        np.testing.assert_allclose(f[..., 0], la, atol=1e-12)
        np.testing.assert_allclose(f[..., 1], ln, atol=1e-12)


def test_aria_and_pinhole_actually_differ_at_the_rim():
    """If the two lenses' fields were close, there would be nothing for the
    conditioning to carry and the hypothesis would be dead on arrival. Measure
    the gap rather than assuming it."""
    th = np.array([ARIA_THETA_MAX])
    la_a, ln_a = J.log_area_aniso(th, lambda t: J.kb4_d(t, ARIA_K),
                                  lambda t: J.kb4_dprime(t, ARIA_K))
    la_p, ln_p = J.log_area_aniso(th, J.pinhole_d, J.pinhole_dprime)
    assert abs(la_a[0] - la_p[0]) > 0.5, (la_a[0], la_p[0])
    assert abs(ln_a[0] - ln_p[0]) > 0.2, (ln_a[0], ln_p[0])


def test_non_monotone_lens_is_refused_not_interpolated():
    """A KB4 fit can turn over inside the imaged field. Interpolating past a
    turnover returns a plausible WRONG angle -- an error shaped exactly like
    'the model is bad at the rim'. It must raise."""
    bad_k = (0.0, -5.0, 0.0, 0.0)          # turns over well inside 1 rad
    with pytest.raises(ValueError, match="not monotone"):
        J.theta_from_radius(np.array([0.1]), lambda t: J.kb4_d(t, bad_k), 1.0)


def test_field_shape_and_outside_cone_saturates():
    """Rays past theta_max are outside the imaged cone. They must saturate, not
    extrapolate the polynomial into territory it was never fitted on."""
    f = J.jacobian_field(32, 48, 20.0, 20.0, 23.5, 15.5,
                         lambda t: J.kb4_d(t, ARIA_K),
                         lambda t: J.kb4_dprime(t, ARIA_K), ARIA_THETA_MAX)
    assert f.shape == (32, 48, 3)
    assert np.isfinite(f).all()
    assert f[..., 2].max() <= 1.0 + 1e-12 and f[..., 2].min() >= 0.0


def test_aria_field_is_NOT_monotone_and_the_anisotropy_changes_sign():
    """The load-bearing measurement of this experiment.

    Aria's own calibration of record does NOT vary monotonically across the
    field. `log_area` rises to a peak near 49 deg and then FALLS toward the rim,
    and `log_aniso` peaks near 40 deg, crosses ZERO near 50 deg, and is strongly
    negative at 54.83 deg. The sign change means the lens stretches radially
    more than tangentially inside ~50 deg and the other way outside it -- the
    rim band is different in KIND, not merely in degree.

    Why that matters here: a scalar rim weight (H5) is monotone in theta by
    construction, so it cannot represent a quantity that turns over and changes
    sign inside the very band it is weighting. That is a concrete mechanism for
    why rim-weighted losses did not beat plain LoRA, and it is the reason to
    condition on the Jacobian rather than on theta.

    If this ever starts passing as monotone, the calibration changed and the
    motivation for this whole experiment needs re-reading.
    """
    th = np.linspace(0, ARIA_THETA_MAX, 2000)
    la, ln = J.log_area_aniso(th, lambda t: J.kb4_d(t, ARIA_K),
                              lambda t: J.kb4_dprime(t, ARIA_K))
    assert not (np.all(np.diff(la) > 0) or np.all(np.diff(la) < 0))
    peak = th[int(np.argmax(la))]
    assert np.deg2rad(45) < peak < np.deg2rad(53), np.rad2deg(peak)
    assert ln.max() > 0.05 and ln[-1] < -0.1, (ln.max(), ln[-1])
    sign_changes = np.sum(np.diff(np.sign(ln[1:])) != 0)
    assert sign_changes == 1, f"expected one sign change, got {sign_changes}"


def test_theta_alone_cannot_stand_in_for_the_jacobian():
    """The conditioning must carry something theta does not, or it is a
    reparameterisation of the rim weight that already failed.

    Because log_aniso turns over, there exist pairs of DIFFERENT theta with the
    SAME log_aniso. A network given only theta sees those as far apart; given
    log_aniso it sees them as alike. That is the extra information, stated as a
    property rather than asserted in prose.
    """
    th = np.linspace(0.02, ARIA_THETA_MAX, 4000)
    _, ln = J.log_area_aniso(th, lambda t: J.kb4_d(t, ARIA_K),
                             lambda t: J.kb4_dprime(t, ARIA_K))
    target = 0.05
    hits = np.where(np.diff(np.sign(ln - target)) != 0)[0]
    assert len(hits) >= 2, "expected log_aniso=0.05 to be reached twice"
    a, b = np.rad2deg(th[hits[0]]), np.rad2deg(th[hits[-1]])
    assert b - a > 10.0, (a, b)
