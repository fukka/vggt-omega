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


def test_a_quarter_turn_sends_pixels_where_the_rotated_image_puts_them():
    """One CCW quarter turn maps ``(u, v) -> (N - v, u)``. The turn is carried on
    the ray and the pixel, never on the parameters — ``p1, p2`` and ``s1..s4``
    are not isotropic, and projectaria_tools makes the same call: its
    ``rotate_camera_calib_cw90deg`` takes the Linear model only."""
    c = cam()
    n = c.width - 1
    u, v = _pixels(c, n=8, margin=0.3)
    d = c.unproject(u, v)                      # rays in the unrotated frame
    r = c.rotated(1)
    assert r.params == c.params, "a quarter turn must not touch the parameters"
    ru, rv = r.project(C._rot_xyz(d, 1))       # the same rays, seen rotated
    assert np.nanmax(np.abs(ru - (n - v))) < 1e-6
    assert np.nanmax(np.abs(rv - u)) < 1e-6


def test_the_quarter_turn_round_trip_closes():
    """unproject then project through a rotated camera has to be the identity, or
    rect_derect samples every point from the wrong place."""
    c = cam().rotated(1)
    u, v = _pixels(c, n=10, margin=0.28)
    ru, rv = c.project(c.unproject(u, v))
    assert np.nanmax(np.hypot(ru - u, rv - v)) < 1e-6


def test_four_quarter_turns_are_the_identity():
    c = cam()
    u, v = _pixels(c, n=8, margin=0.3)
    d = c.unproject(u, v)
    assert np.allclose(C._rot_xyz(d, 4), d)
    assert c.rotated(4).rotation == 0


def test_the_sensor_frame_is_inferred_from_the_principal_point():
    """MPS does not store the frame its calibration is in, so it is read off the
    principal point. The two Aria RGB sizes are far enough apart that the
    inference cannot land between them — and a file that is not this kind of
    calibration must be refused rather than guessed at."""
    assert C._sensor_frame(1440.0, 1440.0) == C.SENSOR_FULL
    assert C._sensor_frame(704.0, 704.0) == C.STREAM_RES
    with pytest.raises(C.CalibrationUnavailable):
        C._sensor_frame(120.0, 120.0)


def test_the_crop_reproduces_the_numbers_in_projectaria_tools_322():
    """The 1408 stream is a 2816 window of the 2880 sensor, binned 2x — not a
    plain downsample. projectaria_tools issue #322 reports an MPS calibration at
    full resolution beside the same calibration regenerated at 1408, and those
    two are the only published measurement of this offset there is.

    A plain halving misses by exactly 16 px, which is the 32 px crop halved."""
    f, cx, cy = 1221.88, 1462.73, 1465.93                 # reported, full res
    q = C._rescale_params((f, cx, cy) + (0.0,) * 12,
                          scale=C.STREAM_RES / C.SENSOR_WINDOW,
                          origin_offset=C.CROP_ORIGIN)
    assert q[0] == pytest.approx(610.94, abs=0.005)       # reported, at 1408
    assert q[1] == pytest.approx(715.11, abs=0.005)
    assert q[2] == pytest.approx(716.71, abs=0.005)

    naive = C._rescale_params((f, cx, cy) + (0.0,) * 12,
                              scale=0.5, origin_offset=0.0)
    assert naive[1] - q[1] == pytest.approx(C.CROP_ORIGIN / 2, abs=0.01)


def test_the_rescale_is_the_librarys_rescale_exactly():
    """``_rescale_params`` is written out rather than called because
    projectaria_tools needs Python >= 3.9 and this repo's interpreter may be
    older. The library is the authority; this is what enforces it."""
    try:
        from projectaria_tools.core import calibration as pcal
        from projectaria_tools.core.sophus import SE3
    except ImportError:
        pytest.skip("projectaria_tools not installed")
    prm = np.array(PARAMS, float)
    try:
        lib = pcal.CameraCalibration("camera-rgb",
                                     pcal.CameraModelType.FISHEYE624, prm,
                                     SE3(), 2880, 2880, None, np.pi, "")
    except TypeError:                       # older binding, no serial argument
        lib = pcal.CameraCalibration("camera-rgb",
                                     pcal.CameraModelType.FISHEYE624, prm,
                                     SE3(), 2880, 2880, None, np.pi)
    if not hasattr(lib, "rescale"):
        pytest.skip("this projectaria_tools predates the rescale binding")
    want = np.asarray(lib.rescale(np.array([896, 896], np.int32),
                                  896 / C.SENSOR_WINDOW,
                                  np.array([C.CROP_ORIGIN, C.CROP_ORIGIN])
                                  ).get_projection_params(), float)
    got = np.asarray(C._rescale_params(PARAMS, 896 / C.SENSOR_WINDOW,
                                       C.CROP_ORIGIN), float)
    assert np.abs(got - want).max() == 0.0, (
        f"drifted from the library by {np.abs(got - want).max():.3e}")


def test_an_unverified_orientation_is_refused_not_warned():
    """A quarter-turn error does not make the score worse, it makes it about a
    different part of the image. That is worth a hard stop."""
    c = cam().rescale(896)
    c = C.Fisheye624(c.params, c.width, c.height, dataset="oxford")
    assert "oxford" not in C.VERIFIED_ROTATION, (
        "this test encodes the state at the time of writing; if the rotation has "
        "since been verified, point it at a dataset that has not been")
    with pytest.raises(C.OrientationUnverified):
        C.require_verified(c)


def _reference_project(params, xyz):
    """FISHEYE624's forward model, transcribed from the public implementation.

    Line for line from ``Fisheye624.project`` in UniK3D
    (``third_party/UniK3D/unik3d/utils/camera.py``), which is itself Meta's
    reference ``FisheyeRadTanThinPrism``. Written out rather than imported so
    the comparison runs on a machine with neither ``projectaria_tools`` nor
    torch — which is precisely the machine whose ``rect_derect`` runs go through
    :meth:`Fisheye624._pure_project` rather than the reference.
    """
    p = np.asarray(params, float)
    f, cx, cy = p[0], p[1], p[2]
    k, (p0, p1), (s0, s1, s2, s3) = p[3:9], p[9:11], p[11:15]
    xyz = np.asarray(xyz, float)
    z = np.where(np.abs(xyz[:, 2]) < 1e-9, 1e-9, xyz[:, 2])
    ab = xyz[:, :2] / z[:, None]
    r = np.linalg.norm(ab, axis=-1, keepdims=True)
    th = np.arctan(r)
    th_k = th + sum(k[i] * th ** (3 + 2 * i) for i in range(6))
    xr, yr = (th_k * np.where(r < 1e-9, 1.0, ab / np.where(r < 1e-9, 1.0, r))).T
    xr_sq, yr_sq = xr * xr, yr * yr
    rd_sq = xr_sq + yr_sq
    u = xr + (2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1      # tangential
    v = yr + (2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0
    u = u + s0 * rd_sq + s1 * rd_sq * rd_sq                       # thin prism
    v = v + s2 * rd_sq + s3 * rd_sq * rd_sq
    return f * u + cx, f * v + cy


def test_the_fallback_projection_is_the_reference_model_not_opencvs():
    """FISHEYE624's tangential term is not OpenCV's, and the difference is a
    p1/p2 swap that no *internal* check can see.

    Written OpenCV-style, ``project`` and ``unproject`` shared the same wrong
    term, so the round trip still closed to 1e-9 and the rays still came back
    unit and forward — every test in this file passed. Only a comparison against
    an outside implementation goes red, and it goes red by 2.9 px at the rim of
    a 2880 sensor (0.90 px at 896), against the 0.29 px twin residual
    ``verify_camera`` accepts a camera at.

    The reference is transcribed rather than imported, so this runs on the
    machines that actually use the fallback.
    """
    c = cam()
    d = _rays_to(62.0)
    mine = np.stack(C.Fisheye624._pure_project(c, d), axis=1)
    want = np.stack(_reference_project(PARAMS, d), axis=1)
    err = np.hypot(mine[:, 0] - want[:, 0], mine[:, 1] - want[:, 1])
    assert np.nanmax(err) < 1e-9, (
        f"fallback drifts {np.nanmax(err):.3f} px from the reference model "
        f"({np.nanmax(err) * 896 / 2880:.3f} px at ego-synth's 896 frame)")


def _rays_to(deg, n=120, n_az=36):
    """Unit rays out to ``deg`` of incidence, swept over azimuth.

    Azimuth is not decoration: the tangential and thin-prism terms are the only
    anisotropic part of this model, so a radial fan along one azimuth cannot
    tell p1 from p2.
    """
    th = np.radians(np.linspace(0.5, deg, n))
    az = np.linspace(0.0, 2 * np.pi, n_az, endpoint=False)
    T, A = np.meshgrid(th, az)
    return np.stack([np.sin(T) * np.cos(A), np.sin(T) * np.sin(A),
                     np.cos(T)], axis=-1).reshape(-1, 3)


def test_the_fallback_projection_tracks_projectaria_tools_itself():
    """The same claim against the real package, where it is installed.

    ``projectaria_tools`` is the authority; the test above is what enforces it
    on an interpreter too old to import it (this repo runs 3.8, the package
    needs 3.9+), and this is the check that the transcription is faithful."""
    if not C.reference_available():
        pytest.skip("projectaria_tools not installed")
    from projectaria_tools.core import calibration as pcal
    ref = pcal.CameraProjection(pcal.CameraModelType.FISHEYE624,
                                np.array(PARAMS, float))
    c = cam()
    u, v = _pixels(c, n=14, margin=0.28)
    d = c.unproject(u, v)
    mine = np.stack(C.Fisheye624._pure_project(c, d), axis=1)
    got = np.array([ref.project(p) for p in d])
    err = np.hypot(mine[:, 0] - got[:, 0], mine[:, 1] - got[:, 1])
    assert np.nanmax(err) < 1e-6, f"fallback drifts {np.nanmax(err):.2e} px"


def test_project_uses_the_reference_when_it_is_available():
    if not C.reference_available():
        pytest.skip("projectaria_tools not installed")
    from projectaria_tools.core import calibration as pcal
    ref = pcal.CameraProjection(pcal.CameraModelType.FISHEYE624,
                                np.array(PARAMS, float))
    c = cam()
    u, v = _pixels(c, n=10, margin=0.25)
    d = c.unproject(u, v)
    pu, pv = c.project(d)
    got = np.array([ref.project(p) for p in d])
    assert np.nanmax(np.hypot(pu - got[:, 0], pv - got[:, 1])) < 1e-9
