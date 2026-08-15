# Copyright (c) 2026.
"""The two lens strategies, and the claim that separates them from the harness.

``rect_derect`` is the arm that can be wrong quietly. It rectifies, predicts, and
maps back — three geometric steps whose failure mode is not a worse score but a
score computed at the wrong place, which looks like a merely mediocre model.

The load-bearing claim is that the rectification is **co-axial**, so planar z is
unchanged by it and no depth conversion is applied on the way back. That claim is
measured on the real release (the producer's own rectified and fisheye point sets
carry identical depths for the same points), and pinned here against an analytic
scene where the right answer is known in closed form.

CPU-only, no data, no weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from slambench import baselines as B
from slambench import camera as C
from slambench import data as D
from slambench.tests.test_camera import PARAMS

RES = 896


def cam896():
    return C.Fisheye624(PARAMS, 2880, 2880, dataset="test").rescale(RES)


def slanted_plane(dirs, normal=(0.25, -0.15, 1.0), offset=3.0):
    """Planar z of a slanted plane along each ray — the closed-form answer.

    Slanted, not fronto-parallel: on a plane at constant z every mapping error
    returns the right depth anyway, so the test would pass with the geometry
    removed.
    """
    n = np.asarray(normal, float)
    d = np.asarray(dirs, float)
    t = offset / (d @ n)
    return t * d[..., 2]


def fisheye_points(c, n=24, margin=0.18):
    """GT points on a grid near the axis, with their true planar z."""
    r = margin * c.width
    t = np.linspace(-r, r, n)
    xv, yv = np.meshgrid(t, t)
    keep = np.hypot(xv, yv) <= r
    u = (xv[keep] + c.cx).ravel()
    v = (yv[keep] + c.cy).ravel()
    z = slanted_plane(c.unproject(u, v))
    return D.FramePoints(u=u.astype(np.float32), v=v.astype(np.float32),
                         d=z.astype(np.float32),
                         inv_dist_std=np.zeros(u.size, np.float32),
                         dist_std=np.zeros(u.size, np.float32))


def true_pinhole_depth(pin):
    """The depth map a perfect model would emit on the rectified image."""
    return slanted_plane(pin.rays()).astype(np.float32)


def test_derectify_recovers_the_true_depth_at_the_fisheye_points():
    """The whole round trip, against a closed form.

    A perfect prediction on the rectified image, sampled back at the fisheye
    points, must equal the true planar z at those points. Any error in
    unproject, in the pinhole projection, or an unnecessary depth conversion
    shows up here.
    """
    c = cam896()
    bl = B.RectDerectBaseline(model=None, cam=c, fov_deg=110.0, rect_size=518)
    pts = fisheye_points(c)
    got = bl.derectify(true_pinhole_depth(bl.pin), pts)
    ok = np.isfinite(got)
    assert ok.mean() > 0.95, f"only {100*ok.mean():.0f}% of points landed"
    rel = np.abs(got[ok] - pts.d[ok]) / pts.d[ok]
    assert np.median(rel) < 2e-3, f"median relative error {np.median(rel):.2e}"
    assert np.max(rel) < 2e-2, f"max relative error {np.max(rel):.2e}"


def test_planar_z_is_unchanged_by_the_co_axial_rectification():
    """The claim that licenses applying no depth conversion.

    Stated directly rather than through the sampler: for the same physical ray,
    planar z about the fisheye's axis and about the pinhole's axis are the same
    number, because the two axes are the same axis.
    """
    c = cam896()
    pin = B.Pinhole(518, 110.0)
    u, v = np.array([448.0, 500.0, 380.0]), np.array([448.0, 402.0, 520.0])
    d_fish = c.unproject(u, v)
    pu, pv = pin.project(d_fish)
    # the pinhole ray at that pixel is the same direction
    j = np.stack([(pu - pin.c) / pin.focal, (pv - pin.c) / pin.focal,
                  np.ones_like(pu)], axis=-1)
    d_pin = j / np.linalg.norm(j, axis=-1, keepdims=True)
    assert np.allclose(d_fish, d_pin, atol=1e-9)
    assert np.allclose(slanted_plane(d_fish), slanted_plane(d_pin), atol=1e-12)


def test_derectify_holds_up_at_the_rim_where_the_pinhole_stretches():
    """The same round trip, but out where it is hard.

    The test above lives inside 18 % of the frame, and every step of the chain
    is nearly linear there — it would pass with the tangential and thin-prism
    terms deleted. Out at the edge of the pinhole's own field the projection
    goes as ``tan``, so the pinhole pixel moves ~3x faster per degree than it
    does on axis and any residual in ``unproject`` is magnified by the same
    factor. This is also the only part of the field where the two arms are
    scored on different physics, so it is the part a regression would reach
    first.
    """
    c = cam896()
    bl = B.RectDerectBaseline(model=None, cam=c, fov_deg=110.0, rect_size=518)
    pts = fisheye_points(c, n=40, margin=0.46)          # out to theta ~53 deg
    got = bl.derectify(true_pinhole_depth(bl.pin), pts)
    ok = np.isfinite(got)
    assert ok.mean() > 0.90, f"only {100 * ok.mean():.0f}% of rim points landed"
    theta = c.theta_of(pts.u[ok], pts.v[ok])
    assert theta.max() > 45.0, f"this grid only reached {theta.max():.0f} deg"
    rel = np.abs(got[ok] - pts.d[ok]) / pts.d[ok]
    assert np.median(rel) < 2e-3, f"median relative error {np.median(rel):.2e}"
    assert np.max(rel) < 3e-2, f"max relative error {np.max(rel):.2e}"


def test_the_run_settings_leave_no_void_for_the_sampler_to_average_over():
    """At 110 deg the pinhole is fully backed by real pixels, and that is load-bearing.

    ``derectify`` reads 2x2 bilinear. Where the render has a black region
    outside the imaged cone, a point next to that region is averaged with the
    model's response to padding, and nothing downstream can tell. It never
    happens at the settings every published run used, because the 896 fisheye
    reaches further than a 110 deg pinhole asks for in every direction — but
    that is a fact about ``--rect-fov``, not a property of the code, so widening
    it must fail here rather than quietly start mixing in padding.

    Measured on the release itself: 100 % of the pinhole grid is backed, and
    zero of 120 028 real ground-truth points have a stencil touching a void.
    """
    c = cam896()
    for size in (512, 518):
        _, _, in_cone = B.RectDerectBaseline(None, c, 110.0, size)._remap()
        assert in_cone.all(), (
            f"a 110 deg / {size} px pinhole left {100 * (~in_cone).mean():.2f}% "
            f"of its grid unbacked; derectify's bilinear read would average "
            f"real depth with the model's answer to black padding")
    # ...and the guard is not vacuous: ask for more than the lens has.
    _, _, wide = B.RectDerectBaseline(None, c, 150.0, 192)._remap()
    assert not wide.all()


def test_a_prediction_on_the_wrong_grid_is_refused_not_sampled():
    """The addresses are the pinhole's. A map on another grid would be read at
    the wrong pixel for every point, which looks like a mediocre model rather
    than a broken harness — the one failure this class is written against."""
    c = cam896()
    bl = B.RectDerectBaseline(model=None, cam=c, fov_deg=110.0, rect_size=518)
    with pytest.raises(SystemExit) as e:
        bl.derectify(np.ones((256, 256), np.float32), fisheye_points(c, n=6))
    assert "518" in str(e.value) and "256" in str(e.value)


def test_points_outside_the_pinhole_field_get_nan_not_a_guess():
    """A 110 deg pinhole cannot cover the whole fisheye cone. The rim points it
    cannot see must be absent, not extrapolated — they are then dropped from
    *both* arms so the comparison stays like-for-like."""
    c = cam896()
    bl = B.RectDerectBaseline(model=None, cam=c, fov_deg=70.0, rect_size=256)
    r = 0.44 * c.width                       # out at the rim of the imaged disc
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pts = D.FramePoints(u=(c.cx + r * np.cos(ang)).astype(np.float32),
                        v=(c.cy + r * np.sin(ang)).astype(np.float32),
                        d=np.full(ang.size, 3.0, np.float32),
                        inv_dist_std=np.zeros(ang.size, np.float32),
                        dist_std=np.zeros(ang.size, np.float32))
    got = bl.derectify(true_pinhole_depth(bl.pin), pts)
    assert np.isnan(got).all(), "rim points must not be given a value"


def test_a_narrow_rectification_gives_up_coverage_and_says_so():
    """The cost of rectifying, made visible: the narrower the pinhole, the fewer
    ground-truth points survive it. The number is what ``common_support``
    reports; here it only has to move in the right direction."""
    c = cam896()
    # Out to theta ~53 deg, so that even a 90 deg pinhole leaves some of it
    # behind; on a grid confined to the inner field, 90 and 120 both cover
    # everything and the comparison has nothing to say.
    pts = fisheye_points(c, n=30, margin=0.46)
    kept = []
    for fov in (60.0, 90.0, 120.0):
        bl = B.RectDerectBaseline(model=None, cam=c, fov_deg=fov, rect_size=256)
        got = bl.derectify(true_pinhole_depth(bl.pin), pts)
        kept.append(float(np.isfinite(got).mean()))
    assert kept[0] < kept[1] < kept[2], kept
    assert kept[0] < 0.5, f"a 60 deg pinhole should lose most of this: {kept[0]}"


def test_the_rectified_render_is_zero_outside_the_imaged_cone():
    """Zero-filled, not border-replicated: a replicated pixel is a fabricated
    observation and the model would be scored on its response to one."""
    c = cam896()
    bl = B.RectDerectBaseline(model=None, cam=c, fov_deg=150.0, rect_size=192)
    frame = np.full((RES, RES, 3), 200, np.uint8)
    out, in_cone = bl.rectify(frame)
    assert not in_cone.all(), "a 150 deg pinhole must overhang this lens"
    assert out[~in_cone].max() == 0, "outside the cone must stay black"
    assert out[in_cone].mean() > 150


def test_raw_baseline_gathers_the_pixel_the_point_names():
    """The raw arm's only geometry is the gather, so that is what it must get
    right. At the ground truth's own grid it is literally ``pred[v, u]``."""
    rng = np.random.default_rng(0)
    pred = rng.random((RES, RES)).astype(np.float32) + 0.5
    u = np.array([0.0, 895.5, 447.4, 12.6], np.float32)
    v = np.array([895.5, 0.0, 200.2, 640.9], np.float32)
    pts = D.FramePoints(u=u, v=v, d=np.ones(4, np.float32),
                        inv_dist_std=np.zeros(4, np.float32),
                        dist_std=np.zeros(4, np.float32))
    vi, ui = pts.index
    assert ui.max() <= RES - 1 and vi.max() <= RES - 1
    assert np.allclose(D.sample_prediction(pred, pts), pred[vi, ui])


def test_rect_derect_refuses_the_per_point_stand_in():
    """The oracle answers per point, which cannot be sampled off a rectified
    map. Refusing beats silently scoring the raw arm's answer under this arm's
    name."""
    from slambench import models as M
    c = cam896()
    model = M.load_model(M.ORACLE, device=None)
    model.key = M.ORACLE
    bl = B.RectDerectBaseline(model=model, cam=c, rect_size=128)
    pts = fisheye_points(c, n=6)
    with pytest.raises(SystemExit):
        bl.predict(np.zeros((RES, RES, 3), np.uint8), pts)


def test_rect_derect_without_a_camera_refuses_rather_than_defaults():
    """There is no nominal Aria calibration to fall back on: a nominal one is
    wrong by more than the effect being measured."""
    with pytest.raises(SystemExit) as e:
        B.build(B.RECT_DERECT, model=None, cam=None)
    assert "calibration" in str(e.value).lower() or "camera" in str(e.value).lower()


def test_the_two_rectifiers_agree():
    """``rectify`` prefers ``projectaria_tools.distort_by_calibration`` — the call
    the Aria ecosystem rectifies with — and falls back to ``cv2.remap`` over this
    module's own geometry. They are independent implementations of the same map,
    so agreeing to interpolation rounding is a real cross-check on the camera
    chain: a wrong crop, scale or quarter turn would separate them immediately.

    Measured on a real ego-synth frame at the time of writing: median 0 levels,
    max 3, on 0-255 uint8.
    """
    cv2 = pytest.importorskip("cv2")
    cam = cam896().rotated(1)
    lib_probe = B._library_rectify(np.zeros((RES, RES, 3), np.uint8),
                                   cam, B.Pinhole(64, 110.0))
    if lib_probe is None:
        pytest.skip("projectaria_tools not installed")

    # A frame with structure at every scale, so a misalignment cannot hide in
    # flat regions. Deterministic.
    j = np.arange(RES, dtype=np.float32)
    xv, yv = np.meshgrid(j, j)
    tex = (127 + 90 * np.sin(xv / 11.0) * np.cos(yv / 13.0)
           + 30 * np.sin((xv + yv) / 3.0))
    frame = np.repeat(np.clip(tex, 0, 255).astype(np.uint8)[:, :, None], 3, 2)

    bl = B.RectDerectBaseline.__new__(B.RectDerectBaseline)
    bl.cam, bl.pin, bl._map = cam, B.Pinhole(256, 110.0), None
    mapx, mapy, in_cone = bl._remap()
    ours = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    theirs = B._library_rectify(frame, cam, bl.pin)

    d = np.abs(ours[in_cone].astype(np.int16) - theirs[in_cone].astype(np.int16))
    assert np.median(d) <= 1, f"median differs by {np.median(d)} levels"
    assert d.max() <= 12, f"max differs by {d.max()} levels — not interpolation"
