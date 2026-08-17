# Copyright (c) 2026.
"""The VGGT-360 arm's geometry, on a lens the port was not written for.

``vggt360`` is the only baseline whose correctness is a *geometric* claim rather
than a plumbing one: it warps a FISHEYE624 frame into tangent views, runs a
network on them, and fuses the answer back through the inverse of the same lens.
If either direction is wrong, the arm still returns a full array of plausible
depths and simply scores badly — which reads as a mediocre method, not as a bug.
That failure mode is what this file exists against.

So the decisive test here fuses a **known** field round-trip through the real
FISHEYE624 model and checks the value comes back, with no network anywhere. It
is the ego-synth counterpart of ``checks/check_fisheye2persp.py``'s test C, which
makes the same argument for the KB4 lens on ADT.

CPU-only: no weights, no ego-synth, no torch, under a few seconds.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from slambench import baselines as B  # noqa: E402
from slambench import camera as C  # noqa: E402
from slambench import vggt360 as V  # noqa: E402
from slambench.tests.test_camera import PARAMS  # noqa: E402

from utils.fisheye_fusion import fuse_views_to_fisheye  # noqa: E402
from utils.fisheye_views import base_views, fisheye_to_persp  # noqa: E402
from utils.pipeline import VGGT360Config  # noqa: E402

SIZE = 256


def _cam(rotation: int = 0) -> C.Fisheye624:
    return C.Fisheye624(PARAMS, 2880, 2880, rotation=rotation,
                        dataset="aea", take="t").rescale(SIZE)


def _lens(rotation: int = 0) -> V.Fisheye624Lens:
    return V.Fisheye624Lens(_cam(rotation))


# --------------------------------------------------------------------------- #
# The lens itself
# --------------------------------------------------------------------------- #

def test_the_bulk_projection_is_the_same_lens_as_the_per_point_one():
    """``project_bulk`` is a calling convention, not a second camera model.

    The whole reason it exists is that ``project`` loops one point at a time for
    ``projectaria_tools``' binding, which a 21-million-ray warp cannot afford. If
    the two ever disagreed, every VGGT-360 number would be measured through a
    lens no other arm uses, and nothing else in the suite would notice.
    """
    cam = _cam()
    rng = np.random.default_rng(0)
    th = rng.uniform(0, math.radians(55), 400)
    az = rng.uniform(0, 2 * math.pi, 400)
    d = np.stack([np.sin(th) * np.cos(az), np.sin(th) * np.sin(az),
                  np.cos(th)], axis=-1)

    u0, v0 = cam.project(d)
    u1, v1 = cam.project_bulk(d)
    assert np.nanmax(np.hypot(u0 - u1, v0 - v1)) < 1e-9

    d0 = cam.unproject(u0, v0)
    d1 = cam.unproject_bulk(u0, v0)
    assert np.nanmax(np.linalg.norm(d0 - d1, axis=-1)) < 1e-9


def test_the_bulk_projection_keeps_the_shape_a_warp_hands_it():
    """``(H, W, 3)`` rays in, two ``(H, W)`` maps out — cv2.remap's shape."""
    cam = _cam()
    rays = cam.unproject_bulk(*np.meshgrid(np.arange(SIZE, dtype=float),
                                           np.arange(SIZE, dtype=float)))
    assert rays.shape == (SIZE, SIZE, 3)
    u, v = cam.project_bulk(rays)
    assert u.shape == v.shape == (SIZE, SIZE)


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_the_imaged_cone_does_not_depend_on_how_the_frame_is_stored(rotation):
    """A quarter turn permutes the margins; it cannot change the lens' FOV.

    Getting this wrong is quiet and total: the layout is sized against
    ``theta_max``, so a cone that collapsed under rotation would produce views
    that image nothing and a fusion that covers no pixel — on the two datasets
    whose orientation is *not* in ``VERIFIED_ROTATION`` and would be tried first.
    """
    assert _cam(rotation).max_imaged_theta() == pytest.approx(
        _cam(0).max_imaged_theta(), rel=1e-12)


def test_dropping_the_non_radial_terms_costs_more_than_a_verified_camera_may():
    """Why the real lens, rather than a KB4 fitted to it. Measured, not asserted.

    A KB4 is radially symmetric and FISHEYE624 is not, so any fit of one to the
    other carries the tangential and thin-prism terms as residual. This measures
    that residual directly — the full model against its own radial part — and the
    bar it has to clear is the repository's own: ``verify_camera`` accepts a
    camera at **0.29 px**, so a lens description that is further out than that
    cannot be the one a warp runs through.

    At ego-synth's 896 frame this comes to about 1.4 px on the suite's test
    calibration. The size is calibration-dependent and a given take may be
    milder; the point is that it is not known to be small, and using the take's
    own model costs nothing but a ``project`` hook.
    """
    cam = C.Fisheye624(PARAMS, 2880, 2880).rescale(896)
    th = np.radians(np.linspace(0.0, 55.0, 60))
    worst = 0.0
    for az in np.linspace(0.0, 2 * np.pi, 16, endpoint=False):
        d = np.stack([np.sin(th) * np.cos(az), np.sin(th) * np.sin(az),
                      np.cos(th)], axis=-1)
        u, v = cam.project_bulk(d)
        rxy = np.hypot(d[..., 0], d[..., 1])
        inv = np.divide(1.0, rxy, out=np.zeros_like(rxy), where=rxy > 1e-12)
        thd = th * cam._radial(th)
        ur = cam.f * thd * d[..., 0] * inv + cam.cx
        vr = cam.f * thd * d[..., 1] * inv + cam.cy
        worst = max(worst, float(np.nanmax(np.hypot(u - ur, v - vr))))
    assert worst > 0.29, (
        f"the non-radial terms displace by only {worst:.3f} px here, inside the "
        f"tolerance verify_camera accepts — if that holds on real calibrations, "
        f"the argument in vggt360.py for the project hook is weaker than stated")


def test_the_cone_is_derived_from_the_lens_and_not_borrowed_from_aria():
    """54.83 deg is ADT's nominal Aria answer, and this is a different lens."""
    t = math.degrees(_lens().theta_max())
    assert 30.0 < t < 90.0
    assert abs(t - 54.83) > 1e-3, (
        "the FISHEYE624 cone came back at ADT's nominal KB4 value, which means "
        "it was assumed rather than derived")


# --------------------------------------------------------------------------- #
# The round trip — the claim that matters
# --------------------------------------------------------------------------- #

def test_a_known_field_survives_the_warp_and_the_fusion_on_a_fisheye624_lens():
    """Render a known per-ray field into the 60 deg layout, fuse it back.

    No network: the "prediction" for each view is the exact field sampled on
    that view's own grid, so anything the round trip loses is geometry. A
    forward/inverse mismatch — the failure this arm is most exposed to, because
    it is the only one that warps in both directions — shows up here as a large
    residual, while a real run would only score badly.

    The field is a smooth, strictly monotone function of incidence angle, so it
    is sensitive to exactly the radial error a mis-fitted lens would introduce.

    What it is sensitive *enough* for, measured: the true lens returns 0.0004,
    and a principal point 3 px out returns 0.015 — a clean fail. Dropping only
    the tangential and prism terms returns 0.001, which this threshold does
    **not** catch, so that specific claim is pinned separately and in pixels by
    ``test_dropping_the_non_radial_terms_costs_more_than_a_verified_camera_may``.
    This test's job is the forward/inverse consistency of whatever lens it is
    given, not the choice of lens.
    """
    lens = _lens()
    cfg = VGGT360Config()
    rays, cone = lens.ray_lut()
    theta = np.arccos(np.clip(rays[..., 2], -1.0, 1.0))
    field = (1.0 + 2.0 * np.sin(theta)).astype(np.float32)   # 1.0 .. ~2.7

    params = base_views(cfg.fov, cfg.ring_tilt, cfg.n_ring)
    values, valids = [], []
    for (psi, tilt, fov) in params:
        v, valid = fisheye_to_persp(field, lens, psi, tilt, fov,
                                    height=128, width=128, supersample=1,
                                    project=lens.project)
        values.append(v)
        valids.append(valid)

    fused, coverage = fuse_views_to_fisheye(
        values, params, lens, view_valids=valids, interp="linear",
        erode_valid_px=1, ray_lut=(rays, cone))

    got = cone & (coverage > 0) & np.isfinite(fused) & (fused > 0)
    # The layout must actually cover the lens it was handed, or the residual
    # below is measured on whatever fraction happened to survive.
    assert got.sum() / cone.sum() > 0.97, (
        f"the 60 deg layout covered only {got.sum() / cone.sum():.1%} of a "
        f"{lens.theta_max_deg:.1f} deg cone")
    rel = np.abs(fused[got] - field[got]) / field[got]
    assert float(rel.mean()) < 0.01, f"mean relative error {rel.mean():.4f}"


def test_the_layout_report_says_when_the_ring_does_not_reach_the_rim():
    """The ADT layout on a wider lens leaves an annulus, and must say so.

    ego-synth is Aria too, but a per-take calibration is not the nominal one. A
    ring that stops short covers the rim with the centre view's corners alone,
    which would read as the method degrading at eccentricity when it is the
    layout not arriving — the same shape as the dead-pixel confound this repo
    has already been caught by once.
    """
    lens = _lens()
    tight = VGGT360Config(fov=60.0, ring_tilt=26.0)
    assert tight.covers_cone(lens.theta_max_deg) < 1.0
    assert "short of it" in V.layout_report(tight, lens)

    wide = VGGT360Config(fov=60.0, ring_tilt=lens.theta_max_deg - 30.0 + 1.0)
    assert "reaches the rim" in V.layout_report(wide, lens)


# --------------------------------------------------------------------------- #
# The cache, which must be a cache and nothing else
# --------------------------------------------------------------------------- #

def test_the_view_map_cache_returns_the_same_maps_it_would_have_built():
    """Caching is a speed decision, so it must not be a numerical one."""
    lens = _lens()
    first = lens.maps_for(45.0, 26.0, 60.0, 64)
    again = lens.maps_for(45.0, 26.0, 64 * 0 + 60.0, 64)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert first[0] is again[0], "the second call rebuilt the maps"

    fresh = V.Fisheye624Lens(_cam()).maps_for(45.0, 26.0, 60.0, 64)
    assert all(np.array_equal(a, b) for a, b in zip(first, fresh))


def test_maps_built_for_another_view_are_refused_rather_than_sampled():
    """Wrong-shaped maps address real pixels, so they cannot fail quietly."""
    lens = _lens()
    img = np.zeros((SIZE, SIZE), np.float32)
    wrong = lens.maps_for(0.0, 0.0, 60.0, 64)
    with pytest.raises(ValueError, match="cached maps"):
        fisheye_to_persp(img, lens, 0.0, 0.0, 60.0, height=128, width=128,
                         supersample=1, project=lens.project, maps=wrong)


# --------------------------------------------------------------------------- #
# The arm's contract with the harness
# --------------------------------------------------------------------------- #

def test_the_published_default_baselines_do_not_include_the_new_arm():
    """Adding it to the default would change what every existing command means."""
    assert B.VGGT360 in B.BASELINES
    assert B.VGGT360 not in B.DEFAULT_BASELINES
    assert B.DEFAULT_BASELINES == (B.RAW, B.RECT_DERECT)


def test_the_arm_refuses_a_temporal_context_instead_of_widening_the_pass():
    """N frames would be 9N views in one pass while the column header said N."""
    lens_cam = _cam()

    class _Pipe:                      # never reached; the refusal is first
        def range_map(self, *a, **k):
            raise AssertionError("the context should have been refused")

    arm = B.VGGT360Baseline(None, lens_cam, _Pipe(), VGGT360Config())
    frames = [np.zeros((SIZE, SIZE, 3), np.uint8) for _ in range(3)]
    with pytest.raises(SystemExit, match="one frame at a time"):
        arm.predict(frames, None, target=0)


def test_building_the_arm_without_a_camera_or_a_pipeline_is_refused():
    with pytest.raises(SystemExit, match="camera model"):
        B.build(B.VGGT360, None, cam=None)
    with pytest.raises(SystemExit, match="loaded pipeline"):
        B.build(B.VGGT360, None, cam=_cam(), vggt360_pipe=None)
