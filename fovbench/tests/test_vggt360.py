# Copyright (c) 2026.
"""The lens-aware entry's contract with the FOV harness.

``vggt360`` is the one model in this benchmark that is *given* the lens, and that
buys it three ways to be quietly wrong that none of the vanilla four have. Each
is pinned here, and none needs weights, ADT or a GPU:

1. it can only answer on the raw fisheye, and a run that asked it for the
   rectified arm or the window sweep would otherwise return a full column;
2. it re-renders the frame it is handed, so it needs *that* frame's camera —
   the resized one, on the pixel-centre convention, not a calibration rescaled
   naively;
3. it answers for the imaged cone and not the square frame, and the harness
   fits its alignment affine before it applies its own mask.

The forward pass itself is tested where it lives: ``slambench/tests/
test_vggt360.py`` puts a known field through the warp and the fusion, and
``VGGT-360-fisheye/checks/check_fisheye2persp.py`` does the same on ADT's lens.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fovbench import geometry as G  # noqa: E402
from fovbench import models as M  # noqa: E402
from fovbench import run as RUN  # noqa: E402

from finetune.eval.metrics import align_depth  # noqa: E402
from utils.pipeline import (VGGT360Config, fill_uncovered,  # noqa: E402
                            range_to_planar_z)

FRAME = 176
OUT = 88


def _args(**kw):
    a = RUN.build_parser().parse_args(["--adt-root", "/nonexistent"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# --------------------------------------------------------------------------- #
# 1. Where it may and may not be asked to answer
# --------------------------------------------------------------------------- #

def test_only_the_lens_aware_model_is_restricted():
    assert M.restricted_views(M.VGGT360) == ("fisheye",)
    for k in M.DEFAULT_MODELS + (M.ANALYTIC,):
        assert M.restricted_views(k) is None


@pytest.mark.parametrize("views,protocols,msg", [
    ("rect,fisheye", "radial", "cannot answer on view"),
    ("rect", "radial", "cannot answer on view"),
    ("fisheye", "radial,window", "cannot run the window protocol"),
    ("fisheye", "window", "cannot run the window protocol"),
])
def test_the_undefined_combinations_are_refused_before_anything_loads(
        views, protocols, msg):
    """A rectified pinhole is not an input this model has, and a 40 deg crop is
    not a 55 deg cone. Either would produce plausible numbers about nothing."""
    with pytest.raises(SystemExit, match=msg):
        RUN.run(_args(models=M.VGGT360, views=views, protocols=protocols,
                      out="/tmp/fovbench-should-not-exist"))


def test_the_fisheye_radial_combination_gets_past_the_view_guard():
    """The guard must not be a blanket refusal: this is the cell it exists for.

    It stops at model availability instead, which is the next check and the
    right one on a machine with no weights.
    """
    with pytest.raises(SystemExit, match="cannot run|vggt360"):
        RUN.run(_args(models=M.VGGT360, views="fisheye", protocols="radial",
                      out="/tmp/fovbench-should-not-exist"))


def test_the_default_line_up_is_still_the_four_vanilla_models():
    """Adding a lens-aware fifth to the default would change every published
    command in GPU_EXPERIMENTS.md without any of them being edited."""
    assert M.VGGT360 not in M.DEFAULT_MODELS
    assert M.DEFAULT_MODELS == ("vggt_1b", "vggt_omega", "dav2_large",
                                "da3_large")


# --------------------------------------------------------------------------- #
# 2. The camera the view actually has
# --------------------------------------------------------------------------- #

def test_the_view_carries_the_resized_camera_not_a_rescaled_calibration():
    """``scaled_cam``, not ``aria_cam(N, N)`` — they differ by half a pixel.

    ``cv2.resize`` maps output centre ``j + 0.5`` to source ``(j + 0.5) * s``,
    so the principal point moves by ``0.5 * (1 - s)`` more than a direct scaling
    of the calibration would put it. Under a pixel, and at this working size a
    third of a degree of incidence angle — which is enough to render nine
    tangent views off the axis their errors are then binned by. The model has no
    way to detect being handed the wrong one.
    """
    rgb = np.zeros((FRAME, FRAME, 3), np.uint8)
    gt = np.ones((FRAME, FRAME), np.float32)
    cam = G.aria_cam(FRAME, FRAME)
    fv = G.full_frame_view(rgb, gt, gt > 0, cam, OUT, "fisheye")

    assert fv.cam is not None
    want = G.scaled_cam(cam, OUT)
    assert fv.cam.cx == pytest.approx(want.cx)
    assert fv.cam.cy == pytest.approx(want.cy)
    assert fv.cam.fx == pytest.approx(want.fx)
    naive = G.aria_cam(OUT, OUT)
    assert abs(fv.cam.cx - naive.cx) > 1e-6, (
        "the two conventions coincided, so this test proves nothing here")


def test_the_fisheye_view_carries_the_native_frame_it_was_resampled_from():
    """The nine tangent views are cut from these, not from the resized frame.

    A 60 deg view at 518 px is a 0.62x downsample of the 1408 source and a 1.69x
    *up*sample of the 518 one, so which frame it is cut from decides whether the
    crops carry detail or interpolation. The answer still lands on the resized
    grid, because that is where ``gt_z``, ``valid`` and ``theta`` live — so both
    have to travel, and this pins that they do.
    """
    rgb = np.arange(FRAME * FRAME * 3, dtype=np.uint8).reshape(FRAME, FRAME, 3)
    gt = np.ones((FRAME, FRAME), np.float32)
    cam = G.aria_cam(FRAME, FRAME)
    fv = G.full_frame_view(rgb, gt, gt > 0, cam, OUT, "fisheye")

    assert fv.source_rgb is not None and fv.source_rgb.shape == (FRAME, FRAME, 3)
    assert fv.source_cam is not None and fv.source_cam.W == FRAME
    assert fv.rgb.shape == (OUT, OUT, 3)          # the scoring grid is still OUT
    assert fv.cam.W == OUT
    assert fv.source_rgb is rgb, "the native frame was copied rather than carried"


def test_the_rectified_view_offers_no_fisheye_camera():
    """A pinhole has no fisheye model, and ``None`` is the honest answer.

    Paired with the view guard above, this is why a rect run cannot reach the
    pipeline even if the guard were removed: it would fail loudly at the
    missing lens rather than warp through a wrong one.
    """
    rgb = np.zeros((FRAME, FRAME, 3), np.uint8)
    gt = np.ones((FRAME, FRAME), np.float32)
    fv = G.full_frame_view(rgb, gt, gt > 0, G.aria_cam(FRAME, FRAME), OUT,
                           "rect")
    assert fv.cam is None
    assert fv.source_rgb is None and fv.source_cam is None


def test_a_model_that_needs_the_lens_refuses_to_guess_it():
    m = M.Model(key="x", family="f", size="s", align_mode="scale_shift",
                input_size=OUT, needs_view=True,
                _predict=lambda v: np.ones((OUT, OUT), np.float32))
    with pytest.raises(SystemExit, match="without the view's camera"):
        m.predict(np.zeros((OUT, OUT, 3), np.uint8))

    rgb = np.zeros((FRAME, FRAME, 3), np.uint8)
    gt = np.ones((FRAME, FRAME), np.float32)
    fv = G.full_frame_view(rgb, gt, gt > 0, G.aria_cam(FRAME, FRAME), OUT,
                           "fisheye")
    assert m.predict(fv.rgb, view=fv).shape == (OUT, OUT)


def test_the_vanilla_models_still_ignore_the_camera_they_are_offered():
    """The premise of the whole benchmark: no lens is given to the four."""
    seen = []
    m = M.Model(key="v", family="f", size="s", align_mode="scale_shift",
                input_size=OUT,
                _predict=lambda rgb, gt, th: seen.append((gt, th)) or
                np.ones(rgb.shape[:2], np.float32))
    m.predict(np.zeros((OUT, OUT, 3), np.uint8), view="a whole FrameView")
    assert seen == [(None, None)]


# --------------------------------------------------------------------------- #
# 3. Range, planar z, and the holes
# --------------------------------------------------------------------------- #

def test_the_depth_convention_is_radial_and_no_affine_can_absorb_it():
    """``range`` scored against planar-z GT is not a scale error, it is a curve.

    The residual after the *best possible* affine is what matters: if a
    scale-and-shift could absorb the conversion, skipping it would cost a
    constant and nothing else. It cannot, which is why the conversion happens
    inside the model wrapper rather than being left to the alignment.

    The scene has to earn that. A constant-depth GT would let the affine set
    scale to zero and fit the mean exactly, so the depth here varies with
    position *independently* of incidence angle, as a real one does.

    What the leftover looks like is worth knowing, because it is a standing trap
    in this repository: under one frozen affine a **monotone** radial bias comes
    back **U-shaped**, since least squares parks the scale in the middle of the
    bias range and both ends then rise. So this asserts the residual is large and
    *structured in theta* — not that it increases with theta, which it does not.
    """
    n = 64
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) / ((n - 1) / 2)
    theta = (np.clip(r, 0, 1) * math.radians(55.0)).astype(np.float32)
    cos_t = np.cos(theta).astype(np.float32)
    gt_z = np.clip(2.0 + 1.2 * np.sin(3 * xx / n * np.pi)
                   + 0.8 * np.cos(2 * yy / n * np.pi), 0.8, None).astype(np.float32)

    rng_map = (gt_z / cos_t).astype(np.float32)   # what fusion would produce
    mask = np.ones((n, n), bool)

    right = range_to_planar_z(rng_map, cos_t)
    assert np.abs(align_depth(right, gt_z, mask) - gt_z).mean() < 1e-3

    wrong = align_depth(rng_map, gt_z, mask)      # best affine, unconverted
    assert np.abs(wrong - gt_z).mean() > 0.1, (
        "an affine absorbed the cos(theta) conversion, so this scene does not "
        "exercise the failure the conversion exists to prevent")

    def _profile(p):
        td = np.degrees(theta)
        rel = (p - gt_z) / gt_z
        return np.array([rel[(td >= lo) & (td < hi)].mean()
                         for lo, hi in [(0, 10), (10, 20), (20, 30),
                                        (30, 40), (40, 50), (50, 56)]])

    assert np.ptp(_profile(wrong)) > 0.2, "the leftover is flat in theta"
    assert np.ptp(_profile(align_depth(right, gt_z, mask))) < 1e-3


def test_uncovered_pixels_get_a_constant_and_the_count_comes_back():
    """A NaN inside the harness's own mask breaks the frame's alignment fit.

    The harness owns its validity mask and fits the affine over it *before* the
    model's coverage is known to it, so an honest NaN is not an option on this
    side. What it costs depends on the alignment mode, and neither outcome is
    acceptable: ``scale_shift`` — this model's mode — takes ``lstsq`` down with
    ``LinAlgError`` and ends the run, while ``scale_only`` returns an all-NaN
    frame and simply drops it. Both are pinned, because the fill exists to
    prevent whichever one a future default would hit.

    The fill is a constant, which cannot manufacture a radial trend — only
    dilute one — and the count is returned so a run reports the hole rather
    than absorbing it.
    """
    pred = np.array([[1.0, 2.0], [3.0, np.nan]], np.float32)
    covered = np.array([[True, True], [True, False]])
    filled, n = fill_uncovered(pred, covered)
    assert n == 1
    assert np.isfinite(filled).all()
    assert filled[1, 1] == pytest.approx(2.0)          # median of 1, 2, 3

    gt = np.ones((2, 2), np.float32)
    mask = np.ones((2, 2), bool)
    with pytest.raises(np.linalg.LinAlgError):
        align_depth(pred, gt, mask, mode="scale_shift")
    assert np.isnan(align_depth(pred, gt, mask, mode="scale_only")).any()

    for mode in ("scale_shift", "scale_only"):
        assert np.isfinite(align_depth(filled, gt, mask, mode=mode)).all()


def test_the_fill_is_confined_to_where_the_caller_asks():
    """Outside the cone the harness masks anyway; filling there is wasted work
    and would change the median the in-cone holes are filled with."""
    pred = np.array([[1.0, np.nan], [np.nan, 4.0]], np.float32)
    covered = np.array([[True, False], [False, True]])
    where = np.array([[True, True], [False, False]])
    filled, n = fill_uncovered(pred, covered, where=where)
    assert n == 1
    assert filled[0, 1] == pytest.approx(2.5)
    assert np.isnan(filled[1, 0]), "a pixel outside `where` was filled"


# --------------------------------------------------------------------------- #
# The configuration that is being called "our model"
# --------------------------------------------------------------------------- #

def test_the_shipped_defaults_are_the_sixty_degree_layout():
    """``--models vggt360`` with no other flag must be main_adt.py's model."""
    a = RUN.build_parser().parse_args(["--adt-root", "/x"])
    assert (a.vggt360_fov, a.vggt360_ring_tilt, a.vggt360_n_ring) == (60.0, 26.0, 8)
    assert (a.vggt360_fuse, a.vggt360_head, a.vggt360_dtype) == ("attn", "depth",
                                                                 "bf16")
    assert not a.vggt360_no_adaptive and not a.vggt360_no_sa_mask

    cfg = VGGT360Config()
    assert (cfg.fov, cfg.ring_tilt, cfg.n_ring) == (a.vggt360_fov,
                                                    a.vggt360_ring_tilt,
                                                    a.vggt360_n_ring)

    # The view is rendered straight onto the backbone's token grid, so that
    # VGGT's own preprocessing has nothing left to resample. 512 -- main_adt's
    # value -- is not a multiple of 14, and load_and_preprocess_images (mode
    # "crop", target 518) would bicubic it up by 1.0117x on every one of the
    # nine views for no gain.
    assert cfg.persp_size == a.vggt360_persp_size == M.native_size(M.VGGT360)
    assert cfg.persp_size % 14 == 0

    # The views are cut from ADT's own 1408 frame, not from the 518 the other
    # four models get. Matching that 518 would measure a handicapped version of
    # the method and report it as the method.
    assert a.vggt360_source == "native"
    # The layout rule: the ring's outer edge reaches the Aria usable cone.
    assert cfg.covers_cone(54.83) == pytest.approx(1.0, abs=0.03)
