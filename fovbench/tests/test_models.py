# Copyright (c) 2026.
"""The four vanilla models resolve, and the registry stays honest without weights.

Nothing here loads a network — that needs the GPU box. What it does check is
that the benchmark's model set is reachable by the names the CLI documents, that
availability reporting degrades to an instruction rather than an exception when a
dependency is missing, and that the depth-space alignment the VGGT family and
DA3 need is what the registry hands them.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from finetune.eval.baselines import model_zoo as zoo  # noqa: E402
from fovbench import models as M  # noqa: E402


def test_the_four_vanilla_models_resolve_by_their_cli_names():
    keys = [s.key for s in zoo.get_specs(list(M.DEFAULT_MODELS))]
    assert keys == list(M.DEFAULT_MODELS)


def test_vggt_family_and_da3_align_in_depth_space_not_disparity():
    """They emit DEPTH, up to scale. A disparity-space affine (the MiDaS/DAv2
    protocol) fits the wrong quantity, and a median scale cannot absorb their
    offset."""
    for key in ("vggt_1b", "vggt_omega", "da3_large", "da3_small"):
        spec = zoo.get_specs([key])[0]
        assert spec.align_modes == ("none", "scale_shift"), key


def test_dav2_keeps_the_disparity_protocol():
    spec = zoo.get_specs(["dav2_large"])[0]
    assert spec.align_modes == ("none", "disparity_scale_shift")


def test_status_reports_missing_dependencies_without_raising():
    for spec in zoo.get_specs(list(M.DEFAULT_MODELS)):
        state, detail = zoo.status(spec)
        assert state in ("ready", "download", "unavailable")
        assert detail                      # never a bare state with no instruction


def test_missing_vggt_omega_checkpoint_is_unavailable_not_downloadable():
    """The weights are gated: telling a user to run ``--download`` would send
    them into a 401 rather than to the access request."""
    spec = zoo.get_specs(["vggt_omega"])[0]
    if os.path.isfile(spec.ref):
        pytest.skip("checkpoint present on this machine")
    state, detail = zoo.status(spec)
    assert state == "unavailable"
    assert "huggingface.co/facebook/VGGT-Omega" in detail


def test_build_adapter_routes_the_three_backbone_kinds():
    for key in ("vggt_1b", "vggt_omega", "da3_large"):
        spec = zoo.get_specs([key])[0]
        assert isinstance(zoo.build_adapter(spec), zoo.BackboneAdapter)


def test_patch_align_rounds_to_the_token_grid():
    assert zoo.patch_align(518, 518, 14) == (518, 518)
    assert zoo.patch_align(512, 512, 16) == (512, 512)
    assert zoo.patch_align(500, 500, 14) == (504, 504)
    assert zoo.patch_align(3, 3, 14) == (14, 14)       # never degenerate


def test_native_size_matches_each_models_token_grid():
    """Views are rendered at the model's own input size so nothing is resampled
    between the view construction and the network — a resize would change the
    view's effective field of view, which is the variable under study."""
    assert M.native_size("vggt_1b") % 14 == 0
    assert M.native_size("vggt_omega") % 16 == 0
    assert M.native_size("da3_large") % 14 == 0
    assert M.native_size("dav2_large") % 14 == 0


def test_analytic_model_is_a_usable_stand_in_for_a_real_one():
    """``--models analytic`` runs the whole harness with no weights and no GPU,
    which is how the pipeline is exercised on a laptop. It must behave like a
    model: take an image, return a positive planar-z map of the same size."""
    m = M.load_model("analytic", device=None)
    rgb = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    d = m.predict(rgb)
    assert d.shape == (64, 64)
    assert np.isfinite(d).all() and (d > 0).all()


def test_analytic_model_can_inject_a_known_radial_bias():
    """It is also the end-to-end fixture for the metric: given GT it returns GT
    warped by a known function of eccentricity, so the reported curve can be
    checked against the bias that was put in."""
    gt = np.full((64, 64), 2.0, np.float32)
    theta = np.linspace(0, 55, 64, dtype=np.float32)[None, :].repeat(64, 0)
    m = M.load_model("analytic", device=None, radial_bias=0.6)
    d = m.predict(np.zeros((64, 64, 3), np.uint8), gt_z=gt, theta_deg=theta)
    assert d[:, 0].mean() == pytest.approx(2.0, rel=1e-3)
    # 1 + 0.6 * (55 deg in rad)^2 = 1.55
    assert d[:, -1].mean() / d[:, 0].mean() == pytest.approx(1.553, rel=1e-2)
