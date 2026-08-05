"""Unit tests for the RayTun3R reproduction.

Focused on the properties the paper's equations actually assert, so a regression
shows up as a failed invariant rather than a slightly different number:

* the pinhole-bias premise of Sec. 3 (Fig. 2),
* zero-init adapters being exact no-ops (Sec. 4.2),
* the losses vanishing at ground-truth geometry (Eq. 8, 9),
* the metrics being scale-invariant where Eq. 16-18 say they are.

Run with ``pytest raytun3r/tests``. The backbone tests build tiny random models
and take a few seconds; everything else is instant.
"""

from __future__ import annotations

import math

import pytest
import torch

from raytun3r import corrections as C
from raytun3r.adapter import RadialAngularPE, RadialRoPE, RayTun3RAdapter, patch_polar_coords
from raytun3r.backbones import Prediction
from raytun3r.cameras import EUCM, KannalaBrandt, Pinhole, convert_depth, pixel_grid
from raytun3r.losses import (backproject, l2_penalty, pose_loss, reprojection_loss,
                             smoothness_loss, tv_penalty)
from raytun3r.matching import Matches, mean_flow_magnitude, relative_pose_magsac
from raytun3r.metrics import (align_scale, depth_metrics, reprojection_depth_error,
                              rotation_error_deg, translation_error_deg)
from raytun3r.testing import toy_camera


# --------------------------------------------------------------------- cameras


@pytest.mark.parametrize("cam", [
    Pinhole(fx=300, fy=300, cx=63.5, cy=63.5, width=128, height=128,
            theta_max=math.radians(50)),
    toy_camera(128, 128, fov_deg=180.0),
    EUCM(fx=60, fy=60, cx=63.5, cy=63.5, width=128, height=128,
         alpha=0.6, beta=1.1, theta_max=math.radians(85)),
])
def test_projection_round_trip(cam):
    uv = pixel_grid(cam.height, cam.width)
    m = cam.valid_mask()
    back = cam.project(cam.unproject(uv))
    assert float((back - uv)[m].abs().max()) < 1e-3


def test_rays_are_unit_norm():
    cam = toy_camera(64, 64, fov_deg=180.0)
    n = cam.ray_grid().norm(dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5)


def test_pinhole_jacobian_is_constant_but_fisheye_is_not():
    """Sec. 3 "On pinhole bias" -- the premise the whole method rests on."""
    ph = Pinhole(fx=300, fy=300, cx=63.5, cy=63.5, width=128, height=128,
                 theta_max=math.radians(50))
    fe = toy_camera(128, 128, fov_deg=180.0)
    r = ((pixel_grid(128, 128) - torch.tensor([63.5, 63.5])) ** 2).sum(-1).sqrt()
    inner, outer = r < 15, (r > 40) & (r < 55)

    s_ph = torch.linalg.svdvals(ph.backproject_jacobian())[..., 0]
    s_fe = torch.linalg.svdvals(fe.backproject_jacobian())[..., 0]
    assert abs(float(s_ph[outer].mean() / s_ph[inner].mean()) - 1.0) < 1e-3
    assert float(s_fe[outer].mean() / s_fe[inner].mean()) > 2.0


def test_jacobian_finite_at_principal_point():
    """unproject divides by the image radius; the centre pixel must not be NaN."""
    cam = toy_camera(127, 127, fov_deg=180.0)
    J = cam.backproject_jacobian()
    assert bool(torch.isfinite(J).all())


def test_resized_camera_preserves_rays():
    cam = toy_camera(128, 128, fov_deg=180.0)
    small = cam.resized(64, 64)
    a = cam.incidence_grid(128, 128)[64, 64]
    b = small.incidence_grid(64, 64)[32, 32]
    assert abs(float(a - b)) < 0.02


# --------------------------------------------------------------------- adapter


def test_param_count_matches_paper():
    """Abstract: "only 10,752 trainable parameters" on DA3-Small (C = 384)."""
    bd = RayTun3RAdapter(384).param_breakdown()
    assert bd["pe_radial"] == 20 * 384
    assert bd["pe_angular"] == 8 * 384
    assert bd["pe_radial"] + bd["pe_angular"] == 10752
    assert bd["rope_radial"] == 20


def test_zero_init_residual_is_zero():
    cam = toy_camera(112, 112, fov_deg=180.0)
    ad = RayTun3RAdapter(32).bind(cam, 8, 8, 14)
    assert float(ad.pe_residual().abs().max()) == 0.0


def test_rope_correction_is_a_rotation():
    rope = RadialRoPE(20)
    with torch.no_grad():
        rope.delta_r.normal_(0, 0.5)
    rho = torch.rand(64)
    a = torch.linspace(0, 6, 32).expand(64, 32).contiguous()
    sin, cos = torch.sin(a), torch.cos(a)
    s2, c2 = rope.rotate(sin, cos, rho)
    assert float((s2 ** 2 + c2 ** 2 - 1).abs().max()) < 1e-5


def test_rope_token_rotation_preserves_norm_within_each_axial_half():
    rope = RadialRoPE(20)
    with torch.no_grad():
        rope.delta_r.normal_(0, 0.5)
    tokens = torch.randn(2, 4, 25, 64)
    out = rope.rotate_tokens(tokens, torch.rand(25))
    a = tokens.reshape(2, 4, 25, 2, 32).norm(dim=-1)
    b = out.reshape(2, 4, 25, 2, 32).norm(dim=-1)
    assert torch.allclose(a, b, atol=1e-4)


def test_angular_term_vanishes_at_the_centre():
    """Eq. 5's rho factor: the angle is ill-defined at the principal point."""
    cam = toy_camera(126, 126, fov_deg=180.0)
    pe = RadialAngularPE(16, 20, 8)
    with torch.no_grad():
        pe.delta_theta.normal_(0, 1.0)          # radial table stays zero
    rho, theta = patch_polar_coords(cam, 9, 9, 14)
    res = pe(rho, theta)
    assert float(res[4, 4].abs().max()) < 1e-6   # centre patch
    assert float(res[0, 4].abs().max()) > 1e-3   # rim patch


def test_lookup_interpolation_endpoints():
    pe = RadialAngularPE(4, n_radial=3, n_angular=0)
    with torch.no_grad():
        pe.t_r.copy_(torch.tensor([[0.0] * 4, [1.0] * 4, [2.0] * 4]))
    out = pe(torch.tensor([0.0, 0.5, 1.0]), torch.zeros(3))
    assert torch.allclose(out[:, 0], torch.tensor([0.0, 1.0, 2.0]), atol=1e-5)


# ------------------------------------------------------- param-free corrections


def test_patch_undistortion_identity_on_axis():
    cam = toy_camera(126, 126, fov_deg=180.0)
    J = C.local_undistort_jacobian(cam, 126, 126, 14)
    assert torch.allclose(J[4, 4], torch.eye(2), atol=2e-3)
    assert abs(float(J[0, 4].det()) - 1.0) > 0.05     # anisotropic at the rim


def test_preserve_scale_normalises_the_determinant():
    cam = toy_camera(126, 126, fov_deg=180.0)
    g_scaled = C.patch_undistort_grid(cam, 126, 126, 14, preserve_scale=True)
    g_raw = C.patch_undistort_grid(cam, 126, 126, 14, preserve_scale=False)
    assert bool(torch.isfinite(g_scaled).all()) and bool(torch.isfinite(g_raw).all())
    assert not torch.allclose(g_scaled, g_raw)


def test_border_tokens_use_the_mean_valid_token():
    valid = torch.tensor([True, True, False, True])
    tok = torch.tensor([[[1.0], [3.0], [99.0], [5.0]]])
    out = C.fill_border_tokens(tok, valid)
    assert float(out[0, 2, 0]) == pytest.approx(3.0)     # mean of 1, 3, 5
    assert torch.equal(out[0, valid], tok[0, valid])


def test_border_fill_is_a_noop_when_all_valid():
    tok = torch.randn(2, 5, 3)
    assert torch.equal(C.fill_border_tokens(tok, torch.ones(5, dtype=torch.bool)), tok)


def test_dpt_grid_matches_reference_span_and_is_non_uniform():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "VGGT-360-fisheye"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from vggt_visfeat.heads.utils import create_uv_grid

    cam = toy_camera(112, 112, fov_deg=180.0)
    ours = C.camera_aware_uv_grid(cam, 8, 8)
    ref = create_uv_grid(8, 8, aspect_ratio=1.0)
    assert ours.shape == ref.shape

    # The grid is normalised over cells inside the imaged cone, so *those* match
    # the span the pretrained head expects. Corner cells fall outside the cone of
    # a 180 deg lens and are allowed to overshoot.
    theta = cam.incidence_grid(cam.height, cam.width)
    idx = ((torch.arange(8) + 0.5) * (cam.height / 8) - 0.5).round().long().clamp(0, cam.height - 1)
    inside = theta[idx][:, idx] <= cam.theta_max
    assert float(ours[..., 0][inside].abs().max()) == pytest.approx(
        float(ref[..., 0].abs().max()), abs=1e-4)

    d = ours[4, :, 0].diff()                        # x coordinate along a row
    assert bool((d > 0).all())                      # monotone
    assert float(d.std() / d.mean()) > 1e-3         # but not uniform


def test_dpt_grid_layout_matches_reference_on_a_non_square_grid():
    """(H, W, 2), not (W, H, 2) -- the shape ``_apply_pos_embed`` adds to x.

    ``create_uv_grid``'s docstring says ``(width, height, 2)`` but it builds the
    grid with ``indexing="xy"`` and so returns ``(height, width, 2)``. Trusting
    the docstring is a shape error on a non-square token grid and a silent x/y
    transpose on a square one, which is why this has to be checked off-square.
    """
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "VGGT-360-fisheye"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from vggt_visfeat.heads.utils import create_uv_grid

    gh, gw = 24, 36
    cam = toy_camera(gh * 14, gw * 14, fov_deg=100.0)
    ours = C.camera_aware_uv_grid(cam, gw, gh, aspect_ratio=gw / gh)
    ref = create_uv_grid(gw, gh, aspect_ratio=gw / gh)
    assert ours.shape == ref.shape == (gh, gw, 2)

    # x sweeps along a row, y sweeps down a column. Both also drift slightly on
    # the other axis -- that radial coupling is exactly what the correction adds
    # over the separable pinhole grid -- so assert that the sweep dominates the
    # drift by a wide margin, which a transposed grid cannot do.
    assert bool((ours[3, :, 0].diff() > 0).all())
    assert bool((ours[:, 3, 1].diff() > 0).all())
    assert float(ours[3, :, 0].std()) > 20 * float(ours[:, 3, 0].std())
    assert float(ours[:, 3, 1].std()) > 20 * float(ours[3, :, 1].std())


def test_wide_lens_falls_back_to_angular_grid():
    """A 180 deg lens must not saturate under the tan() map (Appendix C failure)."""
    cam = toy_camera(112, 112, fov_deg=180.0)
    tan = C.camera_aware_uv_grid(cam, 8, 8, mode="tan")
    auto = C.camera_aware_uv_grid(cam, 8, 8, mode="auto")
    ang = C.camera_aware_uv_grid(cam, 8, 8, mode="angular")
    assert torch.allclose(auto, ang)
    assert float(tan[4, :, 0].diff().min()) < float(ang[4, :, 0].diff().min())


# ---------------------------------------------------------------------- losses


@pytest.fixture(scope="module")
def scene():
    """Exact geometry: a known depth map and pose, and matches derived from them."""
    torch.manual_seed(0)
    h = w = 112
    cam = toy_camera(h, w, fov_deg=180.0)
    valid = cam.valid_mask()
    rng = 2.0 + torch.rand(h, w) * 3.0

    ax = torch.tensor([0.10, -0.05, 0.03])
    ang = ax.norm()
    u = ax / ang
    K = torch.tensor([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    R = torch.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
    t = torch.tensor([0.30, 0.10, -0.20])

    X = backproject(rng, cam, convention="range")
    uv = cam.project(X @ R.T + t)
    inb = ((uv[..., 0] >= 0) & (uv[..., 0] <= w - 1)
           & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1))
    return cam, rng, R, t, Matches(target=uv, weight=(valid & inb).float()), valid


def test_depth_conventions_are_related_by_one_over_cos():
    """The pairing Eq. 7 requires: D and kappa^-1 must describe the same thing.

    ``backproject(z_map, "z")`` and ``backproject(z_map/cos, "range")`` are the
    same point cloud. Pinning the *relationship* rather than the choice means a
    future silent switch of either side shows up here.
    """
    cam = toy_camera(112, 112, fov_deg=170.0)
    z = 1.0 + torch.rand(112, 112)
    # Restrict to where planar z is physically meaningful. This lens grazes 90 deg
    # at the frame corner, and there 1/cos exceeds 1700 -- the equivalence still
    # holds but float32 has no significant digits left to show it in. That
    # degeneracy is exactly why "range" is the default convention.
    cone = cam.valid_mask(112, 112) & (cam.ray_grid(112, 112)[..., 2] > math.cos(math.radians(80)))

    rng = convert_depth(z, cam, src="z", dst="range")
    X_z = backproject(z, cam, convention="z")
    X_r = backproject(rng, cam, convention="range")
    assert bool(cone.any())
    assert torch.allclose(X_z[cone], X_r[cone], rtol=1e-5, atol=1e-6)

    # and the conversion is a genuine round trip
    assert torch.allclose(convert_depth(rng, cam, src="range", dst="z")[cone],
                          z[cone], rtol=1e-5, atol=1e-6)


def test_depth_conversion_is_radial_not_global():
    """1/cos varies across the frame, so no single scale can absorb it.

    This is why the mismatch had to be fixed rather than left to the scale
    alignment in Eq. 16-18, which solves for one scalar per pair.
    """
    cam = toy_camera(112, 112, fov_deg=170.0)
    cone = cam.valid_mask(112, 112)
    ones = torch.ones(112, 112)
    ratio = convert_depth(ones, cam, src="z", dst="range")[cone]
    assert float(ratio.min()) == pytest.approx(1.0, abs=1e-3)
    assert float(ratio.max()) > 3.0            # a 170 deg lens reaches ~11x at the corner


def test_prediction_convention_guard_rejects_a_mismatch():
    """A wrong convention is invisible without this guard: both are (H, W) depths."""
    pred = Prediction(depth=torch.ones(1, 8, 8), conf=torch.ones(1, 8, 8),
                      R=torch.eye(3)[None], t=torch.zeros(1, 3),
                      depth_convention="z")
    pred.require_convention("z")               # no raise
    with pytest.raises(ValueError, match="range"):
        pred.require_convention("range")


def test_reprojection_loss_zero_at_truth(scene):
    cam, rng, R, t, m, valid = scene
    loss, wsum = reprojection_loss(rng, R, t, m, cam, valid=valid)
    assert float(wsum) > 0
    assert float(loss) < 1e-3


def test_reprojection_loss_penalises_wrong_depth(scene):
    cam, rng, R, t, m, valid = scene
    good, _ = reprojection_loss(rng, R, t, m, cam, valid=valid)
    bad, _ = reprojection_loss(rng * 1.2, R, t, m, cam, valid=valid)
    assert float(bad) > float(good) + 0.1


def test_pose_loss_zero_at_truth_and_positive_otherwise(scene):
    _, _, R, t, _, _ = scene
    assert float(pose_loss(R, t, R, t)) < 1e-3
    assert float(pose_loss(torch.eye(3), t, R, t)) > 1e-2
    assert float(pose_loss(R, -t, R, t)) > 3.0        # opposite direction ~ pi


def test_pose_loss_is_translation_scale_invariant(scene):
    _, _, R, t, _, _ = scene
    assert float(pose_loss(R, t * 7.3, R, t)) < 1e-3


def test_smoothness_prefers_flat_depth(scene):
    cam, rng, _, _, _, valid = scene
    img = torch.rand(3, *rng.shape)
    flat = torch.ones_like(rng) * 3.0
    assert float(smoothness_loss(flat, img, valid=valid)) < \
           float(smoothness_loss(rng, img, valid=valid))


def test_regularisers_zero_at_zero_residual():
    res = torch.zeros(64, 16)
    assert float(l2_penalty(res)) == 0.0
    assert float(tv_penalty(res, (8, 8))) == 0.0


def test_tv_uses_the_absolute_table_when_available():
    """Eq. 12 is written on P', not on the residual."""
    res = torch.zeros(64, 4)
    table = torch.randn(64, 4)
    assert float(tv_penalty(res, (8, 8), table)) > 0.0


# --------------------------------------------------------------------- metrics


def test_pose_errors_are_exact(scene):
    _, _, R, t, _, _ = scene
    assert rotation_error_deg(R, R) < 1e-4
    assert translation_error_deg(t, t) < 1e-4
    assert translation_error_deg(-t, t) == pytest.approx(180.0, abs=1e-2)


def test_d_reproj_zero_and_scale_invariant(scene):
    cam, rng, R, t, m, valid = scene
    a = reprojection_depth_error(rng, cam, m, R, t, valid=valid)
    b = reprojection_depth_error(rng * 4.0, cam, m, R, t, valid=valid)
    assert a < 1e-2 and b < 1e-2


def test_d_reproj_detects_wrong_shape(scene):
    cam, rng, R, t, m, valid = scene
    warped = rng * (1.0 + 0.4 * torch.rand_like(rng))
    assert reprojection_depth_error(warped, cam, m, R, t, valid=valid) > 1e-2


def test_depth_metrics_scale_invariant(scene):
    _, rng, _, _, _, valid = scene
    d = depth_metrics(rng * 2.5, rng, valid=valid)
    assert d["AbsRel"] < 1e-5 and d["delta_1.25"] > 0.999
    assert d["scale"] == pytest.approx(0.4, rel=1e-3)


def test_align_scale_recovers_a_known_factor():
    g = torch.rand(1000) + 1.0
    assert align_scale(g * 3.0, g) == pytest.approx(1 / 3.0, rel=1e-4)


# -------------------------------------------------------------------- matching


def test_magsac_recovers_pose_from_exact_matches(scene):
    cam, _, R, t, m, _ = scene
    out = relative_pose_magsac(m, cam)
    assert out is not None
    assert rotation_error_deg(out[0], R) < 0.5
    assert translation_error_deg(out[1], t) < 1.0


def test_magsac_returns_none_when_starved():
    cam = toy_camera(64, 64, fov_deg=180.0)
    empty = Matches(target=pixel_grid(64, 64), weight=torch.zeros(64, 64))
    assert relative_pose_magsac(empty, cam) is None


def test_mean_flow_magnitude_matches_a_known_shift():
    grid = pixel_grid(32, 32)
    m = Matches(target=grid + torch.tensor([3.0, 4.0]), weight=torch.ones(32, 32))
    assert mean_flow_magnitude(m) == pytest.approx(5.0, rel=1e-5)


# ------------------------------------------------------------------- backbones


@pytest.mark.parametrize("name", ["vggt_omega", "vggt"])
def test_backbone_hooks_are_transparent_and_reversible(name):
    from raytun3r.testing import tiny_vggt, tiny_vggt_omega

    bb, hw = (tiny_vggt_omega(), (64, 64)) if name == "vggt_omega" else (tiny_vggt(), (70, 70))
    cam = toy_camera(*hw, fov_deg=180.0)
    imgs = torch.rand(1, 2, 3, *hw)

    with torch.no_grad():
        base = bb.forward(imgs)
    assert bool(torch.isfinite(base.depth).all())

    ad = bb.make_adapter()
    assert (ad.pe is not None) == bb.has_abs_pe
    assert (ad.rope is not None) == bb.has_rope

    # Install in the head's own convention so this isolates the adapter: with
    # depth_convention="range" the depth would also pick up a deliberate 1/cos,
    # and a no-op check could not tell that apart from a broken hook.
    bb.install(ad, cam, hw, patch_undistort=False, border_token=False, dpt_grid=False,
               depth_convention=bb.native_depth)
    with torch.no_grad():
        z = bb.forward(imgs)
    assert z.depth_convention == bb.native_depth
    assert torch.allclose(z.depth, base.depth, atol=1e-6)     # zero-init no-op

    bb.remove()
    with torch.no_grad():
        r = bb.forward(imgs)
    assert torch.equal(r.depth, base.depth)                   # exact restore


def test_da3_hooks_fire_on_the_real_package():
    """DA3-Small is the paper's primary backbone, so its hooks need real coverage.

    Every assertion here corresponds to something that was wrong when written
    blind against an assumed layout: the DPT method was ``_apply_pos_embed`` (it
    is ``_add_pos_embed``), the helper import path was ``model.heads.utils`` (it
    is ``model.utils.head_utils``), and the wrapper's ``forward`` is ``no_grad``
    so the adapter could never have been trained through it.
    """
    pytest.importorskip("depth_anything_3", reason="da3 backbone is an optional dep")
    pytest.importorskip("omegaconf")
    from raytun3r.testing import tiny_da3

    try:
        bb = tiny_da3()
    except ImportError as exc:
        pytest.skip(f"depth_anything_3 present but not importable: {exc}")

    assert bb.embed_dim == 384                    # read from the ViT, not hardcoded
    assert bb.has_abs_pe and bb.has_rope

    hw = (70, 70)
    cam = toy_camera(*hw, fov_deg=140.0)
    imgs = torch.rand(1, 2, 3, *hw)

    with torch.no_grad():
        base = bb.forward(imgs)
    assert base.depth.shape == (2, *hw)
    assert bool(torch.isfinite(base.depth).all())

    ad = bb.make_adapter()
    # Corrections off: this isolates the adapter, which must be an exact no-op
    # at zero init.
    bb.install(ad, cam, hw, patch_undistort=False, border_token=False, dpt_grid=False,
               depth_convention=bb.native_depth)
    with torch.no_grad():
        z = bb.forward(imgs)
    assert torch.allclose(z.depth, base.depth, atol=1e-5)

    # Corrections on: these are *meant* to change the prediction. If the DPT-grid
    # hook silently missed its target -- which is exactly what the wrong method
    # name did -- the output would be unchanged, so this is the regression test.
    bb.install(ad, cam, hw, depth_convention=bb.native_depth)
    with torch.no_grad():
        corrected = bb.forward(imgs)
    assert not torch.allclose(corrected.depth, base.depth, atol=1e-5)

    # gradients must reach every table through the frozen model
    out = bb.forward(imgs)
    out.depth.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in ad.parameters())
    assert all(p.grad is None for p in bb.model.parameters())

    bb.remove()
    with torch.no_grad():
        assert torch.equal(bb.forward(imgs).depth, base.depth)


@pytest.mark.parametrize("name", ["vggt_omega", "vggt"])
def test_installed_convention_is_applied_to_the_direct_path(name):
    """The direct fisheye path must convert, exactly as the pinhole baselines do.

    Before this, only ``baselines.py`` divided by cos, so vanilla/param_free/
    raytun3r were scored in planar z against range ground truth.
    """
    from raytun3r.testing import tiny_vggt, tiny_vggt_omega

    bb, hw = (tiny_vggt_omega(), (64, 64)) if name == "vggt_omega" else (tiny_vggt(), (70, 70))
    cam = toy_camera(*hw, fov_deg=140.0)
    imgs = torch.rand(1, 2, 3, *hw)

    bb.install(None, cam, hw, patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="z")
    with torch.no_grad():
        as_z = bb.forward(imgs)
    bb.install(None, cam, hw, patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="range")
    with torch.no_grad():
        as_range = bb.forward(imgs)

    assert as_z.depth_convention == "z" and as_range.depth_convention == "range"
    cos = cam.ray_grid(*hw)[..., 2]
    assert torch.allclose(as_range.depth, as_z.depth / cos.clamp_min(1e-6), rtol=1e-5, atol=1e-6)
    # and it is a real change, not a no-op that happens to pass
    assert not torch.allclose(as_range.depth, as_z.depth, rtol=1e-2)


@pytest.mark.parametrize("name", ["vggt_omega", "vggt"])
def test_gradient_reaches_adapter_and_backbone_stays_frozen(name):
    from raytun3r.testing import tiny_vggt, tiny_vggt_omega

    bb, hw = (tiny_vggt_omega(), (64, 64)) if name == "vggt_omega" else (tiny_vggt(), (70, 70))
    cam = toy_camera(*hw, fov_deg=180.0)
    ad = bb.make_adapter()
    bb.install(ad, cam, hw)
    with torch.no_grad():
        for p in ad.parameters():
            p.normal_(0, 0.05)

    out = bb.forward(torch.rand(1, 2, 3, *hw))
    (out.depth.mean() + out.t.pow(2).mean()).backward()

    for p in ad.parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all())
        assert float(p.grad.abs().sum()) > 0
    assert all(p.grad is None for p in bb.model.parameters())
    bb.remove()


def test_vggt_omega_has_no_absolute_pe():
    """The structural fact that limits this backbone to the "RoPE only" row."""
    from raytun3r.testing import tiny_vggt_omega

    bb = tiny_vggt_omega()
    assert bb.has_abs_pe is False
    assert bb.make_adapter().param_breakdown()["total"] == 20
