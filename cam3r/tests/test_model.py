"""The two-view network: Ray Module + Cross-view Module.

Run at tiny width so the whole thing fits in a CPU test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cam3r.model import CAM3R, CAM3RConfig


def _tiny(**kw) -> CAM3R:
    cfg = CAM3RConfig(
        img_size=64, patch_size=16, ray_embed_dim=32, ray_depth=2, ray_heads=2,
        cv_embed_dim=32, cv_enc_depth=2, cv_dec_embed_dim=24,
        cv_dec_depth=2, cv_heads=2, cv_dec_heads=2, **kw
    )
    torch.manual_seed(0)
    return CAM3R(cfg).eval()


def test_forward_shapes():
    model = _tiny()
    img = torch.rand(2, 3, 64, 64)
    out = model(img, img.clone())

    for v in range(2):
        assert out["rays"][v].shape == (2, 3, 64, 64)
        assert out["radial"][v].shape == (2, 64, 64)
        assert out["conf"][v].shape == (2, 64, 64)
        assert out["points"][v].shape == (2, 3, 64, 64)
    assert out["R"].shape == (2, 3, 3)
    assert out["t_dir"].shape == (2, 3)
    assert out["scale"].shape == (2,)


def test_outputs_satisfy_their_domains():
    out = _tiny()(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64))
    assert torch.allclose(out["rays"][0].norm(dim=1), torch.ones(2, 64, 64), atol=1e-4)
    assert torch.allclose(out["t_dir"].norm(dim=-1), torch.ones(2), atol=1e-5)
    assert float(out["radial"][0].min()) > 0, "radial distance must be positive"
    assert float(out["conf"][0].min()) > 0, "confidence must be positive"
    assert float(out["scale"].min()) > 0, "scale must be positive"

    R = out["R"]
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand_as(R), atol=1e-4)
    assert torch.allclose(torch.linalg.det(R), torch.ones(2), atol=1e-4)


def test_pointmap_is_ray_times_radial():
    """X = d * r, paper Eq. 1 -- the decoupling the whole method rests on."""
    out = _tiny()(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    expected = out["rays"][0] * out["radial"][0].unsqueeze(1)
    assert torch.allclose(out["points"][0], expected, atol=1e-6)


def test_rays_depend_only_on_their_own_view():
    """The Ray Module is per-image: view 1's content must not move view 0's rays.

    Camera geometry is a property of the lens, not of what the other camera saw.
    """
    model = _tiny()
    a, b, c = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        r_ab = model(a, b)["rays"][0]
        r_ac = model(a, c)["rays"][0]
    assert torch.allclose(r_ab, r_ac, atol=1e-6)


def test_cross_view_output_does_depend_on_the_other_view():
    """Conversely, radial distance must use cross-view information."""
    model = _tiny()
    a, b, c = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        d_ab = model(a, b)["radial"][0]
        d_ac = model(a, c)["radial"][0]
    assert not torch.allclose(d_ab, d_ac, atol=1e-5)


def test_encoder_is_siamese():
    """The same image must embed identically whichever slot it is passed in."""
    model = _tiny()
    a = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        f_a0 = model.cross_view.encode(a)
        f_a1 = model.cross_view.encode(a.clone())
    assert torch.allclose(f_a0, f_a1, atol=1e-6)


def test_gradients_reach_every_head():
    model = _tiny().train()
    out = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    loss = (out["points"][0].mean() + out["points"][1].mean() + out["conf"][0].mean()
            + out["R"].sum() + out["t_dir"].sum() + out["scale"].sum() + out["rays"][0].mean())
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, f"no gradient reached: {dead[:8]}"


def test_non_square_and_multiple_of_patch():
    out = _tiny()(torch.rand(1, 3, 64, 96), torch.rand(1, 3, 64, 96))
    assert out["rays"][0].shape == (1, 3, 64, 96)
    assert out["points"][1].shape == (1, 3, 64, 96)


def test_rejects_sizes_that_are_not_a_multiple_of_the_patch():
    with pytest.raises(ValueError, match="patch"):
        _tiny()(torch.rand(1, 3, 70, 64), torch.rand(1, 3, 70, 64))


def test_rejects_mismatched_view_sizes():
    with pytest.raises(ValueError):
        _tiny()(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 96))


def test_eval_mode_is_deterministic():
    model = _tiny()
    a, b = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(a, b)["points"][0], model(a, b)["points"][0], atol=0)


def test_parameter_count_is_reported():
    model = _tiny()
    n = model.num_parameters()
    assert n > 0 and n == sum(p.numel() for p in model.parameters())
