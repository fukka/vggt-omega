"""Weight loaders for the paper's prescribed initializations.

No real DUSt3R/UniK3D checkpoint is available here, so these use synthetic
state dicts built to those checkpoints' key layouts -- which is what the loaders
actually have to cope with (fused qkv, projq/projk/projv, per-degree names).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cam3r.model import CAM3R, CAM3RConfig
from cam3r.pretrained import (
    initialize_cam3r,
    load_dust3r_into_cross_view,
    load_unik3d_into_ray_module,
)


def _tiny() -> CAM3R:
    torch.manual_seed(0)
    return CAM3R(CAM3RConfig(
        img_size=64, patch_size=16, ray_embed_dim=32, ray_depth=2, ray_heads=2,
        cv_embed_dim=32, cv_enc_depth=2, cv_dec_embed_dim=24, cv_dec_depth=2,
        cv_heads=2, cv_dec_heads=2,
    ))


def _fake_dust3r(model: CAM3R) -> dict:
    """A state dict in DUSt3R's layout: fused self-attn qkv, projq/k/v cross-attn."""
    cv = model.cross_view
    sd = {}
    for name, p in cv.named_parameters():
        if ".attn.q." in name or ".attn.k." in name or ".attn.v." in name:
            continue                                    # emitted fused below
        src = name
        for a, b in (("cross_attn.q.", "cross_attn.projq."),
                     ("cross_attn.k.", "cross_attn.projk."),
                     ("cross_attn.v.", "cross_attn.projv.")):
            src = src.replace(a, b)
        sd[src] = torch.randn_like(p)

    seen = set()
    for name, p in cv.named_parameters():
        if ".attn.q.weight" in name:
            prefix = name[: -len(".q.weight")]
            if prefix in seen:
                continue
            seen.add(prefix)
            dim = p.shape[0]
            sd[f"{prefix}.qkv.weight"] = torch.randn(3 * dim, dim)
            sd[f"{prefix}.qkv.bias"] = torch.randn(3 * dim)
    return sd


def _fake_unik3d(model: CAM3R) -> dict:
    """A state dict in UniK3D's layout.

    Three groups, matching the real checkpoint: the DINOv2 ``pixel_encoder``
    trunk (fused ``attn.qkv``), the per-depth ``camera_token_adapter``
    projections, and the ``angular_module`` with its per-degree names.
    """
    rm = model.ray_module
    sd = {}
    for name, p in rm.named_parameters():
        if name.startswith("head."):
            src = name[len("head.") :]
            for i, deg in enumerate((1, 2, 3)):
                src = src.replace(f"projects.{i}.", f"project_deg{deg}.")
                src = src.replace(f"outs.{i}.", f"out_deg{deg}.")
            sd[f"model.pixel_decoder.angular_module.{src}"] = torch.randn_like(p)
        elif name.startswith("camera_token_adapter."):
            tail = name[len("camera_token_adapter.") :]
            sd[f"model.pixel_decoder.camera_token_adapter.input_adapters.{tail}"] = torch.randn_like(p)
        elif ".attn.q." in name or ".attn.k." in name or ".attn.v." in name:
            continue                                    # emitted fused below
        else:
            sd[f"model.pixel_encoder.{name}"] = torch.randn_like(p)

    seen = set()
    for name, p in rm.named_parameters():
        if ".attn.q.weight" in name:
            prefix = name[: -len(".q.weight")]
            if prefix in seen:
                continue
            seen.add(prefix)
            dim = p.shape[0]
            sd[f"model.pixel_encoder.{prefix}.qkv.weight"] = torch.randn(3 * dim, dim)
            sd[f"model.pixel_encoder.{prefix}.qkv.bias"] = torch.randn(3 * dim)

    sd["model.pixel_encoder.mask_token"] = torch.randn(1, 4)   # no counterpart; ignored
    sd["model.pixel_decoder.radial_module.x.weight"] = torch.randn(4, 4)   # ignored
    return sd


def test_dust3r_loader_transfers_weights(tmp_path):
    model = _tiny()
    ckpt = tmp_path / "dust3r.pth"
    torch.save(_fake_dust3r(model), ckpt)

    before = model.cross_view.enc_blocks[0].mlp.fc1.weight.clone()
    report = load_dust3r_into_cross_view(model.cross_view, str(ckpt))
    assert report.n_loaded > 0
    assert not torch.equal(before, model.cross_view.enc_blocks[0].mlp.fc1.weight)


def test_dust3r_loader_splits_fused_qkv(tmp_path):
    model = _tiny()
    sd = _fake_dust3r(model)
    ckpt = tmp_path / "d.pth"
    torch.save(sd, ckpt)
    load_dust3r_into_cross_view(model.cross_view, str(ckpt))

    fused = sd["enc_blocks.0.attn.qkv.weight"]
    dim = fused.shape[0] // 3
    attn = model.cross_view.enc_blocks[0].attn
    assert torch.allclose(attn.q.weight, fused[:dim])
    assert torch.allclose(attn.k.weight, fused[dim : 2 * dim])
    assert torch.allclose(attn.v.weight, fused[2 * dim :])


def test_dust3r_loader_maps_cross_attention_names(tmp_path):
    model = _tiny()
    sd = _fake_dust3r(model)
    ckpt = tmp_path / "d.pth"
    torch.save(sd, ckpt)
    load_dust3r_into_cross_view(model.cross_view, str(ckpt))
    assert torch.allclose(
        model.cross_view.dec_blocks[0].cross_attn.q.weight, sd["dec_blocks.0.cross_attn.projq.weight"]
    )


def test_unik3d_loader_initializes_the_whole_ray_backbone(tmp_path):
    """The paper initializes the Ray Module's *ViT backbone*, not just the head.

    Every parameter of the Ray Module has a UniK3D counterpart, so a correct
    loader leaves nothing at init.
    """
    model = _tiny()
    ckpt = tmp_path / "unik3d.pt"
    torch.save(_fake_unik3d(model), ckpt)

    trunk_before = model.ray_module.blocks[0].mlp.fc1.weight.clone()
    report = load_unik3d_into_ray_module(model.ray_module, str(ckpt))

    assert not torch.equal(trunk_before, model.ray_module.blocks[0].mlp.fc1.weight)
    assert report.missing == [], f"left at init: {report.missing}"
    assert report.skipped_shape == []
    for group in ("blocks.", "patch_embed.", "camera_token_adapter.", "head."):
        assert any(n.startswith(group) for n in report.loaded), group


def test_unik3d_loader_splits_the_trunks_fused_qkv(tmp_path):
    model = _tiny()
    sd = _fake_unik3d(model)
    ckpt = tmp_path / "u.pt"
    torch.save(sd, ckpt)
    load_unik3d_into_ray_module(model.ray_module, str(ckpt))

    fused = sd["model.pixel_encoder.blocks.0.attn.qkv.weight"]
    dim = fused.shape[0] // 3
    attn = model.ray_module.blocks[0].attn
    assert torch.allclose(attn.q.weight, fused[:dim])
    assert torch.allclose(attn.k.weight, fused[dim : 2 * dim])
    assert torch.allclose(attn.v.weight, fused[2 * dim :])


def test_unik3d_loader_maps_per_degree_names(tmp_path):
    model = _tiny()
    sd = _fake_unik3d(model)
    ckpt = tmp_path / "u.pt"
    torch.save(sd, ckpt)
    load_unik3d_into_ray_module(model.ray_module, str(ckpt))
    assert torch.allclose(
        model.ray_module.head.projects[0].weight,
        sd["model.pixel_decoder.angular_module.project_deg1.weight"],
    )


def test_unik3d_loader_maps_the_table_s3_projection(tmp_path):
    """Table S3's 1024 -> 512 is UniK3D's ``camera_token_adapter``, one per depth."""
    model = _tiny()
    sd = _fake_unik3d(model)
    ckpt = tmp_path / "u.pt"
    torch.save(sd, ckpt)
    load_unik3d_into_ray_module(model.ray_module, str(ckpt))
    assert torch.allclose(
        model.ray_module.camera_token_adapter[0].weight,
        sd["model.pixel_decoder.camera_token_adapter.input_adapters.0.weight"],
    )


def test_loader_refuses_a_checkpoint_that_matches_nothing(tmp_path):
    """Silently loading zero tensors would masquerade as a real initialization."""
    ckpt = tmp_path / "wrong.pth"
    torch.save({"totally.unrelated.weight": torch.randn(3, 3)}, ckpt)
    with pytest.raises(RuntimeError, match="no tensors matched"):
        load_dust3r_into_cross_view(_tiny().cross_view, str(ckpt))


def test_report_counts_shape_mismatches(tmp_path):
    model = _tiny()
    sd = _fake_dust3r(model)
    sd["enc_blocks.0.mlp.fc1.weight"] = torch.randn(999, 7)
    ckpt = tmp_path / "d.pth"
    torch.save(sd, ckpt)
    report = load_dust3r_into_cross_view(model.cross_view, str(ckpt))
    assert any("enc_blocks.0.mlp.fc1.weight" == n for n, _ in report.skipped_shape)
    assert "shape-mismatched" in report.summary()


def test_initialize_cam3r_with_no_checkpoints_is_a_no_op():
    model = _tiny()
    before = model.cross_view.enc_blocks[0].mlp.fc1.weight.clone()
    assert initialize_cam3r(model, verbose=False) == []
    assert torch.equal(before, model.cross_view.enc_blocks[0].mlp.fc1.weight)


def test_state_dict_unwrapping(tmp_path):
    """Checkpoints commonly nest the weights under 'model' and prefix 'module.'."""
    model = _tiny()
    sd = {f"module.{k}": v for k, v in _fake_dust3r(model).items()}
    ckpt = tmp_path / "nested.pth"
    torch.save({"model": sd, "epoch": 3}, ckpt)
    assert load_dust3r_into_cross_view(model.cross_view, str(ckpt)).n_loaded > 0
