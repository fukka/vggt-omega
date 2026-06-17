# Copyright (c) 2026.
"""CPU smoke test: exercises every loss, the config/registry layer, and both
training strategies on tiny random data with stand-in models. No checkpoint, no
dataset, no GPU.

Run::

    python finetune/smoke_test.py
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "finetune"

import sys
import tempfile

import torch

from .config import FinetuneConfig
from .data import random_egocentric_batch
from .engine import AlternatingTrainer  # backward-compat alias (-> SSITrainer)
from .losses import (
    affine_invariant_l1,
    compute_self_supervised_losses,
    gradient_matching_loss,
    multiview_consistency_loss,
    ssi_loss,
    to_disparity,
)
from .models import DummyDepthModel, LoRALinear, apply_lora, count_parameters, mark_trainable
from .models.dummy import DummyVGGT
from .options import apply_overrides, build_config
from .registry import TRAINER_REGISTRY
from .trainers import MetricAnchorTrainer, SSITrainer, build_trainer


def _check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok  {name}")


def _finite(x) -> bool:
    return bool(torch.isfinite(torch.as_tensor(x)).all())


def test_self_supervised():
    print("[1] self-supervised losses (photometric + geometric + smoothness)")
    torch.manual_seed(0)
    B, S, H, W = 1, 5, 48, 64
    imgs = random_egocentric_batch(B, S, H, W)["images"]
    vggt = DummyVGGT()
    vggt.train()
    preds = vggt(imgs)
    depth = preds["depth"][..., 0]
    losses, dyn = compute_self_supervised_losses(imgs, depth, preds["pose_enc"], conf=preds["depth_conf"])
    for k, v in losses.items():
        _check(f"{k} finite", _finite(v))
    _check("dyn_mask shape", tuple(dyn.shape) == (B, S, H, W))
    _check("dyn_mask in [0,1]", float(dyn.min()) >= 0.0 and float(dyn.max()) <= 1.0 + 1e-5)
    total = sum(losses.values())
    total.backward()
    g = vggt.depth_head.weight.grad
    _check("gradient flows to depth head", g is not None and _finite(g) and float(g.abs().sum()) > 0)


def test_distillation():
    print("[2] distillation losses (affine / SSI / gradient-matching / multiview)")
    torch.manual_seed(1)
    B, S, H, W = 1, 4, 48, 64
    imgs = random_egocentric_batch(B, S, H, W)["images"]
    vggt = DummyVGGT().eval()
    dav2 = DummyDepthModel()
    with torch.no_grad():
        depth_v = vggt(imgs)["depth"][..., 0]
        pose = vggt(imgs)["pose_enc"]
    dav2_depth = dav2(imgs)

    v = to_disparity(depth_v).reshape(B * S, H, W)
    d = to_disparity(dav2_depth).reshape(B * S, H, W)
    aff = affine_invariant_l1(d, v)
    ssi = ssi_loss(v, d)
    grad = gradient_matching_loss(v, d)
    mv = multiview_consistency_loss(dav2_depth, pose)
    for name, val in [("affine", aff), ("ssi", ssi), ("grad_match", grad), ("multiview", mv)]:
        _check(f"{name} finite", _finite(val))
    (aff + mv).backward()
    g = dav2.net[0].weight.grad
    _check("gradient flows to DAv2", g is not None and _finite(g) and float(g.abs().sum()) > 0)


def test_lora():
    print("[3] LoRA injection + freezing")
    torch.manual_seed(2)
    m = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.GELU(), torch.nn.Linear(16, 8))
    x = torch.randn(4, 16)
    y0 = m(x)
    n = apply_lora(m, target_substrings=("0", "2"), r=4, alpha=8)
    _check("layers adapted", n == 2)
    _check("is LoRALinear", isinstance(m[0], LoRALinear))
    _check("delta zero at init (B=0)", torch.allclose(m(x), y0, atol=1e-5))
    mark_trainable(m, train_head_substrings=())
    trainable, total = count_parameters(m)
    _check("only LoRA trainable", 0 < trainable < total)
    with torch.no_grad():
        m[0].lora_B.add_(0.1)
    _check("delta changes output", not torch.allclose(m(x), y0, atol=1e-4))


def test_mark_trainable_skips_lora_base():
    print("[4] mark_trainable keeps LoRA base frozen even under a head name")
    torch.manual_seed(7)
    head = torch.nn.Module()
    head.qkv = torch.nn.Linear(8, 24)   # will be LoRA-wrapped
    head.out = torch.nn.Linear(8, 8)    # plain head linear
    model = torch.nn.Module()
    model.camera_head = head
    apply_lora(model, target_substrings=("qkv",), r=2, alpha=4)
    mark_trainable(model, train_head_substrings=("camera_head",))
    base_w = model.camera_head.qkv.base.weight
    lora_a = model.camera_head.qkv.lora_A
    out_w = model.camera_head.out.weight
    _check("LoRA base weight frozen", base_w.requires_grad is False)
    _check("LoRA adapter trainable", lora_a.requires_grad is True)
    _check("plain head weight trainable", out_w.requires_grad is True)


def test_config_and_registry():
    print("[5] config overrides + trainer registry")
    _check("ssi registered", "ssi" in TRAINER_REGISTRY)
    _check("metric_anchor registered", "metric_anchor" in TRAINER_REGISTRY)
    cfg = build_config(config_path=None, overrides=["lr_vggt_lora=1.0e-4", "rectify=true", "offsets=-1,1"],
                       name="unit")
    _check("override float coerced", abs(cfg.lr_vggt_lora - 1.0e-4) < 1e-12)
    _check("override bool coerced", cfg.rectify is True)
    _check("override tuple coerced", tuple(cfg.offsets) == (-1, 1))
    _check("name applied", cfg.name == "unit")
    raised = False
    try:
        apply_overrides(cfg, ["nonsense_field=1"])
    except KeyError:
        raised = True
    _check("unknown field rejected", raised)


def _make_models():
    return DummyVGGT(), DummyDepthModel()


def test_engine_variants():
    print("[6] both trainers: phase A updates DAv2, phase B updates VGGT")
    for trainer_name, cls in (("ssi", SSITrainer), ("metric_anchor", MetricAnchorTrainer)):
        torch.manual_seed(3)
        cfg = FinetuneConfig(
            trainer=trainer_name, vggt_dummy=True, dav2_dummy=True, seq_len=5, batch_size=1,
            offsets=(-1, 1), log_every=1, steps_per_phase=2, rounds=1, amp=False, warmup_steps=0,
            w_metric_anchor=0.5, out_dir=tempfile.mkdtemp(prefix=f"smoke_{trainer_name}_"),
        )
        vggt, dav2 = _make_models()
        trainer = build_trainer(cfg, vggt, dav2, device="cpu")
        _check(f"{trainer_name}: build_trainer -> right class", isinstance(trainer, cls))
        imgs = random_egocentric_batch(1, 5, 48, 64)["images"]

        d_before = dav2.net[0].weight.detach().clone()
        a_logs = trainer.phase_a_step(imgs)
        _check(f"{trainer_name}: phaseA logs finite", all(_finite(v) for v in a_logs.values()))
        _check(f"{trainer_name}: phaseA has photometric anchor", "photometric" in a_logs)
        _check(f"{trainer_name}: phaseA updated DAv2", not torch.allclose(d_before, dav2.net[0].weight))

        v_before = vggt.depth_head.weight.detach().clone()
        b_logs = trainer.phase_b_step(imgs)
        _check(f"{trainer_name}: phaseB logs finite", all(_finite(v) for v in b_logs.values()))
        _check(f"{trainer_name}: phaseB updated VGGT", not torch.allclose(v_before, vggt.depth_head.weight))
        if trainer_name == "metric_anchor":
            _check("metric_anchor term present in phase B", "metric_anchor" in b_logs)


def test_backcompat_alias():
    print("[7] engine.AlternatingTrainer alias still works")
    _check("alias is SSITrainer", AlternatingTrainer is SSITrainer)


def main() -> int:
    torch.autograd.set_detect_anomaly(True)
    for fn in (test_self_supervised, test_distillation, test_lora,
               test_mark_trainable_skips_lora_base, test_config_and_registry,
               test_engine_variants, test_backcompat_alias):
        fn()
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
