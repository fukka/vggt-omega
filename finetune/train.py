# Copyright (c) 2026.
"""CLI entry point for alternating egocentric finetuning.

Examples
--------
Offline dry run (no checkpoint, no data, CPU/GPU)::

    python -m finetune.train --dummy --rounds 1 --steps-per-phase 20

Real run::

    python -m finetune.train \
        --data-root /path/to/egocentric_frames \
        --vggt-checkpoint checkpoints/vggt_omega_1b_512.pt \
        --image-resolution 512 --seq-len 8 --batch-size 1 \
        --rounds 3 --steps-per-phase 500
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from .config import FinetuneConfig
from .data import EgocentricVideoDataset, collate_windows, random_egocentric_batch
from .engine import AlternatingTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_vggt(cfg: FinetuneConfig):
    if cfg.vggt_dummy:
        from .models.dummy import DummyVGGT

        return DummyVGGT()

    from vggt_omega.models import VGGTOmega

    from .models import apply_lora, mark_trainable

    model = VGGTOmega()
    if cfg.vggt_checkpoint:
        sd = torch.load(cfg.vggt_checkpoint, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[finetune] loaded VGGT checkpoint (missing={len(missing)}, unexpected={len(unexpected)})")
    n = apply_lora(model, r=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    mark_trainable(model, ("dense_head", "camera_head"))
    print(f"[finetune] injected LoRA into {n} VGGT-Omega linear layers")
    return model


def build_dav2(cfg: FinetuneConfig):
    from .models import apply_lora, build_depth_anything, mark_trainable

    model = build_depth_anything(use_dummy=cfg.dav2_dummy, model_name=cfg.dav2_model_name)
    if (not cfg.dav2_dummy) and cfg.finetune_dav2_lora_only:
        n = apply_lora(model, r=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
        mark_trainable(model, ("head",))
        print(f"[finetune] injected LoRA into {n} DAv2 linear layers (LoRA-only)")
    return model


class _RandomLoader:
    """Yields random egocentric windows for dry runs (no files)."""

    def __init__(self, n: int, cfg: FinetuneConfig, h: int = 64, w: int = 96) -> None:
        self.n, self.cfg, self.h, self.w = n, cfg, h, w

    def __iter__(self):
        for i in range(self.n):
            yield random_egocentric_batch(
                batch_size=self.cfg.batch_size, seq_len=self.cfg.seq_len, height=self.h, width=self.w, seed=i
            )


def build_loader(cfg: FinetuneConfig):
    if not cfg.data_root:
        if not (cfg.vggt_dummy or cfg.dav2_dummy):
            raise ValueError("--data-root is required for real runs")
        return _RandomLoader(cfg.steps_per_phase, cfg)
    dataset = EgocentricVideoDataset(
        cfg.data_root,
        seq_len=cfg.seq_len,
        stride=cfg.stride,
        image_resolution=cfg.image_resolution,
        patch_size=cfg.patch_size,
    )
    print(f"[finetune] dataset: {len(dataset)} windows from {cfg.data_root}")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_windows,
        drop_last=True,
    )


def parse_args() -> FinetuneConfig:
    p = argparse.ArgumentParser(description="Alternating DAv2 <-> VGGT-Omega egocentric finetuning")
    p.add_argument("--data-root", default="")
    p.add_argument("--vggt-checkpoint", default="")
    p.add_argument("--dav2-model-name", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--dummy", action="store_true", help="shortcut for --vggt-dummy --dav2-dummy")
    p.add_argument("--vggt-dummy", action="store_true")
    p.add_argument("--dav2-dummy", action="store_true")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--image-resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--steps-per-phase", type=int, default=500)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--finetune-dav2-lora-only", action="store_true")
    p.add_argument("--ema-teacher", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="finetune_outputs")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    return FinetuneConfig(
        data_root=a.data_root,
        vggt_checkpoint=a.vggt_checkpoint,
        dav2_model_name=a.dav2_model_name,
        vggt_dummy=a.vggt_dummy or a.dummy,
        dav2_dummy=a.dav2_dummy or a.dummy,
        seq_len=a.seq_len,
        stride=a.stride,
        image_resolution=a.image_resolution,
        batch_size=a.batch_size,
        rounds=a.rounds,
        steps_per_phase=a.steps_per_phase,
        lora_rank=a.lora_rank,
        finetune_dav2_lora_only=a.finetune_dav2_lora_only,
        ema_teacher=a.ema_teacher,
        out_dir=a.out_dir,
        seed=a.seed,
    ), a.device


def main() -> None:
    cfg, device = parse_args()
    set_seed(cfg.seed)
    vggt = build_vggt(cfg)
    dav2 = build_dav2(cfg)
    trainer = AlternatingTrainer(vggt, dav2, cfg, device=device)
    loader = build_loader(cfg)
    trainer.train(loader)
    print("[finetune] done.")


if __name__ == "__main__":
    main()
