# Copyright (c) 2026.
"""CLI entry point for alternating egocentric finetuning.

Examples
--------
Single-GPU::

    python finetune/train.py \
        --data-root /path/to/egocentric_frames \
        --vggt-checkpoint checkpoints/vggt_omega_1b_512.pt \
        --batch-size 2 --rounds 3 --steps-per-phase 500

Multi-GPU (torchrun)::

    torchrun --nproc_per_node=4 finetune/train.py \
        --data-root /path/to/egocentric_frames \
        --vggt-checkpoint checkpoints/vggt_omega_1b_512.pt \
        --batch-size 1 --rounds 3 --steps-per-phase 500

Offline dry run (no checkpoint, no data)::

    python finetune/train.py --dummy --rounds 1 --steps-per-phase 20
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "finetune"

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .config import FinetuneConfig
from .data import EgocentricVideoDataset, collate_windows, random_egocentric_batch
from .engine import AlternatingTrainer


def setup_dist():
    """Init NCCL from torchrun env vars. Returns (rank, world_size, local_rank).

    When not launched by torchrun (RANK not set) returns (0, 1, None) so the
    rest of the code needs no special-casing for single-GPU runs.
    """
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return 0, 1, None
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size(), local_rank


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


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


def build_loader(cfg: FinetuneConfig, rank: int = 0, world_size: int = 1):
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
    if is_main():
        print(f"[finetune] dataset: {len(dataset)} windows from {cfg.data_root}")
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            collate_fn=collate_windows,
        )
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


def _save_checkpoint(cfg: FinetuneConfig, vggt: torch.nn.Module, dav2: torch.nn.Module) -> None:
    """Save trainable parameters only (LoRA deltas + prediction heads)."""
    os.makedirs(cfg.out_dir, exist_ok=True)

    def _trainable_state(m):
        raw = m.module if hasattr(m, "module") else m
        return {k: v.cpu() for k, v in raw.state_dict().items() if v.requires_grad or
                any(tag in k for tag in ("lora_A", "lora_B", "dense_head", "camera_head", "head"))}

    path = os.path.join(cfg.out_dir, "checkpoint.pt")
    torch.save({"vggt": _trainable_state(vggt), "dav2": _trainable_state(dav2)}, path)
    print(f"[finetune] checkpoint saved to {path}")


def main() -> None:
    cfg, cli_device = parse_args()
    rank, world_size, local_rank = setup_dist()

    # Per-rank seed for data sampling diversity; model init uses base seed
    set_seed(cfg.seed + rank)

    if local_rank is not None:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cli_device)

    # Build models on CPU then move to device (avoids double alloc on rank != 0)
    set_seed(cfg.seed)  # identical init across all ranks
    vggt = build_vggt(cfg).to(device)
    dav2 = build_dav2(cfg).to(device)

    if world_size > 1:
        # VGGT: most params frozen (base weights); only LoRA+heads get gradients
        # → find_unused_parameters=True required
        vggt = DDP(vggt, device_ids=[local_rank], find_unused_parameters=True)
        # DAv2: fully trainable (or LoRA-only, same reason)
        dav2 = DDP(dav2, device_ids=[local_rank], find_unused_parameters=True)
        if is_main():
            print(f"[finetune] DDP enabled: {world_size} GPUs")

    trainer = AlternatingTrainer(vggt, dav2, cfg, device=device)
    loader = build_loader(cfg, rank=rank, world_size=world_size)
    trainer.train(loader)

    if is_main():
        _save_checkpoint(cfg, vggt, dav2)
        print("[finetune] done.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
