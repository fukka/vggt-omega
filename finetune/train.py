# Copyright (c) 2026.
"""CLI entry point for alternating egocentric finetuning.

--data-root, --val-data-root, and --vggt-checkpoint default to the user's
cluster paths, so on that cluster a bare invocation already trains on
Ego-Exo4D; override them on the CLI elsewhere. Device defaults to ``cuda``.

Examples
--------
Single-GPU (uses the default cluster paths + checkpoint)::

    python finetune/train.py --batch-size 2 --rounds 3 --steps-per-phase 500

Recommended full run (8 GPUs, ~1 pass over the data, bf16, cosine schedule)::

    torchrun --nproc_per_node=8 finetune/train.py \
        --epochs 1 --rounds 6 --window-stride 8 --batch-size 1 --grad-accum 2 \
        --val-every 1000 --viz-every 500 --tensorboard
    # --epochs auto-sizes steps-per-phase; warmup+cosine LR, EMA teacher, and
    # layer-group LRs (LoRA 2e-4 / heads 2e-5) are on by default.

Offline dry run on CPU (no checkpoint, no data; exercises val + viz + logging)::

    python finetune/train.py --dummy --device cpu --amp-dtype fp32 \
        --rounds 1 --steps-per-phase 20 --warmup-steps 4 --val-every 10 --viz-every 5

Outputs land under ``--out-dir``:
    metrics.jsonl / metrics.csv   per-step train + val losses
    loss_curves.png               train-vs-val total loss per phase (at end)
    viz/phase{A,B}_step*.jpg       input | VGGT depth | DAv2 depth | dyn-mask
    checkpoint_{last,best,final}.pt  trainable weights (LoRA + heads)
    train_log.txt                 console mirror; tb/ if --tensorboard
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

# Default cluster paths (override on the CLI for other environments).
_EGOEXO_ROOT = "/sdp-rgb-perception/tristan-space-s3/egoexo4d/lmeec_data/egoexo_dataset"
_DEFAULT_DATA_ROOT = _EGOEXO_ROOT + "/train"
_DEFAULT_VAL_DATA_ROOT = _EGOEXO_ROOT + "/val"
_DEFAULT_VGGT_CKPT = "/group-volume/Fengjia/projects/vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt"


def setup_dist():
    """Init the process group from torchrun env vars. Returns (rank, world_size, local_rank).

    Uses NCCL on CUDA, else gloo (CPU). When not launched by torchrun (RANK
    unset) returns (0, 1, None) so single-process runs need no special-casing.
    """
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return 0, 1, None
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        dist.init_process_group(backend="gloo")
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
    # Dummy/dry runs use random windows and ignore --data-root entirely (so the
    # default cluster path doesn't need to exist locally).
    if cfg.vggt_dummy or cfg.dav2_dummy:
        return _RandomLoader(cfg.steps_per_phase, cfg)
    if not cfg.data_root:
        raise ValueError("--data-root is required for real runs")
    dataset = EgocentricVideoDataset(
        cfg.data_root,
        seq_len=cfg.seq_len,
        stride=cfg.stride,
        window_stride=cfg.window_stride,
        clip_pattern=cfg.clip_pattern,
        image_resolution=cfg.image_resolution,
        patch_size=cfg.patch_size,
    )
    if is_main():
        print(f"[finetune] train: {len(dataset)} windows from {len(dataset.clips)} clips "
              f"(pattern {cfg.clip_pattern!r}, skipped {dataset.num_skipped_dirs} non-matching dirs) "
              f"under {cfg.data_root}")
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


def build_val_loader(cfg: FinetuneConfig):
    """Held-out loader for periodic validation, or None if not configured.

    Not sharded with DistributedSampler: every rank evaluates the same first
    ``val_steps`` batches and the trainer averages scalars across ranks, so the
    reported val loss is identical and reduction calls stay in lockstep.
    """
    # Dummy/dry runs exercise the validation path with random windows.
    if cfg.vggt_dummy or cfg.dav2_dummy:
        return _RandomLoader(cfg.val_steps, cfg)
    if not cfg.val_data_root:
        return None
    dataset = EgocentricVideoDataset(
        cfg.val_data_root,
        seq_len=cfg.seq_len,
        stride=cfg.stride,
        window_stride=cfg.window_stride,
        clip_pattern=cfg.clip_pattern,
        image_resolution=cfg.image_resolution,
        patch_size=cfg.patch_size,
    )
    if is_main():
        print(f"[finetune] val: {len(dataset)} windows from {len(dataset.clips)} clips "
              f"(pattern {cfg.clip_pattern!r}, skipped {dataset.num_skipped_dirs}) under {cfg.val_data_root}")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_windows,
        drop_last=True,
    )


def apply_epoch_sizing(cfg: FinetuneConfig, loader, world_size: int) -> None:
    """If --epochs > 0 and the loader wraps a sized dataset, set steps_per_phase
    so the alternating loop covers cfg.epochs passes over the (per-rank) data.

    total_micro = epochs * (windows_per_rank / batch_size)
    steps_per_phase = total_micro / (rounds * (1 + dav2_steps_mult))
    """
    if cfg.epochs <= 0:
        return
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__len__"):
        return
    windows_per_rank = max(1, len(dataset) // max(1, world_size))
    total_micro = cfg.epochs * windows_per_rank / max(1, cfg.batch_size)
    denom = cfg.rounds * (1.0 + cfg.dav2_steps_mult)
    cfg.steps_per_phase = max(1, int(round(total_micro / max(1e-9, denom))))
    if is_main():
        print(f"[finetune] epochs={cfg.epochs}: steps_per_phase={cfg.steps_per_phase} "
              f"(rounds={cfg.rounds}, A:B={cfg.dav2_steps_mult:g}:1, "
              f"~{int(total_micro)} micro-iters/rank over {len(dataset)} train windows)")


def parse_args() -> FinetuneConfig:
    p = argparse.ArgumentParser(description="Alternating DAv2 <-> VGGT-Omega egocentric finetuning")
    p.add_argument("--data-root", default=_DEFAULT_DATA_ROOT)
    p.add_argument("--val-data-root", default=_DEFAULT_VAL_DATA_ROOT,
                   help="held-out split for validation (e.g. <root>/val)")
    p.add_argument("--vggt-checkpoint", default=_DEFAULT_VGGT_CKPT)
    p.add_argument("--dav2-model-name", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--dummy", action="store_true", help="shortcut for --vggt-dummy --dav2-dummy")
    p.add_argument("--vggt-dummy", action="store_true")
    p.add_argument("--dav2-dummy", action="store_true")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--window-stride", type=int, default=1,
                   help="frames between window starts (1 = max overlap; seq_len*stride = non-overlapping)")
    p.add_argument("--clip-pattern", default="*214-1",
                   help="fnmatch glob; keep only matching camera dirs (egocentric RGB aria*_214-1). "
                        "Pass '' to load every camera (incl. exocentric/SLAM).")
    p.add_argument("--image-resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    # schedule: prefer --epochs (auto-sizes steps-per-phase to cover the data)
    p.add_argument("--epochs", type=float, default=0.0,
                   help=">0 auto-sizes --steps-per-phase to this many passes over the train set")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--steps-per-phase", type=int, default=2000,
                   help="micro-iterations per phase (ignored when --epochs > 0)")
    p.add_argument("--dav2-steps-mult", type=float, default=1.0,
                   help="phase A (DAv2) length = steps-per-phase * this")
    p.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    # optimizer + LR schedule
    p.add_argument("--lr-vggt-head", type=float, default=2e-5)
    p.add_argument("--lr-vggt-lora", type=float, default=2e-4)
    p.add_argument("--lr-dav2", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"])
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--finetune-dav2-lora-only", action="store_true")
    # EMA teacher (on by default)
    p.add_argument("--no-ema", action="store_true", help="disable the EMA teacher")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--no-ema-distill", action="store_true",
                   help="keep EMA but distill from the live DAv2 instead of the EMA copy")
    p.add_argument("--device", default="cuda",
                   help="cuda (default; real training needs a GPU). Pass 'cpu' for dummy/CPU runs.")
    p.add_argument("--out-dir", default="finetune_outputs")
    p.add_argument("--seed", type=int, default=0)
    # validation
    p.add_argument("--val-every", type=int, default=0,
                   help="steps between validation; 0 = once at the end of each phase")
    p.add_argument("--val-steps", type=int, default=50, help="val batches averaged per validation pass")
    # monitoring
    p.add_argument("--log-every", type=int, default=50, help="steps between train-loss logs")
    p.add_argument("--save-every", type=int, default=2000, help="steps between checkpoints (0 disables)")
    p.add_argument("--viz-every", type=int, default=500, help="steps between depth montages (0 disables)")
    p.add_argument("--num-viz-frames", type=int, default=4, help="frames per saved montage")
    p.add_argument("--tensorboard", action="store_true", help="also log to <out-dir>/tb/")
    a = p.parse_args()

    return FinetuneConfig(
        data_root=a.data_root,
        val_data_root=a.val_data_root,
        vggt_checkpoint=a.vggt_checkpoint,
        dav2_model_name=a.dav2_model_name,
        vggt_dummy=a.vggt_dummy or a.dummy,
        dav2_dummy=a.dav2_dummy or a.dummy,
        seq_len=a.seq_len,
        stride=a.stride,
        window_stride=a.window_stride,
        clip_pattern=a.clip_pattern,
        image_resolution=a.image_resolution,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        epochs=a.epochs,
        rounds=a.rounds,
        steps_per_phase=a.steps_per_phase,
        dav2_steps_mult=a.dav2_steps_mult,
        grad_accum=a.grad_accum,
        lr_vggt_head=a.lr_vggt_head,
        lr_vggt_lora=a.lr_vggt_lora,
        lr_dav2=a.lr_dav2,
        weight_decay=a.weight_decay,
        warmup_steps=a.warmup_steps,
        lr_schedule=a.lr_schedule,
        min_lr_ratio=a.min_lr_ratio,
        amp_dtype=a.amp_dtype,
        amp=(a.amp_dtype != "fp32"),
        lora_rank=a.lora_rank,
        finetune_dav2_lora_only=a.finetune_dav2_lora_only,
        ema_teacher=not a.no_ema,
        ema_decay=a.ema_decay,
        use_ema_teacher_in_distill=not a.no_ema_distill,
        out_dir=a.out_dir,
        seed=a.seed,
        val_every=a.val_every,
        val_steps=a.val_steps,
        log_every=a.log_every,
        save_every=a.save_every,
        viz_every=a.viz_every,
        num_viz_frames=a.num_viz_frames,
        tensorboard=a.tensorboard,
    ), a.device


def main() -> None:
    cfg, cli_device = parse_args()
    rank, world_size, local_rank = setup_dist()

    # Per-rank seed for data sampling diversity; model init uses base seed
    set_seed(cfg.seed + rank)

    use_cuda = torch.cuda.is_available()
    if local_rank is not None and use_cuda:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cli_device)

    # Build models on CPU then move to device (avoids double alloc on rank != 0)
    set_seed(cfg.seed)  # identical init across all ranks
    vggt = build_vggt(cfg).to(device)
    dav2 = build_dav2(cfg).to(device)

    if world_size > 1:
        # device_ids only for CUDA DDP; gloo/CPU DDP takes device_ids=None.
        # find_unused_parameters=True: VGGT has frozen base weights (only LoRA+
        # heads get grads); DAv2 LoRA-only is the same situation.
        ddp_kwargs = dict(find_unused_parameters=True)
        if use_cuda:
            ddp_kwargs["device_ids"] = [local_rank]
        vggt = DDP(vggt, **ddp_kwargs)
        dav2 = DDP(dav2, **ddp_kwargs)
        if is_main():
            print(f"[finetune] DDP enabled: {world_size} ranks ({'nccl' if use_cuda else 'gloo'})")

    # Build loaders first; --epochs finalizes steps_per_phase BEFORE the trainer
    # constructs its LR schedulers (which depend on the total step count).
    loader = build_loader(cfg, rank=rank, world_size=world_size)
    apply_epoch_sizing(cfg, loader, world_size)
    val_loader = build_val_loader(cfg)

    trainer = AlternatingTrainer(vggt, dav2, cfg, device=device)
    # trainer.train() handles per-step logging, viz, validation, and saves
    # checkpoint_{last,best,final}.pt + loss_curves.png under cfg.out_dir.
    trainer.train(loader, val_loader)

    if is_main():
        print("[finetune] done.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
