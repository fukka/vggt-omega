# Copyright (c) 2026.
"""CLI entry point for alternating egocentric finetuning.

Runs are driven by a YAML config (``finetune/configs/*.yaml``). The config's
``trainer:`` field selects the training strategy; outputs go to
``<runs_root>/<name>/`` so multiple experiments coexist and are recorded in
``<runs_root>/index.csv``.

Examples
--------
Two strategies in parallel (each writes to its own runs/<name>/)::

    python -m finetune.train --config finetune/configs/ssi.yaml
    python -m finetune.train --config finetune/configs/metric_anchor.yaml

8-GPU run with a quick override and a custom run name::

    torchrun --nproc_per_node=8 -m finetune.train \
        --config finetune/configs/metric_anchor.yaml \
        --name metric_anchor_lr1e4 --set lr_vggt_lora=1.0e-4

Offline CPU dry run (no checkpoint, no data; exercises the whole loop)::

    python -m finetune.train --dummy --name smoke \
        --set rounds=1 --set steps_per_phase=20 --set warmup_steps=4 \
        --set val_every=10 --set viz_every=5 --device cpu

Outputs land under ``runs/<name>/``:
    config.yaml provenance.json     resolved config + git SHA / argv / time
    metrics.jsonl / metrics.csv     per-step train + val losses
    loss_curves.png                 train-vs-val total loss per phase (at end)
    viz/phase{A,B}_step*.jpg         input | VGGT depth | DAv2 depth | dyn-mask
    checkpoint_{last,best,final}.pt  trainable weights (LoRA + heads)
    train_log.txt                   console mirror; tb/ if tensorboard enabled
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
from .data import EgocentricVideoDataset, FisheyeRectifier, collate_windows, looks_like_fisheye, random_egocentric_batch
from .options import build_config, setup_run_dir
from .trainers import build_trainer


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
    # LoRA only in the aggregator backbone; the prediction heads are trained
    # fully (mark_trainable), so we do NOT want redundant LoRA inside them.
    n = apply_lora(model.aggregator, r=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    mark_trainable(model, ("dense_head", "camera_head"))
    print(f"[finetune] injected LoRA into {n} VGGT-Omega aggregator linear layers")
    return model


def build_dav2(cfg: FinetuneConfig):
    from .models import apply_lora, build_depth_anything, mark_trainable

    model = build_depth_anything(use_dummy=cfg.dav2_dummy, model_name=cfg.dav2_model_name)
    if (not cfg.dav2_dummy) and cfg.finetune_dav2_lora_only:
        n = apply_lora(model, r=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
        mark_trainable(model, ("head",))
        print(f"[finetune] injected LoRA into {n} DAv2 linear layers (LoRA-only)")
    return model


def build_rectifier(cfg: FinetuneConfig):
    """A FisheyeRectifier when cfg.rectify, else None (+ a loud warning if the
    data looks like raw fisheye and rectification is off)."""
    if cfg.rectify:
        if is_main():
            print(f"[finetune] rectifying fisheye -> pinhole (preset={cfg.camera_preset!r})")
        return FisheyeRectifier(cfg.camera_preset, cfg.fisheye_k, cfg.fisheye_d)
    if cfg.warn_unrectified and not (cfg.vggt_dummy or cfg.dav2_dummy) \
            and looks_like_fisheye(cfg.clip_pattern, cfg.data_root) and is_main():
        print("[finetune] WARNING: data looks like Aria fisheye but rectify=false. "
              "The geometric/photometric losses assume PINHOLE; set rectify=true "
              "(camera_preset: aria-214-1) or use pinhole-undistorted frames.")
    return None


class _RandomLoader:
    """Yields random egocentric windows for dry runs (no files)."""

    def __init__(self, n: int, cfg: FinetuneConfig, h: int = 64, w: int = 96) -> None:
        self.n, self.cfg, self.h, self.w = n, cfg, h, w

    def __iter__(self):
        for i in range(self.n):
            yield random_egocentric_batch(
                batch_size=self.cfg.batch_size, seq_len=self.cfg.seq_len, height=self.h, width=self.w, seed=i
            )


def build_loader(cfg: FinetuneConfig, rectifier=None, rank: int = 0, world_size: int = 1):
    # Dummy/dry runs use random windows and ignore data_root entirely (so the
    # default cluster path doesn't need to exist locally).
    if cfg.vggt_dummy or cfg.dav2_dummy:
        return _RandomLoader(cfg.steps_per_phase, cfg)
    if not cfg.data_root:
        raise ValueError("data_root is required for real runs (set it in the --config YAML)")
    dataset = EgocentricVideoDataset(
        cfg.data_root,
        seq_len=cfg.seq_len,
        stride=cfg.stride,
        window_stride=cfg.window_stride,
        clip_pattern=cfg.clip_pattern,
        image_resolution=cfg.image_resolution,
        patch_size=cfg.patch_size,
        rectifier=rectifier,
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


def build_val_loader(cfg: FinetuneConfig, rectifier=None):
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
        rectifier=rectifier,
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
    """If epochs > 0 and the loader wraps a sized dataset, set steps_per_phase
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


def parse_args():
    p = argparse.ArgumentParser(description="Alternating DAv2 <-> VGGT-Omega egocentric finetuning")
    p.add_argument("--config", default=None,
                   help="run config YAML (e.g. finetune/configs/metric_anchor.yaml)")
    p.add_argument("--name", default=None, help="override the run name (output dir = runs_root/name)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="FIELD=VALUE",
                   help="override any config field, e.g. --set lr_vggt_lora=1.0e-4 (repeatable)")
    p.add_argument("--dummy", action="store_true", help="stand-in models, random data (CPU dry run)")
    p.add_argument("--device", default="cuda",
                   help="cuda (default; real training needs a GPU). Pass 'cpu' for dummy/CPU runs.")
    p.add_argument("--resume", action="store_true", help="allow writing into an existing run dir")
    p.add_argument("--overwrite", action="store_true", help="allow clobbering an existing run dir")
    a = p.parse_args()

    cfg = build_config(config_path=a.config, overrides=a.overrides, name=a.name)
    if a.dummy:
        cfg.vggt_dummy = True
        cfg.dav2_dummy = True
        if a.name is None and not a.config:
            cfg.name = "dummy"
    return cfg, a


def main() -> None:
    cfg, args = parse_args()
    rank, world_size, local_rank = setup_dist()

    # Resolve + create the run dir; rank 0 writes provenance and guards clobber.
    if not cfg.out_dir:
        cfg.out_dir = os.path.join(cfg.runs_root, cfg.name)
    if is_main():
        setup_run_dir(cfg, resume=args.resume, overwrite=args.overwrite)
    if dist.is_initialized():
        dist.barrier()

    # Per-rank seed for data sampling diversity; model init uses base seed
    set_seed(cfg.seed + rank)

    use_cuda = torch.cuda.is_available()
    if local_rank is not None and use_cuda:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

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

    # Build loaders first; epochs finalizes steps_per_phase BEFORE the trainer
    # constructs its LR schedulers (which depend on the total step count).
    rectifier = build_rectifier(cfg)
    loader = build_loader(cfg, rectifier=rectifier, rank=rank, world_size=world_size)
    apply_epoch_sizing(cfg, loader, world_size)
    val_loader = build_val_loader(cfg, rectifier=rectifier)

    trainer = build_trainer(cfg, vggt, dav2, device=device)
    # trainer.train() handles per-step logging, viz, validation, and saves
    # checkpoint_{last,best,final}.pt + loss_curves.png under cfg.out_dir.
    trainer.train(loader, val_loader)

    if is_main():
        print("[finetune] done.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
