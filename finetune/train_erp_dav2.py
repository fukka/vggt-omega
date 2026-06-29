# Copyright (c) 2026.
"""Entry point for DAC-style **unsupervised ERP finetuning of DAv2** on EgoExo4D.

Unlike ``finetune.train`` (which alternates DAv2 ↔ VGGT-Omega), this trains a
*single* model — DAv2 — in DAC's ERP canonical space with label-free losses
(equivariance consistency + frozen-DAv2 anchor + smoothness). It reuses the same
YAML config system and ``runs/<name>/`` output layout, so ``erp_dav2_*.yaml`` look
just like the ``dav2_*.yaml`` configs.

Examples
--------
Real run (GPU box; needs the EgoExo4D sample + DAC cloned via setup_baselines.sh)::

    python -m finetune.train_erp_dav2 --config finetune/configs/erp_dav2_base.yaml

Multi-GPU::

    torchrun --nproc_per_node=8 -m finetune.train_erp_dav2 \
        --config finetune/configs/erp_dav2_full_ft.yaml

Offline CPU dry run (random ERP tensors, dummy DAv2; exercises the whole loop)::

    python -m finetune.train_erp_dav2 --dummy --name erp_smoke \
        --set rounds=1 --set steps_per_phase=8 --set warmup_steps=2 \
        --set log_every=2 --set viz_every=0 --device cpu
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "finetune"

import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .config import FinetuneConfig
from .data.erp_egoexo import DummyErpDataset, ErpEgoExoDataset, collate_erp
from .options import build_config, setup_run_dir
from .train import build_dav2, is_main, set_seed, setup_dist
from .trainers.erp_distill import ErpDistillTrainer


def build_erp_dataset(cfg: FinetuneConfig, root: str, seed: int):
    if cfg.dav2_dummy:
        return DummyErpDataset(n=max(8, cfg.steps_per_phase), seed=seed)
    return ErpEgoExoDataset(
        root, stream=cfg.egoexo_stream, frames_per_clip=cfg.egoexo_frames_per_clip,
        erp_cano=cfg.erp_cano, erp_fwd_sz=(cfg.erp_fwd_h, cfg.erp_fwd_w),
        crop_wfov=cfg.erp_crop_wfov, focal_scale=cfg.erp_focal_scale,
        input_scale_jitter=cfg.erp_input_scale_jitter, seed=seed)


def build_loader(cfg: FinetuneConfig, root: str, rank: int, world_size: int,
                 shuffle: bool, seed: int):
    if not root and not cfg.dav2_dummy:
        return None
    ds = build_erp_dataset(cfg, root, seed)
    if world_size > 1 and not cfg.dav2_dummy:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle, drop_last=True)
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers,
            collate_fn=collate_erp, drop_last=True)
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle and not cfg.dav2_dummy,
        num_workers=cfg.num_workers, collate_fn=collate_erp, drop_last=True)


def parse_args():
    p = argparse.ArgumentParser(description="DAC-style unsupervised ERP finetuning of DAv2 (EgoExo4D)")
    p.add_argument("--config", default=None, help="run config YAML (finetune/configs/erp_dav2_*.yaml)")
    p.add_argument("--name", default=None, help="override run name (out dir = runs_root/name)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="FIELD=VALUE",
                   help="override any config field (repeatable)")
    p.add_argument("--dummy", action="store_true", help="dummy DAv2 + random ERP tensors (CPU dry run)")
    p.add_argument("--device", default="cuda", help="cuda (default) or cpu (dummy)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    cfg = build_config(config_path=a.config, overrides=a.overrides, name=a.name)
    cfg.trainer = "erp_distill"
    if a.dummy:
        cfg.dav2_dummy = True
        if a.name is None and not a.config:
            cfg.name = "erp_dummy"
    return cfg, a


def main() -> None:
    cfg, args = parse_args()
    rank, world_size, local_rank = setup_dist()

    if not cfg.out_dir:
        cfg.out_dir = _os.path.join(cfg.runs_root, cfg.name)
    if is_main():
        setup_run_dir(cfg, resume=args.resume, overwrite=args.overwrite)
    if dist.is_initialized():
        dist.barrier()

    set_seed(cfg.seed + rank)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}") if (local_rank is not None and use_cuda) \
        else torch.device(args.device)

    set_seed(cfg.seed)
    dav2 = build_dav2(cfg).to(device)
    if world_size > 1:
        ddp_kwargs = dict(find_unused_parameters=True)
        if use_cuda:
            ddp_kwargs["device_ids"] = [local_rank]
        dav2 = DDP(dav2, **ddp_kwargs)
        if is_main():
            print(f"[finetune] DDP enabled: {world_size} ranks")

    loader = build_loader(cfg, cfg.egoexo_root, rank, world_size, shuffle=True, seed=cfg.seed)
    val_loader = build_loader(cfg, cfg.val_data_root, rank, world_size, shuffle=False, seed=cfg.seed + 7)

    trainer = ErpDistillTrainer(dav2, cfg, device=device)
    trainer.train(loader, val_loader)

    if is_main():
        print("[finetune] done.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
