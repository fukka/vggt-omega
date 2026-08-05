# Copyright (c) 2026.
"""Train CAM3R -- paper Sec. 4.2.

Optimizer settings follow the paper: AdamW ``beta = (0.9, 0.95)``, lr ``5e-5``
with linear warmup then cosine decay, batch 4 per GPU with 2-step accumulation.
The two-phase curriculum runs homogeneous pairs first (``--phase 1``) and mixes
lens families second (``--phase 2``); ``--phase both`` runs them back to back.

Only ADT is wired up here -- see :mod:`cam3r.data` for the interface the other
three corpora plug into.  With a single lens family available, phase 2 has
nothing to mix and degenerates to phase 1; the run log says so rather than
pretending the curriculum ran.

    python -m cam3r.train --adt-root /path/to/adt --out runs/cam3r --epochs 10

The paper trains 300-500 epochs on 4xH200.  This script is single-process; it is
the recipe, not a claim that a laptop reproduces the tables.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cam3r.adt import ADTPairDataset, find_adt_sequences
from cam3r.cli import add_adt_args, add_model_args, config_from_args, require_adt_root
from cam3r.data import CurriculumSampler, TwoViewSource
from cam3r.losses import cam3r_loss
from cam3r.model import CAM3R
from cam3r.pretrained import initialize_cam3r


def build_sources(args: argparse.Namespace) -> List[TwoViewSource]:
    adt_root = require_adt_root(args)
    # Same --seq spelling as eval_adt, so a held-out sequence can be named on
    # both sides and the two cannot silently overlap.
    seq_dirs = [os.path.join(adt_root, s) for s in args.seq] if args.seq \
        else find_adt_sequences(adt_root, rgb_subdir=args.rgb_subdir)
    if not seq_dirs:
        raise SystemExit(f"no ADT sequences under {adt_root}")
    ds = ADTPairDataset(
        seq_dirs,
        resolution=args.resolution,
        max_frames=args.max_frames,
        rgb_subdir=args.rgb_subdir,
        baseline_m=(args.min_baseline, args.max_baseline),
        angle_deg=(args.min_angle, args.max_angle),
        max_pairs=args.max_pairs,
        extrinsics_json=args.extrinsics_json,
    )
    print(f"  ADT: {len(ds.poses_cw)} frames -> {len(ds)} pairs "
          f"(extrinsics: {ds.extrinsics_source})")
    return [TwoViewSource(ds, kind="kb4", name="adt")]


def to_batch(sample: Dict, device: torch.device) -> Dict:
    """One dataset item -> a batch of 1 on ``device``."""
    return {
        "images": [sample["images"][v].unsqueeze(0).to(device) for v in range(2)],
        "rays": [sample["rays"][v].unsqueeze(0).to(device) for v in range(2)],
        "points": [sample["points"][v].unsqueeze(0).to(device) for v in range(2)],
        "valid": [sample["valid"][v].unsqueeze(0).to(device) for v in range(2)],
        "R": sample["R"].unsqueeze(0).to(device),
        "t": sample["t"].unsqueeze(0).to(device),
    }


def lr_at(step: int, total: int, base_lr: float, warmup: int, min_lr: float = 1e-6) -> float:
    """Table S2: linear warmup then cosine decay onto a ``min_lr`` floor.

    ``warmup`` is in *steps*; the caller converts the paper's 10 epochs.
    """
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr + (base_lr - min_lr) * cosine


def run_phase(
    model: CAM3R,
    sources: List[TwoViewSource],
    phase: int,
    args: argparse.Namespace,
    device: torch.device,
    opt: torch.optim.Optimizer,
) -> List[Dict]:
    sampler = CurriculumSampler(sources, phase=phase, hetero_ratio=args.hetero_ratio, seed=args.seed)
    if phase == 2 and not sampler.can_do_heterogeneous():
        print("  NOTE: no source declares supports_heterogeneous, so phase 2 cannot "
              "form cross-lens pairs and is equivalent to phase 1. The paper's "
              "ablation says this costs a lot (65.4 vs 97.7 RRA@15 on 2D3DS).")

    steps_per_epoch = args.steps_per_epoch or max(1, len(sampler))
    total = steps_per_epoch * args.epochs
    # Table S2 gives warmup in epochs; a short run must not spend all of it warming up.
    warmup = min(args.warmup_epochs * steps_per_epoch, max(1, total // 2))
    history: List[Dict] = []
    step = 0
    model.train()

    for epoch in range(args.epochs):
        t0 = time.time()
        running: Dict[str, float] = {}
        for _ in range(steps_per_epoch):
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total, args.lr, warmup, args.min_lr)

            draw = sampler.sample()
            source = sources[draw.source]
            if draw.heterogeneous:
                sample = source.dataset.get(draw.index, heterogeneous=True)  # type: ignore[attr-defined]
            else:
                sample = source.dataset[draw.index]              # type: ignore[index]
            batch = to_batch(sample, device)

            out = model(batch["images"][0], batch["images"][1])
            loss, logs = cam3r_loss(
                out, batch,
                lambda_a=args.lambda_a, lambda_regr=args.lambda_regr, lambda_pose=args.lambda_pose,
                beta=args.beta, conf_mode=args.conf_mode,
            )
            (loss / args.accum).backward()

            if (step + 1) % args.accum == 0:
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                opt.step()
                opt.zero_grad(set_to_none=True)

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            step += 1

        entry = {"phase": phase, "epoch": epoch,
                 **{k: round(v / steps_per_epoch, 5) for k, v in running.items()},
                 "lr": round(opt.param_groups[0]["lr"], 8),
                 "sec": round(time.time() - t0, 1)}
        history.append(entry)
        print(f"  [p{phase} ep{epoch:03d}] total {entry['loss/total']:.4f}  "
              f"ang {entry['loss/angular']:.4f}  regr {entry['loss/regr']:.4f}  "
              f"pose {entry['loss/pose']:.4f}  ({entry['sec']}s)", flush=True)
        checkpoint(model, opt, args, history, "cam3r_last.pt", phase=phase, epoch=epoch)
    return history


def checkpoint(model, opt, args, history: List[Dict], name: str, **extra) -> str:
    """Write a resumable snapshot, atomically.

    Every epoch, not just at the end -- a long run on a shared box should not
    lose hours to a reboot, and ``.tmp`` + rename means a crash mid-write leaves
    the previous snapshot intact rather than a truncated file.
    """
    path = os.path.join(args.out, name)
    torch.save(
        {"model": model.state_dict(), "optimizer": opt.state_dict(),
         "config": vars(config_from_args(args)), "args": vars(args),
         "history": history, **extra},
        path + ".tmp",
    )
    os.replace(path + ".tmp", path)
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return path


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    sources = build_sources(args)
    cfg = config_from_args(args)
    model = CAM3R(cfg).to(device)
    initialize_cam3r(model, dust3r=args.dust3r, unik3d=args.unik3d)
    print(f"  CAM3R: {model.num_parameters() / 1e6:.1f}M parameters", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)

    phases = [1, 2] if args.phase == "both" else [int(args.phase)]
    history: List[Dict] = []
    for phase in phases:
        history += run_phase(model, sources, phase, args, device, opt)

    ckpt = checkpoint(model, opt, args, history, "cam3r_final.pt")
    print(f"\n  saved {ckpt}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_adt_args(p)
    add_model_args(p)
    p.add_argument("--seq", nargs="*", default=None,
                   help="sequence dir names to train on (default: auto-discover)")
    p.add_argument("--out", default="runs/cam3r")

    p.add_argument("--phase", default="both", choices=["1", "2", "both"])
    p.add_argument("--hetero-ratio", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--lr", type=float, default=5e-5, help="Sec. 4.2 initial lr")
    p.add_argument("--warmup-epochs", type=int, default=10, help="Table S2: 10 epochs")
    p.add_argument("--min-lr", type=float, default=1e-6, help="Table S2 cosine floor")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    p.add_argument("--clip-grad", type=float, default=1.0)

    p.add_argument("--lambda-a", type=float, default=1.0)
    p.add_argument("--lambda-regr", type=float, default=1.0)
    p.add_argument("--lambda-pose", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--conf-mode", default="dust3r", choices=["none", "dust3r"])

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main()
