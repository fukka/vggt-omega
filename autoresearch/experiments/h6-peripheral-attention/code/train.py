"""H6 trainer: peripheral cross-frame attention module on frozen DA3-Small.

Protocol: ../protocol.md. Module mechanics verified by module_smoke.py
(zero-init identity; camera outputs bit-identical by construction, so the
H5 rim-feature-preservation loss is structurally unnecessary here).

Losses: compression-weighted depth (both frames, module applied in both
temporal directions) + multi-frame rim consistency. Saves module weights
(~12 MB) + config.

Usage (box):
    python autoresearch/experiments/h6-peripheral-attention/code/train.py \
        --train-seqs <4 clean seq dirs> --epochs 20 \
        --out-dir results/autoresearch-h6-train/full
CPU smoke:
    ... --train-seqs <seq131> --epochs 1 --steps-per-epoch 2 --size 252
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import losses  # noqa: E402  (from h5 code dir)
# both trainers are named train.py — load the H5 one by explicit path
import importlib.util as _ilu  # noqa: E402
_h5_spec = _ilu.spec_from_file_location(
    "h5_train", Path(__file__).resolve().parents[2]
    / "h5-rim-finetune" / "code" / "train.py")
_h5 = _ilu.module_from_spec(_h5_spec)
_h5_spec.loader.exec_module(_h5)
Seq, camera_conjugation = _h5.Seq, _h5.camera_conjugation
from peripheral_attn import (PeripheralCrossFrameAttention,  # noqa: E402
                             apply_to_final_level, rim_mask_for)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-seqs", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rim-deg", type=float, default=35.0)
    p.add_argument("--all-token-control", action="store_true",
                   help="efficiency control: queries = ALL tokens, same params")
    p.add_argument("--depth-alpha", type=float, default=2.0)
    p.add_argument("--lambda-m", type=float, default=0.5)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    seqs = [Seq(s.strip(), args.size, args.max_frames)
            for s in args.train_seqs.split(",") if s.strip()]
    C = camera_conjugation()
    print(f"[h6] {len(seqs)} sequences, "
          f"{sum(len(s.pairs) for s in seqs)} training pairs")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=args.device,
                        variant="small")
    cam = seqs[0].src.camera
    h = w = args.size
    bb.install(None, cam, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb
    for prm in net.parameters():
        prm.requires_grad_(False)

    theta = cam.incidence_grid(h, w)
    cone = theta <= cam.theta_max
    cos_t = torch.cos(theta).to(args.device)
    gh, gw = h // 14, w // 14
    theta_p = theta.reshape(gh, 14, gw, 14).mean((1, 3)).ravel()
    rim = (torch.ones_like(theta_p, dtype=torch.bool)
           if args.all_token_control else rim_mask_for(theta_p, args.rim_deg))
    print(f"[h6] queries: {int(rim.sum())}/{len(rim)} patches "
          f"({'ALL-TOKEN CONTROL' if args.all_token_control else 'rim only'})")

    # probe dim
    with torch.no_grad():
        f0, _ = net.backbone(seqs[0].src.image(seqs[0].frames[0])
                             [None, None].to(args.device),
                             cam_token=None, export_feat_layers=[])
    module = PeripheralCrossFrameAttention(f0[-1][0].shape[-1]).to(args.device)
    print(f"[h6] module {sum(x.numel() for x in module.parameters())/1e6:.2f}M params")

    def depth_from(feats) -> torch.Tensor:
        out = net._process_depth_head(list(feats), h, w)
        z = (out["depth"] if isinstance(out, dict) else out.depth)
        z = z.reshape(h, w)
        return z / cos_t.clamp(min=1e-6)      # planar z -> range, once

    opt = torch.optim.Adam(module.parameters(), lr=args.lr)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log: List[Dict] = []
    for ep in range(args.epochs):
        pairs = [(s, a, b) for s in seqs for a, b in s.pairs]
        random.shuffle(pairs)
        if args.steps_per_epoch:
            pairs = pairs[:args.steps_per_epoch]
        agg = {"depth": 0.0, "mv": 0.0, "total": 0.0}
        t0 = time.time()
        for s, a, b in pairs:
            opt.zero_grad()
            with torch.no_grad():
                fa, _ = net.backbone(s.src.image(a)[None, None].to(args.device),
                                     cam_token=None, export_feat_layers=[])
                fb, _ = net.backbone(s.src.image(b)[None, None].to(args.device),
                                     cam_token=None, export_feat_layers=[])
            da = depth_from(apply_to_final_level(module, fa, fb, rim))
            db = depth_from(apply_to_final_level(module, fb, fa, rim))
            gta = s.gt_range(a, cos_t.cpu()).to(args.device)
            gtb = s.gt_range(b, cos_t.cpu()).to(args.device)
            R_rel, t_rel = s.rel_pose(a, b, C)
            va = cone.to(args.device) & (gta > 0) & (gta <= args.depth_max_m)
            vb = cone.to(args.device) & (gtb > 0) & (gtb <= args.depth_max_m)
            l_d = (losses.depth_loss(da, gta, va, theta, alpha=args.depth_alpha)
                   + losses.depth_loss(db, gtb, vb, theta,
                                       alpha=args.depth_alpha))
            l_m = losses.multiframe_rim_loss(da, db, cam, R_rel, t_rel, theta)
            loss = l_d + args.lambda_m * l_m
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(module.parameters()), 1.0)
            opt.step()
            for k, v in (("depth", l_d), ("mv", l_m), ("total", loss)):
                agg[k] += float(v)
        n = max(1, len(pairs))
        row = {k: v / n for k, v in agg.items()}
        row.update(epoch=ep, n_pairs=len(pairs), sec=round(time.time() - t0, 1))
        log.append(row)
        print(f"[h6] epoch {ep:3d}: total {row['total']:.4f} "
              f"(d {row['depth']:.4f} m {row['mv']:.4f}) {row['sec']}s",
              flush=True)
        torch.save({"module": module.state_dict(), "config": vars(args),
                    "epoch": ep}, out / "module_last.pt")
        with open(out / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)
    print(f"[h6] done; checkpoint + log in {out}")


if __name__ == "__main__":
    main()
