"""BENCH: joint-table row for a RayTun3R-adapted backbone on an ADT sequence.

Loads adapter.pt the same way raytun3r/eval.py does (make_adapter with the
n_radial/n_angular recorded in the checkpoint's args, load_state_dict,
install), then runs per-frame depth and accumulates the protocol-of-record
joint table. --adapter omitted = the vanilla row on the same frames.

Usage:
    python autoresearch/experiments/bench/code/raytun3r_row.py \
        --path <ADT seq dir> [--adapter runs/rt3r_<seq>/adapter.pt] \
        --out results/rt3r_<seq>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from finetune.eval.metrics import align_depth  # noqa: E402
from raytun3r.backbones import build_backbone  # noqa: E402
from raytun3r.data import ADTSequence  # noqa: E402

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--backbone", default="da3")
    p.add_argument("--variant", default="small")
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--extrinsics-json",
                   default="cam3r/data/adt_camera_rgb_calibration.json")
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    src = ADTSequence(args.path, max_size=args.max_size,
                      extrinsics_json=args.extrinsics_json)
    bb = build_backbone(args.backbone, weights="pretrained",
                        device=args.device,
                        **({"variant": args.variant}
                           if args.backbone == "da3" else {}))

    adapter = None
    if args.adapter:
        ck = torch.load(args.adapter, map_location=args.device)
        a = ck["args"]
        adapter = bb.make_adapter(n_radial=a["n_radial"],
                                  n_angular=a["n_angular"])
        adapter.load_state_dict(ck["adapter"])
        adapter = adapter.to(args.device)
        print(f"[rt3r] adapter loaded: {ck.get('param_breakdown')}")
    bb.install(adapter, src.camera, (src.h, src.w),
               patch_undistort=bool(args.adapter), border_token=bool(args.adapter),
               dpt_grid=bool(args.adapter), depth_convention="range")

    theta = src.camera.incidence_grid(src.h, src.w)
    cone = (theta <= src.camera.theta_max).numpy()
    cos_t = np.cos(theta.numpy())
    t_edges = np.linspace(0.0, float(src.camera.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)

    nb_d = len(GT_DEPTH_EDGES) - 1
    s_ = np.zeros((THETA_BINS, nb_d))
    n_ = np.zeros((THETA_BINS, nb_d), dtype=np.int64)
    frames = list(range(len(src)))[:args.max_frames]
    for k in frames:
        gtd = src.depth(k)
        if gtd is None:
            continue
        gt_range, valid_gt = gtd
        t0 = time.time()
        with torch.no_grad():
            pred = bb.forward(src.image(k)[None, None].to(args.device))
        pred.require_convention("range")
        d = pred.depth[0].cpu().numpy()
        # ADTSequence.depth() already returns euclidean range (converted on
        # load, data.py:389-392). Dividing by cos again double-converts and
        # inflates rim GT by up to 1/cos(theta_max)=1.73x — the bug behind
        # issue #38's first delivery (found 2026-08-19 reconciling #37 vs #38).
        gr = gt_range.numpy()
        valid = (cone & valid_gt.numpy().astype(bool) & (gr > 0)
                 & (gr <= args.depth_max_m) & (d > 1e-6))
        if valid.sum() < 1000:
            continue
        aligned = align_depth(d, gr, valid, mode="scale_shift")
        absrel = (np.abs(aligned - gr) / np.clip(gr, 1e-6, None))[valid]
        ti = t_idx[valid]
        di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
        flat = ti * nb_d + di
        s_ += np.bincount(flat, weights=absrel,
                          minlength=THETA_BINS * nb_d).reshape(THETA_BINS, nb_d)
        n_ += np.bincount(flat, minlength=THETA_BINS * nb_d
                          ).reshape(THETA_BINS, nb_d)
        print(f"  frame {k + 1}/{len(frames)} ({time.time() - t0:4.1f}s)",
              flush=True)

    cell = s_ / np.maximum(n_, 1)
    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]
    tag = "adapted" if args.adapter else "vanilla"
    print(f"\n[rt3r/{tag}] joint AbsRel (rows=GT depth, cols=theta):")
    for j in range(nb_d):
        print(f"{GT_DEPTH_EDGES[j]:4.0f}-{GT_DEPTH_EDGES[j + 1]:2.0f} m  "
              + " ".join(f"{cell[i, j]:5.3f}" for i in range(THETA_BINS)))
    if args.out:
        dst = Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump({"path": args.path, "adapter": args.adapter,
                       "backbone": args.backbone, "joint": cell.tolist(),
                       "counts": n_.tolist(), "theta_bin_mid_deg": t_mid,
                       "config": vars(args)}, f, indent=2)
        print(f"[rt3r] wrote {dst}")


if __name__ == "__main__":
    main()
