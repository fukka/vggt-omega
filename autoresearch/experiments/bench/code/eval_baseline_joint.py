"""BENCH: joint (theta x GT-depth) tables for any model_zoo baseline on ADT.

Reuses the baselines' own loaders (ADTWindowDataset via _load_frames,
aria_intrinsics(rotated=True), build_adapter -> planar z), and adds the
protocol-of-record accumulation (range domain, scale_shift per frame frozen
before binning). Theta comes from the repo's KB4 with the 2026-08-23
bisection-safeguarded inversion.

Usage:
    python autoresearch/experiments/bench/code/eval_baseline_joint.py \
        --model unik3d_vitl --adt-root <root with the held-out seq> \
        --out results/bench_<model>_<seq>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from finetune.eval.metrics import align_depth  # noqa: E402
from finetune.eval.baselines.model_zoo import REGISTRY as SPECS, build_adapter  # noqa: E402
from finetune.eval.baselines.aria_fisheye import aria_intrinsics  # noqa: E402
from finetune.eval.baselines.benchmark_adt import _load_frames  # noqa: E402
from raytun3r.cameras import KannalaBrandt  # noqa: E402

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   choices=[s.key for s in SPECS])
    p.add_argument("--adt-root", required=True)
    p.add_argument("--rgb-subdir", default="videos_rgb")
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    spec = next(s for s in SPECS if s.key == args.model)
    adapter = build_adapter(spec).load(args.device)

    frames = _load_frames(args.adt_root, args.rgb_subdir, False, args.res,
                          args.max_frames)
    # the loader snaps to its own token-friendly shape; the camera and the
    # theta grid must follow the frames, not the requested --res
    fh, fw = frames[0][1].shape
    cam = aria_intrinsics(fh, fw, rotated=True)
    kb = KannalaBrandt(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
                       width=fw, height=fh, k=tuple(cam.k),
                       theta_max=cam.usable_theta_max())
    theta = kb.incidence_grid(fh, fw).numpy()
    cone = theta <= float(kb.theta_max)
    cos_t = np.cos(theta)
    t_edges = np.linspace(0.0, float(kb.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta, t_edges) - 1, 0, THETA_BINS - 1)
    print(f"[bench] {args.model}: {len(frames)} frames at {fh}x{fw}")

    nb_d = len(GT_DEPTH_EDGES) - 1
    s_ = np.zeros((THETA_BINS, nb_d))
    n_ = np.zeros((THETA_BINS, nb_d), dtype=np.int64)
    whole = {"absrel": 0.0, "d1": 0.0, "n": 0}
    for k, (rgb, gt_z, valid_in) in enumerate(frames):
        t0 = time.time()
        with torch.no_grad():
            z = adapter.predict_frame(rgb, cam, f"f{k}")   # planar z, doc'd
        z = np.asarray(z, np.float32)
        if z.shape != (fh, fw):
            # adapters may snap to their own patch multiple; resize back
            z = torch.nn.functional.interpolate(
                torch.from_numpy(z)[None, None], size=(fh, fw),
                mode="bilinear", align_corners=False)[0, 0].numpy()
        d = z / np.clip(cos_t, 1e-6, None)
        gr = gt_z / np.clip(cos_t, 1e-6, None)
        valid = (cone & (gt_z > 0) & (valid_in > 0.5)
                 & (gr <= args.depth_max_m) & (d > 1e-6))
        if valid.sum() < 1000:
            continue
        aligned = align_depth(d, gr, valid, mode="scale_shift")
        absrel_px = (np.abs(aligned - gr) / np.clip(gr, 1e-6, None))[valid]
        ratio = np.maximum(aligned / np.clip(gr, 1e-6, None),
                           gr / np.clip(aligned, 1e-6, None))[valid]
        whole["absrel"] += float(absrel_px.sum())
        whole["d1"] += float((ratio < 1.25).sum())
        whole["n"] += int(valid.sum())
        ti = t_idx[valid]
        di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
        flat = ti * nb_d + di
        s_ += np.bincount(flat, weights=absrel_px,
                          minlength=THETA_BINS * nb_d).reshape(THETA_BINS, nb_d)
        n_ += np.bincount(flat, minlength=THETA_BINS * nb_d
                          ).reshape(THETA_BINS, nb_d)
        print(f"  frame {k + 1}/{len(frames)} ({time.time() - t0:4.1f}s)",
              flush=True)

    cell = s_ / np.maximum(n_, 1)
    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]
    print(f"\n{args.model}: whole-image AbsRel "
          f"{whole['absrel'] / max(whole['n'], 1):.4f}, "
          f"d1 {whole['d1'] / max(whole['n'], 1):.4f}")
    print("joint AbsRel (rows=GT depth, cols=theta):")
    for j in range(nb_d):
        print(f"{GT_DEPTH_EDGES[j]:4.0f}-{GT_DEPTH_EDGES[j + 1]:2.0f} m  "
              + " ".join(f"{cell[i, j]:5.3f}" for i in range(THETA_BINS)))

    if args.out:
        dst = Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump({"model": args.model, "adt_root": args.adt_root,
                       "whole_absrel": whole["absrel"] / max(whole["n"], 1),
                       "whole_d1": whole["d1"] / max(whole["n"], 1),
                       "joint": cell.tolist(), "counts": n_.tolist(),
                       "theta_bin_mid_deg": t_mid,
                       "config": vars(args)}, f, indent=2)
        print(f"[bench] wrote {dst}")


if __name__ == "__main__":
    main()
