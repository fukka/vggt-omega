"""H2.0: DA3-Small depth error vs incidence angle on real Aria, with the
distance control (theta x GT-depth joint table).

Protocol: ../protocol-h2.0-baseline.md (committed before this ran). CPU.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h2-center-safe-adapter/code/depth_baseline.py \
        --out results/run_008.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))

from adt_pose_value import AriaLocalPairs, DEFAULT_SEQ   # noqa: E402
from finetune.eval.metrics import align_depth            # noqa: E402

THETA_BINS = 8
DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
MIN_CELL_PX = 500


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", default=DEFAULT_SEQ)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--align", default="scale_shift",
                   choices=["scale_only", "scale_shift", "disparity_scale_shift"],
                   help="fovbench's protocol of record is scale_shift; "
                        "run_008 used scale_only and is superseded")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    src = AriaLocalPairs(os.path.expanduser(args.seq), size=args.size)
    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device="cpu",
                        variant="small")
    bb.install(None, src.camera, (src.h, src.w), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="range")

    theta = src.camera.incidence_grid(src.h, src.w)
    cone = theta <= src.camera.theta_max
    cos_t = torch.cos(theta)
    t_edges = torch.linspace(0.0, float(src.camera.theta_max), THETA_BINS + 1)
    t_idx = torch.bucketize(theta, t_edges[1:-1])          # 0..THETA_BINS-1

    depth_paths = {os.path.basename(q).replace(".npy", ""): q for q in
                   glob.glob(os.path.join(os.path.expanduser(args.seq),
                                          "depth_npy", "*.npy"))}

    nb_d = len(DEPTH_EDGES) - 1
    abs_sum = np.zeros((THETA_BINS, nb_d))
    rel_cnt = np.zeros((THETA_BINS, nb_d), dtype=np.int64)
    scales: List[float] = []
    for n, path in enumerate(src.paths):
        stem = os.path.basename(path).replace(".jpg", "")
        if stem not in depth_paths:
            continue
        gt_z = torch.from_numpy(np.load(depth_paths[stem]).astype(np.float32))
        gt_z = torch.nn.functional.interpolate(
            gt_z[None, None], size=(src.h, src.w), mode="nearest")[0, 0] / 1000.0
        gt_r = gt_z / cos_t.clamp_min(1e-6)                # planar z -> range
        t0 = time.time()
        with torch.no_grad():
            pred = bb.forward(src.image(n)[None, None])
        pred.require_convention("range")
        d = pred.depth[0]
        valid = cone & (gt_z > 0) & (gt_r <= args.depth_max_m) & (d > 0)
        if int(valid.sum()) < 1000:
            continue
        # one affine per frame, frozen before binning; finetune/eval/metrics.py
        # is the single protocol authority (fovbench align_mode="scale_shift")
        aligned = align_depth(d.numpy(), gt_r.numpy(), valid.numpy(),
                              mode=args.align)
        scales.append(float(np.median(gt_r[valid].numpy()
                                      / np.maximum(d[valid].numpy(), 1e-9))))
        absrel = (np.abs(aligned - gt_r.numpy()) / gt_r.numpy())[valid.numpy()]
        ti = t_idx[valid].numpy()
        di = np.clip(np.digitize(gt_r[valid].numpy(), DEPTH_EDGES) - 1, 0, nb_d - 1)
        flat = ti * nb_d + di
        abs_sum += np.bincount(flat, weights=absrel,
                               minlength=THETA_BINS * nb_d).reshape(THETA_BINS, nb_d)
        rel_cnt += np.bincount(flat,
                               minlength=THETA_BINS * nb_d).reshape(THETA_BINS, nb_d)
        print(f"  frame {n + 1}/{len(src.paths)} ({time.time() - t0:4.1f}s) "
              f"scale {s:.3f} valid {int(valid.sum())}", flush=True)

    with np.errstate(invalid="ignore"):
        cell = abs_sum / np.maximum(rel_cnt, 1)
    marg_theta = abs_sum.sum(1) / np.maximum(rel_cnt.sum(1), 1)
    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1])) for i in range(THETA_BINS)]

    print("\nuncontrolled AbsRel per theta bin (pooled over frames):")
    for i in range(THETA_BINS):
        print(f"  {t_mid[i]:5.1f} deg: {marg_theta[i]:.4f}  (n={rel_cnt.sum(1)[i]})")
    print(f"uncontrolled spread max/min: {marg_theta.max() / marg_theta.min():.3f}")

    print("\njoint AbsRel table (rows=depth bands, cols=theta bins; "
          f"cells <{MIN_CELL_PX}px flagged *):")
    print("depth\\theta  " + " ".join(f"{t:5.1f}" for t in t_mid))
    row_spread = {}
    for j in range(nb_d):
        cells = []
        ok = []
        for i in range(THETA_BINS):
            v = cell[i, j]
            flag = "*" if rel_cnt[i, j] < MIN_CELL_PX else " "
            cells.append(f"{v:5.3f}{flag}")
            if rel_cnt[i, j] >= MIN_CELL_PX:
                ok.append(v)
        spread = (max(ok) / min(ok)) if len(ok) >= 2 and min(ok) > 0 else float("nan")
        row_spread[f"{DEPTH_EDGES[j]}-{DEPTH_EDGES[j + 1]}m"] = spread
        print(f"{DEPTH_EDGES[j]:4.0f}-{DEPTH_EDGES[j + 1]:2.0f} m   "
              + " ".join(cells) + f"   row spread {spread:.3f}")

    out = {
        "n_frames": len(scales),
        "scale_median": float(np.median(scales)) if scales else None,
        "theta_bin_mid_deg": t_mid,
        "uncontrolled_absrel": marg_theta.tolist(),
        "uncontrolled_spread": float(marg_theta.max() / marg_theta.min()),
        "joint_absrel": cell.tolist(),
        "joint_counts": rel_cnt.tolist(),
        "depth_edges_m": list(DEPTH_EDGES),
        "row_spread": row_spread,
        "config": vars(args),
    }
    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[h2.0] wrote {dst}")


if __name__ == "__main__":
    main()
