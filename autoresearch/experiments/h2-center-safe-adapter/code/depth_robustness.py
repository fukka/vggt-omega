"""H2.0b: alignment-free (theta x depth) bias and dispersion maps.

Protocol: ../protocol-h2.0b-robustness.md (committed before this ran). CPU.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h2-center-safe-adapter/code/depth_robustness.py \
        --out results/run_009.json
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
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))

from adt_pose_value import AriaLocalPairs, DEFAULT_SEQ   # noqa: E402

THETA_BINS = 8
DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
MIN_CELL_PX = 500


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", default=DEFAULT_SEQ)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--depth-max-m", type=float, default=10.0)
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
    t_idx = torch.bucketize(theta, t_edges[1:-1])

    depth_paths = {os.path.basename(q).replace(".npy", ""): q for q in
                   glob.glob(os.path.join(os.path.expanduser(args.seq),
                                          "depth_npy", "*.npy"))}
    nb_d = len(DEPTH_EDGES) - 1

    # Per-cell pooled residual lists are too big; accumulate per-frame cell
    # medians and per-frame cell MADs, then take medians across frames (each
    # frame counts once — the coarse-bin convention fovbench documents).
    cell_bias: List[List[List[float]]] = [[[] for _ in range(nb_d)]
                                          for _ in range(THETA_BINS)]
    cell_disp: List[List[List[float]]] = [[[] for _ in range(nb_d)]
                                          for _ in range(THETA_BINS)]
    cell_n = np.zeros((THETA_BINS, nb_d), dtype=np.int64)

    for n, path in enumerate(src.paths):
        stem = os.path.basename(path).replace(".jpg", "")
        if stem not in depth_paths:
            continue
        gt_z = torch.from_numpy(np.load(depth_paths[stem]).astype(np.float32))
        gt_z = torch.nn.functional.interpolate(
            gt_z[None, None], size=(src.h, src.w), mode="nearest")[0, 0] / 1000.0
        gt_r = gt_z / cos_t.clamp_min(1e-6)
        t0 = time.time()
        with torch.no_grad():
            pred = bb.forward(src.image(n)[None, None])
        pred.require_convention("range")
        d = pred.depth[0]
        valid = cone & (gt_z > 0) & (gt_r <= args.depth_max_m) & (d > 1e-6)
        if int(valid.sum()) < 1000:
            continue
        r = (torch.log(gt_r) - torch.log(d))[valid].numpy()
        frame_med = float(np.median(r))
        ti = t_idx[valid].numpy()
        di = np.clip(np.digitize(gt_r[valid].numpy(), DEPTH_EDGES) - 1, 0, nb_d - 1)
        for i in range(THETA_BINS):
            for j in range(nb_d):
                sel = (ti == i) & (di == j)
                cnt = int(sel.sum())
                cell_n[i, j] += cnt
                if cnt < 50:          # per-frame minimum for a stable median
                    continue
                cm = float(np.median(r[sel]))
                cell_bias[i][j].append(cm - frame_med)
                cell_disp[i][j].append(float(np.median(np.abs(r[sel] - cm))))
        print(f"  frame {n + 1}/{len(src.paths)} ({time.time() - t0:4.1f}s)",
              flush=True)

    bias = np.full((THETA_BINS, nb_d), np.nan)
    disp = np.full((THETA_BINS, nb_d), np.nan)
    for i in range(THETA_BINS):
        for j in range(nb_d):
            if cell_bias[i][j]:
                bias[i, j] = float(np.median(cell_bias[i][j]))
                disp[i, j] = float(np.median(cell_disp[i][j]))

    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]
    for name, M in (("BIAS (log-depth, vs frame level)", bias),
                    ("DISPERSION (MAD of log residual)", disp)):
        print(f"\n{name}: rows=depth bands, cols=theta")
        print("depth\\theta  " + " ".join(f"{t:6.1f}" for t in t_mid))
        for j in range(nb_d):
            cells = " ".join(
                f"{M[i, j]:+6.3f}" if np.isfinite(M[i, j]) else "  --- "
                for i in range(THETA_BINS))
            print(f"{DEPTH_EDGES[j]:4.0f}-{DEPTH_EDGES[j + 1]:2.0f} m  {cells}")

    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump({"theta_bin_mid_deg": t_mid,
                       "depth_edges_m": list(DEPTH_EDGES),
                       "bias_log": bias.tolist(), "dispersion_mad": disp.tolist(),
                       "cell_counts": cell_n.tolist(),
                       "config": vars(args)}, f, indent=2)
        print(f"\n[h2.0b] wrote {dst}")


if __name__ == "__main__":
    main()
