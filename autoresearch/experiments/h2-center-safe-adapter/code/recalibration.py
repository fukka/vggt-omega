"""H2.1: the 48-parameter (theta x predicted-depth) recalibration table.

Protocol: ../protocol-h2.1-recalibration.md (committed before this ran). CPU.

Fit c[i,j] on training frames, apply d' = d * exp(c) on held-out frames,
evaluate corrected vs uncorrected under the protocol of record (run_008b).

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h2-center-safe-adapter/code/recalibration.py \
        --split even_odd --out results/run_010_even_odd.json
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
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))

from adt_pose_value import AriaLocalPairs, DEFAULT_SEQ   # noqa: E402
from finetune.eval.metrics import align_depth            # noqa: E402

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)         # eval table rows
PRED_EDGES_M = (0.3, 0.6, 1.2, 2.4, 4.8, 9.6, 20.0)      # table's pred-depth bins
MIN_FIT_PX = 2000
CACHE = Path(os.environ.get("H2_CACHE",
             "/private/tmp/claude-501/-Users-fengjiazhang-Desktop-ADT-vggt-omega/"
             "c1591315-71f0-4ca2-9183-9f58fff82acc/scratchpad/h2_pred_cache"))


def predictions(src) -> Dict[int, np.ndarray]:
    """Per-frame predicted range maps, cached on disk (deterministic model)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    preds: Dict[int, np.ndarray] = {}
    todo = [n for n in range(len(src.paths))
            if not (CACHE / f"{n}.npy").exists()]
    if todo:
        from raytun3r.backbones import build_backbone
        bb = build_backbone("da3", weights="pretrained", device="cpu",
                            variant="small")
        bb.install(None, src.camera, (src.h, src.w), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="range")
        for n in todo:
            t0 = time.time()
            with torch.no_grad():
                pr = bb.forward(src.image(n)[None, None])
            pr.require_convention("range")
            np.save(CACHE / f"{n}.npy", pr.depth[0].numpy())
            print(f"  cached pred {n} ({time.time() - t0:4.1f}s)", flush=True)
    for n in range(len(src.paths)):
        preds[n] = np.load(CACHE / f"{n}.npy")
    return preds


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", default=DEFAULT_SEQ)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--split", default="even_odd", choices=["even_odd", "halves"])
    p.add_argument("--fixed-affine", action="store_true",
                   help="fit the eval affine on the UNCORRECTED prediction and "
                        "apply it to both arms — isolates the table's local "
                        "effect from the re-alignment coupling")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    src = AriaLocalPairs(os.path.expanduser(args.seq), size=args.size)
    theta = src.camera.incidence_grid(src.h, src.w)
    cone = (theta <= src.camera.theta_max).numpy()
    cos_t = torch.cos(theta).numpy()
    t_edges = np.linspace(0.0, float(src.camera.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)

    depth_paths = {os.path.basename(q).replace(".npy", ""): q for q in
                   glob.glob(os.path.join(os.path.expanduser(args.seq),
                                          "depth_npy", "*.npy"))}
    frames = [n for n in range(len(src.paths))
              if os.path.basename(src.paths[n]).replace(".jpg", "") in depth_paths]
    if args.split == "even_odd":
        train = [n for k, n in enumerate(frames) if k % 2 == 0]
        test = [n for k, n in enumerate(frames) if k % 2 == 1]
    else:
        half = len(frames) // 2
        train, test = frames[:half], frames[half:]
    print(f"[h2.1] split {args.split}: {len(train)} train / {len(test)} test")

    preds = predictions(src)

    def gt_range(n: int) -> np.ndarray:
        stem = os.path.basename(src.paths[n]).replace(".jpg", "")
        gz = np.load(depth_paths[stem]).astype(np.float32)
        gz = torch.nn.functional.interpolate(
            torch.from_numpy(gz)[None, None], size=(src.h, src.w),
            mode="nearest")[0, 0].numpy() / 1000.0
        return gz / np.clip(cos_t, 1e-6, None), gz

    nb_p = len(PRED_EDGES_M) - 1
    # ---- fit ----
    num = np.zeros((THETA_BINS, nb_p))
    cnt = np.zeros((THETA_BINS, nb_p), dtype=np.int64)
    res_store: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for n in train:
        gr, gz = gt_range(n)
        d = preds[n]
        valid = cone & (gz > 0) & (gr <= args.depth_max_m) & (d > 1e-6)
        r = np.log(gr[valid]) - np.log(d[valid])
        r = r - np.median(r)                       # frame level removed
        ti = t_idx[valid]
        pi = np.clip(np.digitize(d[valid], PRED_EDGES_M) - 1, 0, nb_p - 1)
        for i in range(THETA_BINS):
            for j in range(nb_p):
                sel = (ti == i) & (pi == j)
                if sel.any():
                    res_store.setdefault((i, j), []).append(r[sel])
                    cnt[i, j] += int(sel.sum())
    c = np.zeros((THETA_BINS, nb_p))
    for (i, j), chunks in res_store.items():
        if cnt[i, j] >= MIN_FIT_PX:
            c[i, j] = float(np.median(np.concatenate(chunks)))
    print(f"[h2.1] table: {int((c != 0).sum())}/{THETA_BINS * nb_p} cells fit, "
          f"|c| max {np.abs(c).max():.3f}")

    # ---- evaluate on held-out frames, protocol of record ----
    nb_d = len(GT_DEPTH_EDGES) - 1

    def joint_absrel(correct: bool) -> Tuple[np.ndarray, np.ndarray]:
        s_ = np.zeros((THETA_BINS, nb_d))
        n_ = np.zeros((THETA_BINS, nb_d), dtype=np.int64)
        for n in test:
            gr, gz = gt_range(n)
            d = preds[n].copy()
            if correct:
                pi = np.clip(np.digitize(d, PRED_EDGES_M) - 1, 0, nb_p - 1)
                d = d * np.exp(c[t_idx, pi])
            valid = cone & (gz > 0) & (gr <= args.depth_max_m) & (d > 1e-6)
            if args.fixed_affine:
                d0 = preds[n]
                a0 = align_depth(d0, gr, valid, mode="scale_shift")
                m = valid & (d0 > 1e-6)
                A = np.polyfit(d0[m], a0[m], 1)     # recover (a,b) of the fit
                aligned = A[0] * d + A[1]
            else:
                aligned = align_depth(d, gr, valid, mode="scale_shift")
            absrel = (np.abs(aligned - gr) / gr)[valid]
            ti = t_idx[valid]
            di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
            flat = ti * nb_d + di
            s_ += np.bincount(flat, weights=absrel,
                              minlength=THETA_BINS * nb_d).reshape(THETA_BINS, nb_d)
            n_ += np.bincount(flat,
                              minlength=THETA_BINS * nb_d).reshape(THETA_BINS, nb_d)
        return s_ / np.maximum(n_, 1), n_

    before, counts = joint_absrel(False)
    after, _ = joint_absrel(True)
    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]
    print("\nheld-out joint AbsRel, BEFORE -> AFTER (rows=GT depth, cols=theta):")
    print("depth\\theta  " + " ".join(f"{t:12.1f}" for t in t_mid))
    for j in range(nb_d):
        row = " ".join(f"{before[i, j]:5.3f}>{after[i, j]:5.3f}"
                       for i in range(THETA_BINS))
        print(f"{GT_DEPTH_EDGES[j]:4.0f}-{GT_DEPTH_EDGES[j + 1]:2.0f} m  {row}")

    near_rim = [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                if t_mid[i] >= 38 and GT_DEPTH_EDGES[j + 1] <= 2.0]
    ctr = [(i, j) for i in range(THETA_BINS) for j in range(nb_d) if t_mid[i] <= 11]
    far = [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
           if GT_DEPTH_EDGES[j] >= 3.0]

    def zone(cells):
        w = np.array([counts[i, j] for i, j in cells], dtype=float)
        b = np.array([before[i, j] for i, j in cells])
        a = np.array([after[i, j] for i, j in cells])
        return float((b * w).sum() / w.sum()), float((a * w).sum() / w.sum())

    summary = {"split": args.split, "table": c.tolist(),
               "pred_edges_m": list(PRED_EDGES_M),
               "before": before.tolist(), "after": after.tolist(),
               "counts": counts.tolist(), "theta_bin_mid_deg": t_mid,
               "zones": {}}
    for name, cells in (("near_rim(<=2m,>=38deg)", near_rim),
                        ("center(<=11deg)", ctr), ("far(>=3m)", far)):
        b, a = zone(cells)
        summary["zones"][name] = {"before": b, "after": a}
        print(f"{name}: {b:.3f} -> {a:.3f}  ({(a - b) / b * 100:+.1f}%)")

    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[h2.1] wrote {dst}")


if __name__ == "__main__":
    main()
