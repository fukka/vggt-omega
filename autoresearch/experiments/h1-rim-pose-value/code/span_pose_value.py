"""H1.1: is the periphery's pose value angular SPAN rather than per-point quality?

Protocol: ../protocol-h1.1.md (committed before this ran). Classical-only, CPU.

Cumulative disks theta <= T, count-matched to the smallest condition per pair
(5 seeded resamples, per-pair median), rotation error + translation-direction
error, real and synthetic arms, same denominator discipline as H1.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h1-rim-pose-value/code/span_pose_value.py \
        --path ~/Desktop/ADT/scannetpp_example/3f15a9266d --out results/run_003.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from raytun3r.data import ScanNetPPFisheye            # noqa: E402
from raytun3r.matching import relative_pose_magsac    # noqa: E402
from raytun3r.metrics import rotation_error_deg, translation_error_deg  # noqa: E402

from rim_pose_value import (_gain, _median, scatter, sift_matches,  # noqa: E402
                            synth_targets, theta_of)

SPANS = (35.0, 45.0, 55.0, 65.0, 85.0)
RESAMPLES = 5
MIN_PER_COND = 20


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", required=True)
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--max-gt-rot-deg", type=float, default=30.0)
    p.add_argument("--max-pairs", type=int, default=120)
    p.add_argument("--nfeatures", type=int, default=6000)
    p.add_argument("--ratio", type=float, default=0.8)
    p.add_argument("--magsac-thresh-deg", type=float, default=0.5)
    p.add_argument("--min-gt-trans-m", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    src = ScanNetPPFisheye(os.path.expanduser(args.path), max_size=args.max_size,
                           patch=14, keep_bad=False)
    src.frames = [fr for fr in src.frames
                  if os.path.exists(os.path.join(src._image_root, fr["file_path"]))]
    print(f"[h1.1] {len(src)} frames, grid {src.w}x{src.h}, "
          f"theta_max {math.degrees(src.camera.theta_max):.1f} deg")

    cand: List[Tuple[int, int, float]] = []
    for i in range(len(src)):
        for j in range(i + 1, len(src)):
            gi, gj = src.pose(i), src.pose(j)
            if gi is None or gj is None:
                continue
            R_gt = gj[0] @ gi[0].transpose(-1, -2)
            ang = rotation_error_deg(torch.eye(3, dtype=R_gt.dtype), R_gt)
            if 0.5 < ang <= args.max_gt_rot_deg:
                cand.append((i, j, float(ang)))
    cand.sort(key=lambda c: c[2])
    step = max(1, len(cand) // args.max_pairs)
    cand = cand[::step][:args.max_pairs]
    print(f"[h1.1] {len(cand)} pairs")

    conds = [f"t{int(T)}" for T in SPANS]
    arms = ("real", "synth")
    rot: Dict[str, Dict[str, List[float]]] = {a: {c: [] for c in conds} for a in arms}
    prd: Dict[str, Dict[str, List[float]]] = {a: {c: [] for c in conds} for a in arms}
    trn: Dict[str, Dict[str, List[float]]] = {a: {c: [] for c in conds} for a in arms}
    gts: Dict[str, List[float]] = {a: [] for a in arms}
    nstar: List[int] = []
    dropped = {a: 0 for a in arms}
    rng = np.random.default_rng(args.seed)

    for i, j, ang in cand:
        m = sift_matches(src, i, j, args.nfeatures, args.ratio)
        if m is None:
            continue
        ua, ub = m
        gi, gj = src.pose(i), src.pose(j)
        R_gt = gj[0] @ gi[0].transpose(-1, -2)
        t_rel = (gj[1] - R_gt @ gi[1]).float()
        t_mag = float(t_rel.norm())
        eye = torch.eye(3, dtype=R_gt.dtype)

        for arm in arms:
            if arm == "real":
                sa, sb = ua, ub
            else:
                sa, sb = synth_targets(src, ua, R_gt.float(), t_rel, rng)
            if sa is None or len(sa) < MIN_PER_COND:
                dropped[arm] += 1
                continue
            tha = theta_of(src, sa)
            sels = [np.flatnonzero(tha <= T) for T in SPANS]
            n_star = min(len(s) for s in sels)
            if n_star < MIN_PER_COND:
                dropped[arm] += 1
                continue
            res: Dict[str, Optional[Tuple[float, float, float]]] = {}
            for c, sel in zip(conds, sels):
                rerrs, perrs, terrs = [], [], []
                for k in range(RESAMPLES):
                    sub = rng.choice(sel, size=n_star, replace=False) \
                        if len(sel) > n_star else sel
                    out = relative_pose_magsac(
                        scatter(src, sa[sub], sb[sub]), src.camera,
                        threshold_deg=args.magsac_thresh_deg)
                    if out is None:
                        continue
                    R_hat, t_hat = out[0].to(R_gt), out[1]
                    rerrs.append(rotation_error_deg(R_hat, R_gt))
                    perrs.append(rotation_error_deg(eye, R_hat))
                    if t_mag >= args.min_gt_trans_m:
                        terrs.append(translation_error_deg(
                            t_hat.to(t_rel), t_rel))
                if len(rerrs) < (RESAMPLES + 1) // 2:
                    res[c] = None
                else:
                    res[c] = (_median(rerrs), _median(perrs),
                              _median(terrs) if terrs else float("nan"))
            if any(res[c] is None for c in conds):
                dropped[arm] += 1
                continue
            for c in conds:
                rot[arm][c].append(res[c][0])
                prd[arm][c].append(res[c][1])
                if not math.isnan(res[c][2]):
                    trn[arm][c].append(res[c][2])
            gts[arm].append(ang)
            if arm == "real":
                nstar.append(n_star)

    summary: Dict[str, Dict] = {}
    for arm in arms:
        n = len(gts[arm])
        print(f"\n=== {arm} arm: {n} pairs (dropped {dropped[arm]}), "
              f"N* median {_median([float(x) for x in nstar]):.0f} ===")
        print(f"{'cond':>6} {'med rot err':>12} {'gain':>7} {'med t err':>10} {'n_t':>4}")
        summary[arm] = {"n_pairs": n, "dropped": dropped[arm], "conds": {}}
        for c in conds:
            g = _gain(prd[arm][c], gts[arm])
            print(f"{c:>6} {_median(rot[arm][c]):12.3f} {g:7.3f} "
                  f"{_median(trn[arm][c]):10.3f} {len(trn[arm][c]):4d}")
            summary[arm]["conds"][c] = {
                "median_rot_err_deg": _median(rot[arm][c]), "gain": g,
                "median_t_err_deg": _median(trn[arm][c]),
                "n_t_pairs": len(trn[arm][c])}
        d = np.asarray(rot[arm][conds[-1]]) - np.asarray(rot[arm][conds[0]])
        summary[arm]["paired_widest_minus_narrowest"] = {
            "median_diff_deg": _median(list(d)),
            "n_wide_better": int((d < 0).sum()),
            "n_narrow_better": int((d > 0).sum())}
        print(f"paired {conds[-1]}-{conds[0]} rot: median {_median(list(d)):+.3f} deg, "
              f"wide better on {(d < 0).sum()}/{len(d)} pairs")
    summary["n_star_median"] = _median([float(x) for x in nstar])
    summary["config"] = vars(args)

    if args.out:
        out = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[h1.1] wrote {out}")


if __name__ == "__main__":
    main()
