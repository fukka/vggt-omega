"""H1: does the fisheye rim carry pose value per correspondence?

Protocol: ../protocol.md (committed before this ran). Classical-only, CPU.

For each usable frame pair: SIFT-match, split matches into incidence-angle (theta)
quartile bins of that pair's matches (equal count by construction), estimate the
relative pose per bin with the repo's lens-fair MAGSAC++ wrapper, and compare
rotation error per bin. A synthetic-bearing control repeats the estimation on the
same source pixels with GT-consistent targets and 1 px noise, isolating bin
geometry from SIFT feature quality.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h1-rim-pose-value/code/rim_pose_value.py \
        --path ~/Desktop/ADT/scannetpp_example/3f15a9266d --out results/run_001.json
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
from raytun3r.matching import Matches, relative_pose_magsac  # noqa: E402
from raytun3r.metrics import rotation_error_deg       # noqa: E402

BINS = 4


def _median(v: List[float]) -> float:
    return float(np.median(v)) if len(v) else float("nan")


def _gain(preds: List[float], gts: List[float]) -> float:
    p, g = np.asarray(preds, float), np.asarray(gts, float)
    ok = ~np.isnan(p) & (g > 0)
    return float((p[ok] * g[ok]).sum() / (g[ok] ** 2).sum()) if ok.any() else float("nan")


def sift_matches(src, a: int, b: int, nfeat: int, ratio: float):
    """Matched (uv_a, uv_b) pixel arrays, or None."""
    import cv2

    sift = cv2.SIFT_create(nfeatures=nfeat)
    bf = cv2.BFMatcher()

    def gray(i):
        arr = (src.image(i).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    ka, da = sift.detectAndCompute(gray(a), None)
    kb, db = sift.detectAndCompute(gray(b), None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None
    good = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < ratio * n.distance]
    if len(good) < 4 * 20:          # need >=20 per quartile bin
        return None
    ua = np.array([ka[m.queryIdx].pt for m in good], np.float64)
    ub = np.array([kb[m.trainIdx].pt for m in good], np.float64)
    return ua, ub


def scatter(src, ua: np.ndarray, ub: np.ndarray) -> Matches:
    """Sparse match list -> the dense field relative_pose_magsac consumes."""
    tgt = torch.zeros(src.h, src.w, 2)
    wgt = torch.zeros(src.h, src.w)
    for (u, v), p in zip(np.round(ua).astype(int), ub):
        if 0 <= u < src.w and 0 <= v < src.h:
            tgt[v, u] = torch.tensor(p, dtype=torch.float32)
            wgt[v, u] = 1.0
    return Matches(target=tgt, weight=wgt)


def theta_of(src, uv: np.ndarray) -> np.ndarray:
    b = src.camera.unproject(torch.from_numpy(uv).float())
    b = torch.nn.functional.normalize(b, dim=-1)
    return np.degrees(np.arccos(np.clip(b[:, 2].numpy(), -1.0, 1.0)))


def synth_targets(src, ua: np.ndarray, R_rel: torch.Tensor, t_rel: torch.Tensor,
                  rng: np.random.Generator, noise_px: float = 1.0):
    """GT-consistent target pixels for the source pixels: unproject, assign a
    random depth in [1, 5] m along the ray, transform with the GT relative pose,
    reproject, add noise. Returns (ua_kept, ub_synth)."""
    b = src.camera.unproject(torch.from_numpy(ua).float())
    b = torch.nn.functional.normalize(b, dim=-1)
    d = torch.from_numpy(rng.uniform(1.0, 5.0, size=len(ua))).float()
    Xi = b * d[:, None]
    Xj = Xi @ R_rel.T + t_rel[None, :]
    theta_j = torch.atan2(Xj[:, :2].norm(dim=-1), Xj[:, 2])
    uv_j = src.camera.project(Xj)
    uv_j = uv_j + torch.from_numpy(rng.normal(0, noise_px, size=(len(ua), 2))).float()
    inside = ((theta_j <= src.camera.theta_max)
              & (uv_j[:, 0] >= 0) & (uv_j[:, 0] < src.w)
              & (uv_j[:, 1] >= 0) & (uv_j[:, 1] < src.h)
              & (Xj[:, 2] > 1e-3))
    keep = inside.numpy()
    return ua[keep], uv_j.numpy()[keep]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", required=True)
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--max-gt-rot-deg", type=float, default=30.0)
    p.add_argument("--max-pairs", type=int, default=80)
    p.add_argument("--nfeatures", type=int, default=6000)
    p.add_argument("--ratio", type=float, default=0.8)
    p.add_argument("--magsac-thresh-deg", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    src = ScanNetPPFisheye(os.path.expanduser(args.path), max_size=args.max_size,
                           patch=14, keep_bad=False)
    src.frames = [fr for fr in src.frames
                  if os.path.exists(os.path.join(src._image_root, fr["file_path"]))]
    print(f"[h1] {len(src)} frames on disk, grid {src.w}x{src.h}, "
          f"theta_max {math.degrees(src.camera.theta_max):.1f} deg")

    # Candidate pairs: all (i, j), GT rotation in (0.5, max] deg, spread evenly.
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
    print(f"[h1] {len(cand)} pairs, GT rotation "
          f"{cand[0][2]:.2f}..{cand[-1][2]:.2f} deg" if cand else "[h1] no pairs")

    conds = ["all"] + [f"q{k}" for k in range(BINS)]
    arms = ("real", "synth")
    per: Dict[str, Dict[str, List[float]]] = {
        a: {c: [] for c in conds} for a in arms}
    prd: Dict[str, Dict[str, List[float]]] = {
        a: {c: [] for c in conds} for a in arms}
    gts: Dict[str, List[float]] = {a: [] for a in arms}
    binstat: Dict[str, List[float]] = {f"q{k}_theta": [] for k in range(BINS)}
    counts: List[int] = []
    rng = np.random.default_rng(args.seed)

    for i, j, ang in cand:
        m = sift_matches(src, i, j, args.nfeatures, args.ratio)
        if m is None:
            continue
        ua, ub = m
        gi, gj = src.pose(i), src.pose(j)
        R_gt = gj[0] @ gi[0].transpose(-1, -2)
        # cam-from-world: X_j = R_rel X_i + t_rel with t_rel = t_j - R_rel t_i
        t_rel = (gj[1] - R_gt @ gi[1]).float()
        eye = torch.eye(3, dtype=R_gt.dtype)
        th = theta_of(src, ua)
        edges = np.quantile(th, np.linspace(0, 1, BINS + 1))
        edges[-1] += 1e-6

        for arm in arms:
            if arm == "real":
                sa, sb = ua, ub
            else:
                sa, sb = synth_targets(src, ua, R_gt.float(), t_rel, rng)
                if len(sa) < 4 * 20:
                    sa = None
            got: Dict[str, Optional[torch.Tensor]] = {}
            if sa is None:
                got = {c: None for c in conds}
            else:
                tha = theta_of(src, sa)
                for c in conds:
                    if c == "all":
                        sel = np.ones(len(sa), bool)
                    else:
                        k = int(c[1])
                        sel = (tha >= edges[k]) & (tha < edges[k + 1])
                    if sel.sum() < 20:
                        got[c] = None
                        continue
                    out = relative_pose_magsac(
                        scatter(src, sa[sel], sb[sel]), src.camera,
                        threshold_deg=args.magsac_thresh_deg)
                    got[c] = None if out is None else out[0]
            if any(got[c] is None for c in conds):
                # denominator discipline: every condition answers, or the pair
                # is out of this arm entirely
                got = None
            if got is None:
                per[arm]["_drop"] = per[arm].get("_drop", 0) + 1  # type: ignore
                continue
            for c in conds:
                R_hat = got[c].to(R_gt)
                per[arm][c].append(rotation_error_deg(R_hat, R_gt))
                prd[arm][c].append(rotation_error_deg(eye, R_hat))
            gts[arm].append(ang)
            if arm == "real":
                counts.append(int(len(ua)))
                for k in range(BINS):
                    sel = (th >= edges[k]) & (th < edges[k + 1])
                    binstat[f"q{k}_theta"].append(float(np.median(th[sel])))

    summary: Dict[str, Dict] = {}
    for arm in arms:
        n = len(gts[arm])
        drop = per[arm].pop("_drop", 0)
        print(f"\n=== {arm} arm: {n} pairs (dropped {drop}) ===")
        print(f"{'cond':>6} {'med err':>9} {'gain':>7} {'med theta':>10}")
        summary[arm] = {"n_pairs": n, "dropped": drop, "conds": {}}
        for c in conds:
            errs = per[arm][c]
            g = _gain(prd[arm][c], gts[arm])
            mth = (_median(binstat[f"{c}_theta"]) if c != "all" else float("nan"))
            print(f"{c:>6} {_median(errs):9.3f} {g:7.3f} {mth:10.1f}")
            summary[arm]["conds"][c] = {
                "median_err_deg": _median(errs), "gain": g,
                "median_theta_deg": mth}
        # Paired comparison (same pairs, so pairwise differences are the
        # sharper estimator of the locked prediction than pooled medians).
        d = np.asarray(per[arm][f"q{BINS-1}"]) - np.asarray(per[arm]["q0"])
        summary[arm]["paired_rim_minus_center"] = {
            "median_diff_deg": _median(list(d)),
            "n_rim_better": int((d < 0).sum()), "n_center_better": int((d > 0).sum())}
        print(f"paired q{BINS-1}-q0: median {_median(list(d)):+.3f} deg, "
              f"rim better on {(d < 0).sum()}/{len(d)} pairs")
    summary["match_count_median"] = _median([float(c) for c in counts])
    summary["config"] = vars(args)

    if args.out:
        out = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[h1] wrote {out}")


if __name__ == "__main__":
    main()
