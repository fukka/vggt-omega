"""H9 RayCal-TTA: remove the radial compression field using only video.

TEST-TIME ADAPTATION, PER SCENE. Nothing is trained and nothing is transferred:
for each held-out sequence the pipeline matches its own frames, triangulates
metric anchors from parallax, fits the radial field those anchors imply, and
applies it. **No depth labels are used at any point** -- pose comes from the
device's SLAM trajectory, which Aria ships, and matches come from the images.
That is the same protocol class as RayTun3R's (unsupervised test-scene TTA), so
this row can sit beside theirs; the adaptation-data column reads "test-scene
RGB + pose" rather than "other-scene GT".

THE LOCKED BAR, from the hypothesis registration and not negotiable here:

> the scale_shift <-> frozen-affine gap must COLLAPSE, not just the aligned
> number drop.

The reason that is the bar and not AbsRel: this project measured that ~82% of
seq131's near-rim "penalty" is the eval affine's PLACEMENT (refit on near
pixels: 1.47 -> 0.26), and that no single affine serves near and far. So a
method that merely lets the per-frame affine sit better has changed nothing
about the geometry. If the compression field is really gone, ONE affine for the
whole sequence should do nearly as well as a per-frame least-squares fit -- and
that gap, not the headline number, is what this reports.

Usage (box):
    python .../h9-raycal-tta/code/run_h9.py --seq $ADT/<held-out seq> \\
        --out results/autoresearch-h9-raycal/<name>.json
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

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "h1-rim-pose-value" / "code"))
sys.path.append(str(_HERE.parents[1] / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "common"))

import importlib.util as _ilu  # noqa: E402
import anchors as AN  # noqa: E402
import raycal as RC  # noqa: E402
import upright as U  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
_h5_train = _load("h5_train", _H5 / "train.py")
_h5_eval = _load("h5_eval", _H5 / "eval_lora.py")
Seq = _h5_train.Seq
camera_conjugation = _h5_train.camera_conjugation
THETA_BINS = _h5_eval.THETA_BINS
GT_DEPTH_EDGES = _h5_eval.GT_DEPTH_EDGES

from finetune.eval.metrics import align_depth  # noqa: E402

ZONES = {
    "near_rim(<=2m,>=38deg)": lambda tm, j: tm >= 38 and GT_DEPTH_EDGES[j + 1] <= 2.0,
    "near_center(<=2m,<=11deg)": lambda tm, j: tm <= 11 and GT_DEPTH_EDGES[j + 1] <= 2.0,
    "center(<=11deg)": lambda tm, j: tm <= 11,
    "far(>=3m)": lambda tm, j: GT_DEPTH_EDGES[j] >= 3.0,
}


def zones_from(table, counts, t_mid) -> Dict[str, float]:
    nb = len(GT_DEPTH_EDGES) - 1
    out = {}
    for name, keep in ZONES.items():
        cells = [(i, j) for i in range(THETA_BINS) for j in range(nb) if keep(t_mid[i], j)]
        w = np.array([counts[i, j] for i, j in cells], float)
        if w.sum() == 0:
            continue
        out[name] = float((np.array([table[i, j] for i, j in cells]) * w).sum() / w.sum())
    return out


def frozen_affine(preds: List[np.ndarray], gts: List[np.ndarray],
                  valids: List[np.ndarray]):
    """ONE least-squares (scale, shift) over every valid pixel of the sequence.

    Pooled, not averaged per frame: the claim under test is that a single
    affine can serve the whole sequence, and averaging per-frame fits would
    quietly give the method the per-frame freedom the bar is about.
    """
    xs = np.concatenate([p[v] for p, v in zip(preds, valids)])
    ys = np.concatenate([g[v] for g, v in zip(gts, valids)])
    A = np.stack([xs, np.ones_like(xs)], 1)
    s, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(s), float(b)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--stride", type=int, default=10,
                   help="partner offset in FRAME index. #22 measured that "
                        "adjacent frames buy ~nothing (0.98-1.07) while stride "
                        "10 buys 10-13%%: parallax needs a baseline.")
    p.add_argument("--max-anchors-per-frame", type=int, default=3000)
    p.add_argument("--min-parallax-deg", type=float, default=1.0)
    p.add_argument("--agree-tol", type=float, default=0.10)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--matcher", default="auto")
    p.add_argument("--variant", default="small")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    t0 = time.time()
    s = Seq(os.path.expanduser(a.seq), a.size, a.max_frames)
    cam = s.src.camera
    C = camera_conjugation()
    h = w = a.size
    theta = cam.incidence_grid(h, w)
    theta_np = theta.numpy()
    cone = (theta <= cam.theta_max).numpy()
    cos_t = torch.cos(theta)
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta_np, t_edges) - 1, 0, THETA_BINS - 1)
    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi

    from raytun3r.backbones import build_backbone
    from raytun3r.matching import build_matcher
    bb = build_backbone("da3", weights="pretrained", device=a.device,
                        variant=a.variant)
    bb.install(None, cam, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="z")
    matcher = build_matcher(a.matcher, device=a.device)
    print(f"[h9] {s.name}: {len(s.frames)} frames, matcher={matcher.name}, "
          f"stride={a.stride}")

    # ---- pass 1: predictions, and anchors from parallax -------------------
    gen = torch.Generator().manual_seed(a.seed)
    preds, gts, valids = [], [], []
    A_theta, A_pred, A_true, A_par = [], [], [], []
    valid_mask_t = torch.from_numpy(cone)
    n_pairs_used = 0
    for n in s.frames:
        with torch.no_grad():
            pr = U.forward_range(bb, s.src.image(n).to(a.device), cos_t)
        pred = pr.float().cpu().numpy()
        gt = s.gt_range(n, cos_t).numpy()
        v = cone & (gt > 0) & (gt <= a.depth_max_m) & (pred > 1e-6)
        preds.append(pred); gts.append(gt); valids.append(v)

        # Two partners at +/- stride. Both must exist, both must have a GT
        # pose, and both must match the SAME reference pixels -- the agreement
        # gate has nothing to compare otherwise.
        packed = []
        for off in (-a.stride, +a.stride):
            m = n + off
            if not (0 <= m < len(s.src.paths)):
                continue
            if s.src.pose(n) is None or s.src.pose(m) is None:
                continue
            with torch.no_grad():
                mm = matcher(s.src.image(n).to(a.device),
                             s.src.image(m).to(a.device),
                             valid=valid_mask_t.to(a.device))
            packed.append((mm.target.detach().cpu().double().reshape(-1, 2),
                           mm.weight.detach().cpu().reshape(-1),
                           s.rel_pose(n, m, C)))
        if len(packed) < 2:
            continue
        good = packed[0][1] > 0
        for tgt, wgt, _ in packed[1:]:
            good = good & (wgt > 0)
        good = good & torch.from_numpy(cone.reshape(-1))
        idx = torch.nonzero(good, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        if idx.numel() > a.max_anchors_per_frame:
            sel = torch.randperm(idx.numel(), generator=gen)[:a.max_anchors_per_frame]
            idx = idx[sel]
        uv_ref = torch.stack([(idx % w).double(), (idx // w).double()], dim=-1)
        plist = [(tgt[idx], R.double(), tt.double()) for tgt, _, (R, tt) in packed]
        aset = AN.anchors_from_pairs(cam, uv_ref, plist,
                                     min_parallax_deg=a.min_parallax_deg,
                                     agree_tol=a.agree_tol,
                                     max_range_m=a.depth_max_m + 2.0)
        n_pairs_used += len(packed)
        if len(aset) == 0:
            continue
        px = np.clip(aset.uv[:, 0].round().long().numpy(), 0, w - 1)
        py = np.clip(aset.uv[:, 1].round().long().numpy(), 0, h - 1)
        pv = pred[py, px]
        keep = pv > 1e-6
        A_theta.append(aset.theta.numpy()[keep])
        A_pred.append(pv[keep])
        A_true.append(aset.rng.numpy()[keep])
        A_par.append(aset.parallax.numpy()[keep])

    if not A_theta:
        raise SystemExit("[h9] no anchors survived the gates -- nothing to fit. "
                         "Check the matcher and the pose availability.")
    th_a = np.concatenate(A_theta); pr_a = np.concatenate(A_pred)
    tr_a = np.concatenate(A_true); par_a = np.concatenate(A_par)
    print(f"[h9] {len(th_a)} anchors from {n_pairs_used} pairs; "
          f"median parallax {np.degrees(np.median(par_a)):.2f} deg; "
          f"rim share {(np.degrees(th_a) >= 38).mean() * 100:.1f}%")

    # ---- pass 2: fit each arm, apply, and score under two alignments ------
    results = {}
    for arm in RC.ARMS:
        field = RC.fit_field(arm, th_a, pr_a, tr_a, float(cam.theta_max),
                             n_bins=THETA_BINS, seed=a.seed)
        corr = [RC.apply_field(p, theta_np, field) for p in preds]
        sfz, bfz = frozen_affine(corr, gts, valids)
        nb = len(GT_DEPTH_EDGES) - 1
        acc = {k: [np.zeros((THETA_BINS, nb)), np.zeros((THETA_BINS, nb))]
               for k in ("scale_shift", "frozen")}
        for p_, g_, v_ in zip(corr, gts, valids):
            if v_.sum() < 1000:
                continue
            for mode in ("scale_shift", "frozen"):
                al = (align_depth(p_, g_, v_, mode="scale_shift") if mode == "scale_shift"
                      else sfz * p_ + bfz)
                ar = (np.abs(al - g_) / np.clip(g_, 1e-6, None))[v_]
                di = np.clip(np.digitize(g_[v_], GT_DEPTH_EDGES) - 1, 0, nb - 1)
                flat = t_idx[v_] * nb + di
                acc[mode][0] += np.bincount(flat, weights=ar,
                                            minlength=THETA_BINS * nb).reshape(THETA_BINS, nb)
                acc[mode][1] += np.bincount(flat, minlength=THETA_BINS * nb
                                            ).reshape(THETA_BINS, nb)
        tabs = {m: np.divide(acc[m][0], acc[m][1], out=np.zeros_like(acc[m][0]),
                             where=acc[m][1] > 0) for m in acc}
        counts = acc["scale_shift"][1]
        z_ss = zones_from(tabs["scale_shift"], counts, t_mid)
        z_fz = zones_from(tabs["frozen"], counts, t_mid)
        gap = {k: z_fz[k] - z_ss[k] for k in z_ss}
        results[arm] = {"field": field, "scale_shift": z_ss, "frozen": z_fz,
                        "gap": gap, "frozen_affine": [sfz, bfz],
                        "table_scale_shift": tabs["scale_shift"].tolist(),
                        "counts": counts.tolist()}
        print(f"[h9/{arm}] " + "  ".join(
            f"{k.split('(')[0]} {z_ss[k]:.4f} (gap {gap[k]:+.4f})" for k in z_ss))

    key = "near_rim(<=2m,>=38deg)"
    if key in results["none"]["gap"]:
        g0, g1 = results["none"]["gap"][key], results["raycal"]["gap"][key]
        print(f"[h9] LOCKED BAR near_rim gap: none {g0:+.4f} -> raycal {g1:+.4f} "
              f"({'COLLAPSED' if abs(g1) < 0.5 * abs(g0) else 'did NOT collapse'})")

    out = {"seq": s.name, "n_anchors": int(len(th_a)), "n_pairs": n_pairs_used,
           "median_parallax_deg": float(np.degrees(np.median(par_a))),
           "rim_anchor_share": float((np.degrees(th_a) >= 38).mean()),
           "matcher": matcher.name, "arms": results,
           "theta_bin_mid_deg": t_mid.tolist(),
           "seconds": round(time.time() - t0, 1), "config": vars(a),
           "used_gt_depth": False}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"[h9] wrote {a.out}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
