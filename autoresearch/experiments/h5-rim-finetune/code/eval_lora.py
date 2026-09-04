"""H5 evaluation: three-axis read of a LoRA checkpoint on held-out scenes.

One model, LoRA toggled: "before" = lora_disabled (bit-identical to the
frozen backbone, verified by train_smoke), "after" = LoRA enabled. Reports:
  depth — protocol-of-record joint (theta x GT-depth) AbsRel table + zones;
  pose  — median rotation error, rotation gain, RRA@15/RTA@15 over
          adjacent-frame pairs against the official-calibration GT.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h5-rim-finetune/code/eval_lora.py \
        --seq <held-out seq dir> --lora <lora_last.pt> --out results/eval_<seq>.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from adt_pose_value import AriaLocalPairs  # noqa: E402
from finetune.eval.metrics import align_depth  # noqa: E402
from raytun3r.metrics import rotation_error_deg, translation_error_deg  # noqa: E402
import lora  # noqa: E402
import upright as U  # noqa: E402
from train import Seq, camera_conjugation  # noqa: E402

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)


def load_lora(net, ckpt_path: str) -> int:
    ck = torch.load(ckpt_path, map_location="cpu")
    hits = lora.inject(net, ck["patterns"],
                       r=ck["config"]["lora_r"],
                       alpha=2 * ck["config"]["lora_r"])
    by_name = dict(hits)
    n = 0
    for name, ab in ck["lora"].items():
        assert name in by_name, f"checkpoint layer {name} not found"
        by_name[name].A.data.copy_(ab["A"])
        by_name[name].B.data.copy_(ab["B"])
        n += 1
    assert n == len(hits), (n, len(hits))
    return n


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--rra-deg", type=float, default=15.0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    s = Seq(os.path.expanduser(args.seq), args.size, args.max_frames)
    cam = s.src.camera
    h = w = args.size
    theta = cam.incidence_grid(h, w)
    cone = (theta <= cam.theta_max)
    cos_t = torch.cos(theta)
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)
    C = camera_conjugation()

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=args.device,
                        variant="small")
    bb.install(None, cam, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="z")
    net = bb.model if hasattr(bb, "model") else bb
    n_loaded = load_lora(net, args.lora)
    print(f"[eval] loaded LoRA into {n_loaded} layers from {args.lora}")

    nb_d = len(GT_DEPTH_EDGES) - 1
    tables = {}
    per_frame: Dict[str, list] = {}
    for arm in ("before", "after"):
        s_ = np.zeros((THETA_BINS, nb_d))
        n_ = np.zeros((THETA_BINS, nb_d), dtype=np.int64)
        per_frame.setdefault(arm, [])
        for k, n in enumerate(s.frames):
            with torch.no_grad():
                im = s.src.image(n).to(args.device)
                if arm == "before":
                    with lora.lora_disabled(net):
                        d = U.forward_range(bb, im, cos_t)
                else:
                    d = U.forward_range(bb, im, cos_t)
            d = d.cpu().numpy()
            gr = s.gt_range(n, cos_t).numpy()
            valid = (cone.numpy() & (gr > 0)
                     & (gr <= args.depth_max_m) & (d > 1e-6))
            aligned = align_depth(d, gr, valid, mode="scale_shift")
            absrel = (np.abs(aligned - gr) / np.clip(gr, 1e-6, None))[valid]
            ti = t_idx[valid]
            di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
            flat = ti * nb_d + di
            s_ += np.bincount(flat, weights=absrel,
                              minlength=THETA_BINS * nb_d
                              ).reshape(THETA_BINS, nb_d)
            fn_ = np.bincount(flat, minlength=THETA_BINS * nb_d
                              ).reshape(THETA_BINS, nb_d)
            fs_ = np.bincount(flat, weights=absrel,
                              minlength=THETA_BINS * nb_d
                              ).reshape(THETA_BINS, nb_d)
            # per-frame zone AbsRel for bootstrap error bars (review
            # change #3, 2026-08-19); frame is the sampling unit
            nr = [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                  if math.degrees(0.5 * (t_edges[i] + t_edges[i + 1])) >= 38
                  and GT_DEPTH_EDGES[j + 1] <= 2.0]
            ce = [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                  if math.degrees(0.5 * (t_edges[i] + t_edges[i + 1])) <= 11]
            def _z(cells):
                wg = sum(fn_[i, j] for i, j in cells)
                return (float(sum(fs_[i, j] for i, j in cells) / wg)
                        if wg else None)
            per_frame[arm].append({"frame": int(n),
                                   "near_rim": _z(nr), "center": _z(ce)})
            n_ += fn_
        tables[arm] = s_ / np.maximum(n_, 1)
        if arm == "before":
            counts = n_
    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]
    print("\ndepth joint AbsRel BEFORE -> AFTER:")
    for j in range(nb_d):
        row = " ".join(f"{tables['before'][i, j]:5.3f}>"
                       f"{tables['after'][i, j]:5.3f}"
                       for i in range(THETA_BINS))
        print(f"{GT_DEPTH_EDGES[j]:4.0f}-{GT_DEPTH_EDGES[j + 1]:2.0f} m  {row}")

    zones: Dict[str, Dict] = {}
    # (per_frame collected above)
    for zname, cells in {
        "near_rim(<=2m,>=38deg)": [(i, j) for i in range(THETA_BINS)
                                   for j in range(nb_d)
                                   if t_mid[i] >= 38
                                   and GT_DEPTH_EDGES[j + 1] <= 2.0],
        "near_center(<=2m,<=11deg)": [(i, j) for i in range(THETA_BINS)
                                      for j in range(nb_d)
                                      if t_mid[i] <= 11
                                      and GT_DEPTH_EDGES[j + 1] <= 2.0],
        "center(<=11deg)": [(i, j) for i in range(THETA_BINS)
                            for j in range(nb_d) if t_mid[i] <= 11],
        "far(>=3m)": [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                      if GT_DEPTH_EDGES[j] >= 3.0],
    }.items():
        wgt = np.array([counts[i, j] for i, j in cells], float)
        b = float((np.array([tables["before"][i, j] for i, j in cells])
                   * wgt).sum() / wgt.sum())
        a = float((np.array([tables["after"][i, j] for i, j in cells])
                   * wgt).sum() / wgt.sum())
        zones[zname] = {"before": b, "after": a}
        print(f"{zname}: {b:.3f} -> {a:.3f} ({(a - b) / b * 100:+.1f}%)")

    # ---- pose: adjacent pairs, both arms on the same pairs ----
    pose: Dict[str, Dict[str, List[float]]] = {
        arm: {"rot": [], "tdir": [], "gt": []} for arm in ("before", "after")}
    for a, b in s.pairs:
        R_gt, t_gt = s.rel_pose(a, b, C)
        eye = torch.eye(3, dtype=R_gt.dtype)
        imgs = torch.stack([s.src.image(a), s.src.image(b)])[None].to(args.device)
        with torch.no_grad():
            for arm in ("before", "after"):
                if arm == "before":
                    with lora.lora_disabled(net):
                        pr = bb.forward(U.to_model(imgs))
                else:
                    pr = bb.forward(U.to_model(imgs))
                # the turn rolled the camera frame; put the pose back where
                # R_gt lives before comparing
                R_hat, t_hat = pr.relative(0, 1)
                R_hat, t_hat = U.unroll_R(R_hat), U.unroll_t(t_hat)
                pose[arm]["rot"].append(
                    rotation_error_deg(R_hat.cpu().to(R_gt), R_gt))
                pose[arm]["tdir"].append(
                    translation_error_deg(t_hat.cpu().to(t_gt), t_gt))
                if arm == "before":
                    pose[arm]["gt"].append(
                        rotation_error_deg(eye, R_gt))
    summ_pose = {}
    print("\npose (adjacent pairs, official calibration GT):")
    for arm in ("before", "after"):
        rot = np.array(pose[arm]["rot"])
        td = np.array(pose[arm]["tdir"])
        row = {"median_rot_deg": float(np.median(rot)),
               "rra15": float((rot <= args.rra_deg).mean()),
               "median_tdir_deg": float(np.median(td)),
               "rta15": float((td <= args.rra_deg).mean()),
               "n_pairs": int(len(rot))}
        summ_pose[arm] = row
        print(f"  {arm:6s}: rot median {row['median_rot_deg']:.2f} deg, "
              f"RRA@15 {row['rra15']:.3f}, tdir median "
              f"{row['median_tdir_deg']:.2f}, RTA@15 {row['rta15']:.3f} "
              f"(n={row['n_pairs']})")

    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump({"seq": s.name, "lora": args.lora,
                       "before": tables["before"].tolist(),
                       "after": tables["after"].tolist(),
                       "counts": counts.tolist(), "zones": zones,
                       "per_frame": per_frame,
                       "pose": summ_pose,
                       "theta_bin_mid_deg": t_mid,
                       "config": vars(args)}, f, indent=2)
        print(f"\n[eval] wrote {dst}")


if __name__ == "__main__":
    main()
