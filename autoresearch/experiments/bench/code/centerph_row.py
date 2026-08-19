"""Center-PH baseline row: rectify the KB4 center to a pinhole, run the
frozen backbone, evaluate back on the FISHEYE pixel grid with per-zone
coverage. Definition and locked predictions: ../protocol.md addendum
2026-08-19.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/bench/code/centerph_row.py \
        --seq <seq dir> --out results/centerph_<seq>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))

from finetune.eval.metrics import align_depth  # noqa: E402
from raytun3r.cameras import Pinhole  # noqa: E402
import importlib.util as _ilu  # noqa: E402
_h5_spec = _ilu.spec_from_file_location(
    "h5_train", Path(__file__).resolve().parents[2]
    / "h5-rim-finetune" / "code" / "train.py")
_h5 = _ilu.module_from_spec(_h5_spec)
_h5_spec.loader.exec_module(_h5)
Seq = _h5.Seq

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)


def bilinear(img: torch.Tensor, uv: torch.Tensor, size: int) -> torch.Tensor:
    """Sample (C,H,W) at pixel coords uv (H,W,2); align_corners grid."""
    g = (uv / (size - 1) * 2 - 1)[None]
    return F.grid_sample(img[None], g, mode="bilinear",
                         align_corners=True)[0]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    h = w = args.size
    s = Seq(os.path.expanduser(args.seq), args.size, args.max_frames)
    fish = s.src.camera
    # protocol: f = size/2 (45 deg on the axes, 54.7 deg on the diagonal)
    pin = Pinhole(fx=h / 2, fy=h / 2, cx=(w - 1) / 2, cy=(h - 1) / 2,
                  width=w, height=h,
                  theta_max=math.atan(math.sqrt(2.0)))

    theta_f = fish.incidence_grid(h, w)
    cone = (theta_f <= fish.theta_max).numpy()
    cos_f = torch.cos(theta_f)
    theta_pin = pin.incidence_grid(h, w)
    cos_pin = torch.cos(theta_pin)

    # rectification map: pinhole pixel -> ray -> fisheye pixel
    rect_uv = fish.project(pin.ray_grid(h, w))               # (H,W,2)
    # eval map: fisheye pixel -> ray -> pinhole pixel
    pin_uv = pin.project(fish.ray_grid(h, w))                # (H,W,2)
    covered = (cone
               & (theta_f.numpy() < float(pin.theta_max))
               & (pin_uv[..., 0].numpy() >= 0) & (pin_uv[..., 0].numpy() <= w - 1)
               & (pin_uv[..., 1].numpy() >= 0) & (pin_uv[..., 1].numpy() <= h - 1))

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=args.device,
                        variant="small")
    bb.install(None, pin, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb
    print(f"[centerph] {s.name}: {len(s.frames)} frames, "
          f"pixel coverage of cone = {covered.sum() / cone.sum():.3f}")

    t_edges = np.linspace(0.0, float(fish.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta_f.numpy(), t_edges) - 1, 0,
                    THETA_BINS - 1)
    nb_d = len(GT_DEPTH_EDGES) - 1
    err_acc = np.zeros((THETA_BINS, nb_d))
    cnt_acc = np.zeros((THETA_BINS, nb_d), dtype=np.int64)
    gt_acc = np.zeros((THETA_BINS, nb_d), dtype=np.int64)  # all valid-GT px

    for n in s.frames:
        img = s.src.image(n)                                # (3,H,W)
        rect = bilinear(img, rect_uv, args.size)            # (3,H,W)
        with torch.no_grad():
            feats, _ = net.backbone(rect[None, None].to(args.device),
                                    cam_token=None, export_feat_layers=[])
            out = net._process_depth_head(list(feats), h, w)
            z = (out["depth"] if isinstance(out, dict)
                 else out.depth).reshape(h, w)
            rng_pin = (z / cos_pin.to(z).clamp(min=1e-6))    # range, pin grid
        # pull prediction back onto the fisheye grid
        pred = bilinear(rng_pin[None].cpu(), pin_uv, args.size)[0].numpy()
        gr = s.gt_range(n, cos_f).numpy()
        gt_ok = cone & (gr > 0) & (gr <= args.depth_max_m)
        valid = gt_ok & covered & (pred > 1e-6)
        aligned = align_depth(pred, gr, valid, mode="scale_shift")
        absrel = (np.abs(aligned - gr) / np.clip(gr, 1e-6, None))[valid]
        ti, tg = t_idx[valid], t_idx[gt_ok]
        di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
        dg = np.clip(np.digitize(gr[gt_ok], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
        err_acc += np.bincount(ti * nb_d + di, weights=absrel,
                               minlength=THETA_BINS * nb_d
                               ).reshape(THETA_BINS, nb_d)
        cnt_acc += np.bincount(ti * nb_d + di, minlength=THETA_BINS * nb_d
                               ).reshape(THETA_BINS, nb_d)
        gt_acc += np.bincount(tg * nb_d + dg, minlength=THETA_BINS * nb_d
                              ).reshape(THETA_BINS, nb_d)

    table = err_acc / np.maximum(cnt_acc, 1)
    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]
    print("\ncenter-PH joint AbsRel (— = zero coverage):")
    for j in range(nb_d):
        row = " ".join(f"{table[i, j]:5.3f}" if cnt_acc[i, j] else "  —  "
                       for i in range(THETA_BINS))
        print(f"{GT_DEPTH_EDGES[j]:4.0f}-{GT_DEPTH_EDGES[j + 1]:2.0f} m  {row}")

    zones = {}
    for zname, cells in {
        "near_rim(<=2m,>=38deg)": [(i, j) for i in range(THETA_BINS)
                                   for j in range(nb_d) if t_mid[i] >= 38
                                   and GT_DEPTH_EDGES[j + 1] <= 2.0],
        "near_center(<=2m,<=11deg)": [(i, j) for i in range(THETA_BINS)
                                      for j in range(nb_d) if t_mid[i] <= 11
                                      and GT_DEPTH_EDGES[j + 1] <= 2.0],
        "center(<=11deg)": [(i, j) for i in range(THETA_BINS)
                            for j in range(nb_d) if t_mid[i] <= 11],
        "far(>=3m)": [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                      if GT_DEPTH_EDGES[j] >= 3.0],
    }.items():
        wg = np.array([cnt_acc[i, j] for i, j in cells], float)
        tot = np.array([gt_acc[i, j] for i, j in cells], float)
        cov = float(wg.sum() / max(tot.sum(), 1))
        a = (float((np.array([table[i, j] for i, j in cells]) * wg).sum()
                   / wg.sum()) if wg.sum() else None)
        zones[zname] = {"absrel_covered": a, "coverage": cov}
        astr = f"{a:.3f}" if a is not None else "  —"
        print(f"{zname}: AbsRel(covered) {astr}, coverage {cov:.1%}")

    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump({"seq": s.name, "method": "center-ph(f=size/2)",
                       "table": table.tolist(), "counts": cnt_acc.tolist(),
                       "gt_counts": gt_acc.tolist(), "zones": zones,
                       "theta_bin_mid_deg": t_mid,
                       "pixel_coverage_of_cone":
                           float(covered.sum() / cone.sum()),
                       "config": vars(args)}, f, indent=2)
        print(f"\n[centerph] wrote {dst}")


if __name__ == "__main__":
    main()
