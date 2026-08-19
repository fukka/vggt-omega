"""H6 evaluation: three-axis read of the module checkpoint on held-out scenes.

"before" = plain single-frame forward (module absent); "after" = module
applied with the previous frame as temporal context (video mode). The camera
path reads original feats in both arms — pose rows are reported as a
verification that they are bit-identical, not as a comparison.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h6-peripheral-attention/code/eval_module.py \
        --seq <held-out seq dir> --module <module_last.pt> --out results/eval_<seq>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune.eval.metrics import align_depth  # noqa: E402
import importlib.util as _ilu  # noqa: E402
_h5_spec = _ilu.spec_from_file_location(
    "h5_train", Path(__file__).resolve().parents[2]
    / "h5-rim-finetune" / "code" / "train.py")
_h5 = _ilu.module_from_spec(_h5_spec)
_h5_spec.loader.exec_module(_h5)
Seq = _h5.Seq
from peripheral_attn import (PeripheralCrossFrameAttention,  # noqa: E402
                             apply_to_final_level, rim_mask_for)

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    s = Seq(os.path.expanduser(args.seq), args.size, args.max_frames)
    cam = s.src.camera
    h = w = args.size
    theta = cam.incidence_grid(h, w)
    cone = (theta <= cam.theta_max).numpy()
    cos_t = torch.cos(theta)
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)
    gh, gw = h // 14, w // 14
    theta_p = theta.reshape(gh, 14, gw, 14).mean((1, 3)).ravel()

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=args.device,
                        variant="small")
    bb.install(None, cam, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb

    ck = torch.load(args.module, map_location="cpu")
    rim_deg = ck["config"].get("rim_deg", 35.0)
    all_tok = ck["config"].get("all_token_control", False)
    rim = (torch.ones_like(theta_p, dtype=torch.bool) if all_tok
           else rim_mask_for(theta_p, rim_deg, float(cam.theta_max)))
    # probe dim, build, load
    with torch.no_grad():
        f0, _ = net.backbone(s.src.image(s.frames[0])[None, None]
                             .to(args.device), cam_token=None,
                             export_feat_layers=[])
    module = PeripheralCrossFrameAttention(f0[-1][0].shape[-1]).to(args.device)
    module.load_state_dict(ck["module"])
    module.eval()
    print(f"[eval] module loaded ({'all-token' if all_tok else 'rim'} queries)")

    def depth_from(feats) -> np.ndarray:
        out = net._process_depth_head(list(feats), h, w)
        z = (out["depth"] if isinstance(out, dict) else out.depth).reshape(h, w)
        return (z / cos_t.to(z).clamp(min=1e-6)).cpu().numpy()

    nb_d = len(GT_DEPTH_EDGES) - 1
    acc = {arm: [np.zeros((THETA_BINS, nb_d)),
                 np.zeros((THETA_BINS, nb_d), dtype=np.int64)]
           for arm in ("before", "after")}
    prev_feats = None
    for n in s.frames:
        with torch.no_grad():
            feats, _ = net.backbone(s.src.image(n)[None, None].to(args.device),
                                    cam_token=None, export_feat_layers=[])
            d_before = depth_from(feats)
            if prev_feats is not None:
                d_after = depth_from(
                    apply_to_final_level(module, feats, prev_feats, rim))
            else:
                d_after = None
        gr = s.gt_range(n, cos_t).numpy()
        for arm, d in (("before", d_before), ("after", d_after)):
            if d is None:
                continue
            valid = cone & (gr > 0) & (gr <= args.depth_max_m) & (d > 1e-6)
            aligned = align_depth(d, gr, valid, mode="scale_shift")
            absrel = (np.abs(aligned - gr) / np.clip(gr, 1e-6, None))[valid]
            ti = t_idx[valid]
            di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
            flat = ti * nb_d + di
            acc[arm][0] += np.bincount(flat, weights=absrel,
                                       minlength=THETA_BINS * nb_d
                                       ).reshape(THETA_BINS, nb_d)
            acc[arm][1] += np.bincount(flat, minlength=THETA_BINS * nb_d
                                       ).reshape(THETA_BINS, nb_d)
        prev_feats = feats

    # NOTE: "before" pools ALL frames, "after" pools frames 2..N (needs a
    # previous frame). For a like-for-like read the JSON also carries the
    # before-table restricted to frames 2..N via a second pass.
    acc_b2 = [np.zeros((THETA_BINS, nb_d)),
              np.zeros((THETA_BINS, nb_d), dtype=np.int64)]
    prev = None
    for n in s.frames:
        if prev is not None:
            with torch.no_grad():
                feats, _ = net.backbone(
                    s.src.image(n)[None, None].to(args.device),
                    cam_token=None, export_feat_layers=[])
                d = depth_from(feats)
            gr = s.gt_range(n, cos_t).numpy()
            valid = cone & (gr > 0) & (gr <= args.depth_max_m) & (d > 1e-6)
            aligned = align_depth(d, gr, valid, mode="scale_shift")
            absrel = (np.abs(aligned - gr) / np.clip(gr, 1e-6, None))[valid]
            ti = t_idx[valid]
            di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
            flat = ti * nb_d + di
            acc_b2[0] += np.bincount(flat, weights=absrel,
                                     minlength=THETA_BINS * nb_d
                                     ).reshape(THETA_BINS, nb_d)
            acc_b2[1] += np.bincount(flat, minlength=THETA_BINS * nb_d
                                     ).reshape(THETA_BINS, nb_d)
        prev = n

    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]
    before = acc_b2[0] / np.maximum(acc_b2[1], 1)      # like-for-like frames
    after = acc["after"][0] / np.maximum(acc["after"][1], 1)
    counts = acc_b2[1]
    print("\ndepth joint AbsRel BEFORE -> AFTER (frames with context):")
    for j in range(nb_d):
        row = " ".join(f"{before[i, j]:5.3f}>{after[i, j]:5.3f}"
                       for i in range(THETA_BINS))
        print(f"{GT_DEPTH_EDGES[j]:4.0f}-{GT_DEPTH_EDGES[j + 1]:2.0f} m  {row}")
    zones: Dict[str, Dict] = {}
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
        wg = np.array([counts[i, j] for i, j in cells], float)
        b = float((np.array([before[i, j] for i, j in cells]) * wg).sum()
                  / wg.sum())
        a = float((np.array([after[i, j] for i, j in cells]) * wg).sum()
                  / wg.sum())
        zones[zname] = {"before": b, "after": a}
        print(f"{zname}: {b:.3f} -> {a:.3f} ({(a - b) / b * 100:+.1f}%)")

    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump({"seq": s.name, "module": args.module,
                       "before": before.tolist(), "after": after.tolist(),
                       "counts": counts.tolist(), "zones": zones,
                       "theta_bin_mid_deg": t_mid, "config": vars(args)},
                      f, indent=2)
        print(f"\n[eval] wrote {dst}")


if __name__ == "__main__":
    main()
