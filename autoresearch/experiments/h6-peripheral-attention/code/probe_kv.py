"""H6.1: KV-compression probe on the delivered rim module. Protocol
addendum 2026-08-19. EXPLORATORY (seq131 is a #36 training scene).

Variants: kv=full (as trained), cone, rim (cone & theta>rim_deg),
center (cone & theta<=rim_deg). Reports zone AbsRel of the after-arm per
variant plus mean |after - after_fullKV| output drift.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h6-peripheral-attention/code/probe_kv.py \
        --seq <seq dir> --module <module_last.pt> --out results/probe_kv_<seq>.json
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
                             rim_mask_for)

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)


def apply_with_kv(module, feats_t, feats_prev, rim_mask, kv_mask):
    """apply_to_final_level with a KV token subset."""
    patches_t, cls_t = feats_t[-1]
    patches_p, _ = feats_prev[-1]
    rim_idx = torch.nonzero(rim_mask, as_tuple=False).squeeze(-1)
    kv_idx = torch.nonzero(kv_mask, as_tuple=False).squeeze(-1)
    rim = patches_t[:, 0, rim_idx, :]
    prev = patches_p[:, 0, kv_idx, :]
    updated = module(rim, prev)
    new_patches = patches_t.clone()
    new_patches[:, 0, rim_idx, :] = updated
    out = list(feats_t)
    out[-1] = (new_patches, cls_t)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    s = Seq(os.path.expanduser(args.seq), args.size, args.max_frames,
            dense=True)
    cam = s.src.camera
    h = w = args.size
    theta = cam.incidence_grid(h, w)
    cone_px = (theta <= cam.theta_max).numpy()
    cos_t = torch.cos(theta)
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
    rim = rim_mask_for(theta_p, rim_deg, float(cam.theta_max))
    cone_tok = theta_p <= float(cam.theta_max)
    kv_variants = {
        "full": torch.ones_like(cone_tok),
        "cone": cone_tok,
        "rim": cone_tok & (theta_p > math.radians(rim_deg)),
        "center": cone_tok & (theta_p <= math.radians(rim_deg)),
    }
    with torch.no_grad():
        f0, _ = net.backbone(s.src.image(s.frames[0])[None, None]
                             .to(args.device), cam_token=None,
                             export_feat_layers=[])
    module = PeripheralCrossFrameAttention(f0[-1][0].shape[-1]).to(args.device)
    module.load_state_dict(ck["module"])
    module.eval()
    print(f"[kv-probe] rim queries {int(rim.sum())}/{rim.numel()}, "
          f"KV sizes: " + ", ".join(f"{k}={int(m.sum())}"
                                    for k, m in kv_variants.items()))

    def depth_from(feats) -> np.ndarray:
        out = net._process_depth_head(list(feats), h, w)
        z = (out["depth"] if isinstance(out, dict) else out.depth).reshape(h, w)
        return (z / cos_t.to(z).clamp(min=1e-6)).cpu().numpy()

    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)
    nb_d = len(GT_DEPTH_EDGES) - 1
    acc = {k: [np.zeros((THETA_BINS, nb_d)),
               np.zeros((THETA_BINS, nb_d), dtype=np.int64)]
           for k in kv_variants}
    drift = {k: [] for k in kv_variants}
    prev_feats = None
    for n in s.frames:
        with torch.no_grad():
            feats, _ = net.backbone(s.src.image(n)[None, None]
                                    .to(args.device), cam_token=None,
                                    export_feat_layers=[])
            if prev_feats is None:
                prev_feats = feats
                continue
            gr = s.gt_range(n, cos_t).numpy()
            d_ref = None
            for k, kvm in kv_variants.items():
                d = depth_from(apply_with_kv(module, feats, prev_feats,
                                             rim, kvm))
                if k == "full":
                    d_ref = d
                drift[k].append(float(np.abs(d - d_ref)[cone_px].mean()))
                valid = (cone_px & (gr > 0) & (gr <= args.depth_max_m)
                         & (d > 1e-6))
                aligned = align_depth(d, gr, valid, mode="scale_shift")
                absrel = (np.abs(aligned - gr)
                          / np.clip(gr, 1e-6, None))[valid]
                ti = t_idx[valid]
                di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0,
                             nb_d - 1)
                flat = ti * nb_d + di
                acc[k][0] += np.bincount(flat, weights=absrel,
                                         minlength=THETA_BINS * nb_d
                                         ).reshape(THETA_BINS, nb_d)
                acc[k][1] += np.bincount(flat, minlength=THETA_BINS * nb_d
                                         ).reshape(THETA_BINS, nb_d)
        prev_feats = feats

    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]
    zones_def = {
        "near_rim(<=2m,>=38deg)": [(i, j) for i in range(THETA_BINS)
                                   for j in range(nb_d) if t_mid[i] >= 38
                                   and GT_DEPTH_EDGES[j + 1] <= 2.0],
        "center(<=11deg)": [(i, j) for i in range(THETA_BINS)
                            for j in range(nb_d) if t_mid[i] <= 11],
        "far(>=3m)": [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                      if GT_DEPTH_EDGES[j] >= 3.0],
    }
    report = {}
    for k in kv_variants:
        table = acc[k][0] / np.maximum(acc[k][1], 1)
        cnt = acc[k][1]
        z = {}
        for zname, cells in zones_def.items():
            wg = np.array([cnt[i, j] for i, j in cells], float)
            z[zname] = float((np.array([table[i, j] for i, j in cells])
                              * wg).sum() / wg.sum())
        report[k] = {"zones": z, "mean_drift_m": float(np.mean(drift[k]))}
        print(f"kv={k:6s} near_rim {z['near_rim(<=2m,>=38deg)']:.3f}  "
              f"center {z['center(<=11deg)']:.3f}  "
              f"far {z['far(>=3m)']:.3f}  drift {np.mean(drift[k]):.4f} m")

    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump({"seq": s.name, "module": args.module,
                       "kv_sizes": {k: int(m.sum())
                                    for k, m in kv_variants.items()},
                       "report": report, "config": vars(args)}, f, indent=2)
        print(f"[kv-probe] wrote {dst}")


if __name__ == "__main__":
    main()
