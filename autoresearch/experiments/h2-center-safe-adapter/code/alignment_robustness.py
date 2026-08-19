"""Alignment-robustness rows for the protocol of record (external review
change #4, 2026-08-19): the SAME vanilla predictions scored under
(a) scale_shift on all valid px [record], (b) scale_only on all valid px,
(c) scale_shift fit ONLY on near (<=2m GT) px. Zone AbsRel per mode.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h2-center-safe-adapter/code/alignment_robustness.py \
        --seq <seq dir> --out results/alignment_robustness_<seq>.json
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "h1-rim-pose-value" / "code"))
from finetune.eval.metrics import align_depth  # noqa: E402
import importlib.util as _ilu  # noqa: E402
_sp = _ilu.spec_from_file_location("h5_train", Path(__file__).resolve().parents[2] / "h5-rim-finetune" / "code" / "train.py")
_h5 = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_h5)
Seq = _h5.Seq

THETA_BINS, EDGES = 8, (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    s = Seq(os.path.expanduser(args.seq), args.size, args.max_frames)
    cam = s.src.camera; h = w = args.size
    theta = cam.incidence_grid(h, w); cone = (theta <= cam.theta_max).numpy()
    cos_t = torch.cos(theta)
    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device="cpu", variant="small")
    bb.install(None, cam, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)
    nb = len(EDGES) - 1
    modes = ("scale_shift_all", "scale_only_all", "scale_shift_near")
    acc = {m: [np.zeros((THETA_BINS, nb)), np.zeros((THETA_BINS, nb), np.int64)] for m in modes}
    for n in s.frames:
        with torch.no_grad():
            feats, _ = net.backbone(s.src.image(n)[None, None], cam_token=None, export_feat_layers=[])
            out = net._process_depth_head(list(feats), h, w)
            z = (out["depth"] if isinstance(out, dict) else out.depth).reshape(h, w)
            d = (z / cos_t.clamp(min=1e-6)).numpy()
        gr = s.gt_range(n, cos_t).numpy()
        valid = cone & (gr > 0) & (gr <= 10.0) & (d > 1e-6)
        near_fit = valid & (gr <= 2.0)
        for m in modes:
            if m == "scale_shift_all":
                al = align_depth(d, gr, valid, mode="scale_shift")
            elif m == "scale_only_all":
                al = align_depth(d, gr, valid, mode="scale_only")
            else:
                al = align_depth(d, gr, near_fit if near_fit.sum() > 500 else valid, mode="scale_shift")
            ar = (np.abs(al - gr) / np.clip(gr, 1e-6, None))[valid]
            ti = t_idx[valid]; di = np.clip(np.digitize(gr[valid], EDGES) - 1, 0, nb - 1)
            fl = ti * nb + di
            acc[m][0] += np.bincount(fl, weights=ar, minlength=THETA_BINS * nb).reshape(THETA_BINS, nb)
            acc[m][1] += np.bincount(fl, minlength=THETA_BINS * nb).reshape(THETA_BINS, nb)
    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1])) for i in range(THETA_BINS)]
    zones_def = {
        "near_rim(<=2m,>=38deg)": [(i, j) for i in range(THETA_BINS) for j in range(nb) if t_mid[i] >= 38 and EDGES[j + 1] <= 2.0],
        "near_center(<=2m,<=11deg)": [(i, j) for i in range(THETA_BINS) for j in range(nb) if t_mid[i] <= 11 and EDGES[j + 1] <= 2.0],
        "center(<=11deg)": [(i, j) for i in range(THETA_BINS) for j in range(nb) if t_mid[i] <= 11],
        "far(>=3m)": [(i, j) for i in range(THETA_BINS) for j in range(nb) if EDGES[j] >= 3.0],
    }
    rep = {}
    for m in modes:
        tb = acc[m][0] / np.maximum(acc[m][1], 1); ct = acc[m][1]
        rep[m] = {zn: float((np.array([tb[i, j] for i, j in cs]) * np.array([ct[i, j] for i, j in cs])).sum()
                            / max(sum(ct[i, j] for i, j in cs), 1)) for zn, cs in zones_def.items()}
        print(m, {k: round(v, 3) for k, v in rep[m].items()})
    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"seq": s.name, "modes": rep, "config": vars(args)}, open(dst, "w"), indent=2)
        print("wrote", dst)

if __name__ == "__main__":
    main()
