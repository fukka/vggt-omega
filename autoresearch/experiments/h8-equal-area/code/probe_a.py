"""H8 Probe A: equisolid-angle remap, zero training. Protocol: ../protocol.md.

Two arms through one code path: 'plain' = fisheye straight into frozen
DA3-S; 'equalarea' = resample to an equisolid image (uniform solid angle
per pixel), run the model, map the prediction back to the fisheye grid.
Same frames, same masks, same per-frame scale_shift alignment.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h8-equal-area/code/probe_a.py \
        --seq <seq dir> --out results/probe_a_<seq>.json
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

from finetune.eval.metrics import align_depth  # noqa: E402
from raytun3r.cameras import KannalaBrandt  # noqa: E402
import importlib.util as _ilu  # noqa: E402
_h5_spec = _ilu.spec_from_file_location(
    "h5_train", Path(__file__).resolve().parents[2]
    / "h5-rim-finetune" / "code" / "train.py")
_h5 = _ilu.module_from_spec(_h5_spec)
_h5_spec.loader.exec_module(_h5)
Seq = _h5.Seq

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
EQUISOLID_K = (-1.0 / 24, 1.0 / 1920, -1.0 / 322560, 0.0)  # 2 sin(theta/2)


def bilinear(img: torch.Tensor, uv: torch.Tensor, size: int) -> torch.Tensor:
    g = (uv / (size - 1) * 2 - 1)[None]
    return torch.nn.functional.grid_sample(
        img[None], g, mode="bilinear", align_corners=True)[0]


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
    tmax = float(fish.theta_max)
    f_es = (w - 1) / 2 / (2 * math.sin(tmax / 2))
    es = KannalaBrandt(fx=f_es, fy=f_es, cx=(w - 1) / 2, cy=(h - 1) / 2,
                       width=w, height=h, k=EQUISOLID_K, theta_max=tmax)

    theta_f = fish.incidence_grid(h, w)
    cone = (theta_f <= tmax).numpy()
    cos_f = torch.cos(theta_f)
    theta_es = es.incidence_grid(h, w)
    cos_es = torch.cos(theta_es)

    rect_uv = fish.project(es.ray_grid(h, w))      # equisolid px -> fisheye px
    back_uv = es.project(fish.ray_grid(h, w))      # fisheye px -> equisolid px
    in_es = ((back_uv[..., 0] >= 0) & (back_uv[..., 0] <= w - 1)
             & (back_uv[..., 1] >= 0) & (back_uv[..., 1] <= h - 1)).numpy()
    print(f"[h8a] equisolid f={f_es:.1f}, cone coverage through remap "
          f"{(cone & in_es).sum() / cone.sum():.4f}")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=args.device,
                        variant="small")
    bb.install(None, fish, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb

    def forward_range(img: torch.Tensor, cos_grid: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats, _ = net.backbone(img[None, None].to(args.device),
                                    cam_token=None, export_feat_layers=[])
            out = net._process_depth_head(list(feats), h, w)
            z = (out["depth"] if isinstance(out, dict)
                 else out.depth).reshape(h, w)
        return z / cos_grid.to(z).clamp(min=1e-6)

    t_edges = np.linspace(0.0, tmax, THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta_f.numpy(), t_edges) - 1, 0,
                    THETA_BINS - 1)
    nb_d = len(GT_DEPTH_EDGES) - 1
    acc = {a: [np.zeros((THETA_BINS, nb_d)),
               np.zeros((THETA_BINS, nb_d), dtype=np.int64)]
           for a in ("plain", "equalarea")}
    for n in s.frames:
        img = s.src.image(n)
        gr = s.gt_range(n, cos_f).numpy()
        base_valid = cone & in_es & (gr > 0) & (gr <= args.depth_max_m)
        preds = {}
        preds["plain"] = forward_range(img, cos_f).cpu().numpy()
        es_img = bilinear(img, rect_uv, args.size)
        rng_es = forward_range(es_img, cos_es)
        preds["equalarea"] = bilinear(rng_es[None].cpu(), back_uv,
                                      args.size)[0].numpy()
        for arm, d in preds.items():
            valid = base_valid & (d > 1e-6)
            aligned = align_depth(d, gr, valid, mode="scale_shift")
            absrel = (np.abs(aligned - gr) / np.clip(gr, 1e-6, None))[valid]
            ti = t_idx[valid]
            di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0,
                         nb_d - 1)
            flat = ti * nb_d + di
            acc[arm][0] += np.bincount(flat, weights=absrel,
                                       minlength=THETA_BINS * nb_d
                                       ).reshape(THETA_BINS, nb_d)
            acc[arm][1] += np.bincount(flat, minlength=THETA_BINS * nb_d
                                       ).reshape(THETA_BINS, nb_d)

    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]
    tables = {a: acc[a][0] / np.maximum(acc[a][1], 1) for a in acc}
    counts = acc["plain"][1]
    print("\nplain -> equalarea joint AbsRel:")
    for j in range(nb_d):
        row = " ".join(f"{tables['plain'][i, j]:5.3f}>"
                       f"{tables['equalarea'][i, j]:5.3f}"
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
        wg = np.array([counts[i, j] for i, j in cells], float)
        vals = {a: float((np.array([tables[a][i, j] for i, j in cells])
                          * wg).sum() / wg.sum()) for a in tables}
        zones[zname] = vals
        print(f"{zname}: {vals['plain']:.3f} -> {vals['equalarea']:.3f} "
              f"({(vals['equalarea'] - vals['plain']) / vals['plain'] * 100:+.1f}%)")

    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump({"seq": s.name,
                       "tables": {a: t.tolist() for a, t in tables.items()},
                       "counts": counts.tolist(), "zones": zones,
                       "theta_bin_mid_deg": t_mid, "equisolid_f": f_es,
                       "config": vars(args)}, f, indent=2)
        print(f"\n[h8a] wrote {dst}")


if __name__ == "__main__":
    main()
