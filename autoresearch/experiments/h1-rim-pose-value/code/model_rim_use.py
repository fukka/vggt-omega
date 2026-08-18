"""H1.2: does a frozen depth FM actually use the fisheye periphery for alignment?

Protocol: ../protocol-h1.2.md (committed before this ran). DA3-Small on CPU.

Three conditions per pair: vanilla / rim-masked (theta > T to mean color) /
center-masked (theta <= T). If rim-masking leaves the model's rotation gain
unchanged, the model was not using the rim — and H1.1's classically-recoverable
span value is headroom for an adapter.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h1-rim-pose-value/code/model_rim_use.py \
        --path ~/Desktop/ADT/scannetpp_example/3f15a9266d --out results/run_004.json
"""

from __future__ import annotations

import argparse
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

from raytun3r.data import ScanNetPPFisheye            # noqa: E402
from raytun3r.metrics import rotation_error_deg       # noqa: E402

from rim_pose_value import _gain, _median             # noqa: E402


def masked(imgs: torch.Tensor, theta: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Mask pixels outside ``keep`` to each image's mean color over ``keep``."""
    out = imgs.clone()
    for k in range(out.shape[0]):
        mean = out[k][:, keep].mean(dim=1)
        out[k][:, ~keep] = mean[:, None]
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", required=True)
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--mask-deg", type=float, default=45.0)
    p.add_argument("--max-gt-rot-deg", type=float, default=30.0)
    p.add_argument("--max-pairs", type=int, default=16)
    p.add_argument("--variant", default="small")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    src = ScanNetPPFisheye(os.path.expanduser(args.path), max_size=args.max_size,
                           patch=14, keep_bad=False)
    src.frames = [fr for fr in src.frames
                  if os.path.exists(os.path.join(src._image_root, fr["file_path"]))]
    print(f"[h1.2] {len(src)} frames, grid {src.w}x{src.h}")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device="cpu",
                        variant=args.variant)
    bb.install(None, src.camera, (src.h, src.w), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="range")

    theta = src.camera.incidence_grid(src.h, src.w)
    rim_keep = theta <= math.radians(args.mask_deg)      # keep center
    cen_keep = ~rim_keep                                 # keep rim
    fr_rim_masked = float((~rim_keep).float().mean())
    fr_cen_masked = float((~cen_keep).float().mean())
    print(f"[h1.2] mask T={args.mask_deg} deg: rim-masked kills "
          f"{fr_rim_masked:.1%} of pixels, center-masked kills {fr_cen_masked:.1%}")

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
    print(f"[h1.2] {len(cand)} pairs, GT rot {cand[0][2]:.2f}..{cand[-1][2]:.2f} deg")

    conds = ["vanilla", "rim_masked", "center_masked"]
    per: Dict[str, List[float]] = {c: [] for c in conds}
    prd: Dict[str, List[float]] = {c: [] for c in conds}
    gts: List[float] = []

    for n, (i, j, ang) in enumerate(cand):
        gi, gj = src.pose(i), src.pose(j)
        R_gt = gj[0] @ gi[0].transpose(-1, -2)
        eye = torch.eye(3, dtype=R_gt.dtype)
        try:
            imgs = torch.stack([src.image(i), src.image(j)])
        except (FileNotFoundError, OSError):
            continue
        t0 = time.time()
        row = {}
        with torch.no_grad():
            for c, im in (("vanilla", imgs),
                          ("rim_masked", masked(imgs, theta, rim_keep)),
                          ("center_masked", masked(imgs, theta, cen_keep))):
                R_hat = bb.forward(im[None]).relative(0, 1)[0].to(R_gt)
                row[c] = (rotation_error_deg(R_hat, R_gt),
                          rotation_error_deg(eye, R_hat))
        for c in conds:
            per[c].append(row[c][0])
            prd[c].append(row[c][1])
        gts.append(ang)
        print(f"  pair {n+1}/{len(cand)} (GT {ang:5.2f} deg, "
              f"{time.time()-t0:5.1f}s): "
              + " ".join(f"{c} {row[c][0]:6.2f}" for c in conds), flush=True)

    print(f"\n=== {len(gts)} pairs ===")
    print(f"{'cond':>14} {'med err':>9} {'gain':>7}")
    summary: Dict[str, Dict] = {"conds": {}, "n_pairs": len(gts),
                                "rim_masked_pixel_frac": fr_rim_masked,
                                "center_masked_pixel_frac": fr_cen_masked,
                                "config": vars(args)}
    for c in conds:
        g = _gain(prd[c], gts)
        print(f"{c:>14} {_median(per[c]):9.3f} {g:7.3f}")
        summary["conds"][c] = {"median_err_deg": _median(per[c]), "gain": g}

    if args.out:
        out = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[h1.2] wrote {out}")


if __name__ == "__main__":
    main()
