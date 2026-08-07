"""Reproduce the paper's *vanilla* number exactly, and thereby validate the harness.

**Why vanilla first.** It is the only quantity in the paper with no adapter, no
training, no matcher, no randomness: a frozen backbone on a posed pair. If our
vanilla does not match theirs, every adapter comparison downstream is measured on
a different thing and cannot be interpreted. Conversely, once vanilla matches, the
data loader, the camera model, the pose convention and the metric are all
validated at once.

**The target.** The paper gives exactly one *named-scene* vanilla number for a
backbone we have implemented:

    Tab. 2, VGGT, ScanNet++ 3f15 -->  R = 7.21   t = 16.6   d_reproj = 39.4

`pi^3` has one too (6.17 / 19.7 / 38.6) but is not implemented here. **DA3-Small
has none**: Tab. 1's ScanNet++ row is a mean over scenes the paper never names,
and Tab. 6's DA3 baseline is ETH3D. So DA3-S vanilla cannot be checked
per-scene -- only VGGT can, and that is what this script targets.

**Why a sweep instead of a guess.** The paper under-specifies the evaluation
protocol in four ways that all move `R`, and we have been guessing at them one at
a time across several GPU round-trips. This enumerates them in a single run and
reports which combination lands on the target:

* ``stride`` -- not a paper concept. It says "consecutive image pairs", but a
  1.09 cm baseline makes stride 1 nearly static, and `R_err` is an absolute angle
  whose scale is set by how much rotation there is to estimate.
* ``is_bad`` -- ScanNet++ flags 143 of 896 frames on this scene as unusable. The
  paper does not say whether it honours the flag.
* ``resolution`` -- "resized to a maximum patch-aligned resolution of 504 x 504"
  reads either as a 504 cap on the long side (504x336 here) or as a square 504x504.
* ``seq_len`` -- these are multi-view models, so a pair's prediction depends on
  what else is in the window. 2-frame and 3-frame windows are both defensible.

Pose needs no correspondences, so the sweep runs without a matcher and is cheap.
``--with-dreproj`` adds UFM for the best few configurations only.

Usage -- one command, then paste the final table::

    python -m raytun3r.experiments.vanilla_repro \\
        --backbone vggt --weights pretrained \\
        --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d \\
        --out runs/vanilla-repro/3f15a9266d.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from typing import Dict, List, Optional

import torch

__all__ = ["main"]

#: Tab. 2, VGGT, ScanNet++ 3f15. The only named-scene vanilla triple we can target.
PAPER_TARGET = {"R_deg": 7.21, "t_deg": 16.6, "d_reproj": 39.4}

#: Stage 1 sweeps stride at the base protocol; stage 2 varies the rest at the
#: strides that came closest. Keeps the run to a few minutes instead of a grid.
DEFAULT_STRIDES = (1, 2, 5, 10, 20, 40)


def _median(v: List[float]) -> float:
    return sorted(v)[len(v) // 2] if v else float("nan")


def _load_source(path: str, keep_bad: bool, max_size: int, square: bool, patch: int):
    from ..data import ScanNetPPFisheye

    src = ScanNetPPFisheye(path, max_size=max_size, patch=patch, keep_bad=keep_bad)
    if square:
        # The other reading of "504 x 504": letterbox the working grid to a square
        # so the model sees the same token count on both axes.
        side = (max_size // patch) * patch
        src.h = src.w = side
        src.camera = src.camera.resized(side, side)
    return src


def _run_config(bb, src, stride: int, seq_len: int, n_pairs: int,
                device: str) -> Optional[Dict[str, float]]:
    from ..metrics import rotation_error_deg, translation_error_deg

    n = len(src)
    starts = list(range(0, n - (seq_len - 1) * stride))
    if not starts:
        return None
    # Deterministic, evenly spaced across the whole sequence -- an unbiased
    # subsample of "the full sequence", with no static filtering.
    step = max(1, len(starts) // n_pairs)
    starts = starts[::step][:n_pairs]

    bb.install(None, src.camera, (src.h, src.w), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="range")

    R_err, t_err, R_id = [], [], []
    for s in starts:
        idx = [s + k * stride for k in range(seq_len)]
        gi, gj = src.pose(idx[0]), src.pose(idx[-1])
        if gi is None or gj is None:
            continue
        R_gt = gj[0] @ gi[0].transpose(-1, -2)
        t_gt = gj[1] - R_gt @ gi[1]
        try:                      # a staged sample holds only some of the imagery
            imgs = torch.stack([src.image(i) for i in idx]).to(device)
        except (FileNotFoundError, OSError):
            continue
        with torch.no_grad():
            pred = bb.forward(imgs[None])
        R_hat, t_hat = pred.relative(0, len(idx) - 1)
        R_err.append(rotation_error_deg(R_hat, R_gt))
        t_err.append(translation_error_deg(t_hat, t_gt))
        R_id.append(rotation_error_deg(torch.eye(3, dtype=R_gt.dtype), R_gt))
    if not R_err:
        return None
    r = _median(R_err)
    return {"n": len(R_err), "R_deg": r, "t_deg": _median(t_err),
            "R_deg_identity": _median(R_id),
            "R_skill": _median(R_id) / r if r > 0 else float("nan"),
            "dist_to_target": abs(r - PAPER_TARGET["R_deg"])}


def main(argv=None) -> None:
    p = argparse.ArgumentParser("raytun3r.experiments.vanilla_repro",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="vggt", choices=["vggt", "vggt_omega", "da3"])
    p.add_argument("--variant", default="small")
    p.add_argument("--weights", default="pretrained")
    p.add_argument("--path", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--strides", default=",".join(str(s) for s in DEFAULT_STRIDES))
    p.add_argument("--pairs", type=int, default=100,
                   help="pairs per configuration, evenly spaced over the sequence")
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    from ..backbones import build_backbone

    t0 = time.time()
    bb = build_backbone(args.backbone, weights=args.weights, device=args.device,
                        **({"variant": args.variant} if args.backbone == "da3" else {}))
    print(f"[repro] {args.backbone} loaded in {time.time()-t0:.0f}s; "
          f"target R={PAPER_TARGET['R_deg']} t={PAPER_TARGET['t_deg']} "
          f"(Tab. 2, VGGT, ScanNet++ 3f15)", flush=True)

    rows: Dict[str, Dict] = {}

    def go(stride: int, keep_bad: bool, square: bool, seq_len: int) -> None:
        key = f"s{stride}_bad{int(keep_bad)}_sq{int(square)}_L{seq_len}"
        if key in rows:
            return
        src = _load_source(args.path, keep_bad, args.max_size, square, bb.patch_size)
        r = _run_config(bb, src, stride, seq_len, args.pairs, args.device)
        if r is None:
            return
        r.update(stride=stride, keep_bad=keep_bad, square=square, seq_len=seq_len,
                 res=[src.w, src.h])
        rows[key] = r
        print(f"  {key:28s} n={r['n']:3d}  R={r['R_deg']:7.3f}  t={r['t_deg']:7.2f}  "
              f"identity={r['R_deg_identity']:7.3f}  skill={r['R_skill']:5.2f}x  "
              f"|dR|={r['dist_to_target']:6.3f}", flush=True)

    # Stage 1 -- stride at the base protocol.
    print("\n[repro] stage 1: stride", flush=True)
    strides = [int(s) for s in args.strides.split(",") if s.strip()]
    for s in strides:
        go(s, keep_bad=False, square=False, seq_len=2)

    # Stage 2 -- the other three axes, only at the two closest strides.
    best = sorted(rows.values(), key=lambda r: r["dist_to_target"])[:2]
    print(f"\n[repro] stage 2: other axes at stride "
          f"{[b['stride'] for b in best]}", flush=True)
    for b in best:
        for keep_bad, square, seq_len in itertools.product((False, True), (False, True), (2, 3)):
            go(b["stride"], keep_bad, square, seq_len)

    ranked = sorted(rows.values(), key=lambda r: r["dist_to_target"])
    print("\n=== closest to the paper's VGGT vanilla (R=7.21, t=16.6) ===")
    print(f"{'stride':>6} {'is_bad':>7} {'square':>7} {'L':>2} {'res':>10} "
          f"{'R_deg':>8} {'t_deg':>8} {'skill':>7} {'|dR|':>7}")
    for r in ranked[:10]:
        print(f"{r['stride']:6d} {str(r['keep_bad']):>7} {str(r['square']):>7} "
              f"{r['seq_len']:2d} {str(r['res']):>10} {r['R_deg']:8.3f} "
              f"{r['t_deg']:8.2f} {r['R_skill']:6.2f}x {r['dist_to_target']:7.3f}")

    top = ranked[0]
    print(f"\n[repro] best: stride={top['stride']} keep_bad={top['keep_bad']} "
          f"square={top['square']} seq_len={top['seq_len']} -> R={top['R_deg']:.3f} "
          f"vs target {PAPER_TARGET['R_deg']} (off by {top['dist_to_target']:.3f})")
    if top["dist_to_target"] > 1.0:
        print("[repro] NO configuration reproduces the paper's vanilla. That is the "
              "finding: the difference is not the evaluation protocol, and the "
              "backbone or its preprocessing has to be the next suspect.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"target": PAPER_TARGET, "backbone": args.backbone,
                       "scene": os.path.basename(args.path.rstrip("/")),
                       "pairs_per_config": args.pairs, "configs": rows}, f, indent=2)
        print(f"[repro] wrote {args.out}")


if __name__ == "__main__":
    main()
