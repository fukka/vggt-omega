"""H48 diagnostic — does either resampled arm still have resolving power?

The 2x2 came back "neither cell large", which is one of the three outcomes the
protocol named in advance. But an effect can be absent for two very different
reasons, and the numbers cannot tell them apart on their own:

    informative zero   the resampling kept the content and the effect went away
    uninformative zero the scored region no longer contains anything to get
                       wrong, so every model scores the same and no effect of
                       any size could have shown up

The suspicion is the second: in both arms the four backbones agree to three
decimals (R: 0.0844 / 0.0830 / 0.0837 / 0.0835) where the native cells spread
0.051 to 0.119, and per-frame AbsRel barely moves.

The test is a floor, not a range: fit the same scale_shift alignment to a
CONSTANT prediction. A constant is a fronto-parallel plane -- it knows nothing
about the scene. Whatever AbsRel it scores is the score a model gets for
knowing nothing. If the models sit at that floor, the cell has no resolving
power and its zero says nothing about the mirror effect.

Reported per cell, over the same theta <= 28 deg scored region in all four:

    n_valid          scored pixels per frame
    spread           (p95 - p5) / p50 of GT depth inside the mask
    floor            AbsRel of the aligned constant predictor
    headroom         floor / model_absrel  (>1 means the models beat the plane)

Cells: aa = Aria content, Aria lens (H47's cell)
       a  = ScanNet++ content, Aria lens
       r  = Aria content, ScanNet++ lens
       ss = ScanNet++ content, ScanNet++ lens (H46's cell)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "common"))
sys.path.append(str(_HERE.parents[1] / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h14-rect-distill" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h16-orientation" / "code"))

import importlib.util as _ilu  # noqa: E402
from raytun3r.cameras import from_aria  # noqa: E402
from raytun3r.data import ScanNetPPFisheye  # noqa: E402
from autoresearch.data.scannetpp_aria import AriaRemap  # noqa: E402
from finetune.eval.metrics import align_depth  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


Seq = _load("h5_train", _HERE.parents[1] / "h5-rim-finetune" / "code" / "train.py").Seq


def stats(gz, valid):
    g = gz[valid]
    p5, p50, p95 = np.percentile(g, [5, 50, 95])
    # A constant prediction; scale_shift fits the best plane through it, which
    # is the mean of GT over the mask. This is the know-nothing score.
    const = np.ones_like(gz, dtype=np.float32)
    al = align_depth(const, gz.astype(np.float32), valid, mode="scale_shift")
    floor = float(np.mean(np.abs(al[valid] - g) / g))
    return {"n_valid": int(valid.sum()),
            "p5": float(p5), "p50": float(p50), "p95": float(p95),
            "spread": float((p95 - p5) / max(p50, 1e-6)),
            "floor": floor}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", required=True, choices=("aa", "a", "r", "ss"))
    p.add_argument("--scene", default=None, help="ScanNet++ scene (a, ss, and r's lens)")
    p.add_argument("--seq", default=None, help="ADT sequence (aa, r)")
    p.add_argument("--side", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--common-theta-deg", type=float, default=28.0)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    per = []
    if a.cell in ("a", "ss"):
        src = ScanNetPPFisheye(a.scene, max_size=a.side, patch=14,
                               max_frames=a.max_frames, depth_convention="z")
        if a.cell == "a":
            dst = from_aria(a.side, a.side)
            remap = AriaRemap.build(src.camera, dst, (a.side, a.side))
            H = W = a.side
        else:
            dst, remap = src.camera, None
            H, W = src.h, src.w
        theta = dst.incidence_grid(H, W)
        cone = (theta <= dst.theta_max).numpy()
        common = cone & (np.rad2deg(theta.numpy()) <= a.common_theta_deg)
        for i in range(len(src)):
            d = src.depth(i)
            if d is None:
                continue
            dm, _ = (d if isinstance(d, tuple) else (d, None))
            dm = (dm.numpy() if torch.is_tensor(dm) else np.asarray(dm))
            if remap is None:
                gz, vd = dm.astype(np.float32), np.ones_like(common)
                vi = np.ones_like(common)
            else:
                _, vi = remap.image(src.image(i).permute(1, 2, 0).numpy())
                wd, vd = remap.depth((dm * 1000.0).astype(np.float32))
                gz = wd / 1000.0
            valid = common & vi & vd & (gz > 0) & (gz <= a.depth_max_m)
            if valid.sum() < 1000:
                continue
            per.append(stats(gz, valid))
        name = src.name
    else:
        src = Seq(a.seq, a.side, a.max_frames)
        aria = src.src.camera
        if a.cell == "r":
            lens = ScanNetPPFisheye(a.scene, max_size=a.side, patch=14,
                                    max_frames=1, depth_convention="z")
            dst, H, W = lens.camera, lens.h, lens.w
            remap = AriaRemap.build(aria, dst, (H, W))
        else:
            dst, remap = aria, None
            H = W = a.side
        theta = dst.incidence_grid(H, W)
        cone = (theta <= dst.theta_max).numpy()
        common = cone & (np.rad2deg(theta.numpy()) <= a.common_theta_deg)
        for f in src.frames:
            gz_mm = np.load(src.dp[src.stem(f)]).astype(np.float32)
            if remap is None:
                # Native Aria: depth_npy is at the sensor's own 1408 grid while
                # the scored region is built on the 504 camera the images are
                # resized to. Nearest-neighbour index map, never interpolation
                # -- averaging across a depth discontinuity invents surfaces.
                if gz_mm.shape != common.shape:
                    ii = ((np.arange(common.shape[0]) + 0.5)
                          * gz_mm.shape[0] / common.shape[0]).astype(int)
                    jj = ((np.arange(common.shape[1]) + 0.5)
                          * gz_mm.shape[1] / common.shape[1]).astype(int)
                    gz_mm = gz_mm[np.ix_(ii, jj)]
                gz, vd, vi = gz_mm / 1000.0, np.ones_like(common), np.ones_like(common)
            else:
                _, vi = remap.image(src.src.image(f).permute(1, 2, 0).numpy())
                wd, vd = remap.depth(gz_mm)
                gz = wd / 1000.0
            valid = common & vi & vd & (gz > 0) & (gz <= a.depth_max_m)
            if valid.sum() < 1000:
                continue
            per.append(stats(gz, valid))
        name = src.name

    if not per:
        sys.exit(f"[diag] {a.cell}: nothing usable")
    agg = {k: float(np.mean([x[k] for x in per])) for k in per[0]}
    print(f"[diag] {a.cell:2s} {name}: {len(per)} frames  "
          f"n_valid {agg['n_valid']:.0f}  "
          f"depth p5/p50/p95 {agg['p5']:.2f}/{agg['p50']:.2f}/{agg['p95']:.2f} m  "
          f"spread {agg['spread']:.3f}  floor {agg['floor']:.4f}", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"cell": a.cell, "name": name, "n_frames": len(per),
             "agg": agg, "per_frame": per}, indent=2))
        print(f"[diag] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
