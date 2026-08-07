"""Is the depth path any good? `d_reproj` against a floor we can actually compute.

**Why depth, and why now.** Everything verified so far is pose. `d_reproj`
(Eq. 16) tests a completely different half of the pipeline -- the depth head, the
planar-z/range convention, `Omega`, and the matcher -- and the paper's largest
per-method spread is in that column. Tab. 1, DA3-Small on ScanNet++::

    Vanilla 23.82   Center-PH 2.21   Multi-PH 1.63   LoRA 4.98   CalTok 7.02   RayTun3R 4.16

Note the structure: vanilla is ~11x worse than Center-PH, and **RayTun3R loses to
both pinhole baselines**. The paper concedes it. That ordering is a sharp,
falsifiable prediction that does not need the absolute values.

**The trap, stated so it is not walked into again.** `d_reproj` is *not*
protocol-free. Eq. 16 uses ground-truth pose, so it isolates depth from pose --
but at zero parallax any depth reprojects onto itself, so the number still shrinks
with pair separation exactly as `R_deg` does. Chasing 23.82 by tuning stride would
be the same mistake tickets 9/10/12 made with 7.21.

**What makes it different from `R_deg` is that a floor is computable.** Because
Eq. 16 fixes the pose, the best depth consistent with the correspondences can be
triangulated from those same correspondences, and scored by the same metric. That
gives three numbers on identical pixels:

* **triangulated** -- the achievable floor, i.e. matcher + metric noise;
* **model** -- what the backbone predicts;
* **constant** -- a null, "every pixel at the same distance".

The ratio **model / triangulated is protocol-free**: both scale with parallax the
same way, so it divides out. That, and the log-log depth slope below, are the
numbers to quote. Measured on `3f15a9266d` with DA3-Small at sparse SIFT matches:
floor 0.098 px, model 0.826 px, constant 0.492 px -- the model sits **8.5x above
the floor**, and does not clearly beat a flat wall.

**Depth gain.** Regressing `log d_model` on `log d_triangulated` gives a slope
that is the depth analogue of rotation gain: 1.0 means the depth *range* is
recovered, lower means it is compressed. DA3-Small scores **0.406** here (a 10x
true depth variation predicted as 2.5x) with correlation 0.56 -- right ordering,
heavily flattened. Compare its rotation gain of 0.816 in `harness_verify`.

**Caveat on the convention.** A third regressor `log cos(theta)` would read 0 if
the head emits euclidean range and 1 if it emits planar z. It comes out ~0.3 here,
which settles nothing: with the depth signal itself attenuated to 0.4, and with
`theta` correlated with depth in a room (the periphery is the near door), the
term is confounded. **Do not conclude a convention from this script.** Rendered
ground-truth depth (ticket 006) is what settles it.

Usage::

    python -m raytun3r.experiments.depth_verify \\
        --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d \\
        --backbone da3 --weights pretrained --methods vanilla,center_ph
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

__all__ = ["main"]

#: Tab. 1, DA3-Small, ScanNet++ -- the training-free rows, as *ratios* to vanilla.
#: The row is a mean over unnamed scenes so the absolute values are not matchable
#: from one scene, but the between-method structure is.
PAPER_TAB1 = {"vanilla": 23.82, "center_ph": 2.21, "multi_ph": 1.63}

#: A triangulated floor worse than this means the matcher, not the model, is the
#: limiting factor and nothing downstream is interpretable.
FLOOR_SANITY_PX = 1.0


def _median(v) -> float:
    return float(np.median(v)) if len(v) else float("nan")


def _dreproj(rng: torch.Tensor, rays: torch.Tensor, dst: torch.Tensor,
             R: torch.Tensor, t: torch.Tensor, camera) -> float:
    """Median px reprojection error of a per-match euclidean range, scale-aligned.

    The scale is free (Eq. 16 solves for it), so this scans it rather than
    trusting any single closed form.
    """
    X = rng[:, None] * rays
    best = None
    for k in np.arange(-6.0, 6.0, 0.05):
        Xj = (float(2.0 ** k) * X) @ R.transpose(-1, -2) + t
        e = float((camera.project(Xj) - dst).norm(dim=-1).median())
        if best is None or e < best:
            best = e
    return best


def _triangulate(ri, rj, R, t):
    """Range along ``ri`` minimising ``||(d*R ri + t) x rj||``. Closed form."""
    A = ri @ R.transpose(-1, -2)
    axb = torch.cross(A, rj, dim=-1)
    txb = torch.cross(t.expand_as(rj), rj, dim=-1)
    d = -(axb * txb).sum(-1) / axb.pow(2).sum(-1).clamp_min(1e-12)
    par = torch.asin(axb.norm(dim=-1).clamp(0, 1)) * 180.0 / np.pi
    return d, par


def main(argv=None) -> None:
    from ..backbones import BACKBONE_NAMES

    p = argparse.ArgumentParser("raytun3r.experiments.depth_verify",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True)
    p.add_argument("--backbone", default="da3", choices=BACKBONE_NAMES)
    p.add_argument("--variant", default="small")
    p.add_argument("--weights", default="pretrained")
    p.add_argument("--methods", default="vanilla,center_ph")
    p.add_argument("--strides", default="5,10,20,40")
    p.add_argument("--pairs-per-stride", type=int, default=15)
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--ph-fov", type=float, default=110.0)
    p.add_argument("--keep-bad", action="store_true")
    p.add_argument("--min-parallax-deg", type=float, default=1.0)
    p.add_argument("--max-reproj-px", type=float, default=1.0)
    p.add_argument("--existing-only", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    import cv2

    from ..backbones import build_backbone
    from ..baselines import CenterPH, MultiPH
    from ..data import ScanNetPPFisheye

    src = ScanNetPPFisheye(args.path, max_size=args.max_size, patch=14,
                           keep_bad=args.keep_bad)
    if args.existing_only:
        src.frames = [fr for fr in src.frames
                      if os.path.exists(os.path.join(src._image_root, fr["file_path"]))]
    bb = build_backbone(args.backbone, weights=args.weights, device=args.device,
                        **({"variant": args.variant} if args.backbone == "da3" else {}))
    bb.install(None, src.camera, (src.h, src.w), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="range")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    preds = {}
    for m in methods:
        if m == "vanilla":
            preds[m] = lambda im: bb.forward(im[None])
        elif m in ("center_ph", "multi_ph"):
            ctor = CenterPH if m == "center_ph" else MultiPH
            preds[m] = ctor(bb, src.camera, fov_deg=args.ph_fov,
                            depth_convention="range")
        else:
            raise SystemExit(f"depth_verify handles training-free methods only, got {m!r}")

    sift = cv2.SIFT_create(nfeatures=8000)
    bf = cv2.BFMatcher()

    def gray(i):
        a = (src.image(i).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

    cols = ["triangulated"] + methods + ["constant"]
    acc: Dict[str, List[float]] = {c: [] for c in cols}
    lx: List[torch.Tensor] = []
    ly: Dict[str, List[torch.Tensor]] = {m: [] for m in methods}
    rows: Dict[str, Dict] = {}

    print(f"[depth] {len(src)} frames, grid {src.w}x{src.h}, backbone {args.backbone}")
    print(f"\n{'stride':>6} {'pairs':>6} {'matches':>8} " + " ".join(f"{c:>13}" for c in cols))
    for st in [int(s) for s in args.strides.split(",") if s.strip()]:
        starts = list(range(0, len(src) - st))
        if not starts:
            continue
        step = max(1, len(starts) // args.pairs_per_stride)
        per: Dict[str, List[float]] = {c: [] for c in cols}
        nm = 0
        for s in starts[::step][:args.pairs_per_stride]:
            gi, gj = src.pose(s), src.pose(s + st)
            if gi is None or gj is None:
                continue
            R = gj[0] @ gi[0].transpose(-1, -2)
            t = gj[1] - R @ gi[1]
            try:
                imgs = torch.stack([src.image(s), src.image(s + st)])
            except (FileNotFoundError, OSError):
                continue
            ka, da = sift.detectAndCompute(gray(s), None)
            kb, db = sift.detectAndCompute(gray(s + st), None)
            if da is None or db is None:
                continue
            good = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < 0.7 * n.distance]
            if len(good) < 50:
                continue
            p_i = torch.tensor([ka[m.queryIdx].pt for m in good], dtype=torch.float32)
            p_j = torch.tensor([kb[m.trainIdx].pt for m in good], dtype=torch.float32)
            ri = torch.nn.functional.normalize(src.camera.unproject(p_i), dim=-1)
            rj = torch.nn.functional.normalize(src.camera.unproject(p_j), dim=-1)
            d_tri, par = _triangulate(ri, rj, R, t)
            Xj = (d_tri[:, None] * ri) @ R.transpose(-1, -2) + t
            rep = (src.camera.project(Xj) - p_j).norm(dim=-1)
            # A weak filter here silently destroys the reference: near-parallel
            # rays triangulate to garbage that still passes a plain d > 0 test,
            # and the "floor" then looks worse than the model.
            ok = ((d_tri > 0.2) & (d_tri < 12.0) & (Xj[:, 2] > 0)
                  & (rep < args.max_reproj_px) & (par > args.min_parallax_deg))
            if int(ok.sum()) < 30:
                continue
            idx_all = (p_i[:, 1].round().long().clamp(0, src.h - 1),
                       p_i[:, 0].round().long().clamp(0, src.w - 1))
            # Center-PH and Multi-PH only predict part of the fisheye. Scoring a
            # method on pixels it never predicted charges it for zeros; scoring
            # each method on its own coverage compares different regions. Both are
            # wrong for a comparison, so everything is scored on the INTERSECTION.
            # (This is deliberately not Eq. 16, which averages over all of Omega,
            # nor eval.py, which reports per-method coverage separately.)
            depth_at, common = {}, ok.clone()
            for m, fn in preds.items():
                with torch.no_grad():
                    pr = fn(imgs.to(args.device))
                depth_at[m] = pr.depth[0].cpu()[idx_all]
                if pr.covered is not None:
                    common &= pr.covered[0].cpu()[idx_all]
                common &= depth_at[m] > 1e-4
            if int(common.sum()) < 30:
                continue

            rays, dst = ri[common], p_j[common]
            per["triangulated"].append(_dreproj(d_tri[common], rays, dst, R, t, src.camera))
            per["constant"].append(_dreproj(torch.ones(int(common.sum())), rays, dst,
                                            R, t, src.camera))
            for m in preds:
                per[m].append(_dreproj(depth_at[m][common], rays, dst, R, t, src.camera))
                ly[m].append(torch.log(depth_at[m][common].double()))
            lx.append(torch.log(d_tri[common].double()))
            nm += int(common.sum())
        if not per["triangulated"]:
            continue
        for c in cols:
            acc[c] += per[c]
        rows[f"s{st}"] = {"stride": st, "pairs": len(per["triangulated"]),
                          **{c: _median(per[c]) for c in cols}}
        print(f"{st:6d} {len(per['triangulated']):6d} {nm:8d} "
              + " ".join(f"{_median(per[c]):13.3f}" for c in cols))

    floor = _median(acc["triangulated"])
    print(f"\n{'':>22}" + " ".join(f"{c:>13}" for c in cols))
    print(f"{'median d_reproj (px)':>22}" + " ".join(f"{_median(acc[c]):13.3f}" for c in cols))
    print(f"{'ratio to floor':>22}"
          + " ".join(f"{_median(acc[c])/floor:12.1f}x" if floor > 0 else f"{'--':>13}"
                     for c in cols))

    gains = {}
    if lx:
        Xl = torch.cat(lx)
        for m in methods:
            Yl = torch.cat(ly[m])
            n = min(len(Xl), len(Yl))
            A = torch.stack([torch.ones(n, dtype=torch.float64), Xl[:n]], dim=1)
            b = float(torch.linalg.lstsq(A, Yl[:n, None]).solution[1, 0])
            c = float(torch.corrcoef(torch.stack([Xl[:n], Yl[:n]]))[0, 1])
            gains[m] = {"log_slope": b, "corr": c}
            print(f"  depth gain ({m}): log-log slope {b:.3f}, corr {c:.3f}   "
                  f"(1.0 = depth range recovered)")

    print()
    if not np.isfinite(floor) or floor > FLOOR_SANITY_PX:
        print(f"[depth] REFERENCE UNUSABLE: triangulated depth only reaches "
              f"{floor:.2f} px, above {FLOOR_SANITY_PX}. The matcher or the "
              f"triangulation filter is the limit, not the model. Nothing below "
              f"is interpretable -- tighten --min-parallax-deg / --max-reproj-px.")
    else:
        print(f"[depth] floor {floor:.3f} px from triangulated depth on the same "
              f"pixels. Quote the RATIO to it, not the raw px: both scale with "
              f"parallax, so the ratio is protocol-free while d_reproj is not.")
        v = methods[0]
        if len(methods) > 1:
            want = PAPER_TAB1["vanilla"] / PAPER_TAB1.get(methods[1], float("nan"))
            got = _median(acc[methods[0]]) / _median(acc[methods[1]])
            print(f"[depth] structure check, {methods[0]}/{methods[1]}: ours "
                  f"{got:.1f}x, Tab. 1 wants {want:.1f}x. This ratio is the "
                  f"reproducible content of that column -- the absolute px are not.")
        if _median(acc[v]) > _median(acc["constant"]):
            print(f"[depth] NOTE: {v} does not beat a CONSTANT depth "
                  f"({_median(acc[v]):.3f} vs {_median(acc['constant']):.3f} px). "
                  f"Either the fisheye damage is severe or the depth path is wrong; "
                  f"the depth gain above distinguishes them.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"scene": os.path.basename(args.path.rstrip("/")),
                       "backbone": args.backbone, "paper_tab1": PAPER_TAB1,
                       "rows": rows, "floor_px": floor,
                       "median": {c: _median(acc[c]) for c in cols},
                       "depth_gain": gains}, f, indent=2)
        print(f"\n[depth] wrote {args.out}")


if __name__ == "__main__":
    main()
