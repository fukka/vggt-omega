"""Identify the paper's evaluation protocol -- or show that none exists.

**Why ticket 9's hit does not count.** It found a configuration matching the
paper's VGGT vanilla to 0.021 deg (stride 40, square 504x504, seq_len 2 ->
7.189 vs 7.21) and flagged the obvious objection itself. The objection is
correct, and the stage-1 numbers make it quantitative: our vanilla `R` is an
almost perfect affine function of the identity-predictor score `I` (the median
GT rotation, fixed by the frame span alone)::

    R = 0.42 + 0.170 * I        R^2 = 0.998   over spans 1..40

`I` rises monotonically with span, so `R(span)` is a smooth monotone curve and
*every* target between 0.46 and 5.67 deg is hit at exactly one span. Fitting one
free parameter to one number cannot fail, so it carries no evidence. The stride
40 result is a curve crossing.

**The fix is more targets measured under the same protocol.** The paper gives
several training-free ScanNet++ rows, and every one is a free extra constraint
because none of them needs fitting:

===========  =======================  =========================================
backbone     table                    training-free methods
===========  =======================  =========================================
``vggt``     Tab. 2, named ``3f15``   vanilla 7.21, Center-PH 2.45
``pi3``      Tab. 2, named ``3f15``   vanilla 6.17, Center-PH 2.28
``da3``      Tab. 1, ScanNet++ mean   vanilla 10.21, Center-PH 3.27, Multi-PH 1.66
===========  =======================  =========================================

Tab. 1's row is "the mean of per-scene means" over scenes the paper never names,
so its *absolute* level cannot be matched from one scene -- but it is the only
row with **three** training-free methods, and ratios between methods survive both
the unknown span and the unknown scene set. Tab. 2's two backbones are the tight
targets: same sequence, same protocol, four numbers.

Within one run, the methods share exactly one unknown -- the span the paper
evaluates at -- and each contributes its own constraint, which makes the fit
over-determined and therefore able to fail:

* if the two crossings **agree**, that span is the paper's protocol, and the
  harness is validated at two independent operating points instead of one;
* if they **disagree**, no span reproduces Tab. 2, and the discrepancy is not
  the evaluation protocol but the model or its preprocessing.

The crossing is reported as ``I*``: the median GT rotation a method would have
to face in order to score its paper number. Comparing `I*` between methods is
the whole test, and it avoids interpolating a stride back out of the curve.

Usage -- one command, then paste the final block::

    python -m raytun3r.experiments.protocol_identify \\
        --backbone vggt --weights pretrained \\
        --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d \\
        --out runs/protocol-identify/3f15a9266d.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import torch

__all__ = ["main"]

#: Training-free ScanNet++ rows, per backbone. Only methods that need no fitting
#: appear here, because each one has to be runnable at every span for free.
#:
#: `vggt` and `pi3` are Tab. 2, single **named** sequence 3f15a9266d -- the tight
#: targets. `da3` is Tab. 1, whose ScanNet++ row is "the mean of per-scene means"
#: over scenes the paper never names, so its absolute values are NOT expected to
#: match a single-scene run; `aggregate` marks that. It is still worth running,
#: because it is the only row with **three** training-free methods, and the
#: cross-method structure is informative even when the absolute level is not.
PAPER = {
    "vggt": {"table": "Tab. 2 (3f15a9266d)", "aggregate": False,
             "vanilla":   {"R_deg": 7.21, "t_deg": 16.6},
             "center_ph": {"R_deg": 2.45, "t_deg": 27.3}},
    "pi3":  {"table": "Tab. 2 (3f15a9266d)", "aggregate": False,
             "vanilla":   {"R_deg": 6.17, "t_deg": 19.7},
             "center_ph": {"R_deg": 2.28, "t_deg": 25.7}},
    "da3":  {"table": "Tab. 1 (ScanNet++ mean)", "aggregate": True,
             "vanilla":   {"R_deg": 10.21, "t_deg": 30.26},
             "center_ph": {"R_deg": 3.27,  "t_deg": 22.77},
             "multi_ph":  {"R_deg": 1.66,  "t_deg": 10.43}},
}

#: keys of PAPER[bb] that are methods rather than metadata
_META = ("table", "aggregate")

DEFAULT_STRIDES = (1, 2, 5, 10, 20, 40)

#: Two methods agree on a protocol if their required spans are within this
#: relative tolerance. 25% of span is far looser than the ~1 deg rule ticket 9
#: used on R, so a disagreement flagged here is a real one.
AGREE_TOL = 0.25

#: `I*` is read off the fitted line, so a loose fit invalidates the comparison.
#: Ticket 9's vanilla sweep hit R^2 = 0.998, so this is a low bar on real data.
MIN_R2 = 0.90


def _median(v: List[float]) -> float:
    return sorted(v)[len(v) // 2] if v else float("nan")


def _fit(xs: Sequence[float], ys: Sequence[float]):
    """Least-squares ``y = a + b x`` plus R^2. Returns None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    sx, sy = sum(xs), sum(ys)
    den = n * sum(x * x for x in xs) - sx * sx
    if abs(den) < 1e-12:
        return None
    b = (n * sum(x * y for x, y in zip(xs, ys)) - sx * sy) / den
    a = (sy - b * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def _load_source(path: str, keep_bad: bool, max_size: int, square: bool, patch: int,
                 existing_only: bool = False):
    from ..data import ScanNetPPFisheye

    src = ScanNetPPFisheye(path, max_size=max_size, patch=patch, keep_bad=keep_bad)
    if existing_only:
        # `transforms.json` lists the whole capture, but the staged local sample
        # holds a couple of dozen images. Without this the evenly-spaced sampler
        # picks indices whose files are absent and every pair is skipped, which
        # looks exactly like a working run that found nothing.
        src.frames = [fr for fr in src.frames
                      if os.path.exists(os.path.join(src._image_root, fr["file_path"]))]
    if square:
        side = (max_size // patch) * patch
        src.h = src.w = side
        src.camera = src.camera.resized(side, side)
    return src


def _predictor(bb, src, method: str, ph_fov: float):
    """Every method runs the frozen backbone with all corrections switched off."""
    bb.install(None, src.camera, (src.h, src.w), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="range")
    if method == "vanilla":
        return lambda imgs: bb.forward(imgs[None])
    if method in ("center_ph", "multi_ph"):
        from ..baselines import CenterPH, MultiPH
        ctor = CenterPH if method == "center_ph" else MultiPH
        # ProjectionBaseline takes [s,3,H,W] and adds its own batch dim.
        return ctor(bb, src.camera, fov_deg=ph_fov, depth_convention="range")
    raise ValueError(f"unknown method {method!r}")


def _run(pred_fn, src, stride: int, seq_len: int, n_pairs: int,
         device: str) -> Optional[Dict[str, float]]:
    from ..metrics import rotation_error_deg, translation_error_deg

    n = len(src)
    starts = list(range(0, n - (seq_len - 1) * stride))
    if not starts:
        return None
    # Evenly spaced over the whole sequence: an unbiased subsample of "the full
    # sequence", with no static filtering (the paper's 2 px filter is for the
    # adaptation set only).
    step = max(1, len(starts) // n_pairs)
    starts = starts[::step][:n_pairs]

    R_err, t_err, R_id, gt_ang, pred_ang = [], [], [], [], []
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
            pred = pred_fn(imgs)
        R_hat, t_hat = pred.relative(0, len(idx) - 1)
        R_hat, t_hat = R_hat.to(R_gt), t_hat.to(t_gt)
        R_err.append(rotation_error_deg(R_hat, R_gt))
        t_err.append(translation_error_deg(t_hat, t_gt))
        eye = torch.eye(3, dtype=R_gt.dtype)
        R_id.append(rotation_error_deg(eye, R_gt))
        # Rotation *magnitudes*, for the gain below. `rotation_error_deg(I, R)`
        # is exactly the angle of R, so this needs no second formula.
        gt_ang.append(R_id[-1])
        pred_ang.append(rotation_error_deg(eye, R_hat))
    if not R_err:
        return None
    r, ident = _median(R_err), _median(R_id)
    # Rotation gain: the slope through the origin of predicted angle on GT angle.
    # A model that reads a fraction `alpha` of every rotation scores an error of
    # exactly (1-alpha)*I, so `alpha` is the span-invariant content of the affine
    # fit's slope -- and unlike the slope it is measured directly, per span,
    # rather than inferred from how the error grows across spans.
    denom = sum(x * x for x in gt_ang)
    gain = (sum(x * y for x, y in zip(gt_ang, pred_ang)) / denom) if denom > 0 else float("nan")
    return {"n": len(R_err), "R_deg": r, "t_deg": _median(t_err),
            "identity": ident, "R_skill": ident / r if r > 0 else float("nan"),
            "gain": gain}


def main(argv=None) -> None:
    from ..backbones import BACKBONE_NAMES

    p = argparse.ArgumentParser("raytun3r.experiments.protocol_identify",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="vggt", choices=BACKBONE_NAMES)
    p.add_argument("--variant", default="small")
    p.add_argument("--weights", default="pretrained")
    p.add_argument("--path", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--methods", default="vanilla,center_ph")
    p.add_argument("--strides", default=",".join(str(s) for s in DEFAULT_STRIDES))
    p.add_argument("--pairs", type=int, default=100)
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--ph-fov", type=float, default=110.0)
    p.add_argument("--square", action="store_true",
                   help="stretch the working grid to max_size x max_size, the other "
                        "reading of '504 x 504'. Ticket 9's best-R config had this on, "
                        "but it also costs a flat ~27%% of R, so it is off by default")
    p.add_argument("--keep-bad", action="store_true")
    p.add_argument("--seq-len", type=int, default=2)
    p.add_argument("--existing-only", action="store_true",
                   help="drop frames whose image file is missing. For smoke-testing "
                        "against the staged local sample; leave off on a full download")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    from ..backbones import build_backbone

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    strides = [int(s) for s in args.strides.split(",") if s.strip()]

    t0 = time.time()
    bb = build_backbone(args.backbone, weights=args.weights, device=args.device,
                        **({"variant": args.variant} if args.backbone == "da3" else {}))
    print(f"[proto] {args.backbone} loaded in {time.time()-t0:.0f}s  "
          f"square={args.square} keep_bad={args.keep_bad} seq_len={args.seq_len}",
          flush=True)

    src = _load_source(args.path, args.keep_bad, args.max_size, args.square,
                       bb.patch_size, args.existing_only)
    print(f"[proto] {len(src)} frames, working grid {src.w}x{src.h}\n", flush=True)

    target = PAPER.get(args.backbone, {})
    if not target:
        print(f"[proto] WARNING: no ScanNet++ paper row for backbone "
              f"{args.backbone!r}; measuring only, no verdict.", flush=True)
    elif target["aggregate"]:
        print(f"[proto] targets from {target['table']} -- this row is a mean over "
              f"scenes the paper does not name, so the ABSOLUTE level is not "
              f"expected to match a single scene. The cross-method comparison "
              f"below is still meaningful; I* is not.\n", flush=True)
    else:
        print(f"[proto] targets from {target['table']}\n", flush=True)

    rows: Dict[str, Dict[int, Dict]] = {}
    for m in methods:
        pred_fn = _predictor(bb, src, m, args.ph_fov)
        rows[m] = {}
        tgt = target.get(m, {})
        print(f"[proto] {m}   (paper: R={tgt.get('R_deg', '?')} t={tgt.get('t_deg', '?')})",
              flush=True)
        for st in strides:
            r = _run(pred_fn, src, st, args.seq_len, args.pairs, args.device)
            if r is None:
                continue
            rows[m][st] = r
            print(f"    stride {st:3d}  n={r['n']:3d}  I={r['identity']:7.3f}  "
                  f"R={r['R_deg']:7.3f}  t={r['t_deg']:7.2f}  "
                  f"skill={r['R_skill']:5.2f}x  gain={r['gain']:.3f}", flush=True)
        print(flush=True)

    # --- the test -------------------------------------------------------------
    print("=== R as an affine function of the identity score, per method ===")
    fits, need = {}, {}
    for m in methods:
        pts = sorted(rows[m].values(), key=lambda r: r["identity"])
        if len(pts) < 2 or m not in target:
            continue
        f = _fit([r["identity"] for r in pts], [r["R_deg"] for r in pts])
        if f is None:
            continue
        a, b, r2 = f
        fits[m] = {"floor_deg": a, "slope": b, "r2": r2}
        span = (min(r["identity"] for r in pts), max(r["identity"] for r in pts))
        print(f"  {m:10s} R = {a:6.3f} + {b:6.4f} * I     R^2={r2:.4f}   "
              f"(I swept {span[0]:.2f}..{span[1]:.2f} deg)")
        if b > 1e-6:
            istar = (target[m]["R_deg"] - a) / b
            need[m] = istar
            fits[m]["I_star"] = istar
            fits[m]["extrapolated"] = not (span[0] <= istar <= span[1])

    print("\n=== I*: the median GT rotation each method needs to score its paper R ===")
    for m, istar in need.items():
        flag = "  << EXTRAPOLATED beyond the swept range" if fits[m]["extrapolated"] else ""
        print(f"  {m:10s} paper R={target[m]['R_deg']:5.2f}  ->  I* = {istar:8.2f} deg{flag}")

    print()
    worst_r2 = min((fits[m]["r2"] for m in need), default=float("nan"))
    if len(need) < 2:
        print("[proto] INCONCLUSIVE: need two training-free methods to over-determine "
              "the fit. Re-run with --methods vanilla,center_ph.")
    elif not (worst_r2 >= MIN_R2):
        # Every I* is read off the fitted line, so a loose fit makes the whole
        # comparison meaningless -- and it would fail toward DISAGREE, which is
        # the conclusion that costs the most to act on.
        print(f"[proto] INCONCLUSIVE: worst fit is R^2={worst_r2:.3f}, below {MIN_R2}. "
              f"I* is read off these lines, so neither AGREE nor DISAGREE can be "
              f"claimed. Widen --strides or raise --pairs and re-run.")
    else:
        vals = list(need.values())
        lo, hi = min(vals), max(vals)
        rel = (hi - lo) / ((hi + lo) / 2) if (hi + lo) > 0 else float("inf")
        n_m, tbl = len(need), target["table"]
        print(f"[proto] {n_m} methods, required spans differ by {rel*100:.0f}% "
              f"(I* from {lo:.2f} to {hi:.2f} deg)")
        if target["aggregate"]:
            print(f"[proto] NO VERDICT on I*: {tbl} is a mean over unnamed scenes, so "
                  f"the absolute level carries a scene-composition offset this run "
                  f"cannot separate from a protocol offset. Read the spread as "
                  f"indicative and the ratio cross-check below as the real signal.")
        elif rel <= AGREE_TOL:
            print(f"[proto] AGREE: all {n_m} methods want the same protocol, I* ~ "
                  f"{(lo+hi)/2:.1f} deg of GT rotation per pair. That span IS the "
                  f"paper's, and the harness reproduces {tbl} at {n_m} independent "
                  f"operating points. Re-run ticket 003 under it.")
        else:
            print(f"[proto] DISAGREE: no single span reproduces {tbl}. The {n_m} "
                  f"methods need protocols {rel*100:.0f}% apart, so the gap is NOT "
                  f"the evaluation protocol -- it is the backbone or its "
                  f"preprocessing. Ticket 9's stride-40 R match was a curve "
                  f"crossing, as suspected.")

    # --- gain, the span-invariant comparison ---------------------------------
    # `R_deg` ratios between methods are a bad summary once both errors are small:
    # they divide two near-zero numbers and explode. Rotation gain does not. A
    # model reading a fraction `alpha` of every rotation scores (1-alpha)*I, so
    # the paper's own numbers convert into a gain once the span is known -- and
    # the vanilla agreement is what supplies the span.
    if "vanilla" in need:
        I_ref = need["vanilla"]
        print(f"\n=== rotation gain at the span the vanilla agreement implies "
              f"(I = {I_ref:.1f} deg) ===")
        print("  a gain of 1.000 means the full rotation is recovered; the paper's "
              "column\n  assumes our fitted floor, so read it as approximate.")
        print(f"  {'method':10s} {'ours (measured)':>16s} {'paper (implied)':>16s}")
        for m in methods:
            if m not in target or m not in fits:
                continue
            near = min(rows[m].values(), key=lambda r: abs(r["identity"] - I_ref),
                       default=None)
            if near is None:
                continue
            paper_gain = 1.0 - (target[m]["R_deg"] - fits[m]["floor_deg"]) / I_ref
            print(f"  {m:10s} {near['gain']:16.3f} {paper_gain:16.3f}"
                  f"   (ours at I={near['identity']:.1f})")
        print("  a large R ratio between two methods that both sit near gain 1.0 is "
              "an\n  artefact of dividing small errors, not a disagreement about "
              "geometry.")

    # Ratios are the same test with no fitting and no absolute level: they are
    # dimensionless, so they survive both the span question and (for Tab. 1) the
    # unknown scene set. A protocol that explains the paper must reproduce them.
    pairs = [(a, b) for i, a in enumerate(methods) for b in methods[i + 1:]
             if a in target and b in target]
    for m0, m1 in pairs:
        want = target[m0]["R_deg"] / target[m1]["R_deg"]
        print(f"\n=== cross-check without fitting: R({m0}) / R({m1}) ===")
        print(f"  paper wants {want:.2f} at every span it evaluates")
        hit = False
        for st in strides:
            if st in rows.get(m0, {}) and st in rows.get(m1, {}):
                got = rows[m0][st]["R_deg"] / rows[m1][st]["R_deg"]
                ok = abs(got - want) < 0.3
                hit = hit or ok
                print(f"  stride {st:3d}  ours = {got:6.2f}   {'<-- matches' if ok else ''}")
        if not hit:
            print(f"  none of the swept spans reproduces {want:.2f} -- a span-free "
                  f"disagreement between us and the paper on these two methods.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"paper": target, "backbone": args.backbone,
                       "scene": os.path.basename(args.path.rstrip("/")),
                       "square": args.square, "keep_bad": args.keep_bad,
                       "seq_len": args.seq_len, "pairs": args.pairs,
                       "rows": rows, "fits": fits, "I_star": need}, f, indent=2)
        print(f"\n[proto] wrote {args.out}")


if __name__ == "__main__":
    main()
