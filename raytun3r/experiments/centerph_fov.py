"""Which virtual-pinhole FOV is the paper's Center-PH? Two backbones, one unknown.

**Why this is now the blocking question.** Ticket 10 identified the evaluation
span (stride 60 on `3f15a9266d`, `I` ~ 44 deg) by requiring one span to reproduce
the paper's vanilla on VGGT *and* π³ at once. With the span fixed, the paper's
remaining ScanNet++ numbers convert into **rotation gain** — the fraction of each
rotation a method recovers, which is what `R = a + b*I` actually measures
(`b = 1 - gain`). In those units:

===========  =========  =============  ============
backbone     method     paper's gain   ours
===========  =========  =============  ============
VGGT         vanilla    0.84           0.849
VGGT         Center-PH  0.95           **0.992**
VGGT         RayTun3R   0.98           not yet run
π³           vanilla    0.86           0.878
π³           Center-PH  0.95           **0.998**
π³           RayTun3R   0.98           not yet run
===========  =========  =============  ============

Our vanilla matches. **Our Center-PH does not — it is better, and it is better
than the paper's *RayTun3R*.** That single fact explains a result this
reproduction has produced and withdrawn twice: Center-PH beating the adapter,
"the reverse of Tab. 1". It is not a bug in the adapter. Our pinhole baseline is
simply stronger than the paper's, so the comparison is not the paper's comparison.

Until that is understood, training an adapter cannot test the paper's claim: to
beat *our* Center-PH it would have to beat the paper's own published RayTun3R
number (0.56 deg vs their 0.93 on VGGT; 0.20 vs 0.78 on π³). So the adapter run
is blocked on this, not the other way round.

**The unknown.** The paper never states the virtual pinhole's field of view.
Ours is 110 deg over a 504x504 view, covering 66% of the fisheye pixels. A
narrower view discards more periphery and should score worse — so FOV is a
one-parameter knob that moves Center-PH's gain, exactly as span moved vanilla's.

**The test, and why it can fail.** One free parameter against one number cannot
fail (§6c). So this sweeps FOV and requires the *same* FOV to reproduce Center-PH
on both backbones — 2.45 on VGGT and 2.28 on π³, separately published, different
architectures. Agreement pins the baseline configuration and makes Tab. 2's
Center-PH column reproducible; disagreement means the difference is not FOV.

**The confound this reports rather than hides.** Past a certain FOV the virtual
view runs off the edge of the fisheye cone and fills with black. Reaching the
paper's number by feeding the backbone dead pixels is not a configuration match,
so `live` (fraction of the virtual view carrying image) is printed on every row
and flagged past 5%.

Usage -- one command per backbone, then paste the table::

    python -m raytun3r.experiments.centerph_fov \\
        --backbone vggt --weights pretrained \\
        --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d \\
        --out runs/centerph-fov/3f15-vggt.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional

import torch

__all__ = ["main"]

#: Tab. 2, ScanNet++ 3f15. Center-PH is what we are solving for; vanilla is the
#: fixed reference that says the span is still the one ticket 10 identified.
PAPER = {"vggt": {"center_ph": 2.45, "vanilla": 7.21},
         "pi3":  {"center_ph": 2.28, "vanilla": 6.17}}

#: Ticket 10's answer. `I` ~ 43.7 deg on this scene at seq_len 2.
DEFAULT_STRIDE = 60

DEFAULT_FOVS = (60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0)

#: A virtual view with more dead area than this is reported as confounded: any
#: agreement it reaches may come from blanked pixels rather than from geometry.
LIVE_FLOOR = 0.95


def _median(v: List[float]) -> float:
    return sorted(v)[len(v) // 2] if v else float("nan")


def _measure(pred_fn, src, stride: int, n_pairs: int, device: str) -> Optional[Dict]:
    from ..metrics import rotation_error_deg, translation_error_deg

    starts = list(range(0, len(src) - stride))
    if not starts:
        return None
    step = max(1, len(starts) // n_pairs)
    starts = starts[::step][:n_pairs]

    R_err, t_err, gt_ang, pred_ang = [], [], [], []
    for s in starts:
        gi, gj = src.pose(s), src.pose(s + stride)
        if gi is None or gj is None:
            continue
        R_gt = gj[0] @ gi[0].transpose(-1, -2)
        t_gt = gj[1] - R_gt @ gi[1]
        try:
            imgs = torch.stack([src.image(s), src.image(s + stride)]).to(device)
        except (FileNotFoundError, OSError):
            continue
        with torch.no_grad():
            pred = pred_fn(imgs)
        R_hat, t_hat = pred.relative(0, 1)
        R_hat, t_hat = R_hat.to(R_gt), t_hat.to(t_gt)
        eye = torch.eye(3, dtype=R_gt.dtype)
        R_err.append(rotation_error_deg(R_hat, R_gt))
        t_err.append(translation_error_deg(t_hat, t_gt))
        gt_ang.append(rotation_error_deg(eye, R_gt))
        pred_ang.append(rotation_error_deg(eye, R_hat))
    if not R_err:
        return None
    den = sum(x * x for x in gt_ang)
    return {"n": len(R_err), "R_deg": _median(R_err), "t_deg": _median(t_err),
            "identity": _median(gt_ang),
            "gain": (sum(x * y for x, y in zip(gt_ang, pred_ang)) / den)
                    if den > 0 else float("nan")}


def main(argv=None) -> None:
    from ..backbones import BACKBONE_NAMES

    p = argparse.ArgumentParser("raytun3r.experiments.centerph_fov",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="vggt", choices=BACKBONE_NAMES)
    p.add_argument("--variant", default="small")
    p.add_argument("--weights", default="pretrained")
    p.add_argument("--path", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--fovs", default=",".join(str(f) for f in DEFAULT_FOVS))
    p.add_argument("--pairs", type=int, default=200)
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--keep-bad", action="store_true")
    p.add_argument("--existing-only", action="store_true",
                   help="drop frames with no image file; for the staged local sample")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    from ..backbones import build_backbone
    from ..baselines import CenterPH
    from ..data import ScanNetPPFisheye

    target = PAPER.get(args.backbone)
    if target is None:
        print(f"[fov] no Tab. 2 Center-PH target for {args.backbone!r} "
              f"(have {sorted(PAPER)}); measuring only.")

    t0 = time.time()
    bb = build_backbone(args.backbone, weights=args.weights, device=args.device,
                        **({"variant": args.variant} if args.backbone == "da3" else {}))
    src = ScanNetPPFisheye(args.path, max_size=args.max_size, patch=bb.patch_size,
                           keep_bad=args.keep_bad)
    if args.existing_only:
        src.frames = [fr for fr in src.frames
                      if os.path.exists(os.path.join(src._image_root, fr["file_path"]))]
    bb.install(None, src.camera, (src.h, src.w), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="range")
    print(f"[fov] {args.backbone} loaded in {time.time()-t0:.0f}s; {len(src)} frames, "
          f"stride {args.stride}, fisheye theta_max="
          f"{math.degrees(src.camera.theta_max):.1f} deg", flush=True)

    van = _measure(lambda im: bb.forward(im[None]), src, args.stride, args.pairs,
                   args.device)
    if van:
        ref = f" (paper {target['vanilla']})" if target else ""
        print(f"[fov] vanilla reference: R={van['R_deg']:.3f}{ref}  "
              f"gain={van['gain']:.3f}  I={van['identity']:.2f}\n", flush=True)

    rows: Dict[str, Dict] = {}
    print(f"{'fov':>6} {'live':>7} {'cover':>7} {'n':>4} {'R_deg':>8} {'t_deg':>8} "
          f"{'gain':>7} {'|dR|':>7}")
    for fov in [float(f) for f in args.fovs.split(",") if f.strip()]:
        base = CenterPH(bb, src.camera, fov_deg=fov, depth_convention="range")
        g = base.views[0].sampling_grid(src.camera)
        live = float(((g[..., 0].abs() <= 1) & (g[..., 1].abs() <= 1)).float().mean())
        cover = float(base.views[0].coverage(src.camera).float().mean())
        r = _measure(base, src, args.stride, args.pairs, args.device)
        if r is None:
            continue
        r.update(fov=fov, live=live, coverage=cover)
        if target:
            r["dist"] = abs(r["R_deg"] - target["center_ph"])
        rows[f"fov{int(fov)}"] = r
        flag = "  << dead pixels" if live < LIVE_FLOOR else ""
        print(f"{fov:6.0f} {live*100:6.1f}% {cover*100:6.1f}% {r['n']:4d} "
              f"{r['R_deg']:8.3f} {r['t_deg']:8.2f} {r['gain']:7.3f} "
              f"{r.get('dist', float('nan')):7.3f}{flag}", flush=True)

    if target and rows:
        best = min(rows.values(), key=lambda r: r["dist"])
        print(f"\n[fov] closest to the paper's Center-PH ({target['center_ph']}): "
              f"fov={best['fov']:.0f} -> R={best['R_deg']:.3f} "
              f"(off by {best['dist']:.3f}), gain={best['gain']:.3f}, "
              f"live={best['live']*100:.1f}%")
        if best["dist"] > 0.4:
            print("[fov] NO field of view reproduces the paper's Center-PH. The "
                  "difference is then not the virtual view's FOV, and the "
                  "remaining candidates are its resolution, the backbone "
                  "checkpoint, or the paper's baseline itself.")
        elif best["live"] < LIVE_FLOOR:
            print(f"[fov] CONFOUNDED: the closest FOV is only "
                  f"{best['live']*100:.1f}% live, so it may be matching by "
                  f"blanking pixels rather than by geometry. Say so when reporting.")
        else:
            print("[fov] candidate found. It only counts if the OTHER backbone "
                  "lands on the same FOV -- run both before concluding.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"paper": target, "backbone": args.backbone,
                       "scene": os.path.basename(args.path.rstrip("/")),
                       "stride": args.stride, "pairs": args.pairs,
                       "vanilla": van, "rows": rows}, f, indent=2)
        print(f"\n[fov] wrote {args.out}")


if __name__ == "__main__":
    main()
