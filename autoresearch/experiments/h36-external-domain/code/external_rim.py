"""H36 - is the rim penalty an Aria artefact, or does it exist in another domain?

Every number in this line comes from one camera family, one apartment and one
small office. SynWoodScape is synthetic automotive surround-view fisheye,
outdoors, ~190 degree lens, with depth ground truth - about as far from an
indoor Aria recording as a fisheye dataset gets.

Zones are defined as FRACTIONS of theta_max, taken from the Aria thresholds
(38/54.83 = outer 30.7%, 11/54.83 = inner 20.1%), because the absolute angles
mean something completely different on a 190 degree lens. No depth band, since
the Aria band encodes "the wearer's hands" and has no automotive equivalent.

The `depthfisheye` package is used READ-ONLY; it belongs to a different work
stream and is not modified here.
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

RIM_FRAC, CTR_FRAC = 38.0 / 54.83, 11.0 / 54.83


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sws-root", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--size", type=int, default=518)
    p.add_argument("--variant", default="small")
    p.add_argument("--max-depth", type=float, default=40.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from depthfisheye.synwoodscape import SynWoodScape
    from raytun3r.backbones import build_backbone
    from finetune.eval.metrics import align_depth

    ds = SynWoodScape(a.sws_root, split="val", size=a.size, max_depth=a.max_depth)
    print(f"[h36] {len(ds)} val samples, cameras {ds.cameras}", flush=True)

    bb = build_backbone("da3", weights="pretrained", device=a.device, variant=a.variant)

    per_cam, rows = {}, []
    n = min(a.n, len(ds))
    for k in range(n):
        s = ds[k]
        cam = s["cam"]
        C = ds.calib[cam]
        th = C.incidence_grid(a.size, a.size).numpy()
        tmax = float(C.theta_max)
        rim = th >= RIM_FRAC * tmax
        ctr = th <= CTR_FRAC * tmax
        gt = s["depth"].numpy(); ok = s["valid"].numpy()
        bb.install(None, C, (a.size, a.size), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="z")
        with torch.no_grad():
            pr = bb.forward(s["image"][None, None].to(a.device)).depth[0].float().cpu().numpy()
        v = ok & (pr > 1e-6) & np.isfinite(pr)
        if v.sum() < 5000:
            continue
        al = align_depth(pr, gt, v, mode="scale_shift")
        ar = np.abs(al - gt) / np.clip(gt, 1e-6, None)
        e_all = float(ar[v].mean())
        e_rim = float(ar[v & rim].mean()) if (v & rim).sum() > 500 else np.nan
        e_ctr = float(ar[v & ctr].mean()) if (v & ctr).sum() > 500 else np.nan
        rows.append(dict(cam=cam, all=e_all, rim=e_rim, ctr=e_ctr,
                         ratio=e_rim / e_ctr if e_ctr and np.isfinite(e_rim) else np.nan))
        per_cam.setdefault(cam, []).append(rows[-1])
        if k % 40 == 0:
            print(f"[h36] {k}/{n}  {cam}  all {e_all:.3f}  rim {e_rim:.3f}  ctr {e_ctr:.3f}", flush=True)

    R = np.array([r["ratio"] for r in rows], float)
    A = np.array([r["all"] for r in rows], float)
    RIMv = np.array([r["rim"] for r in rows], float)
    CTRv = np.array([r["ctr"] for r in rows], float)
    good = np.isfinite(R)
    print(f"\n[h36] {int(good.sum())} usable frames")
    print(f"  whole-image AbsRel {np.nanmean(A):.4f}")
    print(f"  rim  {np.nanmean(RIMv):.4f}   centre {np.nanmean(CTRv):.4f}")
    print(f"  rim/centre  {np.nanmean(R[good]):.3f} +- {np.nanstd(R[good], ddof=1):.3f}"
          f"   median {np.nanmedian(R[good]):.3f}")
    for cam, rs in sorted(per_cam.items()):
        rr = np.array([x["ratio"] for x in rs], float)
        print(f"    {cam:>4s}: n={len(rs):3d}  ratio {np.nanmean(rr):.3f}")
    m = float(np.nanmean(R[good]))
    print(f"\n  B1  ratio > 1.5 ? {m:.3f} -> {'PASS' if m > 1.5 else ('FAIL' if m <= 1.2 else 'MARGINAL')}")
    print(f"  B2  inside Aria's 1.4-2.7 ? -> {'PASS' if 1.4 <= m <= 2.7 else 'FAIL'}")
    print(f"  B3  whole-image AbsRel < 0.6 ? {np.nanmean(A):.3f} -> "
          f"{'PASS' if np.nanmean(A) < 0.6 else 'FAIL'}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"rows": rows, "config": vars(a),
             "zone_fracs": {"rim": RIM_FRAC, "ctr": CTR_FRAC}}, indent=1))
        print(f"[h36] wrote {a.out}")


if __name__ == "__main__":
    main()
