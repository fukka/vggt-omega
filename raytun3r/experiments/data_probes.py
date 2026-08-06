"""Answer the pixel-level questions about a dataset **without moving any pixels**.

ScanNet++ is distributed under Terms of Use that do not permit redistribution,
and `vggt-omega` is a public repo, so no frame, mask or depth map from it may be
committed. Every question we currently have about the data can be answered by a
number instead, and numbers about a dataset are not the dataset.

Run this on the box; commit the JSON. Each probe is stated so its output is
decisive rather than suggestive.

**Probe 1 — does the frame corner carry real image content?**
The paper calls ScanNet++ 115 deg; our intrinsics imply ~170 deg at the corner.
If the corner is black vignette then the image circle is smaller than the sensor,
``Omega`` should be the circle, and the disagreement is settled. Reports mean and
std of intensity in bins of incidence angle. A dead bin has near-zero std -- a
uniform black region -- while real content has std comparable to the inner bins.

**Probe 2 — what is ``mask_path``?**
ScanNet++ ships a per-frame DSLR mask that our loader ignores. It is either a
lens/vignette mask (then it *defines* ``Omega`` and we have ``Omega`` wrong) or an
anonymisation mask over faces and screens (then ignoring it is defensible). The
two look completely different in these numbers: a lens mask is radially symmetric,
identical across frames, and cuts at a fixed incidence angle; an anonymisation
mask is small, irregular, and varies frame to frame.

**Probe 3 — is ``render_depth`` planar z or euclidean range?**
Settled geometrically rather than by reading docs. Backproject frame i's GT depth
under a candidate convention, move it by the GT relative pose, project into frame
j, and compare against frame j's GT depth read under the same convention. The
wrong convention is off by a per-pixel ``1/cos(theta)``, which is radially varying
and therefore does *not* cancel under a pose change. The convention with the lower
residual is the one the files are stored in.

Usage::

    python -m raytun3r.experiments.data_probes \\
        --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d \\
        --json runs/audit/3f15a9266d-probes.json

Nothing here needs a GPU or any model weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional

import torch

__all__ = ["main"]


def _pct(v: List[float], p: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))] if s else float("nan")


def probe_content(src, frames: List[int], edges: List[float]) -> Dict:
    """Intensity statistics in bins of incidence angle."""
    cam = src.camera
    th = cam.incidence_grid(cam.height, cam.width).rad2deg()
    greys, got = [], []
    for i in frames:                       # a real download may have gaps; skip them
        try:
            greys.append(src.image(i).mean(0))     # (H, W) grey in [0, 1]
            got.append(i)
        except (FileNotFoundError, OSError):
            continue
    if not greys:
        return {"bins": [], "frames": [], "error": "no readable images"}
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        m = (th >= lo) & (th < hi)
        if not bool(m.any()):
            rows.append({"lo": lo, "hi": hi, "n": 0})
            continue
        means = [float(g[m].mean()) for g in greys]
        stds = [float(g[m].std()) for g in greys]
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                     "px_frac": float(m.float().mean()),
                     "mean": sum(means) / len(means), "std": sum(stds) / len(stds)})
    return {"bins": rows, "frames": got}


def probe_mask(src, frames: List[int], edges: List[float]) -> Optional[Dict]:
    """Is the shipped mask a lens mask or an anonymisation mask?"""
    import numpy as np
    from PIL import Image

    root = os.path.join(src.scene_dir, "dslr")
    got, per_frame, agree = [], [], None
    cam = src.camera
    th = cam.incidence_grid(cam.height, cam.width).rad2deg()
    for i in frames:
        rel = src.frames[i].get("mask_path")
        if not rel:
            continue
        for cand in (os.path.join(root, rel), os.path.join(root, "masks", os.path.basename(rel))):
            if os.path.exists(cand):
                break
        else:
            continue
        m = torch.from_numpy(np.asarray(Image.open(cand).convert("L")).astype("float32")) / 255.0
        m = torch.nn.functional.interpolate(m[None, None], size=(cam.height, cam.width),
                                            mode="nearest")[0, 0]
        keep = m > 0.5
        got.append(keep)
        prof = []
        for lo, hi in zip(edges, edges[1:]):
            b = (th >= lo) & (th < hi)
            prof.append(round(float(keep[b].float().mean()), 4) if bool(b.any()) else None)
        per_frame.append({"frame": i, "keep_frac": float(keep.float().mean()),
                          "keep_frac_by_incidence": prof})
    if not got:
        return None
    if len(got) > 1:
        # Identical across frames => lens/vignette. Varying => content-dependent.
        agree = float(torch.stack([(a == got[0]).float().mean() for a in got[1:]]).mean())
    return {"n_masks": len(got), "per_frame": per_frame,
            "pixelwise_agreement_with_first": agree,
            "reading": ("radially symmetric + identical across frames => lens mask, and "
                        "Omega should come from it; irregular + frame-varying => "
                        "anonymisation, and ignoring it is fine")}


#: Slope of log(D) on log(cos theta). Planar z gives 1, euclidean range gives 0.
#: Anything in between is not separated and must not be reported as a verdict.
CONV_SLOPE_Z, CONV_SLOPE_RANGE, CONV_SLOPE_MARGIN = 1.0, 0.0, 0.3


def probe_depth_convention(src, frames: List[int], edges: List[float]) -> Optional[Dict]:
    """Is the stored depth planar z or euclidean range? One frame is enough.

    ``range = z / cos(theta)``. Whatever the room looks like, the *range* to a
    surface has no particular reason to vary with the angle off the optical axis,
    while *planar z* carries an explicit ``cos(theta)`` factor. So regress
    ``log(D)`` on ``log(cos theta)`` across the image:

    * stored as planar z  -> ``D = range * cos(theta)`` -> slope ~ **1**
    * stored as range     -> ``D`` independent of theta -> slope ~ **0**

    At 80 deg off-axis the two differ by 5.7x, far beyond how much room geometry
    drifts with viewing angle, so the separation is wide. This deliberately
    replaces the two-frame reprojection test that was here first: that one draws
    all of its power from parallax, and at the small baselines in this data a
    wrong ``1/cos(theta)`` reprojects each pixel back onto itself and cancels --
    it returned the wrong verdict on a synthetic scene with a known answer.
    """
    lo_deg, hi_deg = 10.0, 80.0        # avoid log(cos)->0 at the axis and ->-inf at the rim
    cam = src.camera
    th = cam.incidence_grid(cam.height, cam.width)
    cos = th.cos()
    band = (th.rad2deg() >= lo_deg) & (th.rad2deg() <= hi_deg)

    slopes, profiles = [], []
    for i in frames:
        d = src.depth(i)
        if d is None:
            continue
        D, v = d
        ok = band & v & (D > 1e-6)
        if int(ok.sum()) < 1000:
            continue
        x = cos[ok].log().double()
        y = D[ok].log().double()
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if float(denom) <= 0:
            continue
        slopes.append(float(((x - xm) * (y - ym)).sum() / denom))
        prof = []
        for a, b in zip(edges, edges[1:]):
            m = v & (D > 1e-6) & (th.rad2deg() >= a) & (th.rad2deg() < b)
            prof.append(round(float(D[m].median()), 4) if int(m.sum()) > 50 else None)
        profiles.append({"frame": i, "slope": round(slopes[-1], 4),
                         "median_depth_by_incidence": prof})

    if not slopes:
        return {"verdict": None, "reading": "no usable GT depth (render_depth/ absent?)"}

    med = sorted(slopes)[len(slopes) // 2]
    out = {"n_frames": len(slopes), "median_slope": round(med, 4),
           "per_frame": profiles, "band_deg": [lo_deg, hi_deg],
           "expected": {"z": CONV_SLOPE_Z, "range": CONV_SLOPE_RANGE}}
    if abs(med - CONV_SLOPE_Z) <= CONV_SLOPE_MARGIN:
        out["verdict"] = "z"
    elif abs(med - CONV_SLOPE_RANGE) <= CONV_SLOPE_MARGIN:
        out["verdict"] = "range"
    else:
        out["verdict"] = None
        out["reading"] = (f"INCONCLUSIVE: slope {med:.3f} sits between the two "
                          f"predictions (z=1, range=0) by more than {CONV_SLOPE_MARGIN}")
    out.setdefault("reading", "slope ~1 => stored as planar z; slope ~0 => stored as range")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser("raytun3r.experiments.data_probes", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True)
    p.add_argument("--frames", default="0,50,200,400,800",
                   help="frame indices to sample (clamped to the sequence)")
    p.add_argument("--depth-pairs", default="0:10,200:210,400:410")
    p.add_argument("--incidence-bins", default="0,30,50,65,75,80,85,90")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    from ..data import ScanNetPPFisheye

    src = ScanNetPPFisheye(args.path)
    n = len(src)
    frames = sorted({min(int(x), n - 1) for x in args.frames.split(",") if x.strip()})
    edges = [float(x) for x in args.incidence_bins.split(",") if x.strip()]
    pairs = [tuple(min(int(v), n - 1) for v in s.split(":"))
             for s in args.depth_pairs.split(",") if ":" in s]

    rep: Dict = {"scene": os.path.basename(args.path.rstrip("/")), "n_frames": n,
                 "working_resolution": [src.camera.width, src.camera.height]}

    print(f"=== probe 1: is there content at the frame corner?  ({n} frames)")
    rep["content"] = probe_content(src, frames, edges)
    print(f"  {'incidence':>14} {'px frac':>9} {'mean':>8} {'std':>8}")
    for b in rep["content"]["bins"]:
        if not b.get("n"):
            continue
        print(f"  {b['lo']:5.0f}-{b['hi']:<5.0f}deg {b['px_frac']:9.3f} "
              f"{b['mean']:8.4f} {b['std']:8.4f}")
    print("  -> a near-zero std in the outer bins means dead vignette, so Omega is a")
    print("     circle and the paper's 115 deg is right; std like the inner bins means")
    print("     real content out to the corner and our ~170 deg is right.")

    print(f"\n=== probe 2: what is mask_path?")
    rep["mask"] = probe_mask(src, frames, edges)
    if rep["mask"] is None:
        print("  no masks found (no mask_path, or the files are elsewhere)")
    else:
        for f in rep["mask"]["per_frame"]:
            print(f"  frame {f['frame']:5d}  keep {f['keep_frac']:.4f}  "
                  f"by incidence {f['keep_frac_by_incidence']}")
        print(f"  pixelwise agreement across frames: {rep['mask']['pixelwise_agreement_with_first']}")

    print(f"\n=== probe 3: planar z or euclidean range?")
    rep["depth_convention"] = probe_depth_convention(src, frames, edges)
    dc = rep["depth_convention"]
    for f in dc.get("per_frame", []):
        print(f"  frame {f['frame']:5d}  slope {f['slope']:7.3f}   "
              f"median depth by incidence {f['median_depth_by_incidence']}")
    if dc.get("median_slope") is not None:
        print(f"  median slope of log(D) on log(cos theta): {dc['median_slope']:.4f}"
              f"   (planar z => 1, range => 0)")
    print(f"  -> verdict: {dc.get('verdict')!r}")
    print(f"     {dc.get('reading')}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\n[data-probes] wrote {args.json}")


if __name__ == "__main__":
    main()
