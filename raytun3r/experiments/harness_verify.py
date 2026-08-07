"""Is the evaluation harness correct? Answered without the paper, and without a backbone.

**Why this exists.** Three tickets were spent trying to reproduce the paper's
vanilla `R°` by varying `stride`, `is_bad`, resolution and window length until the
number matched. That is fitting, not verifying: `R°` is an *absolute* angle, so it
is set by which pairs you evaluate, and the paper does not say. Any pair selection
can be made to produce any number in a wide range, so agreement proves nothing and
disagreement diagnoses nothing.

**What actually verifies the harness.** Classical geometry. Detect SIFT features
on two real frames, unproject the matches to bearings *through our own camera
model*, recover the relative pose with MAGSAC++, and compare to *our own* ground
truth. Every component under suspicion is in that loop:

* the Kannala-Brandt fisheye model and its `k1..k4` (via `unproject`),
* the nerfstudio -> OpenCV pose conversion and the cam-from-world convention,
* the relative-pose algebra `R_rel = R_j R_i^T`,
* `rotation_error_deg` itself,
* and that the images and the poses actually correspond to each other.

If a frozen 1990s algorithm recovers the ground truth from our images through our
camera model, then none of those is broken, and any remaining disagreement with
the paper is about the paper's setup rather than our correctness. No published
number is needed, and none is used here.

**It also supplies the reference the paper cannot.** Classical geometry is the
"a correct method should reach about this" line. A learned backbone that scores far
worse on the same pairs is showing real damage, not a harness bug -- and the
adapter's job is to close that gap. Measured on ScanNet++ `3f15a9266d`:

    SIFT+MAGSAC   median 0.31 deg, gain 0.97
    DA3 vanilla   median 2.63 deg, gain 0.72
    DA3 Center-PH median 1.16 deg, gain 0.82

**Read `gain`, not `R_deg`.** A model recovering a fraction `alpha` of every
rotation scores `(1-alpha)*I`, so `R_deg` slides with pair separation while
`alpha` does not. Gain is comparable across strides, scenes and datasets; bare
`R_deg` is comparable across none of them.

Usage::

    python -m raytun3r.experiments.harness_verify \\
        --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d \\
        --backbone da3 --weights pretrained      # --backbone is optional
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

__all__ = ["main"]

#: A harness in which classical geometry cannot beat this on well-separated pairs
#: has something wrong with it -- the camera model, the poses, or the metric.
CLASSICAL_SANITY_DEG = 1.5


def _median(v: List[float]) -> float:
    return float(np.median(v)) if len(v) else float("nan")


def _gain(preds: List[float], gts: List[float]) -> float:
    """Fraction of rotation recovered: slope of predicted angle on GT angle.

    Deliberately *not* derived from ``|error|``. The error is unsigned, so a
    method that overshoots and one that undershoots look identical, and a single
    large miss drags the estimate down however the rest behaved. Regressing the
    predicted magnitude on the true one through the origin is the actual quantity
    -- 1.0 means the full rotation is recovered.
    """
    p = np.asarray(preds, dtype=float)
    g = np.asarray(gts, dtype=float)
    ok = ~np.isnan(p) & (g > 0)
    if not ok.any():
        return float("nan")
    return float((p[ok] * g[ok]).sum() / (g[ok] ** 2).sum())


def _sift_pose(src, a: int, b: int, nfeat: int, ratio: float, thr_deg: float):
    import cv2

    from ..matching import Matches, relative_pose_magsac

    sift = _sift_pose._sift = getattr(_sift_pose, "_sift", None) or \
        cv2.SIFT_create(nfeatures=nfeat)
    bf = _sift_pose._bf = getattr(_sift_pose, "_bf", None) or cv2.BFMatcher()

    def gray(i):
        arr = (src.image(i).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    ka, da = sift.detectAndCompute(gray(a), None)
    kb, db = sift.detectAndCompute(gray(b), None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None
    good = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < ratio * n.distance]
    if len(good) < 20:
        return None
    # relative_pose_magsac takes a dense field, so scatter the sparse matches.
    tgt = torch.zeros(src.h, src.w, 2)
    wgt = torch.zeros(src.h, src.w)
    for m in good:
        u, v = ka[m.queryIdx].pt
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < src.w and 0 <= vi < src.h:
            tgt[vi, ui] = torch.tensor(kb[m.trainIdx].pt)
            wgt[vi, ui] = 1.0
    if int(wgt.sum()) < 20:
        return None
    out = relative_pose_magsac(Matches(target=tgt, weight=wgt), src.camera,
                               threshold_deg=thr_deg)
    return None if out is None else (out[0], int(wgt.sum()))


def main(argv=None) -> None:
    from ..backbones import BACKBONE_NAMES

    p = argparse.ArgumentParser("raytun3r.experiments.harness_verify",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True)
    p.add_argument("--backbone", default=None, choices=BACKBONE_NAMES,
                   help="optional: also score this backbone on the SAME pairs, "
                        "vanilla and Center-PH. Omit for the harness check alone")
    p.add_argument("--variant", default="small")
    p.add_argument("--weights", default="pretrained")
    p.add_argument("--strides", default="1,2,5,10,20,40,60")
    p.add_argument("--pairs-per-stride", type=int, default=20)
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--ph-fov", type=float, default=110.0)
    p.add_argument("--keep-bad", action="store_true")
    p.add_argument("--nfeatures", type=int, default=6000)
    p.add_argument("--ratio", type=float, default=0.8)
    p.add_argument("--magsac-thresh-deg", type=float, default=0.5)
    p.add_argument("--existing-only", action="store_true",
                   help="drop frames with no image file; for the staged local sample")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    from ..data import ScanNetPPFisheye
    from ..metrics import rotation_error_deg

    src = ScanNetPPFisheye(args.path, max_size=args.max_size, patch=14,
                           keep_bad=args.keep_bad)
    if args.existing_only:
        src.frames = [fr for fr in src.frames
                      if os.path.exists(os.path.join(src._image_root, fr["file_path"]))]
    print(f"[verify] {len(src)} frames, grid {src.w}x{src.h}")

    bb = cph = None
    if args.backbone:
        from ..backbones import build_backbone
        from ..baselines import CenterPH
        bb = build_backbone(args.backbone, weights=args.weights, device=args.device,
                            **({"variant": args.variant} if args.backbone == "da3" else {}))
        src.camera = src.camera            # unchanged; install binds the working grid
        bb.install(None, src.camera, (src.h, src.w), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="range")
        cph = CenterPH(bb, src.camera, fov_deg=args.ph_fov, depth_convention="range")
        print(f"[verify] also scoring {args.backbone} vanilla and Center-PH "
              f"(fov {args.ph_fov:.0f}) on the same pairs")

    strides = [int(s) for s in args.strides.split(",") if s.strip()]
    cols = ["classical"] + (["vanilla", "center_ph"] if bb else [])
    acc: Dict[str, Dict[str, List[float]]] = {c: {"err": [], "pred": [], "gt": []}
                                              for c in cols}
    rows: Dict[str, Dict] = {}

    hdr = f"{'stride':>6} {'n':>4} {'GT rot':>8} " + " ".join(f"{c:>12}" for c in cols)
    print("\n" + hdr)
    for st in strides:
        starts = list(range(0, len(src) - st))
        if not starts:
            continue
        step = max(1, len(starts) // args.pairs_per_stride)
        starts = starts[::step][:args.pairs_per_stride]
        per: Dict[str, List[float]] = {c: [] for c in cols}
        prd: Dict[str, List[float]] = {c: [] for c in cols}
        gts: List[float] = []
        for s in starts:
            gi, gj = src.pose(s), src.pose(s + st)
            if gi is None or gj is None:
                continue
            R_gt = gj[0] @ gi[0].transpose(-1, -2)
            eye = torch.eye(3, dtype=R_gt.dtype)
            try:
                imgs = torch.stack([src.image(s), src.image(s + st)])
            except (FileNotFoundError, OSError):
                continue
            got = _sift_pose(src, s, s + st, args.nfeatures, args.ratio,
                             args.magsac_thresh_deg)
            per["classical"].append(
                rotation_error_deg(got[0].to(R_gt), R_gt) if got else float("nan"))
            prd["classical"].append(
                rotation_error_deg(eye, got[0].to(R_gt)) if got else float("nan"))
            if bb is not None:
                with torch.no_grad():
                    d = imgs.to(args.device)
                    for name, R_hat in (("vanilla", bb.forward(d[None]).relative(0, 1)[0]),
                                        ("center_ph", cph(d).relative(0, 1)[0])):
                        R_hat = R_hat.to(R_gt)
                        per[name].append(rotation_error_deg(R_hat, R_gt))
                        prd[name].append(rotation_error_deg(eye, R_hat))
            gts.append(rotation_error_deg(eye, R_gt))
        if not gts:
            continue
        # Only pairs every method answered may enter the summary. Classical
        # geometry returns nothing when SIFT cannot match, and those are exactly
        # the hard pairs -- keeping the backbone's score for them while dropping
        # classical's compares two different pair sets and inverts the result.
        keep = [k for k in range(len(gts))
                if all(not np.isnan(per[c][k]) for c in cols if k < len(per[c]))]
        for c in cols:
            acc[c]["err"] += [per[c][k] for k in keep]
            acc[c]["pred"] += [prd[c][k] for k in keep]
            acc[c]["gt"] += [gts[k] for k in keep]
        n_drop = len(gts) - len(keep)
        rows[f"s{st}"] = {"stride": st, "n": len(gts), "n_common": len(keep),
                          "identity": _median(gts),
                          **{c: _median(per[c]) for c in cols}}
        print(f"{st:6d} {len(gts):4d} {_median(gts):8.2f} "
              + " ".join(f"{_median(per[c]):12.3f}" for c in cols)
              + (f"   ({n_drop} pair(s) classical could not solve)" if n_drop else ""))

    n_common = len(acc["classical"]["gt"])
    print(f"\nsummary over the {n_common} pairs EVERY method answered:")
    print(f"{'':>20}" + " ".join(f"{c:>12}" for c in cols))
    gains = {c: _gain(acc[c]["pred"], acc[c]["gt"]) for c in cols}
    meds = {c: _median([e for e in acc[c]["err"] if not np.isnan(e)]) for c in cols}
    print(f"{'median error':>20}" + " ".join(f"{meds[c]:12.3f}" for c in cols))
    print(f"{'rotation gain':>20}" + " ".join(f"{gains[c]:12.3f}" for c in cols))

    cls = meds["classical"]
    print()
    if n_common < 8:
        print(f"[verify] INCONCLUSIVE: only {n_common} pairs were solved by every "
              f"method, too few to compare. Widen --strides toward smaller "
              f"separations, or raise --pairs-per-stride.")
    elif np.isnan(cls):
        print("[verify] INCONCLUSIVE: classical geometry produced no pose at all. "
              "Too few matches, or the camera model is rejecting the bearings.")
    elif cls > CLASSICAL_SANITY_DEG:
        print(f"[verify] HARNESS SUSPECT: classical geometry only reaches "
              f"{cls:.2f} deg against our own ground truth. It should be well "
              f"below {CLASSICAL_SANITY_DEG} deg, so something in the camera "
              f"model, the pose convention or the metric is wrong. Fix that "
              f"before interpreting ANY backbone number.")
    else:
        print(f"[verify] HARNESS OK: classical geometry recovers our ground truth "
              f"to {cls:.2f} deg (gain {gains['classical']:.3f}) through our own "
              f"camera model. The fisheye model, the pose convention, the "
              f"relative-pose algebra and the metric are all consistent.")
        if bb is not None:
            if meds["vanilla"] > 2.0 * cls:
                print(f"[verify] {args.backbone} vanilla is {meds['vanilla']:.2f} deg "
                      f"on the same pairs against classical's {cls:.2f}, gain "
                      f"{gains['vanilla']:.3f} vs {gains['classical']:.3f}. That "
                      f"deficit is the fisheye damage the adapter has to remove -- a "
                      f"real property of the frozen model, not an artefact of how we "
                      f"score it.")
            else:
                # Do not narrate a deficit that the numbers do not show. On badly
                # separated pairs SIFT fails or mismatches and classical stops being
                # a reference at all.
                print(f"[verify] but classical is NOT clearly better than "
                      f"{args.backbone} here ({cls:.2f} vs {meds['vanilla']:.2f} deg), "
                      f"so it is not acting as a reference on these pairs -- most "
                      f"likely they are too widely separated for SIFT. Re-run with "
                      f"smaller --strides before reading anything into the "
                      f"backbone's gain.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"scene": os.path.basename(args.path.rstrip("/")),
                       "backbone": args.backbone, "ph_fov": args.ph_fov,
                       "rows": rows, "median": meds, "gain": gains}, f, indent=2)
        print(f"\n[verify] wrote {args.out}")


if __name__ == "__main__":
    main()
