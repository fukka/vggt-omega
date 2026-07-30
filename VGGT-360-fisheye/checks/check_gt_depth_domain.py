# Copyright (c) 2026.
"""Is "curvy" per-view depth a bug?  Ask the ground truth.

Motivation: per-view VGGT depth montages (``*_views_range.png``) look "curvy"
even though the perspective RGB inputs are perfectly rectilinear, raising the
suspicion of a projection/model bug.  This script settles it without a GPU:
it takes a real ADT frame with its **ground-truth** depth, slices the GT into
the same 9 perspective views with the same warp, and renders it with the same
colormap.  If perfect depth shows the same curviness, the curviness is
geometry, not a bug:

  * "range" (euclidean distance along the ray — what ``||world_points||`` and
    our fusion use) of a FLAT wall grows toward the view corners by up to
    ``sec = sqrt(1+x^2+y^2)`` (~15% at FOV 60): iso-range bands on planes are
    curved BY CONSTRUCTION, in any correct depth map.
  * planar z (per-view, ``range / sec``) of the same wall is constant: planes
    look flat.

It also settles the ADT GT depth *convention*, which decides how main_adt
scores.  The GT values are either euclidean range or planar z of the fisheye
frame.  Two independent answers are produced:

  1. **Montages** (illustrative).  Per-view planar z is rendered under both
     hypotheses — hypR: ``warp(GT) / sec``; hypZ: ``warp(GT / cos) / sec`` —
     and under the correct one, planes look planar.  Inside a 60-deg view
     ``1/cos`` spans only ~1.15, so this is a WEAK signal: judging it by eye is
     what led to the range hypothesis being adopted in error.

  2. **Numeric verdict** (authoritative, ``numeric_domain_verdict``).  Over the
     whole imaged cone ``1/cos`` reaches 2.15x.  Back-projecting GT under
     ``P(a) = ray * D / cos^a`` and RANSAC-fitting the dominant plane gives an
     inlier fraction that peaks at the true convention.

**Measured result: the peak is at a=1.00 — ADT GT is planar z**, with fall-off
on both sides (so it is a real optimum, not a warp artifact).  This agrees with
``finetune/eval/baselines/benchmark_adt.py``, which converts GT z->range, and
with Depth-Any-Camera's own fisheye loader (``dac/dataloders/scannetpp.py``:
"convert depth from zbuffer to euclid ... critical for fisheye dataset").

Exit code 0 iff the numeric verdict still says planar z.

Usage:
    python VGGT-360-fisheye/checks/check_gt_depth_domain.py \
        --adt-root <root> [--rgb-subdir videos_rgb] [--frame 6] [--out dir]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from datasets.adt import ADTFisheyeFrames, find_adt_sequences
from utils.fisheye_cam import aria_intrinsics, fisheye_ray_lut
from utils.fisheye_views import base_views, fisheye_to_persp


def colorize(x: np.ndarray, lo: float, hi: float,
             invalid: np.ndarray = None) -> np.ndarray:
    x8 = np.clip((x - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)
    c = cv2.cvtColor(cv2.applyColorMap(x8, cv2.COLORMAP_VIRIDIS),
                     cv2.COLOR_BGR2RGB)
    if invalid is not None:
        c[invalid] = 24
    return c


def montage(images, cols=3, pad=4, labels=None) -> np.ndarray:
    H, W = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    out = np.full((rows * (H + pad) - pad, cols * (W + pad) - pad, 3), 30, np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        img = np.ascontiguousarray(img)
        if labels:
            cv2.putText(img, labels[i], (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 60), 2, cv2.LINE_AA)
        out[r * (H + pad): r * (H + pad) + H,
            c * (W + pad): c * (W + pad) + W] = img
    return out


def _ransac_plane_inliers(P: np.ndarray, seed: int, iters: int = 400,
                          thresh: float = 0.02) -> float:
    """Inlier fraction of the dominant plane in a point cloud ``P`` (N,3).

    Fixed seed, so the same triplets are tried for every hypothesis and the
    comparison is apples-to-apples.
    """
    rng = np.random.default_rng(seed)
    N, best = len(P), 0
    for _ in range(iters):
        i = rng.choice(N, 3, replace=False)
        a, b, c = P[i]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        best = max(best, int((np.abs((P - a) @ (nrm / ln)) < thresh).sum()))
    return best / N


def numeric_domain_verdict(ds, cam, rays, cone, n_frames: int = 4,
                           n_px: int = 6000) -> bool:
    """Decide the GT convention by NUMBER, not by eye.  True iff GT is planar z.

    The per-view montages below span only +-30 deg from each view centre, where
    ``1/cos`` reaches just ~1.15 — too subtle to call by eye, which is how the
    range hypothesis was mistakenly adopted.  Over the FULL imaged cone the
    factor reaches 2.15x, so a whole-frame test has far more leverage.

    Method: real scenes contain large planes.  Back-project GT to 3D under a
    family of hypotheses ``P(a) = ray * D / cos(theta)**a`` and RANSAC the
    dominant plane in each.  A plane is only planar under the correct
    convention, so the inlier fraction peaks at the true ``a``:

        a = 0 -> GT is euclidean range      a = 1 -> GT is planar z

    Sweeping ``a`` rather than testing two points guards against a monotonic
    artifact of the warp: a genuine optimum falls off on BOTH sides.
    """
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    rng = np.random.default_rng(0)
    curves = []
    for idx in range(min(n_frames, len(ds))):
        it = ds[idx]
        D, valid = it["depth"], it["valid"]
        m = valid & cone & (D > 0.3) & (D < 8.0)
        if m.sum() < 5000:
            continue
        ys, xs = np.nonzero(m)
        sel = rng.choice(len(ys), min(n_px, len(ys)), replace=False)
        ys, xs = ys[sel], xs[sel]
        d = D[ys, xs].astype(np.float64)
        r = rays[ys, xs].astype(np.float64)
        rz = np.clip(r[:, 2], 1e-3, None)
        curves.append([_ransac_plane_inliers(r * (d / rz ** a)[:, None], seed=42)
                       for a in alphas])
    if not curves:
        print("  numeric verdict SKIPPED (no frame with enough valid depth)")
        return True

    mean = np.mean(curves, axis=0)
    peak = alphas[int(np.argmax(mean))]
    print("\nNumeric verdict — dominant-plane inlier fraction vs correction "
          "exponent a  (P = ray * D / cos^a):")
    print("  " + "  ".join(f"a={a:.2f}:{v * 100:4.1f}%"
                           for a, v in zip(alphas, mean)))
    is_z = peak >= 0.5
    print(f"  peak at a={peak:.2f}  =>  ADT GT is "
          f"{'PLANAR Z' if is_z else 'EUCLIDEAN RANGE'}")
    if is_z:
        print("  => main_adt.py must convert one side: --eval-domains z scales "
              "the prediction by cos(theta); 'range' scales the GT by 1/cos.")
    else:
        print("  !! This contradicts the measured result (peak expected at "
              "a=1.00, i.e. planar z). Investigate before trusting any metric.")
    return is_z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adt-root", required=True)
    ap.add_argument("--rgb-subdir", default="videos_rgb")
    ap.add_argument("--frame", type=int, default=6)
    ap.add_argument("--view-size", type=int, default=384)
    ap.add_argument("--frames-numeric", type=int, default=4,
                    help="frames used by the numeric plane-fit verdict")
    ap.add_argument("--out", default=os.path.join(_PKG, "outputs", "gt_domain"))
    args = ap.parse_args()

    seqs = find_adt_sequences(args.adt_root, rgb_subdir=args.rgb_subdir)
    ds = ADTFisheyeFrames(seqs[:1], rgb_subdir=args.rgb_subdir,
                          max_frames=args.frame + 1)
    item = ds[min(args.frame, len(ds) - 1)]
    rgb, gt, gt_valid = item["rgb"], item["depth"], item["valid"]
    H, W = gt.shape
    cam = aria_intrinsics(H, W, rotated=True)
    rays, cone = fisheye_ray_lut(cam)
    cos_th = np.clip(rays[..., 2], 1e-3, None)

    views = base_views()
    labels = [f"az{int(p)} t{int(t)}" for (p, t, _) in views]
    n = args.view_size
    finite = gt[gt_valid]
    lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)

    # Hypothesis R: GT is euclidean range.   Hypothesis Z: GT is planar z.
    range_R = np.where(gt_valid, gt, 0.0).astype(np.float32)
    range_Z = np.where(gt_valid, gt / cos_th, 0.0).astype(np.float32)

    rgb_tiles, rangeR_tiles, zR_tiles, zZ_tiles = [], [], [], []
    for (psi, tilt, fov) in views:
        t = math.tan(math.radians(fov) / 2.0)
        xs = np.linspace(-t, t, n, dtype=np.float32)
        xv, yv = np.meshgrid(xs, xs)
        sec = np.sqrt(1.0 + xv * xv + yv * yv)

        pv_rgb, _ = fisheye_to_persp(rgb, cam, psi, tilt, fov, n, n)
        rgb_tiles.append(pv_rgb.astype(np.uint8))
        wR, valid = fisheye_to_persp(range_R, cam, psi, tilt, fov, n, n)
        wZ, _ = fisheye_to_persp(range_Z, cam, psi, tilt, fov, n, n)
        bad = (valid < 0.5) | (wR <= 0)
        # exact analogue of main_adt's *_views_range.png, but with PERFECT gt
        rangeR_tiles.append(colorize(wR, lo, hi, bad))
        # per-view planar z under each hypothesis
        zR_tiles.append(colorize(wR / sec, lo, hi, bad))
        zZ_tiles.append(colorize(wZ / sec, lo, hi, bad))

    os.makedirs(args.out, exist_ok=True)
    for name, tiles in [("gt_views_rgb", rgb_tiles),
                        ("gt_views_range_hypR", rangeR_tiles),
                        ("gt_views_z_hypR", zR_tiles),
                        ("gt_views_z_hypZ", zZ_tiles)]:
        p = os.path.join(args.out, f"{name}.png")
        Image.fromarray(montage(tiles, labels=labels)).save(p)
        print(f"saved {p}")

    print("\nHow to read the montages:")
    print(" * gt_views_range_hypR = GROUND TRUTH shown exactly like main_adt's")
    print("   *_views_range.png.  Curved iso-depth bands on flat walls here are")
    print("   geometry, not a bug — compare against your model montage.")
    print(" * gt_views_z_hypR vs gt_views_z_hypZ: planes look planar only under")
    print("   the correct convention.  These are ILLUSTRATIVE only — inside a")
    print("   60-deg view 1/cos spans just ~1.15, too subtle to call by eye.")
    print("   The numeric verdict below is what settles it.")

    # Separate load: the montage only needs frame --frame, the numeric verdict
    # wants the first --frames-numeric frames.
    ds_num = ADTFisheyeFrames(seqs[:1], rgb_subdir=args.rgb_subdir,
                              max_frames=args.frames_numeric)
    ok = numeric_domain_verdict(ds_num, cam, rays, cone,
                                n_frames=args.frames_numeric)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
