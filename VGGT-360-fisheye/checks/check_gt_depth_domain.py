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

Bonus: the same montages settle the ADT GT depth *convention* empirically.
The GT values are either euclidean range or planar z of the fisheye frame —
we render the per-view z montage under BOTH hypotheses:

  hypothesis R (GT = range):  z_view = warp(GT) / sec_view
  hypothesis Z (GT = z):      z_view = warp(GT / cos(theta_fisheye)) / sec_view

Under the correct hypothesis planes look planar; under the wrong one they
bow by up to 2x at the FOV edge.  Whichever wins tells us what
``--pred-domain`` must default to for the eval to be apples-to-apples.

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adt-root", required=True)
    ap.add_argument("--rgb-subdir", default="videos_rgb")
    ap.add_argument("--frame", type=int, default=6)
    ap.add_argument("--view-size", type=int, default=384)
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

    print("\nHow to read:")
    print(" * gt_views_range_hypR = GROUND TRUTH shown exactly like main_adt's")
    print("   *_views_range.png.  Curved iso-depth bands on flat walls here are")
    print("   geometry, not a bug — compare against your model montage.")
    print(" * gt_views_z_hypR vs gt_views_z_hypZ: planes look planar only under")
    print("   the CORRECT GT-depth convention -> sets --pred-domain:")
    print("     hypR planar  => GT is euclidean range  => --pred-domain range")
    print("     hypZ planar  => GT is planar z         => --pred-domain z")


if __name__ == "__main__":
    main()
