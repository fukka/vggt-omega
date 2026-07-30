# Copyright (c) 2026.
"""E1: does the ANGULAR SIZE of an input view drive depth quality?

Question
--------
A center-view sweep showed that, on matched content (a fixed 40-deg cone), a
model's depth-edge displacement falls monotonically as the render FoV grows:

    FoV      40     60     80    100    120
    disp   0.503  0.481  0.428  0.373  0.224   degrees   (VGGT-Omega, tangent)

So a wide view is better *at its centre* — while its rim degrades from gnomonic
stretch.  That combination predicts a specific fix: cover the scene with
OVERLAPPING views and keep each one's centre.  This script tests it.

Design
------
Total coverage is held FIXED (``--total-fov``, default 100 deg).  Only the
tiling changes:

    1x1  -> one patch at 100 deg          (the baseline: today's behaviour)
    2x2  -> four patches at ~62 deg
    3x3  -> nine patches at ~45 deg
    4x4  -> sixteen at ~34 deg

Every patch is rendered at the backend's NATIVE token grid (512 for
VGGT-Omega's patch-16, 518 for VGGT-1B's patch-14), so the model resamples
nothing and every patch costs the same tokens.  Patch centres are laid out on
the reference view's tangent plane and converted to (azimuth, tilt); the per
patch FoV is grown by ``--overlap`` so the union covers the cone with no seams.

What this does and does NOT isolate
-----------------------------------
Because the source fisheye has finite resolution (~11.35 px/deg), a narrow
patch rendered to 512 px is UPSAMPLED (soft, no new detail) while a wide one is
downsampled.  So E1 varies angular size *and* effective sharpness together.
``--match-detail`` adds the control: every patch is pre-blurred to the
effective angular detail of the WIDEST tiling, so sharpness is equalised and
only angular size varies.  Run both; if the effect survives ``--match-detail``,
it is angular size / context, not input detail.

Metrics — deliberately two of them
----------------------------------
PatchFusion reports tilings that improve every global depth metric while a
boundary metric moves the other way, so one number cannot settle this:

  * **boundary**: median displacement from each fused depth edge to the nearest
    fisheye-RGB edge (degrees), plus the probe's align%.
  * **global**  : AbsRel / delta1 against ADT GT via the repo's shared protocol
    (``finetune/eval/metrics.py``), the same one the DAC / UniK3D rows use.

Per-patch scale drift is reported, and ``--harmonize`` applies the least-squares
per-view scale correction; run both so fusion cannot hide a scale problem.

Usage
-----
    python VGGT-360-fisheye/checks/patch_scale_experiment.py \
        --adt-root <ROOT> --backend vggt_omega \
        --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
        --tilings 1 2 3 4 --total-fov 100
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
_REPO = os.path.dirname(_PKG)
for _p in (_HERE, _PKG, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from depth_probe import (BACKENDS, NATIVE_SIZE, adt_frame, load_backend,
                         materialize, predict_depth, View)
from utils.fisheye_cam import aria_intrinsics, fisheye_ray_lut, ray_cos_incidence
from utils.fisheye_fusion import (fuse_views_to_fisheye, harmonize_view_scales,
                                  pairwise_scale_stats, per_view_fisheye_ranges)
from utils.fisheye_views import fisheye_to_persp
from finetune.eval.metrics import align_depth, depth_metrics


def tiling_views(n: int, total_fov: float, overlap: float):
    """N x N patch (azimuth, tilt, fov) covering a ``total_fov`` cone.

    Centres are placed on the reference view's tangent plane, then converted to
    the (azimuth, tilt) parameterisation the renderer uses.  Per-patch FoV is
    the cell size grown by ``overlap`` so neighbours share a margin.
    """
    if n == 1:
        return [(0.0, 0.0, total_fov)]
    t = math.tan(math.radians(total_fov / 2.0))
    step = 2.0 * t / n                       # cell width in tangent units
    half = (step / 2.0) * (1.0 + overlap)    # grown half-extent
    fov = 2.0 * math.degrees(math.atan(half))
    centres = [-t + step * (i + 0.5) for i in range(n)]
    views = []
    for y in centres:
        for x in centres:
            d = np.array([x, y, 1.0]); d /= np.linalg.norm(d)
            tilt = math.degrees(math.acos(float(np.clip(d[2], -1, 1))))
            azim = math.degrees(math.atan2(d[1], d[0])) % 360.0
            views.append((azim, tilt, fov))
    return views


def _secant(fov_deg: float, h: int, w: int) -> np.ndarray:
    """Per-pixel sqrt(1+x^2+y^2) of a tangent grid: planar z -> euclidean range."""
    t = math.tan(math.radians(fov_deg) / 2.0)
    xs = np.linspace(-t, t, w, dtype=np.float32)
    ys = np.linspace(-t, t, h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    return np.sqrt(1.0 + xv * xv + yv * yv).astype(np.float32)


def boundary_displacement(rgb_fisheye, fused, cone, deg_per_px, pct=96.0):
    """Median distance from a fused depth edge to the nearest RGB edge (degrees)."""
    g = cv2.cvtColor(rgb_fisheye, cv2.COLOR_RGB2GRAY).astype(np.float32)
    re_ = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    de_ = np.hypot(cv2.Sobel(fused, cv2.CV_32F, 1, 0, 3),
                   cv2.Sobel(fused, cv2.CV_32F, 0, 1, 3))
    re_[~cone] = 0.0
    de_[~cone] = 0.0
    rm = (re_ > np.percentile(re_[cone], pct)).astype(np.uint8)
    dm = (de_ > np.percentile(de_[cone], pct)) & cone
    if dm.sum() < 50 or rm.sum() < 50:
        return float("nan"), float("nan")
    dist = cv2.distanceTransform(1 - rm, cv2.DIST_L2, 3)
    recall = float((rm.astype(bool) & (cv2.dilate(dm.astype(np.uint8),
                    np.ones((5, 5), np.uint8)) > 0)).sum()) / max(int(rm.sum()), 1)
    return float(np.median(dist[dm])) * deg_per_px, recall * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="E1: angular size vs depth quality")
    ap.add_argument("--adt-root", required=True)
    ap.add_argument("--rgb-subdir", default="videos_synthetic")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--backend", choices=BACKENDS, default="vggt_omega")
    ap.add_argument("--model-path", default="facebook/VGGT-1B")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tilings", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="N for each NxN tiling of the same cone")
    ap.add_argument("--total-fov", type=float, default=100.0,
                    help="total coverage held fixed across tilings (deg)")
    ap.add_argument("--overlap", type=float, default=0.4,
                    help="fractional growth of each patch beyond its cell")
    ap.add_argument("--patch-size", type=int, default=None,
                    help="render size per patch; default = backend native")
    ap.add_argument("--fisheye-size", type=int, default=512,
                    help="fusion/scoring grid")
    ap.add_argument("--match-detail", action="store_true",
                    help="CONTROL: pre-blur every patch to the effective angular "
                         "detail of the widest tiling, so only angular size varies")
    ap.add_argument("--harmonize", action="store_true",
                    help="least-squares per-patch scale correction before fusion")
    ap.add_argument("--depth-max-m", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(_PKG, "outputs", "patch_scale"))
    args = ap.parse_args()

    if args.patch_size is None:
        args.patch_size = NATIVE_SIZE[args.backend]
    os.makedirs(args.out, exist_ok=True)

    rgb, cam_native = adt_frame(args.adt_root, rgb_subdir=args.rgb_subdir,
                                frame=args.frame)
    S = args.fisheye_size
    rgb_s = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA)
    cam = aria_intrinsics(S, S, rotated=True)
    _, cone = fisheye_ray_lut(cam)
    cos_lut = ray_cos_incidence(cam)
    deg_per_px = math.degrees(2 * math.atan(0.5 / cam.fx))   # approx, on-axis

    # GT on the same grid (ADT GT is planar z -> convert to range for scoring)
    from datasets.adt import ADTFisheyeFrames, find_adt_sequences
    seqs = find_adt_sequences(args.adt_root, rgb_subdir=args.rgb_subdir,
                              depth_subdir="depth_npy")
    ds = ADTFisheyeFrames(seqs[:1], rgb_subdir=args.rgb_subdir,
                          depth_max_m=args.depth_max_m,
                          max_frames=args.frame + 1, working_size=S)
    item = ds[min(args.frame, len(ds) - 1)]
    gt_z, gt_valid = item["depth"], item["valid"]
    gt_range = gt_z / np.clip(cos_lut, 1e-3, None)

    backend = load_backend(args.backend, model_path=args.model_path,
                           checkpoint=args.checkpoint)

    # widest tiling sets the detail floor for --match-detail
    widest_fov = tiling_views(min(args.tilings), args.total_fov, args.overlap)[0][2]

    rows = []
    print(f"\ncoverage held at {args.total_fov:.0f} deg; patches rendered at "
          f"{args.patch_size}px ({args.backend} native)\n")
    print(f"{'tiling':>7s} {'patches':>8s} {'patch FoV':>10s} {'scale spread':>13s} "
          f"{'disp(deg)':>10s} {'align%':>7s} {'AbsRel':>8s} {'delta1':>8s}")

    for n in args.tilings:
        views = tiling_views(n, args.total_fov, args.overlap)
        crops, valids, paths = [], [], []
        for i, (az, tilt, fov) in enumerate(views):
            crop, valid = fisheye_to_persp(rgb, cam_native, az, tilt, fov,
                                           height=args.patch_size,
                                           width=args.patch_size, supersample=3)
            crop = np.clip(crop, 0, 255).astype(np.uint8)
            if args.match_detail and fov < widest_fov:
                # equalise effective angular detail with the widest patch
                sigma = 0.6 * (widest_fov / fov - 1.0)
                if sigma > 0.3:
                    crop = cv2.GaussianBlur(crop, (0, 0), sigma)
            v = View(crop=crop, tag=f"n{n}_p{i:02d}", true_fov=fov,
                     valid=valid.astype(np.float32))
            crops.append(crop); valids.append(v.valid)
            paths.append(materialize(v, os.path.join(args.out, f"tiling{n}")))

        preds = predict_depth(backend, paths)          # independent 1-view scenes
        ranges = [p.depth_z * _secant(views[i][2], *p.depth_z.shape)
                  for i, p in enumerate(preds)]

        # cross-patch scale agreement (all patches share one optical centre)
        maps, ok = per_view_fisheye_ranges(ranges, views, cam, view_valids=valids)
        ratio, _ = pairwise_scale_stats(maps, ok)
        fin = np.isfinite(ratio) & ~np.eye(len(views), dtype=bool)
        spread = (float(np.max(np.abs(np.log(ratio[fin])))) if fin.any() else 0.0)
        if args.harmonize and len(views) > 1:
            s = harmonize_view_scales(maps, ok)
            ranges = [r * s[i] for i, r in enumerate(ranges)]

        fused, cover = fuse_views_to_fisheye(ranges, views, cam,
                                             view_valids=valids, interp="linear")
        mask = gt_valid & cone & (cover > 0) & np.isfinite(fused) & (fused > 0)
        disp, align = boundary_displacement(rgb_s, fused, mask, deg_per_px)
        m = depth_metrics(align_depth(fused, gt_range, mask, "scale_shift"),
                          gt_range, mask)
        rows.append((n, len(views), views[0][2], spread, disp, align,
                     m["AbsRel"], m["delta1"]))
        print(f"{n}x{n:>4s} {len(views):8d} {views[0][2]:9.1f}d "
              f"{(np.exp(spread)-1)*100:12.1f}% {disp:10.3f} {align:7.1f} "
              f"{m['AbsRel']:8.4f} {m['delta1']:8.4f}".replace("x   ", "x  "))
        Image.fromarray(np.uint8(np.clip((fused - np.percentile(fused[mask], 2)) /
                        max(np.ptp(fused[mask]), 1e-6) * 255, 0, 255))).save(
                        os.path.join(args.out, f"fused_{n}x{n}.png"))

    with open(os.path.join(args.out, "summary.csv"), "w") as f:
        f.write("tiling,n_patches,patch_fov,scale_spread_pct,disp_deg,align_pct,"
                "AbsRel,delta1\n")
        for r in rows:
            f.write(f"{r[0]}x{r[0]},{r[1]},{r[2]:.2f},{(math.exp(r[3])-1)*100:.2f},"
                    f"{r[4]:.4f},{r[5]:.2f},{r[6]:.4f},{r[7]:.4f}\n")
    print(f"\nwrote {args.out}/summary.csv")
    print("Read: if disp falls with more/smaller patches, angular size drives it. "
          "Re-run with --match-detail; if the effect survives, it is context/scale, "
          "not input sharpness. Watch AbsRel too — PatchFusion reports tilings "
          "that help boundaries while hurting global depth.")


if __name__ == "__main__":
    main()
