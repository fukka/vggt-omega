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

    1x1  ->  1 patch  at 100.0 deg        (the baseline: today's behaviour)
    2x2  ->  4 patches at  79.7 deg
    3x3  ->  9 patches at  58.2 deg
    4x4  -> 16 patches at  45.3 deg

(per-patch FoV includes the --overlap growth; coverage of the cone verified at
100.00% with zero holes for every tiling.  At 4x4 a 512px patch is 11.31 px/deg
against the sensor's ~11.35, so that is the finest tiling that still carries
real detail -- beyond it the patches only interpolate.)

Every patch is rendered at the backend's NATIVE token grid (512 for
VGGT-Omega's patch-16, 518 for VGGT-1B's patch-14), so the model resamples
nothing and every patch costs the same tokens.  Patch centres are laid out on
the reference view's tangent plane and converted to (azimuth, tilt); the per
patch FoV is grown by ``--overlap`` so the union covers the cone with no seams.

All patches of a tiling go through the model in ONE forward pass
---------------------------------------------------------------
This is the whole premise of VGGT-360 and it is not optional.  The patches
share an optical centre, so cross-view attention can reconcile them into a
single scale-consistent 3D model.  Feeding them separately and merging
afterwards fails badly: each patch has a different FoV, so the model infers a
different camera for each one, and its metric depth scales with that inferred
focal length.  Measured on a real ADT frame with independent passes:

    tiling   scale spread   AbsRel   delta1
    1x1            0.00%    0.0389   0.9755
    2x2          179.11%    0.4047   0.1438
    3x3          284.68%    0.8047   0.2377
    4x4          490.00%    0.9275   0.0928

The single 100-deg view is excellent (AbsRel 0.039); tiling with independent
passes destroys it purely through scale disagreement, because a global
scale_shift on the fused map cannot repair a spatially varying scale error.
``--separate`` reproduces that failure on purpose -- the gap between it and the
default measures exactly what cross-view attention contributes.

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

Coarse-plus-fine: the ``A+B`` spec
----------------------------------
``--tilings 1+4`` feeds the global 100-deg view AND the 16 local patches through
the model TOGETHER (17 views, one pass).  VGGT-Omega only accepts pinhole-like
input, so covering a wide fisheye cone requires tiling; the open question is
whether the fine views can inherit layout and scale from a coarse view present
in the same attention pass.  That is PatchFusion's coarse-plus-fine structure,
done inside attention rather than by a learned merge network.

``--center-weight K`` fuses with weight cos(angle from view axis)**K, so each
view contributes mostly at its centre.  It is the direct consequence of the
measurement above: wide views are faithful at the centre and stretched at the
rim, and uniform fusion averages one view's bad rim into its neighbour's good
centre.

Artefacts written per config (so results can be ANALYSED, not just eyeballed)
----------------------------------------------------------------------------
    fused_<spec>.npy     fused euclidean range, NaN outside the valid mask
    aligned_<spec>.npy   the same after scale_shift alignment to GT
    overlay_<spec>.png   RGB | depth | edge-overlay strip (red = image edges,
                         cyan = depth edges) on the fisheye grid
    fused_<spec>.png     colourised fused depth
    gt_range.npy         GT in the same domain, on the same grid
    overlay_GT.png       the SAME overlay for ground truth -- its align% is the
                         ceiling the metric can reach on this frame, since most
                         strong RGB edges are texture and have no depth edge at
                         all.  Read every align% against that number, not 100.

Usage
-----
    python VGGT-360-fisheye/checks/patch_scale_experiment.py \
        --adt-root <ROOT> --backend vggt_omega \
        --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
        --tilings 1 4 1+4 --total-fov 100
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

from depth_probe import (BACKENDS, EDGE_PERCENTILE, NATIVE_SIZE,
                         edge_alignment, load_backend, materialize,
                         predict_depth, View)
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


def views_for_spec(spec: str, total_fov: float, overlap: float):
    """Parse a tiling spec into the view list fed through the model together.

    ``"4"``   -> a 4x4 tiling (16 views).
    ``"1+4"`` -> the 1x1 GLOBAL view **plus** the 4x4 tiling (17 views), all in
                 one forward pass.  This is the coarse-plus-fine arrangement:
                 the global view carries scene layout and scale, the local
                 patches carry detail, and cross-view attention is what lets
                 the fine views inherit the coarse view's geometry.  It is the
                 same structure PatchFusion and BoostingMonocularDepth use,
                 except done inside one attention pass instead of by a learned
                 merge network.
    """
    views = []
    for part in str(spec).split("+"):
        views.extend(tiling_views(int(part), total_fov, overlap))
    return views


def center_weights(fov_deg: float, h: int, w: int, k: float):
    """``cos(angle from the view axis)**k`` over a view's tangent grid.

    Measured on ADT: a tangent view's depth is most faithful at its centre and
    degrades toward the rim, where the gnomonic projection stretches hardest.
    Uniform fusion averages a view's bad rim into its neighbour's good centre;
    this weighting lets every view contribute mostly where it is reliable.
    ``k=0`` disables it (uniform), ``k~4`` is strongly centre-biased.
    """
    if k <= 0:
        return None
    return ((1.0 / _secant(fov_deg, h, w)) ** k).astype(np.float32)


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


def _save_fused(fused: np.ndarray, mask: np.ndarray, path: str) -> None:
    """Percentile-normalised viridis view of the fused map, blanked off-mask."""
    vals = fused[mask]
    lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
    norm = np.clip((fused - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    col = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
    col[~mask] = 24
    Image.fromarray(col).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="E1: angular size vs depth quality")
    ap.add_argument("--adt-root", required=True)
    ap.add_argument("--rgb-subdir", default="videos_synthetic")
    ap.add_argument("--depth-subdir", default="depth_npy")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--backend", choices=BACKENDS, default="vggt_omega")
    ap.add_argument("--model-path", default="facebook/VGGT-1B")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tilings", nargs="+", default=["1", "2", "3", "4"],
                    help="tiling specs. 'N' = an NxN tiling; 'A+B' = tilings A "
                         "and B fed through the model TOGETHER in one pass "
                         "(e.g. '1+4' = global 100deg view + 16 local patches, "
                         "the coarse-plus-fine arrangement)")
    ap.add_argument("--center-weight", type=float, default=0.0,
                    help="fuse with weight cos(angle from view axis)**k, so "
                         "each view contributes most at its centre where it is "
                         "most reliable. 0 = uniform (default), ~4 = strong")
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
    ap.add_argument("--separate", action="store_true",
                    help="ABLATION: run each patch as an independent 1-view "
                         "scene instead of one joint multi-view pass. This is "
                         "the wrong way round -- without cross-view attention "
                         "each patch gets its own scale (up to 490%% spread at "
                         "4x4) and fusion produces rectangular seams. Kept "
                         "because the gap quantifies what joint inference buys.")
    ap.add_argument("--harmonize", action="store_true",
                    help="least-squares per-patch scale correction before "
                         "fusion; a post-hoc patch for scale drift, which joint "
                         "inference should make unnecessary")
    ap.add_argument("--edge-percentile", type=float, default=EDGE_PERCENTILE,
                    help="edge-strength percentile for align%% and the overlay")
    ap.add_argument("--depth-max-m", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(_PKG, "outputs", "patch_scale"))
    args = ap.parse_args()

    if args.patch_size is None:
        args.patch_size = NATIVE_SIZE[args.backend]
    os.makedirs(args.out, exist_ok=True)

    # ONE dataset load, at native resolution, so the RGB we cut patches from and
    # the GT we score against are guaranteed to be the SAME frame.  (Loading the
    # image and the GT through two separate discovery calls risks them resolving
    # to different sequences, which would corrupt every number silently.)
    from datasets.adt import ADTFisheyeFrames, find_adt_sequences
    seqs = find_adt_sequences(args.adt_root, rgb_subdir=args.rgb_subdir,
                              depth_subdir=args.depth_subdir)
    if not seqs:
        raise SystemExit(f"no ADT sequences with {args.rgb_subdir}/ + "
                         f"{args.depth_subdir}/ under {args.adt_root}")
    ds = ADTFisheyeFrames(seqs[:1], rgb_subdir=args.rgb_subdir,
                          depth_subdir=args.depth_subdir,
                          depth_max_m=args.depth_max_m,
                          max_frames=args.frame + 1)          # native 1408
    item = ds[min(args.frame, len(ds) - 1)]
    rgb = item["rgb"]                                          # patches cut from this
    cam_native = aria_intrinsics(*rgb.shape[:2], rotated=True)

    S = args.fisheye_size
    rgb_s = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA)
    gt_z = cv2.resize(item["depth"], (S, S), interpolation=cv2.INTER_NEAREST)
    gt_valid = cv2.resize(item["valid"].astype(np.uint8), (S, S),
                          interpolation=cv2.INTER_NEAREST) > 0
    cam = aria_intrinsics(S, S, rotated=True)
    _, cone = fisheye_ray_lut(cam)
    cos_lut = ray_cos_incidence(cam)
    # KB4 near the axis: u = cx + fx*theta, so one pixel spans 1/fx radians.
    deg_per_px = math.degrees(1.0 / cam.fx)
    # ADT GT is planar z; score in the range domain the fusion produces.
    gt_range = gt_z / np.clip(cos_lut, 1e-3, None)

    # GT reference on the same grid, so every fused_*.npy can be diffed against
    # a ground truth in the SAME domain without re-deriving it.
    _gt_mask = gt_valid & cone
    np.save(os.path.join(args.out, "gt_range.npy"),
            np.where(_gt_mask, gt_range, np.nan).astype(np.float32))
    _gt_strip, _gt_align = edge_alignment(rgb_s, gt_range,
                                          valid=_gt_mask.astype(np.float32))
    Image.fromarray(_gt_strip).save(os.path.join(args.out, "overlay_GT.png"))
    print(f"\nGT reference written (its own align% = {_gt_align * 100:.1f}% — "
          f"the ceiling this metric can reach on this frame)")

    backend = load_backend(args.backend, model_path=args.model_path,
                           checkpoint=args.checkpoint)

    # the widest patch anywhere sets the detail floor for --match-detail
    widest_fov = max(v[2] for spec in args.tilings
                     for v in views_for_spec(spec, args.total_fov, args.overlap))

    rows = []
    mode = ("SEPARATE 1-view passes (ablation: no cross-view attention)"
            if args.separate else "ONE joint multi-view pass per tiling")
    print(f"\ncoverage held at {args.total_fov:.0f} deg; patches rendered at "
          f"{args.patch_size}px ({args.backend} native)")
    print(f"inference: {mode}")
    _biggest = max(len(views_for_spec(s_, args.total_fov, args.overlap))
                   for s_ in args.tilings)
    if not args.separate and _biggest > 9:
        print(f"  note: the largest spec sends {_biggest} views through one "
              f"pass — reduce --tilings if this runs out of VRAM\n")
    else:
        print()
    print(f"{'tiling':>7s} {'patches':>8s} {'patch FoV':>10s} {'scale spread':>13s} "
          f"{'disp(deg)':>10s} {'align%':>7s} {'AbsRel':>8s} {'delta1':>8s}")

    for spec in args.tilings:
        views = views_for_spec(spec, args.total_fov, args.overlap)
        valids, paths = [], []
        for i, (az, tilt, fov) in enumerate(views):
            crop, valid = fisheye_to_persp(rgb, cam_native, az, tilt, fov,
                                           height=args.patch_size,
                                           width=args.patch_size, supersample=3)
            crop = np.clip(crop, 0, 255).astype(np.uint8)
            if args.match_detail and fov < widest_fov:
                # Equalise effective angular detail with the widest patch.  A
                # patch at FoV f rendered to P px carries P/f px/deg, so it is
                # sharper than the widest patch by r = widest_fov / fov.  Blur
                # it by the standard anti-alias sigma for a decimation by r,
                # 0.5*sqrt(r^2 - 1), which matches the MTF rather than merely
                # softening it.
                r = widest_fov / fov
                sigma = 0.5 * math.sqrt(max(r * r - 1.0, 0.0))
                if sigma > 0.3:
                    crop = cv2.GaussianBlur(crop, (0, 0), sigma)
            v = View(crop=crop, tag=f"n{spec}_p{i:02d}", true_fov=fov,
                     valid=valid.astype(np.float32))
            valids.append(v.valid)
            paths.append(materialize(v, os.path.join(args.out, f"tiling{spec}")))

        # ONE forward pass over all patches (VGGT-360's premise): the patches
        # share an optical centre, so cross-view attention resolves the per-view
        # scale ambiguity and the outputs land in a single consistent 3D model.
        # Running them separately is measurably wrong here -- each patch has a
        # different FoV, so the model infers a different camera per patch, its
        # metric depth scales with that inferred focal, and the scales disagree
        # (measured: up to 490% spread at 4x4).  A single global scale_shift on
        # the fused map cannot repair a spatially varying scale error, which is
        # what produced the rectangular tiling seams.
        preds = predict_depth(backend, paths, multiview=not args.separate)
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

        wts = None
        if args.center_weight > 0:
            wts = [center_weights(views[i][2], *ranges[i].shape,
                                  args.center_weight)
                   for i in range(len(views))]
        fused, cover = fuse_views_to_fisheye(ranges, views, cam, weights=wts,
                                             view_valids=valids, interp="linear")
        mask = gt_valid & cone & (cover > 0) & np.isfinite(fused) & (fused > 0)
        if mask.sum() < 100:
            print(f"{spec:>7s} {len(views):8d} {views[0][2]:9.1f}d  "
                  f"SKIPPED (only {int(mask.sum())} valid px)")
            continue
        disp, _ = boundary_displacement(rgb_s, fused, mask, deg_per_px)
        # canonical align% + the RGB|depth|overlay strip, same definition the
        # probe CLI uses, so numbers here are comparable to numbers there
        strip, align = edge_alignment(rgb_s, fused, valid=mask.astype(np.float32),
                                      percentile=args.edge_percentile)
        align *= 100.0
        m = depth_metrics(align_depth(fused, gt_range, mask, "scale_shift"),
                          gt_range, mask)
        rows.append((spec, len(views), views[0][2], spread, disp, align,
                     m["AbsRel"], m["delta1"]))
        print(f"{spec:>7s} {len(views):8d} {views[0][2]:9.1f}d "
              f"{(math.exp(spread) - 1) * 100:12.1f}% {disp:10.3f} "
              f"{align:7.1f} {m['AbsRel']:8.4f} {m['delta1']:8.4f}")

        # ---- per-config artefacts, so the result can be ANALYSED not just eyeballed
        tag = spec.replace("+", "plus")
        _save_fused(fused, mask, os.path.join(args.out, f"fused_{tag}.png"))
        Image.fromarray(strip).save(os.path.join(args.out, f"overlay_{tag}.png"))
        np.save(os.path.join(args.out, f"fused_{tag}.npy"),
                np.where(mask, fused, np.nan).astype(np.float32))
        np.save(os.path.join(args.out, f"aligned_{tag}.npy"),
                np.where(mask, align_depth(fused, gt_range, mask, "scale_shift"),
                         np.nan).astype(np.float32))

    with open(os.path.join(args.out, "summary.csv"), "w") as f:
        f.write("tiling,n_patches,patch_fov,scale_spread_pct,disp_deg,align_pct,"
                "AbsRel,delta1\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.2f},{(math.exp(r[3])-1)*100:.2f},"
                    f"{r[4]:.4f},{r[5]:.2f},{r[6]:.4f},{r[7]:.4f}\n")
    print(f"\nwrote {args.out}/summary.csv")
    print("Read: if disp falls with more/smaller patches, angular size drives it. "
          "Re-run with --match-detail; if the effect survives, it is context/scale, "
          "not input sharpness. Watch AbsRel too — PatchFusion reports tilings "
          "that help boundaries while hurting global depth.")


if __name__ == "__main__":
    main()
