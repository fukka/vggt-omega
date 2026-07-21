# Copyright (c) 2026.
"""VGGT-360 on ADT Aria fisheye: zero-shot depth via multi-view reconstruction.

The fisheye port of upstream ``main.py`` (kept as ``main_erp_upstream.py``).
Per frame, the pipeline mirrors the paper's three training-free modules:

  1. **Adaptive projection** — split the raw KB4 fisheye frame into a
     center + 8-direction ring of perspective views (default FOV 60 deg,
     ring tilt 32 deg, so the layout tiles the ~62.3-deg imaged cone), plus
     uncertainty-guided neighbor views (``utils/fisheye_views.py``).
  2. **Structure-saliency attention** — per-view Sobel confidence + analytic
     validity masks are injected into VGGT's frame attention as a log-bias
     (``utils/att_utils.py`` -> modified ``vggt_visfeat``); VGGT reconstructs
     one scale-consistent 3D model from all views at once.
  3. **Correlation-weighted fusion** — per-view radial distances
     ``||world_points||`` are fused back onto the fisheye pixel grid, weighted
     by attention-derived confidences and analytic validity
     (``utils/fisheye_fusion.py``), then scored against ADT GT depth.

Depth-domain note: the fused quantity is euclidean *range* along each ray.
``--pred-domain z`` (default) converts to planar z-depth via ``cos(theta)``
before scoring — at the Aria FOV edge this is a >2x factor, so pick the
domain that matches your GT convention consciously (``--pred-domain range``
to score raw range).  Alignment is affine-invariant (scale+shift) by default,
matching this repo's ADT baseline protocol.

Example (GPU box)
-----------------
    python VGGT-360-fisheye/main_adt.py \
        --adt-root /group-volume/Fengjia/data/projectaria_tools_adt_data_clean \
        --max-frames 100 --qual-dir outputs/vggt360_fisheye_qual

Ablations: ``--fuse mean`` (uniform instead of attention weights),
``--no-adaptive`` (base 9 views only), ``--no-sa-mask`` (vanilla attention).
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:              # make `utils`, `vggt_visfeat`, `datasets`
    sys.path.insert(0, _HERE)          # importable when run from anywhere

import numpy as np
import torch
from PIL import Image

from datasets.adt import ADTFisheyeFrames, find_adt_sequences
from utils.att_utils import SA_confidence
from utils.fisheye_cam import aria_intrinsics, fisheye_ray_lut, ray_cos_incidence
from utils.fisheye_fusion import (build_selfview_confidence,
                                  fuse_views_to_fisheye,
                                  harmonize_view_scales,
                                  pairwise_scale_stats,
                                  per_view_fisheye_ranges)
from utils.fisheye_views import fisheye_to_persp, view_generation_fisheye
from utils.metrics_adt import (aggregate_metrics, align_depth, depth_metrics,
                               print_summary)
from vggt_visfeat.models.vggt import VGGT
from vggt_visfeat.utils.load_fn2 import load_and_preprocess_images


def _colorize(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """[H,W] float -> [H,W,3] uint8 viridis (grayscale fallback)."""
    normed = np.clip((x - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    try:
        import matplotlib.cm as cm
        return (cm.viridis(normed)[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        g = (normed * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)


_SECANT_CACHE = {}


def _view_secant(fov_deg: float, H: int, W: int) -> np.ndarray:
    """Per-pixel ``sqrt(1 + x^2 + y^2)`` of a view's tangent grid.

    Converts the depth head's planar z (distance along the VIEW axis) to
    euclidean range along each pixel ray: ``r = z * secant``.  The point
    head needs no such conversion (``||world_points||`` is already range),
    but is empirically noisier — hence the ``--head`` ablation.
    """
    key = (round(fov_deg, 3), H, W)
    if key not in _SECANT_CACHE:
        t = np.tan(np.radians(fov_deg) / 2.0)
        xs = np.linspace(-t, t, W, dtype=np.float32)
        ys = np.linspace(-t, t, H, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        _SECANT_CACHE[key] = np.sqrt(1.0 + xv * xv + yv * yv).astype(np.float32)
    return _SECANT_CACHE[key]


def _montage(images, cols=4, pad=4, labels=None) -> np.ndarray:
    """Tile equally-sized HxWx3 uint8 images into a labeled grid."""
    import cv2
    H, W = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    out = np.full((rows * (H + pad) - pad, cols * (W + pad) - pad, 3), 30, np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        img = np.ascontiguousarray(img)
        if labels:
            cv2.putText(img, labels[i], (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 60), 2, cv2.LINE_AA)
        out[r * (H + pad): r * (H + pad) + H,
            c * (W + pad): c * (W + pad) + W] = img
    return out


def _dump_debug(debug_dir: str, idx: int, view_params, persp_imgs, radial,
                maps, ratio, n_ov, weights, pose_enc, fused,
                coverage) -> None:
    """Per-frame diagnostics for seam/bumpiness debugging (see README).

    Dumps, per frame:
      * per-view inputs and per-view predicted radial depth (view space),
      * each view's range re-projected ALONE onto the fisheye grid — seams in
        the fused map are disagreements visible directly between these tiles,
      * attention fusion weights (if used),
      * the pairwise overlap median-ratio matrix (text) + pose translations,
      * a gradient map of the fused depth (seams/bumps highlighter).
    """
    import cv2
    os.makedirs(debug_dir, exist_ok=True)
    labels = [f"az{int(p)} t{int(t)}" for (p, t, _) in view_params]
    S = len(persp_imgs)

    Image.fromarray(_montage([np.clip(p, 0, 255).astype(np.uint8)
                              for p in persp_imgs], labels=labels)
                    ).save(os.path.join(debug_dir, f"{idx:04d}_views_rgb.png"))

    finite = radial[np.isfinite(radial) & (radial > 0)]
    lo, hi = (np.percentile(finite, 2), np.percentile(finite, 98)) if finite.size else (0, 1)
    Image.fromarray(_montage([_colorize(radial[i], lo, hi) for i in range(S)],
                             labels=labels)
                    ).save(os.path.join(debug_dir, f"{idx:04d}_views_range.png"))

    # Planar z per view (= range / secant).  Range of a FLAT wall is radially
    # "bowl-shaped" by construction (up to +15% at FOV-60 corners) — judge
    # smoothness on this z montage instead to avoid mistaking that geometry
    # for an artifact.
    z_views = [radial[i] / _view_secant(view_params[i][2], *radial[i].shape)
               for i in range(S)]
    Image.fromarray(_montage([_colorize(z, lo, hi) for z in z_views],
                             labels=labels)
                    ).save(os.path.join(debug_dir, f"{idx:04d}_views_z.png"))

    # RGB-edge vs depth-edge overlay — the decisive "is that curvy line a real
    # depth edge or just a smooth iso-range band?" test.  Per view, the input
    # RGB's Sobel edges are drawn RED and the depth's own Sobel edges CYAN, on
    # top of the (dimmed) depth colormap, all on the depth pixel grid.
    #   * a "curvy edge" that has NO red line under it  -> it is a smooth
    #     range/z contour on a flat surface (geometry, not a bug),
    #   * red and cyan lines COINCIDE and both curve                 -> the
    #     input itself is curved (intrinsics/OOD) — depth faithfully follows it,
    #   * red straight but cyan (depth) curved / displaced            -> VGGT is
    #     bending structure relative to its input (a real model/registration
    #     problem to chase upstream).
    # The depth edges are taken on PLANAR Z (range/secant), not range, so the
    # radial "bowl" of a flat surface is removed and cyan marks true depth
    # discontinuities — the fair comparison against the RGB edges.
    ov_tiles = []
    for i in range(S):
        d = radial[i]
        z = d / _view_secant(view_params[i][2], *d.shape)
        Hd, Wd = d.shape
        rgb_i = cv2.resize(np.clip(persp_imgs[i], 0, 255).astype(np.uint8),
                           (Wd, Hd), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(rgb_i, cv2.COLOR_RGB2GRAY).astype(np.float32)
        rgb_e = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))
        dep_e = np.hypot(cv2.Sobel(z, cv2.CV_32F, 1, 0, 3),
                         cv2.Sobel(z, cv2.CV_32F, 0, 1, 3))
        tile = (_colorize(z, lo, hi).astype(np.float32) * 0.45).astype(np.uint8)
        tile[rgb_e > np.percentile(rgb_e, 96)] = (255, 40, 40)   # RGB edges red
        tile[dep_e > np.percentile(dep_e, 96)] = (0, 230, 230)   # z-depth edges cyan
        ov_tiles.append(tile)
    Image.fromarray(_montage(ov_tiles, labels=labels)
                    ).save(os.path.join(debug_dir, f"{idx:04d}_edge_overlay.png"))

    # Export each perspective crop as a standalone PNG so the EXACT input can be
    # fed to the official VGGT (`pip install vggt`) for an apples-to-apples
    # check: if official VGGT curves the depth of this same crop, the curviness
    # is VGGT's behavior on fisheye-derived tangent views, not this port.
    crop_dir = os.path.join(debug_dir, f"{idx:04d}_crops")
    os.makedirs(crop_dir, exist_ok=True)
    for i in range(S):
        Image.fromarray(np.clip(persp_imgs[i], 0, 255).astype(np.uint8)
                        ).save(os.path.join(crop_dir, f"view{i:02d}_{labels[i].replace(' ', '_')}.png"))

    tiles = []
    for i in range(S):
        m = maps[i].copy()
        bad = ~np.isfinite(m)
        m[bad] = 0.0
        tile = _colorize(m, lo, hi)
        tile[bad] = 24
        tiles.append(tile)
    Image.fromarray(_montage(tiles, labels=labels)
                    ).save(os.path.join(debug_dir, f"{idx:04d}_perview_fisheye.png"))

    if weights is not None:
        Image.fromarray(_montage([_colorize(w, 0.0, 1.0) for w in weights],
                                 labels=labels)
                        ).save(os.path.join(debug_dir, f"{idx:04d}_attn_weights.png"))

    gx = cv2.Sobel(fused, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(fused, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad[coverage == 0] = 0
    Image.fromarray(_colorize(grad, 0.0, float(np.percentile(grad, 99) + 1e-6))
                    ).save(os.path.join(debug_dir, f"{idx:04d}_fused_seams.png"))

    with open(os.path.join(debug_dir, f"{idx:04d}_consistency.txt"), "w") as f:
        f.write("pairwise median range ratio maps_i/maps_j (rows=i, cols=j):\n")
        for i in range(S):
            f.write(" ".join("  .  " if not np.isfinite(ratio[i, j])
                             else f"{ratio[i, j]:5.3f}" for j in range(S)) + "\n")
        f.write("\noverlap pixel counts:\n")
        for i in range(S):
            f.write(" ".join(f"{n_ov[i, j]:7d}" for j in range(S)) + "\n")
        if pose_enc is not None:
            f.write("\nVGGT per-view translation norms (should be ~0 — shared "
                    "optical center; large values = scale/seam trouble):\n")
            for i in range(S):
                t = pose_enc[i, :3]
                f.write(f"  view {i:2d} {labels[i]:<12} |t| = {np.linalg.norm(t):.4f}"
                        f"   t = [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}]\n")


def _save_qual(path: str, rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray,
               mask: np.ndarray) -> None:
    """RGB | aligned pred | GT | abs-error strip on the fisheye grid."""
    valid_g = gt[mask]
    vmin = float(np.percentile(valid_g, 2)) if valid_g.size else 0.0
    vmax = float(np.percentile(valid_g, 98)) if valid_g.size else 10.0
    panels = [rgb]
    for img, lo, hi in ((pred, vmin, vmax), (gt, vmin, vmax),
                        (np.abs(pred - gt), 0.0, max(0.3 * (vmax - vmin), 0.1))):
        img = np.where(mask, img, 0.0)
        col = _colorize(img, lo, hi)
        col[~mask] = 24
        panels.append(col)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    device = args.device
    use_attn_fusion = args.fuse == "attn"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    if dtype is torch.bfloat16 and device == "cuda" \
            and torch.cuda.get_device_capability()[0] < 8:
        dtype = torch.float16
    use_autocast = device == "cuda" and dtype is not torch.float32
    # fp16 quality is a KNOWN VGGT failure mode (official repo recommends
    # bf16); make any silent bf16->fp16 fallback loudly visible.
    print(f"autocast: {'OFF (fp32)' if not use_autocast else str(dtype)}"
          + ("   <-- fp16 fallback: pre-Ampere GPU; expect noisy depth, "
             "try --dtype fp32" if use_autocast and dtype is torch.float16
             and args.dtype == "bf16" else ""))

    print(f"loading {args.model_path} ...")
    model = VGGT.from_pretrained(args.model_path).to(device).eval()

    # Weight-loading sanity: PyTorchModelHubMixin can load non-strictly, so a
    # key mismatch (e.g. after code edits to vggt_visfeat) would silently
    # leave layers at random init -> garbage/bumpy predictions.  Compare the
    # checkpoint's keys against the model's and shout if anything is off.
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        ckpt = load_file(hf_hub_download(args.model_path, "model.safetensors"))
        model_keys = set(model.state_dict().keys())
        missing = sorted(model_keys - set(ckpt.keys()))
        unexpected = sorted(set(ckpt.keys()) - model_keys)
        if missing or unexpected:
            print(f"  WEIGHT CHECK FAILED: {len(missing)} model keys not in "
                  f"checkpoint (random init!), {len(unexpected)} checkpoint "
                  f"keys unused. First few missing: {missing[:5]}")
        else:
            print(f"  weight check OK: all {len(model_keys)} keys matched")
    except Exception as e:  # local path / no safetensors — non-fatal
        print(f"  weight check skipped ({type(e).__name__}: {e})")

    seq_dirs = find_adt_sequences(args.adt_root, args.rgb_subdir, args.depth_subdir)
    if not seq_dirs:
        raise SystemExit(f"no ADT sequences with {args.rgb_subdir}/ + "
                         f"{args.depth_subdir}/ under {args.adt_root}")
    dataset = ADTFisheyeFrames(
        seq_dirs, rgb_subdir=args.rgb_subdir, depth_subdir=args.depth_subdir,
        depth_max_m=args.depth_max_m, max_frames=args.max_frames,
        frame_stride=args.frame_stride, working_size=args.fisheye_size)

    frame_metrics = {m: [] for m in args.align_modes}
    cam = None
    cone = cos_lut = None

    for idx in range(len(dataset)):
        item = dataset[idx]
        rgb, gt, gt_valid = item["rgb"], item["depth"], item["valid"]
        H, W = rgb.shape[:2]

        # Intrinsics + cached per-pixel geometry (identical for every frame).
        if cam is None or (cam.H, cam.W) != (H, W):
            cam = aria_intrinsics(H, W, rotated=True)
            _, cone = fisheye_ray_lut(cam)          # imaged-cone mask
            cos_lut = ray_cos_incidence(cam)        # range -> planar z factor

        # ── module 1: view generation ────────────────────────────────────────
        view_params = view_generation_fisheye(
            rgb, cam, fov_deg=args.fov, ring_tilt_deg=args.ring_tilt,
            n_ring=args.n_ring, adaptive=not args.no_adaptive,
            max_total=args.max_views)

        persp_imgs, sa_masks, valid_masks = [], [], []
        for (psi, tilt, fov) in view_params:
            persp, valid = fisheye_to_persp(rgb, cam, psi, tilt, fov,
                                            height=args.persp_size,
                                            width=args.persp_size)
            sa, vm = SA_confidence(persp, valid_mask=valid > 0.5)
            persp_imgs.append(persp)
            sa_masks.append(sa)
            valid_masks.append(vm)

        # ── module 2: one multi-view VGGT pass with mask-biased attention ────
        pil_views = [Image.fromarray(np.clip(p, 0, 255).astype(np.uint8))
                     for p in persp_imgs]
        images = load_and_preprocess_images(pil_views).to(device)
        persp_masks = None if args.no_sa_mask else torch.from_numpy(np.array(sa_masks))
        rgb_masks = None if args.no_sa_mask else torch.from_numpy(np.array(valid_masks))

        with torch.autocast(device_type=device, dtype=dtype, enabled=use_autocast):
            predictions, attention_maps = model(
                images=images, persp_masks=persp_masks, rgb_masks=rgb_masks,
                save_attn=use_attn_fusion)

        # ── module 3: fuse radial distances back onto the fisheye grid ──────
        if args.head == "depth":
            # depth head: per-view planar z along the view axis -> range
            depth_z = predictions["depth"][0, ..., 0].float().cpu().numpy()  # [S,h,w]
            radial = np.stack([
                depth_z[i] * _view_secant(view_params[i][2], *depth_z[i].shape)
                for i in range(depth_z.shape[0])]).astype(np.float32)
        else:
            # point head (upstream's choice): ||world_points|| is range directly
            world_points = predictions["world_points"][0].float().cpu().numpy()
            radial = np.linalg.norm(world_points, axis=-1).astype(np.float32)
        pose_enc = (predictions["pose_enc"][0].float().cpu().numpy()
                    if "pose_enc" in predictions else None)

        weights = None
        if use_attn_fusion:
            w = build_selfview_confidence(attention_maps)[:, 0, :, :].cpu().numpy()
            weights = [w[i] for i in range(w.shape[0])]
        del predictions, attention_maps
        if device == "cuda":
            torch.cuda.empty_cache()

        # Cross-view consistency: all views share one optical center, so their
        # ranges must agree on overlapping rays.  Measured at 512 (scale-free).
        maps = ratio = n_ov = None
        if args.debug_dir or args.harmonize_scales:
            dbg_cam = aria_intrinsics(512, 512, rotated=True)
            maps, ok = per_view_fisheye_ranges(
                [radial[i] for i in range(radial.shape[0])], view_params,
                dbg_cam, view_valids=valid_masks)
            ratio, n_ov = pairwise_scale_stats(maps, ok)
            fin = np.isfinite(ratio) & ~np.eye(ratio.shape[0], dtype=bool)
            if fin.any():
                spread = float(np.max(np.abs(np.log(ratio[fin]))))
                print(f"  [{idx}] cross-view scale spread: max |log ratio| = "
                      f"{spread:.3f} ({(np.exp(spread) - 1) * 100:.1f}%)")
            if args.harmonize_scales:
                s = harmonize_view_scales(maps, ok)
                radial = radial * s[:, None, None]
                print(f"  [{idx}] harmonized view scales: "
                      + " ".join(f"{v:.3f}" for v in s))

        fused_range, coverage = fuse_views_to_fisheye(
            [radial[i] for i in range(radial.shape[0])], view_params, cam,
            weights=weights, view_valids=valid_masks, interp="linear",
            erode_valid_px=args.erode_valid_px)

        if args.debug_dir and idx < args.n_qual:
            _dump_debug(args.debug_dir, idx, view_params, persp_imgs, radial,
                        maps, ratio, n_ov, weights, pose_enc, fused_range,
                        coverage)

        pred = fused_range * cos_lut if args.pred_domain == "z" else fused_range
        mask = gt_valid & cone & (coverage > 0) & np.isfinite(pred) & (pred > 0)
        if mask.sum() < 10:
            print(f"  [{idx}] skipped (no valid overlap)")
            continue

        for mode in args.align_modes:
            aligned = align_depth(pred, gt, mask, mode=mode)
            frame_metrics[mode].append(depth_metrics(aligned, gt, mask,
                                                     max_depth=args.depth_max_m))
        if args.qual_dir and idx < args.n_qual:
            aligned = align_depth(pred, gt, mask, mode="scale_shift")
            _save_qual(os.path.join(args.qual_dir, f"{idx:04d}.png"),
                       rgb, aligned, gt, mask)
        if (idx + 1) % 10 == 0 or idx == 0:
            m = frame_metrics[args.align_modes[0]][-1]
            print(f"  [{idx + 1}/{len(dataset)}] views={len(view_params)}  "
                  f"abs_rel={m['abs_rel']:.4f}  d1={m['d1']:.4f}")

    results = {}
    for mode in args.align_modes:
        results[mode] = aggregate_metrics(frame_metrics[mode])
        print_summary(results[mode], label=f"VGGT-360-fisheye ({args.fuse})",
                      align=mode)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="VGGT-360-fisheye on ADT")
    p.add_argument("--adt-root", required=True,
                   help="dir containing ADT sequence folders")
    p.add_argument("--rgb-subdir", default="videos_synthetic",
                   help="videos_synthetic (GT-aligned) or videos_rgb (real sensor)")
    p.add_argument("--depth-subdir", default="depth_npy")
    p.add_argument("--model-path", default="facebook/VGGT-1B")
    p.add_argument("--fisheye-size", type=int, default=None,
                   help="downsample fisheye+GT to this square size (default: native 1408)")
    p.add_argument("--persp-size", type=int, default=512,
                   help="perspective view size fed to VGGT (518 after its preprocessing)")
    p.add_argument("--fov", type=float, default=60.0, help="per-view FOV (deg)")
    p.add_argument("--ring-tilt", type=float, default=32.0,
                   help="ring-view tilt from the optical axis (deg); "
                        "design rule tilt + fov/2 ~= 62.3 (the Aria cone)")
    p.add_argument("--n-ring", type=int, default=8, help="ring view count")
    p.add_argument("--max-views", type=int, default=13,
                   help="cap after adaptive augmentation (9 base + 4)")
    p.add_argument("--fuse", choices=["attn", "mean"], default="attn",
                   help="attention-weighted fusion (paper) or uniform (ablation)")
    p.add_argument("--head", choices=["point", "depth"], default="depth",
                   help="range source: depth head z * secant (default — "
                        "measurably less bumpy on ADT) or point head "
                        "||world_points|| (upstream's choice; noisier)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="aggregator autocast dtype; fp32 disables autocast "
                        "(rules out mixed-precision noise)")
    p.add_argument("--harmonize-scales", action="store_true",
                   help="least-squares per-view scale correction from overlap "
                        "regions before fusion (fixes seams caused by VGGT "
                        "per-view scale drift)")
    p.add_argument("--erode-valid-px", type=int, default=3,
                   help="valid-mask erosion at fusion; raise to ~7 (half a "
                        "ViT patch) if rim/corner garbage bleeds into seams")
    p.add_argument("--debug-dir", default=None,
                   help="dump per-view diagnostics for the first --n-qual "
                        "frames: inputs, per-view depths, per-view fisheye "
                        "reprojections, attention weights, seam map, pairwise "
                        "consistency matrix, pose translations")
    p.add_argument("--no-adaptive", action="store_true",
                   help="disable uncertainty-guided neighbor views (module 1 ablation)")
    p.add_argument("--no-sa-mask", action="store_true",
                   help="disable structure-saliency attention bias (module 2 ablation)")
    p.add_argument("--pred-domain", choices=["z", "range"], default="range",
                   help="GT depth convention to score in. ADT GT is euclidean "
                        "range (verified empirically by checks/"
                        "check_gt_depth_domain.py: planes look planar only "
                        "under the range hypothesis), so the default compares "
                        "fused range directly; 'z' multiplies by cos(theta) "
                        "first (up to 2x at the FOV edge — only for z-domain GT)")
    p.add_argument("--align-modes", nargs="+",
                   default=["scale_shift", "median"],
                   choices=["scale_shift", "median", "none"])
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--max-frames", type=int, default=100,
                   help="frames per sequence (None-like: pass a huge number)")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--qual-dir", default=None,
                   help="save RGB|pred|GT|err strips for the first --n-qual frames")
    p.add_argument("--n-qual", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.device != "cuda":
        print("WARNING: running VGGT-1B on CPU will be extremely slow.")
    run(args)


if __name__ == "__main__":
    main()
