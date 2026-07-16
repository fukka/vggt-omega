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
from utils.fisheye_fusion import depth_set_to_fisheye_attention
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
    dtype = torch.bfloat16
    if device == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        dtype = torch.float16

    print(f"loading {args.model_path} ...")
    model = VGGT.from_pretrained(args.model_path).to(device).eval()

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

        with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            predictions, attention_maps = model(
                images=images, persp_masks=persp_masks, rgb_masks=rgb_masks,
                save_attn=use_attn_fusion)

        # ── module 3: fuse radial distances back onto the fisheye grid ──────
        world_points = predictions["world_points"][0].float().cpu().numpy()  # [S,h,w,3]
        radial = np.linalg.norm(world_points, axis=-1).astype(np.float32)
        fused_range, coverage = depth_set_to_fisheye_attention(
            depths=[radial[i] for i in range(radial.shape[0])],
            view_params=view_params, cam=cam,
            attention_maps=attention_maps if use_attn_fusion else None,
            view_valids=valid_masks, interp="linear")
        del predictions, attention_maps
        if device == "cuda":
            torch.cuda.empty_cache()

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
    p.add_argument("--no-adaptive", action="store_true",
                   help="disable uncertainty-guided neighbor views (module 1 ablation)")
    p.add_argument("--no-sa-mask", action="store_true",
                   help="disable structure-saliency attention bias (module 2 ablation)")
    p.add_argument("--pred-domain", choices=["z", "range"], default="z",
                   help="score planar z-depth (range*cos(theta)) or raw range")
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
