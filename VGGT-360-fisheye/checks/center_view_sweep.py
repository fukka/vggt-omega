# Copyright (c) 2026.
"""Center-view cropping/processing sweep — how does the CENTER view's FoV/ROI and
its pre-processing change VGGT's depth alignment?

Motivation
----------
``main_adt.py --n-ring 1`` reduces the layout to the single center view (tilt 0),
yet even that center view shows the "depth is a distorted version of the input"
misalignment.  Since the center view is a plain single image, the whole VGGT-360
machinery (multi-view attention, fusion, masks) is out of the loop — what remains
is: *how the center image is cropped/processed from the raw fisheye* × *original
VGGT's behaviour on that image*.  This script isolates exactly those two factors.

Controlled experiment (see the Chinese write-up in the chat)
------------------------------------------------------------
Everything is held fixed — the SAME raw ADT fisheye frame, the SAME optical
center (tilt 0, azimuth 0), the SAME original VGGT-1B, the SAME output
resolution — while sweeping two knobs:

  1. **coverage** ``--fovs`` : the center view's field of view, small -> large.
     A larger FoV covers more of the imaged cone (edge incidence = fov/2), so it
     ingests more of the distortion-heavy periphery.

  2. **processing** ``--modes`` : how the pixels are produced at that coverage:
       * ``tangent``   — gnomonic ``fisheye_to_persp`` crop (straight lines stay
                         straight; the periphery is radially STRETCHED, and the
                         effective resampling gets softer as FoV grows).
       * ``raw_roi``   — a centered square ROI of the RAW fisheye covering the
                         same incidence (fov/2), resized to the model size.  NO
                         undistortion: straight lines stay CURVED (barrel), but
                         there is no gnomonic stretch — the honest "does the
                         undistortion cause it, or is it VGGT on distorted input?"
                         control.
       * ``rectifier`` — the repo's validated ``FisheyeRectifier`` wide (~85 deg)
                         pinhole of the whole frame (a fixed reference; ignores
                         ``--fovs``).

  Optional third axis ``--enhance`` {none,clahe,sharpen,blur} tests the
  "clearness" hypothesis (does sharpening/blurring the SAME crop move the depth
  alignment — i.e. is VGGT depth-edge localisation limited by image clarity?).

Measured per configuration
--------------------------
  * ``align%`` — fraction of RGB edges that have a depth edge within ~2 px
    (restricted to the valid cone).  High = depth structure follows the image.
  * VGGT's inferred camera FoV (``pose_enc[7:9]``) vs the crop's TRUE FoV — the
    suspected mechanism: VGGT couples depth to its own FoV estimate, so a wrong
    estimate bends the geometry.  Watch whether the gap widens with coverage.
  * an ``RGB | depth | edge-overlay`` strip per config, a summary CSV, an
    ``align vs FoV`` plot, and an overlay grid (rows = mode/enhance, cols = FoV).

Runs original VGGT-1B (``vggt_visfeat`` == ``facebook/VGGT-1B``), exactly the
base model of this port.  Use ``--render-only`` to dump the crops WITHOUT a GPU
(geometry preview / this is what can be checked on the Mac).

Examples
--------
    # FoV sweep, tangent vs raw-ROI, on a sharp ADT frame
    python VGGT-360-fisheye/checks/center_view_sweep.py \
        --adt-root <ROOT> --rgb-subdir videos_synthetic --frame 6 \
        --fovs 40 60 80 100 120 --modes tangent raw_roi

    # add the clearness axis
    python VGGT-360-fisheye/checks/center_view_sweep.py --adt-root <ROOT> \
        --fovs 60 90 --modes tangent --enhance none clahe blur

    # geometry preview only, no model / no GPU
    python VGGT-360-fisheye/checks/center_view_sweep.py --adt-root <ROOT> \
        --fovs 40 80 120 --modes tangent raw_roi --render-only
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
_PKG = os.path.dirname(_HERE)          # VGGT-360-fisheye/
_REPO = os.path.dirname(_PKG)          # repo root (for the rectifier)
for _p in (_PKG, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.fisheye_cam import aria_intrinsics, fisheye_ray_lut, kb4_forward_theta
from utils.fisheye_views import fisheye_to_persp


# --------------------------------------------------------------------------- #
# Center-view renderers (each returns crop uint8 [S,S,3] + valid float [S,S])
# --------------------------------------------------------------------------- #

def render_tangent(rgb, cam, fov_deg, out_size, supersample):
    """Gnomonic tangent crop of the center view (straight lines preserved)."""
    crop, valid = fisheye_to_persp(rgb, cam, 0.0, 0.0, fov_deg,
                                   height=out_size, width=out_size,
                                   supersample=supersample)
    return np.clip(crop, 0, 255).astype(np.uint8), valid.astype(np.float32)


def render_raw_roi(rgb, cam, fov_deg, out_size):
    """Centered square ROI of the RAW fisheye covering incidence <= fov/2.

    The ROI half-size is the KB4 image radius at that incidence, ``f * theta_d``.
    No undistortion: the raw (barrel-curved) pixels are only cropped + resized,
    so this shares the SAME angular coverage as ``render_tangent`` at the same
    FoV but keeps fisheye distortion — the clean "undistort vs not" control.
    """
    _, cone = fisheye_ray_lut(cam)                       # imaged-cone mask
    td = float(kb4_forward_theta(np.array(math.radians(fov_deg / 2.0)), cam.k))
    rx, ry = cam.fx * td, cam.fy * td
    # subpixel-centered patch about the principal point (replicates at borders)
    patch = cv2.getRectSubPix(rgb, (int(round(2 * rx)), int(round(2 * ry))),
                              (float(cam.cx), float(cam.cy)))
    vpatch = cv2.getRectSubPix(cone.astype(np.float32),
                               (int(round(2 * rx)), int(round(2 * ry))),
                               (float(cam.cx), float(cam.cy)))
    crop = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(vpatch, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return np.clip(crop, 0, 255).astype(np.uint8), (valid > 0.5).astype(np.float32)


def render_rectifier(rgb, out_size):
    """Repo's validated wide (~85 deg) fisheye->pinhole of the whole frame."""
    from finetune.data.rectify import FisheyeRectifier
    rect = FisheyeRectifier("aria-214-1")
    crop = rect(rgb.astype(np.float32) / 255.0) * 255.0
    crop = cv2.resize(np.clip(crop, 0, 255).astype(np.uint8), (out_size, out_size))
    true_fov = float(np.degrees(2 * np.arctan(0.5 / 0.55)))   # ~84.6
    return crop, np.ones((out_size, out_size), np.float32), true_fov


# --------------------------------------------------------------------------- #
# Processing (the "clearness" axis)
# --------------------------------------------------------------------------- #

def process(crop, enhance):
    if enhance == "none":
        return crop
    if enhance == "clahe":
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if enhance == "sharpen":
        blur = cv2.GaussianBlur(crop, (0, 0), 1.5)
        return np.clip(1.5 * crop.astype(np.float32) - 0.5 * blur, 0, 255).astype(np.uint8)
    if enhance == "blur":
        return cv2.GaussianBlur(crop, (0, 0), 2.0)
    raise ValueError(enhance)


# --------------------------------------------------------------------------- #
# Edge-alignment diagnostic (planar z; restricted to the valid cone)
# --------------------------------------------------------------------------- #

def edge_overlay(rgb, depth_z, valid=None):
    """RGB|depth|overlay strip + alignment fraction over valid pixels."""
    H, W = depth_z.shape
    rgb = cv2.resize(rgb, (W, H))
    if valid is not None and valid.shape != (H, W):
        valid = cv2.resize(valid, (W, H), interpolation=cv2.INTER_NEAREST)
    vmask = None if valid is None else cv2.erode(
        (valid > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    d = depth_z.copy()
    finite = d[np.isfinite(d)] if vmask is None else d[np.isfinite(d) & vmask]
    lo, hi = (np.percentile(finite, 2), np.percentile(finite, 98)) if finite.size else (0, 1)
    d8 = np.clip((d - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    dcol = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rgb_e = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))
    dep_e = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0, 3), cv2.Sobel(d, cv2.CV_32F, 0, 1, 3))
    if vmask is not None:
        rgb_e[~vmask] = 0
        dep_e[~vmask] = 0
    rgb_m = rgb_e > np.percentile(rgb_e[rgb_e > 0], 90) if (rgb_e > 0).any() else rgb_e > 1
    dep_m = dep_e > np.percentile(dep_e[dep_e > 0], 90) if (dep_e > 0).any() else dep_e > 1

    ov = (dcol.astype(np.float32) * 0.45).astype(np.uint8)
    ov[rgb_m] = (255, 40, 40)
    ov[dep_m] = (0, 230, 230)
    if vmask is not None:
        dcol[~vmask] = 24
        ov[~vmask] = 24

    dep_dil = cv2.dilate(dep_m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    align = float((rgb_m & dep_dil).sum()) / max(int(rgb_m.sum()), 1)
    return np.concatenate([rgb, dcol, ov], axis=1), align


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def load_vggt(model_path, device):
    import torch
    from vggt_visfeat.models.vggt import VGGT
    print(f"loading {model_path} on {device} ...")
    model = VGGT.from_pretrained(model_path).to(device).eval()
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        ckpt = load_file(hf_hub_download(model_path, "model.safetensors"))
        mk = set(model.state_dict().keys())
        missing = sorted(mk - set(ckpt.keys()))
        print(f"  weight check: {len(missing)} missing "
              f"({'OK' if not missing else 'RANDOM INIT — suspect results!'})")
    except Exception as e:
        print(f"  weight check skipped ({type(e).__name__})")
    return model


def predict_vggt(model, crop_uint8, device, dtype):
    """Single-image VGGT -> planar-z depth [H,W] + inferred (fov_h,fov_w) deg."""
    import torch
    from vggt_visfeat.utils.load_fn2 import load_and_preprocess_images
    images = load_and_preprocess_images([Image.fromarray(crop_uint8)]).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype,
                                         enabled=device.type == "cuda"):
        pred, _ = model(images=images, save_attn=False)
    depth = pred["depth"][0, 0, ..., 0].float().cpu().numpy()
    fov = None
    if "pose_enc" in pred and pred["pose_enc"] is not None:
        pe = pred["pose_enc"][0, 0].float().cpu().numpy()
        fov = (float(np.degrees(pe[7])), float(np.degrees(pe[8])))
    return depth, fov


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Center-view cropping/processing sweep for VGGT")
    ap.add_argument("--adt-root", required=True)
    ap.add_argument("--rgb-subdir", default="videos_synthetic",
                    help="videos_synthetic = sharp (recommended); videos_rgb = "
                         "real sensor (motion-blurred confound)")
    ap.add_argument("--frame", type=int, default=6)
    ap.add_argument("--fovs", type=float, nargs="+", default=[40, 60, 80, 100, 120],
                    help="center-view FoVs to sweep (deg); edge incidence = fov/2, "
                         "so stay <= ~124 to keep the edge inside the 62.3deg cone")
    ap.add_argument("--modes", nargs="+", default=["tangent", "raw_roi"],
                    choices=["tangent", "raw_roi", "rectifier"])
    ap.add_argument("--enhance", nargs="+", default=["none"],
                    choices=["none", "clahe", "sharpen", "blur"])
    ap.add_argument("--persp-size", type=int, default=518)
    ap.add_argument("--crop-supersample", type=int, default=3)
    ap.add_argument("--model-path", default="facebook/VGGT-1B")
    ap.add_argument("--render-only", action="store_true",
                    help="dump crops + valid masks WITHOUT running the model (no GPU)")
    ap.add_argument("--out", default=os.path.join(_PKG, "outputs", "center_view_sweep"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- one raw ADT fisheye frame ----
    from datasets.adt import ADTFisheyeFrames, find_adt_sequences
    seqs = find_adt_sequences(args.adt_root, rgb_subdir=args.rgb_subdir)
    ds = ADTFisheyeFrames(seqs[:1], rgb_subdir=args.rgb_subdir, max_frames=args.frame + 1)
    rgb = ds[min(args.frame, len(ds) - 1)]["rgb"]
    cam = aria_intrinsics(*rgb.shape[:2], rotated=True)
    Image.fromarray(rgb).save(os.path.join(args.out, "raw_fisheye.png"))

    # ---- enumerate configs: (mode, fov, enhance, true_fov) ----
    configs = []
    for mode in args.modes:
        fov_iter = [None] if mode == "rectifier" else args.fovs
        for fov in fov_iter:
            for enh in args.enhance:
                configs.append((mode, fov, enh))

    model = dtype = device = None
    if not args.render_only:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (torch.bfloat16 if (device.type == "cuda"
                                    and torch.cuda.get_device_capability()[0] >= 8)
                 else torch.float32)
        model = load_vggt(args.model_path, device)

    rows = []   # (tag, mode, fov, enh, true_fov, infer_h, infer_w, align, overlay_third)
    print("\nconfig                          true_fov  infer(h,w)      align")
    for (mode, fov, enh) in configs:
        # render
        if mode == "tangent":
            crop, valid = render_tangent(rgb, cam, fov, args.persp_size, args.crop_supersample)
            true_fov = fov
        elif mode == "raw_roi":
            crop, valid = render_raw_roi(rgb, cam, fov, args.persp_size)
            true_fov = fov
        else:  # rectifier
            crop, valid, true_fov = render_rectifier(rgb, args.persp_size)
        crop = process(crop, enh)

        fov_tag = "rect" if fov is None else f"{int(fov)}"
        tag = f"{mode}_fov{fov_tag}_{enh}"
        Image.fromarray(crop).save(os.path.join(args.out, f"{tag}_input.png"))

        if args.render_only:
            Image.fromarray((valid * 255).astype(np.uint8)).save(
                os.path.join(args.out, f"{tag}_valid.png"))
            print(f"{tag:<32}  {true_fov if true_fov else '?':>6}   (render-only)")
            continue

        # predict + score
        depth, infer = predict_vggt(model, crop, device, dtype)
        strip, align = edge_overlay(crop, depth, valid)
        Image.fromarray(strip).save(os.path.join(args.out, f"{tag}.png"))
        np.save(os.path.join(args.out, f"{tag}_depth.npy"), depth)
        ih, iw = (infer if infer else (float("nan"), float("nan")))
        rows.append((tag, mode, fov, enh, true_fov, ih, iw, align,
                     strip[:, 2 * strip.shape[0]:]))  # overlay third for the grid
        tf = f"{true_fov:.0f}" if true_fov else "?"
        print(f"{tag:<32}  {tf:>6}   {ih:5.1f},{iw:5.1f}   {align*100:5.1f}%")

    if args.render_only:
        print(f"\nrendered {len(configs)} crops -> {args.out}/  (add a GPU + drop "
              "--render-only to score them)")
        return

    _write_summary(rows, args)
    print(f"\ndone — {len(rows)} configs. See {args.out}/ "
          "(summary.csv, align_vs_fov.png, grid.png, per-config *.png)")


def _write_summary(rows, args):
    # CSV
    with open(os.path.join(args.out, "summary.csv"), "w") as f:
        f.write("tag,mode,fov,enhance,true_fov,infer_fov_h,infer_fov_w,align_pct\n")
        for (tag, mode, fov, enh, tf, ih, iw, al, _ov) in rows:
            f.write(f"{tag},{mode},{fov},{enh},{tf},{ih:.2f},{iw:.2f},{al*100:.2f}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # align vs FoV, one line per (mode, enhance) with a real FoV sweep
    fig, ax = plt.subplots(figsize=(7, 5))
    keyed = {}
    for (tag, mode, fov, enh, tf, ih, iw, al, _ov) in rows:
        if fov is None:
            continue
        keyed.setdefault((mode, enh), []).append((fov, al * 100, ih, iw, tf))
    for (mode, enh), pts in sorted(keyed.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], "-o", label=f"{mode}/{enh} align%")
    ax.set_xlabel("center-view FoV (deg) — larger = more of the fisheye cone")
    ax.set_ylabel("edge alignment (%)")
    ax.set_title("Depth–input edge alignment vs center-view coverage")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "align_vs_fov.png"), dpi=140)
    plt.close(fig)

    # overlay grid: rows = (mode,enhance), cols = FoV order
    grid_rows = {}
    for (tag, mode, fov, enh, tf, ih, iw, al, ov) in rows:
        grid_rows.setdefault((mode, enh), []).append((fov if fov else 1e9, tag, al, ov))
    if grid_rows:
        H = min(ov.shape[0] for r in grid_rows.values() for (_, _, _, ov) in r)
        thumbs = []
        for (mode, enh), r in sorted(grid_rows.items()):
            r.sort()
            row_imgs = []
            for (fov, tag, al, ov) in r:
                t = cv2.resize(ov, (H, H))
                cv2.putText(t, f"{tag} {al*100:.0f}%", (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 60), 1, cv2.LINE_AA)
                row_imgs.append(t)
            thumbs.append(np.concatenate(row_imgs, axis=1))
        wmax = max(t.shape[1] for t in thumbs)
        thumbs = [np.pad(t, ((0, 0), (0, wmax - t.shape[1]), (0, 0)), constant_values=24)
                  for t in thumbs]
        Image.fromarray(np.concatenate(thumbs, axis=0)).save(
            os.path.join(args.out, "grid.png"))


if __name__ == "__main__":
    main()
