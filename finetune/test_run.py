# Copyright (c) 2026.
"""Test-run mode: sanity-check data loading + model outputs (no training).

Loads a few windows from the dataset, runs VGGT-Omega and Depth-Anything-V2
forward, and dumps, per saved frame:

    <out>/raw_input/            w00_f00.png   (frame as the loader yields it)
    <out>/raw_input_rectified/  w00_f00.png   (after the configured rectifier)
    <out>/VGGT-Omega_depth/     w00_f00.png   (+ .npy raw metric depth)
    <out>/DAv2_depth/           w00_f00.png   (+ .npy raw depth)
    <out>/montage/              w00_f00.jpg   (raw | rectified | VGGT | DAv2)
    <out>/metadata.json         (clips, paths, shapes, predicted FoV, depth stats)

Use ``--dummy`` to verify data loading + the save pipeline on CPU with no
checkpoint; drop it (and pass ``--vggt-checkpoint``) to see real depth.

Examples
--------
    # check data loading only (CPU, no checkpoint):
    python finetune/test_run.py --data-root <root>/train --dummy --num-windows 2

    # real models (fisheye rectification on by default; Aria 214-1 preset auto-detected):
    python finetune/test_run.py --data-root <root>/train \
        --vggt-checkpoint checkpoints/vggt_omega_1b_512.pt --num-windows 3

    # explicit Aria preset (same as auto-detected when path contains 'aria'):
    python finetune/test_run.py --data-root <root>/train --dummy \
        --camera-preset aria-214-1 --num-windows 3

    # disable rectification:
    python finetune/test_run.py --data-root <root>/train --dummy --no-rectify
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "finetune"

import argparse
import json
import os

import cv2
import numpy as np
import torch

from .data import EgocentricVideoDataset
from .geometry import decode_pose_encoding
from .viz import chw_to_bgr, colorize_depth, label  # shared viz helpers


# --------------------------------------------------------------------------- #
# models (plain inference build; no LoRA)
# --------------------------------------------------------------------------- #
def build_vggt(args):
    if args.vggt_dummy:
        from .models.dummy import DummyVGGT

        return DummyVGGT().eval()
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega().eval()
    if args.vggt_checkpoint:
        sd = torch.load(args.vggt_checkpoint, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[test] loaded VGGT checkpoint (missing={len(missing)}, unexpected={len(unexpected)})")
    return model


def build_dav2(args):
    from .models import build_depth_anything

    return build_depth_anything(use_dummy=args.dav2_dummy, model_name=args.dav2_model_name).eval()


# Depth/image viz helpers (chw_to_bgr, colorize_depth, label) are imported from
# finetune.viz so training and this tool share one implementation.


# The Aria 214-1 lens comes from ``finetune/aria_calibration.py``. This file
# held a fourth copy, with the same off-by-one ``rectify.py`` had — its comment
# said "new_cx = H_native - old_cy", which is a pixel past the last column of an
# H-wide frame. The KB4 coefficients are a least-squares fit over theta in
# [0, 60] deg to the device's own Fisheye624 calibration, max angular error
# 0.22 deg; the provenance is in that module.
from finetune.aria_calibration import KB4 as _ARIA_214_1_KB4, intrinsics as _aria

_ARIA_214_1_D_KB4 = np.array(_ARIA_214_1_KB4, np.float64)
# Output focal (fraction of max(H,W)) that avoids black borders for this camera.
# Derived empirically: 0.55*H at 512px ≈ focal=281, matching projectaria_tools
# linear-camera output at the same resolution (0.22° max error vs GT).
_ARIA_214_1_FOCAL_OUT_NORM = 0.55


def _aria_K(H: int, W: int) -> np.ndarray:
    """Intrinsics for a pre-extracted (rotated 90° CW) Aria RGB 214-1 frame."""
    fx, fy, cx, cy = _aria(H, W, rotated=True)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], np.float64)


def _detect_preset(clip_path: str) -> str:
    """Heuristic: paths containing 'aria' → assume Aria 214-1 fisheye."""
    return "aria-214-1" if "aria" in clip_path.lower() else "none"


def build_rectifier(args, H: int, W: int, clip_path: str = ""):
    """Returns (fn: bgr->bgr, info dict).

    Preset auto-detection: if the clip path contains 'aria' and no explicit
    preset is given, 'aria-214-1' is used automatically.  Pass --no-rectify
    to disable entirely.
    """
    if not args.rectify_fisheye:
        return (lambda bgr: bgr), {"mode": "identity (--no-rectify set)"}

    preset = args.camera_preset
    if preset == "none" and clip_path:
        auto = _detect_preset(clip_path)
        if auto != "none":
            preset = auto
            print(f"[test] auto-detected camera preset '{preset}' from clip path")

    # --- Intrinsics K ---
    if args.fisheye_k:
        fx, fy, cx, cy = (float(v) for v in args.fisheye_k.split(","))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], np.float64)
    elif preset == "aria-214-1":
        K = _aria_K(H, W)
    else:
        f = 0.43 * max(H, W)
        K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]], np.float64)

    # --- Distortion D (Kannala-Brandt k1 k2 k3 k4) ---
    if args.fisheye_d:
        D = np.array([float(v) for v in args.fisheye_d.split(",")], np.float64)
    elif preset == "aria-214-1":
        D = _ARIA_214_1_D_KB4.copy()
    else:
        # D=zeros = equidistant fisheye → pinhole; still a non-trivial transform
        D = np.zeros(4, np.float64)
        print("[test] WARNING: no camera preset; using equidistant fisheye model (D=zeros). "
              "Pass --camera-preset aria-214-1 for Aria RGB 214-1 frames.")

    D4 = D.reshape(4, 1)
    # Fixed output focal: 0.55*max(H,W) avoids black borders for Aria (validated
    # vs projectaria_tools ground-truth on ADT sample — 0 black pixels, <0.22° error).
    # estimateNewCameraMatrixForUndistortRectify is NOT used: for wide-angle cameras
    # it returns a very small focal (130°+ FoV) that causes large black blobs.
    if preset == "aria-214-1":
        focal_out = _ARIA_214_1_FOCAL_OUT_NORM * max(H, W)
    else:
        focal_out = 0.5 * max(H, W)   # ~90° FoV; safe for unknown cameras
    Knew = np.array([[focal_out, 0, W / 2.0], [0, focal_out, H / 2.0], [0, 0, 1.0]], np.float64)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D4, np.eye(3, dtype=np.float64), Knew, (W, H), cv2.CV_16SC2
    )

    def rect(bgr: np.ndarray) -> np.ndarray:
        return cv2.remap(bgr, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    return rect, {
        "mode": "opencv-fisheye KB4",
        "preset": preset,
        "K_src": K.tolist(), "D": D.tolist(), "K_dst": Knew.tolist(),
        "note": "KB4 coefficients fitted from Aria Fisheye624 VRS calibration (max err 0.22 deg)",
    }


# --------------------------------------------------------------------------- #
# Default cluster paths (override on the CLI for other environments).
_DEFAULT_DATA_ROOT = "/sdp-rgb-perception/tristan-space-s3/egoexo4d/lmeec_data/egoexo_dataset/train"
_DEFAULT_VGGT_CKPT = "/group-volume/Fengjia/projects/vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt"


def parse_args():
    p = argparse.ArgumentParser(description="VGGT-Omega egocentric finetuning — test/data-check run")
    p.add_argument("--data-root", default=_DEFAULT_DATA_ROOT)
    p.add_argument("--vggt-checkpoint", default=_DEFAULT_VGGT_CKPT)
    p.add_argument("--dav2-model-name", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--dummy", action="store_true", help="shortcut for --vggt-dummy --dav2-dummy")
    p.add_argument("--vggt-dummy", action="store_true")
    p.add_argument("--dav2-dummy", action="store_true")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--image-resolution", type=int, default=512)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--num-windows", type=int, default=2, help="how many windows to sample")
    p.add_argument("--max-frames-per-window", type=int, default=0, help="0 = all frames in the window")
    p.add_argument("--shuffle", action="store_true", help="random windows (else evenly spaced)")
    p.add_argument("--no-rectify", action="store_true",
                   help="skip fisheye rectification (raw_input_rectified = raw_input)")
    p.add_argument("--camera-preset", default="none", choices=["none", "aria-214-1"],
                   help="Built-in approximate camera model. 'aria-214-1' uses KB4 fitted from "
                        "the Aria RGB Fisheye624 VRS calibration. Auto-detected from clip path when 'none'.")
    p.add_argument("--fisheye-k", default="", help="fx,fy,cx,cy (overrides preset K)")
    p.add_argument("--fisheye-d", default="", help="k1,k2,k3,k4 KB4 distortion coeffs (overrides preset D)")
    p.add_argument("--clip-pattern", default="*214-1",
                   help="fnmatch glob; keep only matching camera dirs (egocentric RGB aria*_214-1). "
                        "Pass '' to load every camera.")
    p.add_argument("--out-dir", default="test_run_outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.vggt_dummy = a.vggt_dummy or a.dummy
    a.dav2_dummy = a.dav2_dummy or a.dummy
    a.rectify_fisheye = not a.no_rectify  # on by default; --no-rectify disables
    return a


def _squeeze_depth(d: torch.Tensor) -> torch.Tensor:
    return d[..., 0] if d.dim() == 5 and d.shape[-1] == 1 else d


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    dataset = EgocentricVideoDataset(
        args.data_root, seq_len=args.seq_len, stride=args.stride,
        clip_pattern=args.clip_pattern,
        image_resolution=args.image_resolution, patch_size=args.patch_size,
    )
    n = len(dataset)
    print(f"[test] dataset: {n} windows from {len(dataset.clips)} clips "
          f"(pattern {args.clip_pattern!r}, skipped {dataset.num_skipped_dirs} non-matching dirs) "
          f"under {args.data_root}")
    k = min(args.num_windows, n)
    if args.shuffle:
        idxs = sorted(np.random.choice(n, k, replace=False).tolist())
    else:
        idxs = sorted(set(np.linspace(0, n - 1, k).astype(int).tolist()))

    vggt = build_vggt(args).to(device)
    dav2 = build_dav2(args).to(device)

    subdirs = ["raw_input", "raw_input_rectified", "VGGT-Omega_depth", "DAv2_depth", "montage"]
    for s in subdirs:
        os.makedirs(os.path.join(args.out_dir, s), exist_ok=True)

    meta = {
        "data_root": args.data_root,
        "num_windows_total": n,
        "models": {"vggt": "dummy" if args.vggt_dummy else (args.vggt_checkpoint or "VGGTOmega(random init)"),
                   "dav2": "dummy" if args.dav2_dummy else args.dav2_model_name},
        "seq_len": args.seq_len, "stride": args.stride, "image_resolution": args.image_resolution,
        "device": str(device), "windows": [],
    }
    rectifier = None
    saved = 0

    for wi, idx in enumerate(idxs):
        sample = dataset[idx]
        images = sample["images"].unsqueeze(0).to(device)  # [1,S,3,H,W]
        S, _, H, W = images.shape[1:]
        if rectifier is None:
            rectifier, meta["rectification"] = build_rectifier(args, H, W, clip_path=sample["clip"])

        with torch.inference_mode():
            preds = vggt(images)
            depth_v = _squeeze_depth(preds["depth"])[0]          # [S,H,W]
            pose_enc = preds["pose_enc"][0]                       # [S,9]
            depth_d = _squeeze_depth(dav2(images))[0]             # [S,H,W]
        _, K = decode_pose_encoding(pose_enc.float().cpu()[None], (H, W))
        fov_h = (2 * np.degrees(np.arctan((H / 2) / K[0, :, 1, 1].numpy())))
        fov_w = (2 * np.degrees(np.arctan((W / 2) / K[0, :, 0, 0].numpy())))

        frames = range(S if args.max_frames_per_window <= 0 else min(S, args.max_frames_per_window))
        win_meta = {"index": int(idx), "clip": sample["clip"], "H": int(H), "W": int(W), "frames": []}
        for s in frames:
            tag = f"w{wi:02d}_f{s:02d}"
            raw_bgr = chw_to_bgr(images[0, s])
            rect_bgr = rectifier(raw_bgr)
            dv_color, dv_stats = colorize_depth(depth_v[s].cpu().numpy())
            dd_color, dd_stats = colorize_depth(depth_d[s].cpu().numpy())

            cv2.imwrite(os.path.join(args.out_dir, "raw_input", tag + ".png"), raw_bgr)
            cv2.imwrite(os.path.join(args.out_dir, "raw_input_rectified", tag + ".png"), rect_bgr)
            cv2.imwrite(os.path.join(args.out_dir, "VGGT-Omega_depth", tag + ".png"), dv_color)
            cv2.imwrite(os.path.join(args.out_dir, "DAv2_depth", tag + ".png"), dd_color)
            np.save(os.path.join(args.out_dir, "VGGT-Omega_depth", tag + ".npy"), depth_v[s].cpu().numpy())
            np.save(os.path.join(args.out_dir, "DAv2_depth", tag + ".npy"), depth_d[s].cpu().numpy())

            panels = [label(raw_bgr, "raw_input"), label(rect_bgr, "rectified"),
                      label(dv_color, "VGGT-Omega"), label(dd_color, "DAv2")]
            cv2.imwrite(os.path.join(args.out_dir, "montage", tag + ".jpg"), np.hstack(panels))

            win_meta["frames"].append({
                "tag": tag, "src": os.path.basename(sample["paths"][s]),
                "vggt_fov_deg": [round(float(fov_h[s]), 2), round(float(fov_w[s]), 2)],
                "vggt_depth_min_max_med": [round(v, 4) for v in dv_stats],
                "dav2_depth_min_max_med": [round(v, 4) for v in dd_stats],
            })
            saved += 1
        meta["windows"].append(win_meta)
        print(f"[test] window {wi} (dataset idx {idx}): {len(win_meta['frames'])} frames "
              f"<- {os.path.relpath(sample['clip'], args.data_root)}")

    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[test] saved {saved} frames x 5 artifacts under {args.out_dir}/")
    print(f"[test] open {args.out_dir}/montage/ for side-by-side raw | rectified | VGGT | DAv2, "
          f"and metadata.json for shapes/FoV/depth stats.")


if __name__ == "__main__":
    main()
