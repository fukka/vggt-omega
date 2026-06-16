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
    # check data loading only (CPU, no checkpoint, stand-in models):
    python -m finetune.test_run --data-root <root>/train --dummy --num-windows 2

    # real models:
    python -m finetune.test_run --data-root <root>/train \
        --vggt-checkpoint checkpoints/vggt_omega_1b_512.pt --num-windows 3

    # also rectify the fisheye frames (approx Kannala-Brandt via OpenCV):
    python -m finetune.test_run --data-root <root>/train --dummy \
        --rectify-fisheye --fisheye-k 150,150,256,256 --fisheye-d 0.0,0.0,0.0,0.0
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from .data import EgocentricVideoDataset
from .geometry import decode_pose_encoding


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


# --------------------------------------------------------------------------- #
# image / depth helpers (OpenCV only)
# --------------------------------------------------------------------------- #
def chw_to_bgr(img: torch.Tensor) -> np.ndarray:
    """[3,H,W] float in [0,1] RGB -> [H,W,3] uint8 BGR."""
    rgb = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def colorize(depth_hw: np.ndarray):
    """[H,W] depth -> ([H,W,3] BGR uint8, (vmin, vmax, median))."""
    d = np.asarray(depth_hw, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.zeros((*d.shape, 3), np.uint8), (0.0, 0.0, 0.0)
    vmin, vmax = float(np.percentile(d[valid], 2)), float(np.percentile(d[valid], 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    u8 = (np.clip((d - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color, (vmin, vmax, float(np.median(d[valid])))


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (max(120, 9 * len(text)), 20), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_rectifier(args, H: int, W: int):
    """Returns (fn: bgr->bgr, info dict). Default identity (no rectification)."""
    if not args.rectify_fisheye:
        return (lambda bgr: bgr), {"mode": "identity (no rectification configured)"}
    if args.fisheye_k:
        fx, fy, cx, cy = (float(v) for v in args.fisheye_k.split(","))
    else:
        fx = fy = 0.45 * max(H, W)
        cx, cy = W / 2.0, H / 2.0
    D = np.array([float(v) for v in args.fisheye_d.split(",")], np.float64) if args.fisheye_d else np.zeros(4)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)

    def rect(bgr):
        return cv2.fisheye.undistortImage(bgr, K, D.reshape(4, 1), Knew=K)

    return rect, {"mode": "opencv-fisheye (Kannala-Brandt approx)", "K": K.tolist(), "D": D.tolist(),
                  "note": "approximate; for exact Aria rectification use projectaria_tools on native-res frames"}


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="VGGT-Omega egocentric finetuning — test/data-check run")
    p.add_argument("--data-root", required=True)
    p.add_argument("--vggt-checkpoint", default="")
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
    p.add_argument("--rectify-fisheye", action="store_true")
    p.add_argument("--fisheye-k", default="", help="fx,fy,cx,cy for OpenCV fisheye undistort")
    p.add_argument("--fisheye-d", default="", help="k1,k2,k3,k4 fisheye distortion coeffs")
    p.add_argument("--out-dir", default="test_run_outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.vggt_dummy = a.vggt_dummy or a.dummy
    a.dav2_dummy = a.dav2_dummy or a.dummy
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
        image_resolution=args.image_resolution, patch_size=args.patch_size,
    )
    n = len(dataset)
    print(f"[test] dataset: {n} windows from {args.data_root}")
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
            rectifier, meta["rectification"] = build_rectifier(args, H, W)

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
            dv_color, dv_stats = colorize(depth_v[s].cpu().numpy())
            dd_color, dd_stats = colorize(depth_d[s].cpu().numpy())

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
