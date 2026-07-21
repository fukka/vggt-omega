# Copyright (c) 2026.
"""Run our vendored VGGT (vggt_visfeat) on single images — no pip vggt needed.

This is the control that settles "is our VGGT distorting the input, or is it the
fisheye crop?" using ONLY the environment you already run main_adt.py in.  It:

  1. loads the SAME ``vggt_visfeat.VGGT`` main_adt uses, with a LOUD weight
     check (missing keys => layers left at random init => distorted depth);
  2. runs the plain forward (no SA mask, no attention save, is_feats=False —
     bit-identical to official VGGT) on whatever images you give it;
  3. dumps ``RGB | depth(planar z) | edge-overlay`` and prints an edge-
     alignment number (fraction of RGB edge pixels with a depth edge within
     2 px).  Straight/aligned depth => high alignment.

How to use it to localize the curviness (run BOTH):

  A. a NORMAL perspective photo (any snapshot; or VGGT's own example images,
     e.g. github facebookresearch/vggt examples/room/images/*.png):
        python VGGT-360-fisheye/checks/vggt_single_image.py --images room.png
     * curved / low alignment here too  => weights/env problem (our model is
       NOT loading the official checkpoint correctly — check the weight line).
     * clean / high alignment here       => our VGGT + weights are FINE.

  B. a fisheye-derived center crop, generated internally from a real ADT frame
     (no dependence on main_adt's crop export):
        python VGGT-360-fisheye/checks/vggt_single_image.py \
            --adt-root <ROOT> --rgb-subdir videos_rgb --frame 6
     * clean in A but curved in B => the tangent crop's image quality
       (fisheye periphery softness / resampling), not the model or the port.

So: A tests the model+weights; B tests the crop.  Together they say exactly
where the curviness comes from.
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


def weight_check(model, model_path: str) -> None:
    """Compare checkpoint keys vs model keys — LOUD on any mismatch."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        ckpt = load_file(hf_hub_download(model_path, "model.safetensors"))
        mk = set(model.state_dict().keys())
        ck = set(ckpt.keys())
        missing = sorted(mk - ck)      # in model, not in ckpt -> random init
        unexpected = sorted(ck - mk)   # in ckpt, not used
        print("=" * 68)
        if missing or unexpected:
            print(f"!!! WEIGHT CHECK FAILED — {len(missing)} model keys NOT in "
                  f"checkpoint (LEFT AT RANDOM INIT), {len(unexpected)} ckpt "
                  f"keys unused.")
            print(f"    first missing: {missing[:8]}")
            print(f"    => THIS alone can cause the distorted/curved depth.")
        else:
            print(f"weight check OK: all {len(mk)} model keys matched the "
                  f"checkpoint (no random-init layers).")
        print("=" * 68)
    except Exception as e:
        print(f"weight check SKIPPED ({type(e).__name__}: {e}) — cannot verify "
              f"loading; treat curvy results with suspicion.")


def edge_overlay(rgb: np.ndarray, depth_z: np.ndarray):
    """RGB|depth|overlay strip + edge-alignment fraction (planar z, bowl-free)."""
    H, W = depth_z.shape
    rgb = cv2.resize(rgb, (W, H))
    lo, hi = np.percentile(depth_z, 2), np.percentile(depth_z, 98)
    d8 = np.clip((depth_z - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    dcol = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rgb_e = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))
    dep_e = np.hypot(cv2.Sobel(depth_z, cv2.CV_32F, 1, 0, 3), cv2.Sobel(depth_z, cv2.CV_32F, 0, 1, 3))
    rgb_m = rgb_e > np.percentile(rgb_e, 96)
    dep_m = dep_e > np.percentile(dep_e, 96)

    ov = (dcol.astype(np.float32) * 0.45).astype(np.uint8)
    ov[rgb_m] = (255, 40, 40)      # RGB edges red
    ov[dep_m] = (0, 230, 230)      # depth (z) edges cyan

    # alignment: fraction of RGB edge pixels with a depth edge within 2 px
    dep_dil = cv2.dilate(dep_m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    align = float((rgb_m & dep_dil).sum()) / max(int(rgb_m.sum()), 1)
    return np.concatenate([rgb, dcol, ov], axis=1), align


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[],
                    help="arbitrary image paths (normal perspective photos)")
    ap.add_argument("--adt-root", default=None,
                    help="if set, also generate a fisheye center crop from a "
                         "real ADT frame and test it")
    ap.add_argument("--rgb-subdir", default="videos_rgb")
    ap.add_argument("--frame", type=int, default=6)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--model-path", default="facebook/VGGT-1B")
    ap.add_argument("--out", default=os.path.join(_PKG, "outputs", "vggt_single"))
    args = ap.parse_args()

    import torch
    from vggt_visfeat.models.vggt import VGGT
    from vggt_visfeat.utils.load_fn2 import load_and_preprocess_images

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda"
                               and torch.cuda.get_device_capability()[0] >= 8) else torch.float32
    print(f"loading {args.model_path} on {device} ({dtype}) ...")
    model = VGGT.from_pretrained(args.model_path).to(device).eval()
    weight_check(model, args.model_path)

    os.makedirs(args.out, exist_ok=True)
    pil_list, tags = [], []

    for p in args.images:
        pil_list.append(Image.open(p).convert("RGB"))
        tags.append(os.path.splitext(os.path.basename(p))[0])

    if args.adt_root:
        from datasets.adt import ADTFisheyeFrames, find_adt_sequences
        from utils.fisheye_cam import aria_intrinsics
        from utils.fisheye_views import fisheye_to_persp
        seqs = find_adt_sequences(args.adt_root, rgb_subdir=args.rgb_subdir)
        ds = ADTFisheyeFrames(seqs[:1], rgb_subdir=args.rgb_subdir, max_frames=args.frame + 1)
        rgb = ds[min(args.frame, len(ds) - 1)]["rgb"]
        cam = aria_intrinsics(*rgb.shape[:2], rotated=True)
        crop, _ = fisheye_to_persp(rgb, cam, 0.0, 0.0, args.fov, 518, 518)
        pil_list.append(Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)))
        tags.append(f"adt_fisheye_center_f{args.frame}")

    if not pil_list:
        raise SystemExit("give --images and/or --adt-root")

    images = load_and_preprocess_images(pil_list).to(device)
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype,
                                         enabled=device == "cuda"):
        pred, _ = model(images=images, save_attn=False)   # plain vanilla path
    depth = pred["depth"][0, ..., 0].float().cpu().numpy()   # [S,H,W] planar z

    print("\nedge-alignment (RGB edges that have a depth edge within 2px; "
          "high = depth structure follows the image):")
    for i, tag in enumerate(tags):
        rgb_i = np.array(pil_list[i].resize((depth.shape[2], depth.shape[1])))
        strip, align = edge_overlay(rgb_i, depth[i])
        out = os.path.join(args.out, f"{tag}.png")
        Image.fromarray(strip).save(out)
        print(f"  {tag:<28} align = {align * 100:5.1f}%   -> {out}")
    print("\nRead: a normal photo should score high (depth follows image). If a "
          "normal photo scores LOW too, the model/weights are the problem "
          "(see the weight-check line above), not the fisheye crop.")


if __name__ == "__main__":
    main()
