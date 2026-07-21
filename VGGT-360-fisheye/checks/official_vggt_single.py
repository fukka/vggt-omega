# Copyright (c) 2026.
"""Run the OFFICIAL VGGT on a single exported crop — the apples-to-apples control.

Our ``vggt_visfeat`` was verified (file-by-file) to be computationally identical
to upstream facebook/vggt on the depth path (rope/attention/block/mlp/
patch_embed/vision_transformer bit-identical; aggregator differs only by a
layer-caching optimisation; dpt_head only by an inert ``is_feats`` branch).  So
if the per-view depth looks curved/bumpy, either it is genuine VGGT behavior on
these fisheye-derived tangent crops, or an environment/weights issue — NOT a
code difference in this port.

This settles it: it loads the *pip-installed* official VGGT (``pip install
vggt``) and runs it on the crop PNGs exported by ``main_adt.py --debug-dir``
(``<debug>/XXXX_crops/viewNN_*.png``).  If official VGGT produces the same
curved depth on ``view00`` (the upright center crop), the port is exonerated and
the curviness is a VGGT/input property to address by other means (sharper
resampling, wider FOV, a different backbone, or fine-tuning).

Usage (on the GPU box, in an env with the official ``vggt`` installed):
    python VGGT-360-fisheye/checks/official_vggt_single.py \
        --crop outputs/dbg_single/0000_crops/view00_az0_t0.png \
        --out  outputs/official_vggt_view00.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True, help="a viewNN_*.png from --debug-dir")
    ap.add_argument("--out", default="official_vggt_depth.png")
    ap.add_argument("--model", default="facebook/VGGT-1B")
    args = ap.parse_args()

    import torch
    # Official package — intentionally NOT our vendored vggt_visfeat.
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda"
                               and torch.cuda.get_device_capability()[0] >= 8) \
        else torch.float32

    model = VGGT.from_pretrained(args.model).to(device).eval()
    images = load_and_preprocess_images([args.crop]).to(device)  # official loader
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype,
                                         enabled=device == "cuda"):
        pred = model(images)

    # depth head planar z (bowl-free), same quantity as our _views_z overlay
    depth = pred["depth"][0, 0, ..., 0].float().cpu().numpy()  # [H,W]
    H, W = depth.shape

    import cv2
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
    d8 = np.clip((depth - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    dcol = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

    rgb = np.array(Image.open(args.crop).convert("RGB").resize((W, H)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rgb_e = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3),
                     cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))
    dep_e = np.hypot(cv2.Sobel(depth, cv2.CV_32F, 1, 0, 3),
                     cv2.Sobel(depth, cv2.CV_32F, 0, 1, 3))
    overlay = (dcol.astype(np.float32) * 0.45).astype(np.uint8)
    overlay[rgb_e > np.percentile(rgb_e, 96)] = (255, 40, 40)
    overlay[dep_e > np.percentile(dep_e, 96)] = (0, 230, 230)

    out = np.concatenate([rgb, dcol, overlay], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Image.fromarray(out).save(args.out)
    print(f"saved {args.out}  (RGB | official-VGGT depth | edge overlay)")
    print("If the cyan (depth) edges curve away from the straight red (RGB) "
          "edges here too, the curviness is official VGGT's own output on this "
          "crop — not a bug in the VGGT-360-fisheye port.")


if __name__ == "__main__":
    main()
