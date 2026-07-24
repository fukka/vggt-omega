# Copyright (c) 2026.
"""Run VGGT-Omega on single images — the VGGT-Omega twin of ``vggt_single_image.py``.

Purpose: test whether **VGGT-Omega** exhibits the same "depth is a distorted
version of the input" behaviour that plain VGGT (``facebook/VGGT-1B``) shows on
fisheye-derived tangent crops.  This is the apples-to-apples control: same
inputs, same edge-alignment metric, same inferred-FoV readout — only the base
model changes (Meta VGGT-Omega, loaded from a local ``.pt``, instead of VGGT-1B
from the HF hub).

Differences vs the VGGT-1B script (all intentional):
  * loads ``vggt_omega.models.VGGTOmega`` from a local checkpoint (``--checkpoint``)
    and reports missing/unexpected keys loudly (random-init layers => distorted
    depth), mirroring the repo's validated ``run_eval._load_vggt_base`` path;
  * uses VGGT-Omega's OWN ``load_and_preprocess_images`` (mode="balanced",
    ``image_resolution`` default 512) — each model must get its native
    preprocessing for the comparison to be fair;
  * runs each image **independently as a 1-view scene** (``[B=1,S=1,...]``).
    The VGGT-1B script batched every input into ONE multi-view forward; for a
    genuine *single-image* test we do not want VGGT-Omega's cross-view attention
    fusing unrelated photos, so we loop one image at a time.

How to localize the curviness (identical logic to the VGGT-1B script):

  A. a NORMAL perspective photo (any snapshot; or a VGGT example image):
        python VGGT-360-fisheye/checks/vggt_omega_single_image.py \
            --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt --images room.png
     * curved / low alignment here too => weights/env problem (check the
       weight-check line — the checkpoint did not load cleanly).
     * clean / high alignment here      => VGGT-Omega + weights are FINE.

  B. a fisheye-derived tangent crop, generated internally from a real ADT frame:
        python VGGT-360-fisheye/checks/vggt_omega_single_image.py \
            --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
            --adt-root <ROOT> --rgb-subdir videos_synthetic --frame 6
     * clean in A but curved in B => the tangent crop's distribution (fisheye
       periphery / narrow FoV), not the model.

Compare the printed ``align`` and ``fov_h/fov_w`` numbers against the VGGT-1B
run on the SAME inputs: if VGGT-Omega aligns where VGGT-1B did not, VGGT-Omega
is the better fisheye backbone; if it curves the same way, the distortion is a
VGGT-family property, not specific to VGGT-1B.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)          # VGGT-360-fisheye/  (fisheye utils, datasets)
_REPO = os.path.dirname(_PKG)          # repo root          (vggt_omega/, finetune/)
for _p in (_PKG, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DEFAULT_CKPT = os.path.join(_REPO, "checkpoints", "VGGT-Omega-1B-512", "model.pt")


def load_vggt_omega(checkpoint: str, device):
    """Load VGGT-Omega from a local ``.pt`` — LOUD on any key mismatch.

    Mirrors ``finetune.eval.run_eval._load_vggt_base``: a non-strict load, then
    report missing (model keys left at random init) / unexpected (checkpoint
    keys unused) so a bad checkpoint can't silently produce garbage depth.
    """
    import torch
    from vggt_omega.models import VGGTOmega

    if not os.path.isfile(checkpoint):
        raise SystemExit(
            f"VGGT-Omega checkpoint not found: {checkpoint}\n"
            f"  pass --checkpoint <path to vggt_omega_1b_512.pt / model.pt> "
            f"(request access at https://huggingface.co/facebook/VGGT-Omega)")

    print(f"loading VGGT-Omega from {checkpoint} ...")
    model = VGGTOmega()
    sd = torch.load(checkpoint, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("=" * 68)
    if missing or unexpected:
        print(f"!!! WEIGHT CHECK: {len(missing)} model keys NOT in checkpoint "
              f"(LEFT AT RANDOM INIT), {len(unexpected)} checkpoint keys unused.")
        if missing:
            print(f"    first missing: {missing[:8]}")
            print("    => random-init layers alone can cause distorted/curved depth.")
    else:
        print("weight check OK: all model keys matched the checkpoint "
              "(no random-init layers).")
    print("=" * 68)
    return model.to(device).eval()


def edge_overlay(rgb: np.ndarray, depth_z: np.ndarray):
    """RGB|depth|overlay strip + edge-alignment fraction (planar z, bowl-free).

    Byte-for-byte the same diagnostic as the VGGT-1B script: RGB Sobel edges in
    red, depth Sobel edges in cyan, on the depth colormap; ``align`` = fraction
    of RGB edge pixels with a depth edge within ~2 px.  High => depth structure
    follows the image.  (VGGT-Omega depth is already planar z, so no de-bowling
    is needed.)
    """
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

    dep_dil = cv2.dilate(dep_m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    align = float((rgb_m & dep_dil).sum()) / max(int(rgb_m.sum()), 1)
    return np.concatenate([rgb, dcol, ov], axis=1), align


def predict_one(model, image_path: str, device, image_resolution: int, dtype):
    """VGGT-Omega on ONE image (1-view scene) -> (planar-z depth [H,W], pose_enc[9]).

    ``preds["depth"]`` is metric planar z (distance along the optical axis);
    ``preds["pose_enc"][...,7:9]`` = (fov_h, fov_w) in radians — VGGT-Omega's
    inferred camera field of view, the quantity that (for VGGT-1B) mis-tracked
    the true render FoV and bent the geometry.
    """
    import torch
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    imgs = load_and_preprocess_images([image_path], image_resolution=image_resolution)
    imgs = imgs.unsqueeze(0).to(device)                     # [S,3,H,W] -> [B=1,S=1,3,H,W]
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype,
                                         enabled=device.type == "cuda"):
        preds = model(imgs)
    depth = preds["depth"]
    if depth.ndim == 5:
        depth = depth.squeeze(-1)                           # [B,S,H,W,1] -> [B,S,H,W]
    depth_np = depth[0, 0].float().cpu().numpy()            # [H,W]
    pose = None
    if "pose_enc" in preds and preds["pose_enc"] is not None:
        pose = preds["pose_enc"][0, 0].float().cpu().numpy()  # [9]
    return depth_np, pose


def main() -> None:
    ap = argparse.ArgumentParser(description="VGGT-Omega single-image depth tester")
    ap.add_argument("--checkpoint", default=_DEFAULT_CKPT,
                    help="VGGT-Omega .pt (e.g. checkpoints/VGGT-Omega-1B-512/model.pt)")
    ap.add_argument("--images", nargs="*", default=[],
                    help="arbitrary image paths (normal perspective photos)")
    ap.add_argument("--adt-root", default=None,
                    help="if set, also generate a fisheye view from a real ADT "
                         "frame and test it")
    ap.add_argument("--rgb-subdir", default="videos_synthetic",
                    help="videos_synthetic = sharp, GT-aligned rendered RGB "
                         "(recommended); videos_rgb = real sensor, MOTION-BLURRED "
                         "(a confound)")
    ap.add_argument("--frame", type=int, default=6)
    ap.add_argument("--fov", type=float, default=60.0,
                    help="render FoV for the --adt-root tangent view (known "
                         "exactly, since this script renders it)")
    ap.add_argument("--assume-fov", type=float, default=None,
                    help="reference 'true' FoV to print for --images inputs whose "
                         "real FoV this script cannot know")
    ap.add_argument("--crop-supersample", type=int, default=3,
                    help="anti-alias factor for the tangent render (1 = old)")
    ap.add_argument("--view-mode", choices=["tangent", "rectifier"], default="tangent",
                    help="how to make the ADT view: 'tangent' = 60deg "
                         "fisheye_to_persp crop; 'rectifier' = the repo's "
                         "validated FisheyeRectifier wide (~85deg) pinhole of the "
                         "WHOLE frame")
    ap.add_argument("--image-resolution", type=int, default=512,
                    help="VGGT-Omega preprocessing resolution (divisible by 16)")
    ap.add_argument("--out", default=os.path.join(_PKG, "outputs", "vggt_omega_single"))
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if (device.type == "cuda"
                                and torch.cuda.get_device_capability()[0] >= 8)
             else torch.float32)
    print(f"device={device}  dtype={dtype}")
    model = load_vggt_omega(args.checkpoint, device)

    os.makedirs(args.out, exist_ok=True)

    # Build the list of (path, tag, true_fov).  VGGT-Omega's preprocessing reads
    # from disk, so the ADT crop is rendered and saved to --out first, then fed
    # by path (this also exports the EXACT crop for cross-model comparison).
    inputs = []   # (path, tag, true_fov_deg_or_None)
    for p in args.images:
        inputs.append((p, os.path.splitext(os.path.basename(p))[0], args.assume_fov))

    if args.adt_root:
        from datasets.adt import ADTFisheyeFrames, find_adt_sequences
        from utils.fisheye_cam import aria_intrinsics
        from utils.fisheye_views import fisheye_to_persp
        seqs = find_adt_sequences(args.adt_root, rgb_subdir=args.rgb_subdir)
        ds = ADTFisheyeFrames(seqs[:1], rgb_subdir=args.rgb_subdir,
                              max_frames=args.frame + 1)
        rgb = ds[min(args.frame, len(ds) - 1)]["rgb"]
        if args.view_mode == "rectifier":
            from finetune.data.rectify import FisheyeRectifier
            rect = FisheyeRectifier("aria-214-1")
            crop = rect(rgb.astype(np.float32) / 255.0) * 255.0
            tag = f"adt_rectifier_f{args.frame}"
            true_fov = float(np.degrees(2 * np.arctan(0.5 / 0.55)))  # ~84.6
        else:
            cam = aria_intrinsics(*rgb.shape[:2], rotated=True)
            crop, _ = fisheye_to_persp(rgb, cam, 0.0, 0.0, args.fov, 518, 518,
                                       supersample=args.crop_supersample)
            tag = f"adt_tangent{int(args.fov)}_f{args.frame}"
            true_fov = args.fov
        crop_path = os.path.join(args.out, f"{tag}_input.png")
        Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).save(crop_path)
        inputs.append((crop_path, tag, true_fov))

    if not inputs:
        raise SystemExit("give --images and/or --adt-root")

    # Run each image independently (true single-image / 1-view mode).
    print("\nVGGT-Omega inferred FoV per image (deg). 'true' is the render FoV "
          "when known (rendered by this script); for --images inputs it is "
          "UNKNOWN unless you pass --assume-fov:")
    depths = {}
    for (path, tag, true_fov) in inputs:
        depth, pose = predict_one(model, path, device, args.image_resolution, dtype)
        depths[tag] = (path, depth)
        if pose is not None:
            fov_h, fov_w = np.degrees(pose[7]), np.degrees(pose[8])
            ref = f"true={true_fov:.0f}" if true_fov is not None else "true=unknown"
            print(f"  {tag:<28} fov_h={fov_h:5.1f}  fov_w={fov_w:5.1f}   ({ref})")
        else:
            print(f"  {tag:<28} (no pose_enc in predictions)")

    print("\nedge-alignment (RGB edges that have a depth edge within 2px; "
          "high = depth structure follows the image):")
    for tag, (path, depth) in depths.items():
        rgb_i = np.array(Image.open(path).convert("RGB").resize(
            (depth.shape[1], depth.shape[0])))
        strip, align = edge_overlay(rgb_i, depth)
        out = os.path.join(args.out, f"{tag}.png")
        Image.fromarray(strip).save(out)
        print(f"  {tag:<28} align = {align * 100:5.1f}%   -> {out}")

    print("\nRead: compare these numbers with the VGGT-1B run "
          "(checks/vggt_single_image.py) on the SAME inputs. A normal photo "
          "should score high for both. If VGGT-Omega aligns on the fisheye "
          "tangent crop where VGGT-1B did NOT, VGGT-Omega is the better fisheye "
          "backbone; if it curves the same way, the distortion is a VGGT-family "
          "trait (camera/FoV coupling), not specific to VGGT-1B.")


if __name__ == "__main__":
    main()
