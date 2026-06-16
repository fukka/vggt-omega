# Copyright (c) 2026.
"""Headless VGGT-Omega inference: run the model and SAVE results to disk.

This is the compute half of ``demo_gradio.py`` with no Gradio and no
``--share`` tunnel. Run it on the GPU machine, then copy the output directory
to wherever you want to visualize (see ``load_and_visualize.py``).

It mirrors ``demo_gradio.run_model`` exactly and writes a portable bundle::

    <out>/
        images/             frames actually fed to the model
        predictions.npz     depth, depth_conf, pose_enc, extrinsic, intrinsic,
                            world_points_from_depth, images
        scene.glb           ready-to-view point cloud + cameras (if trimesh present)
        metadata.json       shapes, resolution, frame count, params

Examples
--------
    python save_demo_results.py --checkpoint ckpt.pt --images-dir my_frames/ --out demo_results/run1
    python save_demo_results.py --checkpoint ckpt.pt --video clip.mp4 --fps 1.0 --out demo_results/run2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_model(checkpoint_path: str) -> VGGTOmega:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run VGGT-Omega inference.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state_dict.get("model", state_dict) if isinstance(state_dict, dict) else state_dict
    model.load_state_dict(state_dict)
    return model.to("cuda")


def gather_frames(images_dir: str | None, video: str | None, fps: float, out_images: str) -> list:
    """Populate ``out_images`` with the frames to reconstruct; return sorted paths."""
    os.makedirs(out_images, exist_ok=True)
    if images_dir:
        srcs = sorted(p for p in glob.glob(os.path.join(images_dir, "*")) if p.lower().endswith(_IMG_EXTS))
        if not srcs:
            raise FileNotFoundError(f"No images found in {images_dir!r}")
        for i, src in enumerate(srcs):
            shutil.copy(src, os.path.join(out_images, f"{i:06d}{os.path.splitext(src)[1].lower()}"))
    elif video:
        import cv2

        cap = cv2.VideoCapture(video)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        sample_fps = max(float(fps), 0.1)
        interval = max(int(round((src_fps if src_fps and src_fps > 0 else 1) / sample_fps)), 1)
        frame_idx = saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % interval == 0:
                cv2.imwrite(os.path.join(out_images, f"{saved:06d}.png"), frame)
                saved += 1
            frame_idx += 1
        cap.release()
        if saved == 0:
            raise RuntimeError(f"No frames extracted from {video!r}")
    else:
        raise ValueError("Provide either --images-dir or --video")
    return sorted(glob.glob(os.path.join(out_images, "*")))


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Identical to demo_gradio.unproject_depth_map_to_point_map."""
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi", np.transpose(rotation, (0, 2, 1)), camera_points - translation[:, None, None, :]
    )


def run_model(image_names: list, model: VGGTOmega, image_resolution: int) -> dict:
    """Mirror of demo_gradio.run_model (no gradio)."""
    images = load_and_preprocess_images(image_names, image_resolution=image_resolution).to("cuda")
    print(f"Preprocessed images shape: {tuple(images.shape)}")
    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    out = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            out[key] = value
    out["world_points_from_depth"] = unproject_depth_map_to_point_map(
        out["depth"], out["extrinsic"], out["intrinsic"]
    )
    torch.cuda.empty_cache()
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Headless VGGT-Omega inference -> saved results")
    p.add_argument("--checkpoint", required=True, help="VGGT-Omega checkpoint path")
    p.add_argument("--images-dir", default=None, help="Folder of input frames")
    p.add_argument("--video", default=None, help="Input video (frames sampled at --fps)")
    p.add_argument("--fps", type=float, default=1.0, help="Video sampling FPS")
    p.add_argument("--image-resolution", type=int, default=512)
    p.add_argument("--out", default="demo_results/run", help="Output bundle directory")
    # default GLB export params (same meaning as the gradio sliders)
    p.add_argument("--conf-thres", type=float, default=50.0)
    p.add_argument("--max-points-k", type=int, default=1000)
    p.add_argument("--no-cam", action="store_true", help="Do not draw camera frusta")
    p.add_argument("--mask-black-bg", action="store_true")
    p.add_argument("--mask-white-bg", action="store_true")
    p.add_argument("--mask-sky", action="store_true")
    p.add_argument("--no-glb", action="store_true", help="Skip GLB export (npz only)")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    out_images = os.path.join(out, "images")

    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint)

    frames = gather_frames(args.images_dir, args.video, args.fps, out_images)
    print(f"Reconstructing {len(frames)} frames")
    predictions = run_model(frames, model, args.image_resolution)

    npz_path = os.path.join(out, "predictions.npz")
    np.savez_compressed(npz_path, **predictions)
    print(f"Saved predictions -> {npz_path}")

    metadata = {
        "num_frames": len(frames),
        "image_resolution": args.image_resolution,
        "source": args.images_dir or args.video,
        "shapes": {k: list(v.shape) for k, v in predictions.items()},
        "glb_params": {
            "conf_thres": args.conf_thres,
            "max_points": int(args.max_points_k * 1000),
            "show_cam": not args.no_cam,
            "mask_black_bg": args.mask_black_bg,
            "mask_white_bg": args.mask_white_bg,
            "mask_sky": args.mask_sky,
        },
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if not args.no_glb:
        try:
            from visual_util import predictions_to_glb

            scene = predictions_to_glb(
                predictions,
                conf_thres=args.conf_thres,
                mask_black_bg=args.mask_black_bg,
                mask_white_bg=args.mask_white_bg,
                show_cam=not args.no_cam,
                mask_sky=args.mask_sky,
                target_dir=out,
                max_points=int(args.max_points_k * 1000),
            )
            glb_path = os.path.join(out, "scene.glb")
            scene.export(file_obj=glb_path)
            print(f"Saved GLB -> {glb_path}")
        except ImportError as exc:
            print(f"[warn] GLB not exported ({exc}); npz is saved. "
                  f"Build the GLB later with load_and_visualize.py --rebuild.")

    print(f"\nDone. Copy '{out}/' to your local machine and run:\n"
          f"    python load_and_visualize.py --results {out}")


if __name__ == "__main__":
    main()
