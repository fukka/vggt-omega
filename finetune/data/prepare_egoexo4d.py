# Copyright (c) 2026.
"""Convert Ego-Exo4D egocentric (Aria) takes into the clip-folder layout that
``EgocentricVideoDataset`` expects.

Ego-Exo4D stores the egocentric stream as
``takes/<take>/frame_aligned_videos/aria0N_214-1.mp4`` (downscaled copies under
``frame_aligned_videos/downscaled/448/``). Those frames are **fisheye and
rotated**; VGGT-Omega and the self-supervised losses assume a **pinhole** camera,
so prefer ``--undistort`` (rectify to a linear camera with projectaria_tools).

Output (one folder per take = one clip)::

    out_root/<take>/000000.jpg 000001.jpg ...

Then point training at it:  ``python -m finetune.train --data-root out_root ...``

Examples
--------
Quick start (fisheye frames, fine for a first shakeout)::

    python -m finetune.data.prepare_egoexo4d \
        --takes-root /data/egoexo4d/takes --out data/egoexo_frames --fps 1 --downscaled

Recommended (pinhole-rectified; needs `pip install projectaria_tools` + the
take's Aria .vrs from the ``take_vrs`` download part)::

    python -m finetune.data.prepare_egoexo4d \
        --takes-root /data/egoexo4d/takes --vrs-root /data/egoexo4d/take_vrs \
        --out data/egoexo_frames --fps 2 --undistort
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2

EGO_GLOB = "*214-1.mp4"  # Aria RGB camera 214-1


def find_ego_videos(takes_root: str, downscaled: bool = True) -> dict:
    """take_name -> egocentric Aria mp4 path."""
    out = {}
    for take in sorted(os.listdir(takes_root)):
        base = os.path.join(takes_root, take, "frame_aligned_videos")
        if not os.path.isdir(base):
            continue
        search_dirs = []
        if downscaled:
            search_dirs.append(os.path.join(base, "downscaled", "448"))
        search_dirs.append(base)
        for d in search_dirs:
            cands = sorted(glob.glob(os.path.join(d, EGO_GLOB)))
            if cands:
                out[take] = cands[0]
                break
    return out


def find_vrs(vrs_root: str, take: str) -> str | None:
    cands = sorted(glob.glob(os.path.join(vrs_root, take, "*.vrs")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(vrs_root, f"{take}*.vrs")))
    return cands[0] if cands else None


def _frame_step(cap, fps: float) -> int:
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 30.0
    return max(1, int(round(src_fps / max(fps, 0.1))))


def extract_plain(video: str, out_dir: str, fps: float, rotate: bool) -> int:
    """Sample frames straight from the mp4 (fisheye, optionally rotated upright)."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    step = _frame_step(cap, fps)
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(out_dir, f"{saved:06d}.jpg"), frame)
            saved += 1
        i += 1
    cap.release()
    return saved


def extract_undistorted(video: str, vrs_path: str, out_dir: str, fps: float, focal: float, size: int) -> int:
    """Rectify Aria fisheye -> linear pinhole with projectaria_tools, then sample.

    NOTE: template — verify against your projectaria_tools version and the
    official Ego-Exo4D "Undistort frames" tutorial. The Aria RGB calibration must
    match the mp4 resolution (scale the calib if you use the downscaled videos;
    undistorting from the full-res VRS frames avoids that mismatch).
    """
    import numpy as np
    from projectaria_tools.core import calibration, data_provider

    provider = data_provider.create_vrs_data_provider(vrs_path)
    src_calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
    dst_calib = calibration.get_linear_camera_calibration(size, size, focal, "camera-rgb")

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    step = _frame_step(cap, fps)
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            undist = calibration.distort_by_calibration(rgb, dst_calib, src_calib)
            undist = np.rot90(undist, -1)  # Aria RGB is stored rotated 90 deg
            cv2.imwrite(os.path.join(out_dir, f"{saved:06d}.jpg"), cv2.cvtColor(undist, cv2.COLOR_RGB2BGR))
            saved += 1
        i += 1
    cap.release()
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description="Ego-Exo4D egocentric takes -> clip frame folders")
    p.add_argument("--takes-root", required=True, help="Ego-Exo4D 'takes' directory")
    p.add_argument("--out", required=True, help="Output data_root for training")
    p.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample")
    p.add_argument("--downscaled", action="store_true", help="Prefer frame_aligned_videos/downscaled/448")
    p.add_argument("--no-rotate", action="store_true", help="Do not rotate frames 90deg upright (plain mode)")
    p.add_argument("--undistort", action="store_true", help="Rectify fisheye->pinhole (needs projectaria_tools + --vrs-root)")
    p.add_argument("--vrs-root", default="", help="Ego-Exo4D 'take_vrs' directory (for --undistort)")
    p.add_argument("--focal", type=float, default=300.0, help="Target pinhole focal length (--undistort)")
    p.add_argument("--size", type=int, default=512, help="Target square size for the linear camera (--undistort)")
    p.add_argument("--max-takes", type=int, default=0, help="Limit number of takes (0 = all)")
    a = p.parse_args()

    videos = find_ego_videos(a.takes_root, a.downscaled)
    if not videos:
        raise SystemExit(f"No egocentric '{EGO_GLOB}' videos found under {a.takes_root!r}")
    if a.max_takes:
        videos = dict(list(videos.items())[: a.max_takes])
    if a.undistort and not a.vrs_root:
        raise SystemExit("--undistort requires --vrs-root (the Ego-Exo4D 'take_vrs' directory)")

    total = 0
    for take, video in videos.items():
        out_dir = os.path.join(a.out, take)
        if a.undistort:
            vrs = find_vrs(a.vrs_root, take)
            if not vrs:
                print(f"[skip] {take}: no .vrs under {a.vrs_root}")
                continue
            n = extract_undistorted(video, vrs, out_dir, a.fps, a.focal, a.size)
        else:
            n = extract_plain(video, out_dir, a.fps, rotate=not a.no_rotate)
        total += n
        print(f"{take}: {n} frames -> {out_dir}")
    print(f"\nDone: {len(videos)} takes, {total} frames under {a.out}")
    if not a.undistort:
        print("NOTE: these frames are still FISHEYE. For best results re-run with "
              "--undistort (pinhole) — see the Ego-Exo4D 'Undistort frames' tutorial.")


if __name__ == "__main__":
    main()
