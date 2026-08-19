# Copyright (c) 2026.
"""Ticket 28: GT-only stats on how much of an ADT frame is dynamic (human/hand/body).

No model inference here -- this only characterizes the ground truth, ahead of
any H4 ("dynamic hand regions dominate depth error") modeling.

ADT ships two segmentation/depth VRS pairs per skeleton sequence: the plain
one (``segmentations.vrs`` / ``depth_images.vrs``) and a
``*_with_skeleton.vrs`` pair that additionally paints in the tracked human
instance. ``AriaDigitalTwinDataPathsProvider.get_datapaths(skeleton_flag=True)``
selects the latter -- the plain pair never has a human pixel because it is
generated before skeleton compositing, so a script that forgot the flag would
silently see 0% dynamic-pixel fraction on every frame and misreport H4 as dead.

Dynamic pixels = pixels whose segmentation instance has
``instance_type == InstanceType.HUMAN`` (the SDK's own classification, not a
category-keyword heuristic -- ADT's plain object catalog in ``instances.json``
never contains a human/hand/body category; the wearer's tracked body is a
separate ``HUMAN``-typed instance layered in only by the skeleton-flag path).
The exact ``category``/``name`` of every such instance actually found is
recorded per sequence.

Incidence angle and the imaged-cone mask (``theta <= 54.83 deg``) reuse
``cam3r.cameras.aria_214_1_kb4`` with ``rotated=False``: these are raw SDK
images in the camera's native calibration frame, not the 270-deg-rotated
convention of the pre-extracted ``videos_rgb``/``depth_npy`` trees that
``cam3r/adt.py`` targets (confirmed by cross-checking the per-sequence
calibration's principal point against the un-rotated native constants).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

import projectaria_tools.projects.adt as adt
from projectaria_tools.core.stream_id import StreamId

from cam3r.cameras import aria_214_1_kb4

RGB_STREAM_ID = StreamId("214-1")
N_THETA_BINS = 8
THETA_MAX_DEG = 54.83
DEPTH_SCALE_M = 0.001  # ADT depth images are uint16 millimetres, planar z.


def _depth_stats(values: np.ndarray) -> Optional[Dict[str, float]]:
    if values.size == 0:
        return None
    return {
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "n_pixels": int(values.size),
    }


def _subsample(items: List, max_frames: int) -> List:
    if len(items) <= max_frames:
        return items
    idx = np.linspace(0, len(items) - 1, max_frames).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def process_sequence(seq_dir: str, max_frames: int = 100) -> dict:
    seq_name = os.path.basename(os.path.normpath(seq_dir))
    paths_provider = adt.AriaDigitalTwinDataPathsProvider(seq_dir)
    data_paths = paths_provider.get_datapaths(True)  # skeleton_flag=True
    gt = adt.AriaDigitalTwinDataProvider(data_paths)

    human_ids = set()
    human_instances_found = []
    all_categories = set()
    for iid in gt.get_instance_ids():
        info = gt.get_instance_info_by_id(iid)
        all_categories.add(str(info.category))
        if info.instance_type == adt.InstanceType.HUMAN:
            human_ids.add(int(iid))
            human_instances_found.append({
                "id": int(iid), "name": str(info.name), "category": str(info.category),
                "motion_type": str(info.motion_type),
            })

    calib = gt.get_aria_camera_calibration(RGB_STREAM_ID)
    W, H = (int(x) for x in calib.get_image_size())
    rays, cone = aria_214_1_kb4(H, W, rotated=False).ray_field(H, W)
    theta_deg = torch.rad2deg(torch.arccos(rays[..., 2].clamp(-1.0, 1.0))).numpy()
    theta_edges = np.linspace(0.0, THETA_MAX_DEG, N_THETA_BINS + 1)
    cone = cone.numpy()

    all_ts = gt.get_aria_device_capture_timestamps_ns(RGB_STREAM_ID)
    sample_ts = _subsample(all_ts, max_frames)
    human_id_arr = np.array(sorted(human_ids), dtype=np.uint64)

    frames_out = []
    for t in sample_ts:
        seg = gt.get_segmentation_image_by_timestamp_ns(t, RGB_STREAM_ID)
        depth = gt.get_depth_image_by_timestamp_ns(t, RGB_STREAM_ID)
        if not (seg.is_valid() and depth.is_valid()):
            continue
        seg_img = seg.data().to_numpy_array()
        depth_img = depth.data().to_numpy_array().astype(np.float64) * DEPTH_SCALE_M

        dyn_pixel = np.isin(seg_img, human_id_arr) if human_id_arr.size else np.zeros_like(seg_img, dtype=bool)
        in_cone = cone & (depth_img > 0)
        dyn_in_cone = dyn_pixel & in_cone
        static_in_cone = (~dyn_pixel) & in_cone

        n_cone = int(in_cone.sum())
        dyn_frac = float(dyn_in_cone.sum()) / n_cone if n_cone else 0.0
        hist, _ = np.histogram(theta_deg[dyn_in_cone], bins=theta_edges)

        frames_out.append({
            "frame_id": str(t),
            "n_cone_pixels": n_cone,
            "dyn_frac": dyn_frac,
            "dyn_theta_hist": hist.tolist(),
            "theta_bin_edges_deg": theta_edges.tolist(),
            "dyn_depth_stats_m": _depth_stats(depth_img[dyn_in_cone]),
            "static_depth_stats_m": _depth_stats(depth_img[static_in_cone]),
        })

    dyn_fracs = [fr["dyn_frac"] for fr in frames_out]
    return {
        "seq": seq_name,
        "n_frames_total": len(all_ts),
        "n_frames_sampled": len(frames_out),
        "categories_found": sorted(all_categories),
        "human_instances_found": human_instances_found,
        "mean_dyn_frac": float(np.mean(dyn_fracs)) if dyn_fracs else 0.0,
        "max_dyn_frac": float(np.max(dyn_fracs)) if dyn_fracs else 0.0,
        "frac_frames_with_any_dynamic_pixel": float(np.mean([f > 0 for f in dyn_fracs])) if dyn_fracs else 0.0,
        "frames": frames_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dirs", nargs="+", required=True, help="ADT sequence directories")
    ap.add_argument("--out-dir", required=True, help="directory to write one JSON per sequence")
    ap.add_argument("--max-frames", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for seq_dir in args.seq_dirs:
        result = process_sequence(seq_dir, max_frames=args.max_frames)
        out_path = os.path.join(args.out_dir, f"{result['seq']}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"{result['seq']}: {result['n_frames_sampled']}/{result['n_frames_total']} frames, "
            f"mean dyn_frac={result['mean_dyn_frac']:.5f}, "
            f"human instances={result['human_instances_found']} -> {out_path}"
        )


if __name__ == "__main__":
    main()
