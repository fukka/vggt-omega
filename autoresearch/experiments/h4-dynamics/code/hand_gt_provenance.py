# Copyright (c) 2026.
"""Ticket 032 (issue #34), raised by the human: verify hand GT-depth provenance,
and use the synthetic RGB stream as a hands-free control.

#28 (``hand_pixel_stats.py``) and #31 (``hands_pose_depth.py``) both build a
*single* ``AriaDigitalTwinDataProvider`` from
``AriaDigitalTwinDataPathsProvider(seq_dir).get_datapaths(True)`` and query
segmentation AND depth from that same object -- so both scripts used the
WITH-skeleton variant for both streams; there is no seg/depth mismatch between
them. What was never checked is whether the WITH-skeleton *depth* stream
actually differs from the WITHOUT-skeleton one at the pixels the WITH-skeleton
*segmentation* calls dynamic -- i.e. whether GT depth "sees" the person at all,
or whether ADT renders depth from an empty room regardless of the flag.

Part 2 uses the synthetic RGB stream (rendered from the twin, so if the human
mesh isn't composited into it the way it is into the real capture, synthetic
frames are a photometrically natural "hands removed" control) to re-test #31's
"hands are just occlusion" conclusion without the mean-fill's visible seam.

Usage (repo root, on the box):
    <venv>/bin/python autoresearch/experiments/h4-dynamics/code/hand_gt_provenance.py \
        --seq-dirs /path/to/*_skeleton_seq*_M1292 ... \
        --out-dir results/autoresearch-h4-provenance
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))

from cam3r.cameras import _ARIA_KB4, aria_214_1_kb4, aria_valid_theta_max  # noqa: E402
from raytun3r.cameras import KannalaBrandt              # noqa: E402
from finetune.eval.metrics import align_depth            # noqa: E402
from rim_pose_value import _median                       # noqa: E402

import projectaria_tools.projects.adt as adt              # noqa: E402
from projectaria_tools.core.stream_id import StreamId     # noqa: E402

RGB_STREAM_ID = StreamId("214-1")
DEPTH_SCALE_M = 0.001
DIFF_GATE_M = 0.05          # "differs" threshold for the provenance check
N_HIGH_DYN_FRAMES = 20
SIZE = 504


def camera_for(size: int) -> KannalaBrandt:
    ref = aria_214_1_kb4(size, size, rotated=False)
    return KannalaBrandt(ref.fx, ref.fy, ref.cx, ref.cy, width=size, height=size,
                         k=tuple(_ARIA_KB4), theta_max=aria_valid_theta_max())


def resize(arr: np.ndarray, size: int, nearest: bool) -> np.ndarray:
    t = torch.from_numpy(arr)
    if t.ndim == 2:
        t = t[None, None].float()
    else:
        t = t.permute(2, 0, 1)[None].float()
    mode = "nearest" if nearest else "bicubic"
    out = torch.nn.functional.interpolate(
        t, size=(size, size), mode=mode, align_corners=None if nearest else False)
    return out[0, 0].numpy() if arr.ndim == 2 else out[0].permute(1, 2, 0).numpy()


def pick_high_dyn_frames(gt_with, human_ids: np.ndarray, cone: np.ndarray,
                         size: int, n: int) -> List[int]:
    ts = list(gt_with.get_aria_device_capture_timestamps_ns(RGB_STREAM_ID))
    sample = ts[::max(1, len(ts) // 200)]      # 200-ish frame scan for dyn_frac
    scored = []
    for t in sample:
        seg = gt_with.get_segmentation_image_by_timestamp_ns(t, RGB_STREAM_ID)
        if not seg.is_valid():
            continue
        seg_img = seg.data().to_numpy_array()
        dyn = np.isin(seg_img, human_ids) if human_ids.size else \
            np.zeros_like(seg_img, dtype=bool)
        dyn_r = resize(dyn.astype(np.float32), size, nearest=True) > 0.5
        df = float((dyn_r & cone).sum()) / max(int(cone.sum()), 1)
        scored.append((df, t))
    scored.sort(reverse=True)
    return [t for _, t in scored[:n]]


def process_sequence(seq_dir: str, out_dir: str, da3_bb, device: str) -> dict:
    seq_name = os.path.basename(os.path.normpath(seq_dir))
    paths_provider = adt.AriaDigitalTwinDataPathsProvider(seq_dir)
    gt_with = adt.AriaDigitalTwinDataProvider(paths_provider.get_datapaths(True))
    gt_without = adt.AriaDigitalTwinDataProvider(paths_provider.get_datapaths(False))

    human_ids = np.array(sorted(
        int(iid) for iid in gt_with.get_instance_ids()
        if gt_with.get_instance_info_by_id(iid).instance_type == adt.InstanceType.HUMAN
    ), dtype=np.uint64)

    camera = camera_for(SIZE)
    cone = camera.incidence_grid(SIZE, SIZE).numpy() <= camera.theta_max

    frames = pick_high_dyn_frames(gt_with, human_ids, cone, SIZE, N_HIGH_DYN_FRAMES)
    print(f"[{seq_name}] {len(frames)} high-dyn_frac frames selected")

    provenance_rows = []
    da3_rows = []
    grid_examples = []
    for t in frames:
        seg = gt_with.get_segmentation_image_by_timestamp_ns(t, RGB_STREAM_ID)
        d_with = gt_with.get_depth_image_by_timestamp_ns(t, RGB_STREAM_ID)
        d_without = gt_without.get_depth_image_by_timestamp_ns(t, RGB_STREAM_ID)
        img_real = gt_with.get_aria_image_by_timestamp_ns(t, RGB_STREAM_ID)
        img_syn = gt_with.get_synthetic_image_by_timestamp_ns(t, RGB_STREAM_ID)
        if not (seg.is_valid() and d_with.is_valid() and d_without.is_valid()
                and img_real.is_valid() and img_syn.is_valid()):
            continue

        seg_img = seg.data().to_numpy_array()
        dyn = np.isin(seg_img, human_ids) if human_ids.size else \
            np.zeros_like(seg_img, dtype=bool)
        dyn_r = resize(dyn.astype(np.float32), SIZE, nearest=True) > 0.5

        zw = resize(d_with.data().to_numpy_array().astype(np.float32)
                   * DEPTH_SCALE_M, SIZE, nearest=True)
        zwo = resize(d_without.data().to_numpy_array().astype(np.float32)
                    * DEPTH_SCALE_M, SIZE, nearest=True)

        dyn_cone = dyn_r & cone & (zw > 0) & (zwo > 0)
        if dyn_cone.sum() >= 10:
            diff = np.abs(zw - zwo)[dyn_cone]
            provenance_rows.append({
                "frame_id": str(t),
                "n_dyn_pixels": int(dyn_cone.sum()),
                "median_abs_diff_m": float(np.median(diff)),
                "frac_differs_gt_5cm": float((diff > DIFF_GATE_M).mean()),
                "median_with_m": float(np.median(zw[dyn_cone])),
                "median_without_m": float(np.median(zwo[dyn_cone])),
            })

        rgb_real = resize(img_real.data().to_numpy_array().astype(np.float32)
                          / 255.0, SIZE, nearest=False)
        rgb_syn = resize(img_syn.data().to_numpy_array().astype(np.float32)
                         / 255.0, SIZE, nearest=False)
        static_cone = cone & ~dyn_r & (zw > 0)
        photo_diff = np.abs(rgb_real - rgb_syn).mean(axis=-1)
        d_stat = float(photo_diff[static_cone].mean()) if static_cone.sum() else None
        d_dyn = float(photo_diff[dyn_r & cone].mean()) if (dyn_r & cone).sum() else None

        cos_t = torch.cos(camera.incidence_grid(SIZE, SIZE)).numpy()
        gr_with = zw / np.clip(cos_t, 1e-6, None)
        rgb_real_t = torch.from_numpy(rgb_real).permute(2, 0, 1).float().to(device)
        rgb_syn_t = torch.from_numpy(rgb_syn).permute(2, 0, 1).float().to(device)
        with torch.no_grad():
            pr_real = da3_bb.forward(rgb_real_t[None, None])
            pr_real.require_convention("range")
            d_real = pr_real.depth[0].numpy()
            pr_syn = da3_bb.forward(rgb_syn_t[None, None])
            pr_syn.require_convention("range")
            d_syn = pr_syn.depth[0].numpy()

        row = {"frame_id": str(t), "photometric_diff_static": d_stat,
              "photometric_diff_dynamic": d_dyn}
        for tag, valid_extra in (("static", ~dyn_r), ("dynamic", dyn_r)):
            valid = cone & valid_extra & (gr_with <= 10.0) & (gr_with > 0)
            for label, d in (("real", d_real), ("synthetic", d_syn)):
                v = valid & (d > 1e-6)
                if v.sum() < 30:
                    row[f"absrel_{tag}_{label}"] = None
                    continue
                aligned = align_depth(d, gr_with, v, mode="scale_shift")
                row[f"absrel_{tag}_{label}"] = float(np.median(
                    np.abs(aligned - gr_with)[v] / gr_with[v]))
        da3_rows.append(row)

        if len(grid_examples) < 3:
            grid_examples.append({
                "frame_id": str(t), "rgb_real": rgb_real, "rgb_syn": rgb_syn,
                "dyn_mask": dyn_r, "depth_with": zw, "depth_without": zwo,
            })

    if grid_examples:
        save_visual_grid(grid_examples, os.path.join(out_dir, f"{seq_name}_grid.png"))

    prov_diffs = [r["median_abs_diff_m"] for r in provenance_rows]
    frac_differ = [r["frac_differs_gt_5cm"] for r in provenance_rows]
    return {
        "seq": seq_name,
        "n_frames_scored_provenance": len(provenance_rows),
        "n_frames_scored_da3": len(da3_rows),
        "provenance": {
            "median_abs_diff_with_vs_without_m": _median(prov_diffs) if prov_diffs else None,
            "median_frac_differs_gt_5cm": _median(frac_differ) if frac_differ else None,
            "median_depth_with_at_dyn_m": _median(
                [r["median_with_m"] for r in provenance_rows]) if provenance_rows else None,
            "median_depth_without_at_dyn_m": _median(
                [r["median_without_m"] for r in provenance_rows]) if provenance_rows else None,
        },
        "provenance_rows": provenance_rows,
        "da3_rows": da3_rows,
        "da3_summary": {
            k: _median([r[k] for r in da3_rows if r.get(k) is not None])
            for k in ("absrel_static_real", "absrel_static_synthetic",
                     "absrel_dynamic_real", "absrel_dynamic_synthetic")
        },
    }


def save_visual_grid(examples: List[Dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(examples)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[None, :]
    cols = ["real", "synthetic", "dyn-seg mask", "depth (with-skel)", "depth (without-skel)"]
    for i, ex in enumerate(examples):
        axes[i, 0].imshow(ex["rgb_real"])
        axes[i, 1].imshow(ex["rgb_syn"])
        axes[i, 2].imshow(ex["dyn_mask"], cmap="gray")
        axes[i, 3].imshow(ex["depth_with"], cmap="turbo", vmin=0, vmax=8)
        axes[i, 4].imshow(ex["depth_without"], cmap="turbo", vmin=0, vmax=8)
        for j, c in enumerate(cols):
            axes[i, j].set_title(f"{ex['frame_id']}: {c}" if j == 0 else c, fontsize=9)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    device = args.device
    from raytun3r.backbones import build_backbone
    da3_bb = build_backbone("da3", weights="pretrained", device=device, variant="small")
    da3_bb.install(None, camera_for(SIZE), (SIZE, SIZE),
                  patch_undistort=False, border_token=False, dpt_grid=False,
                  depth_convention="range")

    os.makedirs(args.out_dir, exist_ok=True)
    for seq_dir in args.seq_dirs:
        result = process_sequence(seq_dir, args.out_dir, da3_bb, device)
        out_path = os.path.join(args.out_dir, f"{result['seq']}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{result['seq']}] wrote {out_path}")
        print(json.dumps(result["provenance"], indent=2))
        print(json.dumps(result["da3_summary"], indent=2))


if __name__ == "__main__":
    main()
