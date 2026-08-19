# Copyright (c) 2026.
"""Ticket 032 (issue #34) addendum: does #31's meal_seq131 pose anomaly
(random-matched mask beating the GT-dynamic mask by ~4 deg median) reproduce
when the synthetic RGB stream stands in for masking?

Reuses #31's ``ADTSkeletonSource`` (unmodified import, no edits to
``hands_pose_depth.py``) and its exact anchor-pair protocol, but with a 4th
"synthetic" condition: the raw synthetic frame for each pair member, no
mean-fill. This is orthogonal to that script's pinned vanilla/gt_dyn_masked/
random_matched numbers (already on `results/autoresearch-h4-pose/`) -- this
addendum writes to a separate path and does not touch them.

Usage (repo root, on the box):
    <venv>/bin/python autoresearch/experiments/h4-dynamics/code/hand_synthetic_pose_check.py \
        --seq-dir /path/to/Apartment_release_meal_skeleton_seq131_M1292 \
        --out results/autoresearch-h4-provenance/pose_synthetic_check_meal_seq131.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))

from raytun3r.metrics import rotation_error_deg          # noqa: E402
from rim_pose_value import _gain, _median                 # noqa: E402

from hands_pose_depth import (ADTSkeletonSource, quat_xyzw_to_matrix,  # noqa: E402
                              pick_anchors, RGB_STREAM_ID)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--calib", default=str(Path(__file__).resolve().parents[4]
                                           / "cam3r" / "data"
                                           / "adt_camera_rgb_calibration.json"))
    ap.add_argument("--n-anchors", type=int, default=41)
    ap.add_argument("--size", type=int, default=504)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    calib = json.load(open(args.calib))
    R_dc = quat_xyzw_to_matrix(calib["T_device_camera"]["quaternion_xyzw"])
    Ct = torch.tensor(R_dc.T, dtype=torch.float64)

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device="cpu", variant="small")

    src = ADTSkeletonSource(args.seq_dir, size=args.size)
    bb.install(None, src.camera, (src.camera.height, src.camera.width),
              patch_undistort=False, border_token=False, dpt_grid=False,
              depth_convention="range")
    anchors = pick_anchors(src.ts, args.n_anchors)
    print(f"[{src.name}] {len(src.ts)} frames, {len(anchors)} anchors")

    def synthetic_rgb(t: int):
        img = src.gt.get_synthetic_image_by_timestamp_ns(t, RGB_STREAM_ID)
        if not img.is_valid():
            return None
        rgb = src._resize(img.data().to_numpy_array().astype(np.float32) / 255.0,
                          nearest=False)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()

    conds = ["vanilla", "synthetic"]
    rot: Dict[str, List[float]] = {c: [] for c in conds}
    gts: List[float] = []
    pair_ids = [(anchors[k], anchors[k + 1]) for k in range(len(anchors) - 1)]
    for n, (i, j) in enumerate(pair_ids):
        fi, fj = src.frame(src.ts[i]), src.frame(src.ts[j])
        si, sj = synthetic_rgb(src.ts[i]), synthetic_rgb(src.ts[j])
        if fi is None or fj is None or si is None or sj is None:
            continue
        R_dev = torch.from_numpy(fj["R_dw"] @ fi["R_dw"].T).double()
        Rg = (Ct @ R_dev @ Ct.T).float()
        imgs_by_cond = {
            "vanilla": torch.stack([fi["rgb"], fj["rgb"]]),
            "synthetic": torch.stack([si, sj]),
        }
        with torch.no_grad():
            for c in conds:
                Rh = bb.forward(imgs_by_cond[c][None]).relative(0, 1)[0].to(Rg)
                rot[c].append(rotation_error_deg(Rh, Rg))
        gts.append(rotation_error_deg(torch.eye(3), Rg))
        print(f"  pair {n + 1}/{len(pair_ids)}: "
              + " ".join(f"{c} {rot[c][-1]:6.2f}" for c in conds), flush=True)

    summary = {"seq": src.name, "n_pairs": len(gts), "conds": {}}
    print(f"\n=== {src.name}: vanilla vs synthetic, {len(gts)} pairs ===")
    for c in conds:
        med = _median(rot[c])
        print(f"{c:>10} median_rot_err {med:7.3f}")
        summary["conds"][c] = {"median_rot_err_deg": med}
    if gts:
        d = np.asarray(rot["synthetic"]) - np.asarray(rot["vanilla"])
        summary["synthetic_minus_vanilla"] = {
            "median_deg": _median(list(d)), "n_synthetic_better": int((d < 0).sum())}
        print(f"synthetic - vanilla: median {_median(list(d)):+.3f} deg, "
              f"synthetic better on {(d < 0).sum()}/{len(d)}")
        print("Reference (#31, real-image conditions): vanilla 12.590 deg, "
              "gt_dyn_masked 12.932 deg, random_matched 8.688 deg -- the "
              "anomaly was random_matched beating both by ~4 deg.")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
