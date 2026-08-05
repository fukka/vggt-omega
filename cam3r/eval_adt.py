# Copyright (c) 2026.
"""Evaluate CAM3R on ADT -- the paper's two-view protocol (Table 1).

Reports RRA@t / RTA@t and mAA@30 over pairs selected by the paper's window
(0.35-1.75 m baseline, 25-65 deg viewing-angle change), plus two diagnostics the
paper does not tabulate but which localize *where* an error comes from:

* ``ray_deg`` -- mean angular error of the Ray Module's field against the Aria
  KB4 ground truth.  Isolates camera-geometry error from scene error.
* ``AbsRel``/``delta1`` on radial distance after median scale alignment.

With ``--multi-view`` it additionally runs Ray-Aware Global Alignment over the
frame graph and reports ATE RMSE.

    python -m cam3r.eval_adt --adt-root /path/to/adt --resolution 256

Without ``--weights`` the model is randomly initialized and the numbers are
chance-level by construction; the report says so rather than presenting them as
a result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.eval.metrics import align_depth, depth_metrics

from cam3r.adt import ADTPairDataset, find_adt_sequences
from cam3r.alignment import PairwiseEdge, prune_scene_graph, ray_aware_global_alignment
from cam3r.cli import add_adt_args, add_model_args, build_model as _build, require_adt_root
from cam3r.geometry import se3_inverse
from cam3r.metrics import ate_rmse, mean_average_accuracy, pose_accuracy, relative_pose_error
from cam3r.model import CAM3R
from cam3r.pretrained import initialize_cam3r

# Scoring mode for the radial-distance diagnostic. CONTEXT.md is explicit that
# `finetune/eval/metrics.py` is the single depth protocol and there is no second
# copy, so the alignment and the metrics both come from there.
DEPTH_ALIGN_MODE = "scale_only"


def build_model(args: argparse.Namespace, device: torch.device) -> CAM3R:
    model = _build(args)
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        print(f"  loaded CAM3R weights from {args.weights}")
    else:
        initialize_cam3r(model, dust3r=args.dust3r, unik3d=args.unik3d)
    return model.to(device).eval()


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict:
    device = torch.device(args.device)
    adt_root = require_adt_root(args)
    seq_dirs = [os.path.join(adt_root, s) for s in args.seq] if args.seq \
        else find_adt_sequences(adt_root, rgb_subdir=args.rgb_subdir)
    if not seq_dirs:
        raise SystemExit(f"no ADT sequences with {args.rgb_subdir}/ + depth_npy/ under {adt_root}")

    dataset = ADTPairDataset(
        seq_dirs,
        resolution=args.resolution,
        patch_size=args.patch_size,
        max_frames=args.max_frames,
        rgb_subdir=args.rgb_subdir,
        baseline_m=(args.min_baseline, args.max_baseline),
        angle_deg=(args.min_angle, args.max_angle),
        max_pairs=args.max_pairs,
        extrinsics_json=args.extrinsics_json,
    )
    print(f"  {len(dataset.poses_cw)} frames -> {len(dataset)} pairs "
          f"(baseline {args.min_baseline}-{args.max_baseline} m, "
          f"angle {args.min_angle}-{args.max_angle} deg)")
    print(f"  extrinsics: {dataset.extrinsics_source} (exact={dataset.extrinsics_exact})")
    if not dataset.extrinsics_exact:
        print("  WARNING: no camera calibration found; GT poses are in the DEVICE frame. "
              "Rotation error is unaffected but translation direction carries a lever-arm bias.")

    model = build_model(args, device)

    rot_errs: List[torch.Tensor] = []
    tra_errs: List[torch.Tensor] = []
    ray_degs: List[float] = []
    absrels: List[float] = []
    delta1s: List[float] = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        img1 = item["images"][0].unsqueeze(0).to(device)
        img2 = item["images"][1].unsqueeze(0).to(device)
        out = model(img1, img2)

        rot, tra = relative_pose_error(
            out["R"].cpu(), out["t_dir"].cpu(), item["R"].unsqueeze(0), item["t"].unsqueeze(0)
        )
        rot_errs.append(rot)
        tra_errs.append(tra)

        for v in (0, 1):
            valid = item["valid"][v].to(device)
            if not bool(valid.any()):
                continue
            dot = (out["rays"][v][0] * item["rays"][v].to(device)).sum(0).clamp(-1, 1)
            ray_degs.append(float(torch.rad2deg(torch.arccos(dot))[valid].mean()))

            gt_r = item["points"][v].norm(dim=0).cpu().numpy()
            pred_r = out["radial"][v][0].cpu().numpy()
            mask = (item["valid"][v].cpu().numpy() & (gt_r > 0))
            if not mask.any():
                continue
            aligned = align_depth(pred_r, gt_r, mask, mode=DEPTH_ALIGN_MODE)
            m = depth_metrics(aligned, gt_r, mask, max_depth=args.metric_max_depth)
            if m["n_valid"]:
                absrels.append(m["AbsRel"])
                delta1s.append(m["delta1"])

        if args.verbose and (idx + 1) % 10 == 0:
            print(f"    pair {idx + 1}/{len(dataset)}")

    rot = torch.cat(rot_errs)
    tra = torch.cat(tra_errs)
    results: Dict = {
        "n_pairs": len(dataset),
        "n_frames": len(dataset.poses_cw),
        "extrinsics_source": dataset.extrinsics_source,
        "extrinsics_exact": dataset.extrinsics_exact,
        "trained": bool(args.weights),
        "resolution": args.resolution,
        "depth_align_mode": DEPTH_ALIGN_MODE,
        **{k: round(v, 4) for k, v in pose_accuracy(rot, tra, thresholds=(5, 15, 30)).items()},
        "mAA@30": round(mean_average_accuracy(rot, tra, 30), 4),
        "rot_err_deg_median": round(float(rot.median()), 3),
        "tra_err_deg_median": round(float(tra.median()), 3),
    }
    if ray_degs:
        results["ray_err_deg_mean"] = round(sum(ray_degs) / len(ray_degs), 3)
    if absrels:
        results["radial_AbsRel"] = round(sum(absrels) / len(absrels), 4)
        results["radial_delta1"] = round(sum(delta1s) / len(delta1s), 4)

    if args.multi_view:
        results["multi_view"] = _multi_view(dataset, model, device, args)
    return results


@torch.no_grad()
def _multi_view(dataset, model, device, args) -> Dict:
    """Run RAGA over the pair graph and score the recovered trajectory."""
    frames = sorted({i for pair in dataset.pairs for i in pair})
    remap = {f: k for k, f in enumerate(frames)}
    edges: List[PairwiseEdge] = []
    overlaps: List[float] = []
    n_attempted = 0

    limit = args.max_pairs or len(dataset.pairs)
    for pair_idx, (i, j) in enumerate(dataset.pairs[:limit]):
        item = dataset[pair_idx]
        out_p = model(item["images"][0].unsqueeze(0).to(device), item["images"][1].unsqueeze(0).to(device))
        R = out_p["R"][0].cpu().double()
        t = out_p["t_dir"][0].cpu().double()
        s = out_p["scale"][0].cpu().double()

        stride = max(1, args.resolution // 32)
        pts1 = out_p["points"][0][0, :, ::stride, ::stride].reshape(3, -1).T.cpu().double()
        pts2 = out_p["points"][1][0, :, ::stride, ::stride].reshape(3, -1).T.cpu().double()
        conf_i = out_p["conf"][0][0, ::stride, ::stride].reshape(-1).cpu().double()
        conf_j = out_p["conf"][1][0, ::stride, ::stride].reshape(-1).cpu().double()
        # Eq. 4: view 2's points expressed in frame 1.
        pts2_in_1 = pts2 @ R.T + (s * t)
        edge = PairwiseEdge.from_pointmaps(
            remap[i], remap[j], pts1, pts2_in_1, pts2, conf_i=conf_i, conf_j=conf_j, R=R, t=s * t,
            radius=args.overlap_radius,
        )
        n_attempted += 1
        overlaps.append(edge.overlap or 0.0)
        if len(edge.points_i) >= args.min_correspondences:
            edges.append(edge)

    kept = prune_scene_graph(edges, min_overlap=args.min_overlap, overlap_radius=args.overlap_radius)
    out: Dict = {
        "edges_attempted": n_attempted,
        "edges_with_correspondences": len(edges),
        "edges_kept": len(kept),
        "median_overlap": round(float(torch.tensor(overlaps).median()), 4) if overlaps else 0.0,
    }
    if len(kept) < 1:
        out["note"] = (
            f"no edge survived: median MNN overlap {out['median_overlap']:.3f} at radius "
            f"{args.overlap_radius}. Expected for an untrained network, whose two views' "
            f"pointmaps are unrelated; raise --overlap-radius to exercise the path anyway."
        )
        return out

    res = ray_aware_global_alignment(len(frames), kept, iters=args.raga_iters)
    gt_positions = torch.stack([se3_inverse(dataset.poses_cw[f].unsqueeze(0))[0][:3, 3] for f in frames])
    out["ate_rmse_m"] = round(ate_rmse(res.poses[:, :3, 3], gt_positions), 4)
    out["raga_final_loss"] = float(res.best_loss)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_adt_args(p)
    add_model_args(p)
    p.add_argument("--seq", nargs="*", default=None, help="sequence dir names (default: auto-discover)")
    p.add_argument("--weights", default=None, help="trained CAM3R checkpoint")
    p.add_argument("--metric-max-depth", type=float, default=100.0,
                   help="exclude out-of-range predictions, as the reference depth eval does")

    p.add_argument("--multi-view", action="store_true", help="also run Ray-Aware Global Alignment")
    p.add_argument("--raga-iters", type=int, default=300)
    p.add_argument("--min-overlap", type=float, default=0.2)
    p.add_argument("--overlap-radius", type=float, default=0.1)
    p.add_argument("--min-correspondences", type=int, default=8,
                   help="drop an edge with fewer surviving MNN matches")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="write results JSON here")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    results = evaluate(args)

    print("\n=== CAM3R on ADT (two-view) ===")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:>22s}: {vv}")
        else:
            print(f"  {k:>22s}: {v}")
    if not results["trained"]:
        print("\n  NOTE: no --weights given, so these are UNTRAINED-network numbers "
              "(chance level), not a reproduction of the paper's 99.0 / 95.0.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
