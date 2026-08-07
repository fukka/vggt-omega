"""Evaluation: Tab. 1/2 (R, t, d_reproj), Tab. 3 (AbsRel, delta_1.25), Tab. 4/8 ablations.

The adapter is fitted on a short segment and evaluated on the full sequence, on
consecutive image pairs, with the same pair sampling for every method.

    python -m raytun3r.eval \
        --backbone vggt --weights pretrained \
        --dataset scannetpp --path /data/scannetpp/data/<scene> \
        --adapter runs/rt3r/<scene>/adapter.pt \
        --methods vanilla,center_ph,multi_ph,raytun3r

Reported numbers are only comparable to the paper when the matcher is UFM;
whichever matcher ran is recorded in the output JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional

import torch

from .adapter import RayTun3RAdapter
from .backbones import build_backbone
from .baselines import CenterPH, MultiPH, attach_caltok, attach_lora
from .data import build_windows, load_sequence
from .matching import build_matcher
from .metrics import (aggregate, depth_metrics, pose_errors,
                      reprojection_depth_error, rotation_error_deg)

__all__ = ["evaluate", "main"]

METHODS = ["vanilla", "param_free", "raytun3r", "center_ph", "multi_ph", "lora", "caltok"]


@torch.no_grad()
def evaluate(runner, windows, camera, *, convention: str = "range",
             device="cpu", label: str = "", max_depth: Optional[float] = None
             ) -> Dict[str, float]:
    """Score one method over the evaluation windows.

    ``runner`` maps ``(S, 3, H, W)`` images to a :class:`~raytun3r.backbones.Prediction`.
    Pose and ``d_reproj`` need ground-truth pose and are skipped without it;
    ``AbsRel``/``delta_1.25`` need ground-truth depth.

    ``max_depth`` caps the ground truth before scoring. The paper states no cap,
    so the default is none; it exists because mesh-rendered depth carries rays
    that leave the room through windows and doorways, and whether those are in
    Tab. 3 is an unstated choice worth measuring rather than assuming.

    Only frame 0 of each window contributes to ``AbsRel``/``delta_1.25``. Window
    starts step by one, so that is a distinct frame per window; scoring all
    ``seq_len`` frames would count the overlap repeatedly.
    """
    valid = camera.valid_mask(*windows[0].images.shape[-2:], device=device)
    rows: List[Dict[str, float]] = []
    n_no_pose = n_no_depth = 0

    coverage_sum, n_cov = 0.0, 0
    for win in windows:
        pred = runner(win.images)
        # d_reproj (Eq. 16) backprojects, and AbsRel compares against ground truth
        # that data.py converted to this same convention.
        pred.require_convention(convention)
        row: Dict[str, float] = {}

        # A method that only predicts part of the frame (Center-PH, Multi-PH) is
        # scored on what it predicted; the rest is reported as coverage instead of
        # being silently charged as depth error. See Prediction.covered.
        def omega(i: int) -> torch.Tensor:
            return valid if pred.covered is None else (valid & pred.covered[i])

        if pred.covered is not None:
            coverage_sum += float((pred.covered & valid).sum()) / max(float(valid.sum()), 1.0) \
                / pred.covered.shape[0]
            n_cov += 1

        # Consecutive pairs only, matching the paper's protocol.
        for i in range(len(win.indices) - 1):
            j = i + 1
            gt = win.gt_relative(i, j)
            if gt is None:
                n_no_pose += 1
                continue
            R_gt, t_gt = gt
            R_hat, t_hat = pred.relative(i, j)
            row.update(pose_errors(R_hat, t_hat, R_gt, t_gt))

            # What a model that predicts *no rotation at all* would score. R_deg
            # is an absolute angular error, so its scale is set by how much
            # rotation there is to estimate -- which `--stride` changes directly.
            # Without this column an R_deg is uninterpretable on its own and not
            # comparable across strides or datasets. Skill is R_deg_identity /
            # R_deg; at or below 1.0 the method carries no information.
            row["R_deg_identity"] = rotation_error_deg(
                torch.eye(3, dtype=R_gt.dtype, device=R_gt.device), R_gt)

            # Both averagings, every run: `d_reproj` is Eq. 16 as written and is
            # the only one comparable to the paper's tables, while
            # `d_reproj_conf` is the confidence-weighted number we reported
            # before 2026-08-05. Carrying both keeps older runs interpretable and
            # measures the ratio between them on real data instead of guessing it.
            d = reprojection_depth_error(pred.depth[i], camera, win.matches[(i, j)],
                                         R_gt, t_gt, convention=convention,
                                         valid=omega(i), weighting="omega")
            if d is not None:
                row["d_reproj"] = d
            dc = reprojection_depth_error(pred.depth[i], camera, win.matches[(i, j)],
                                          R_gt, t_gt, convention=convention,
                                          valid=omega(i), weighting="confidence")
            if dc is not None:
                row["d_reproj_conf"] = dc

        if win.gt_depth is not None:
            dm = depth_metrics(pred.depth[0], win.gt_depth[0],
                               valid=omega(0) & win.gt_valid[0],
                               max_depth=max_depth)
            row["AbsRel"], row["delta_1.25"] = dm["AbsRel"], dm["delta_1.25"]
        else:
            n_no_depth += 1

        if row:
            rows.append(row)

    out = aggregate(rows)
    if n_no_pose:
        out["_pose_skipped"] = n_no_pose
    if n_no_depth:
        out["_depth_skipped"] = n_no_depth
    if n_cov:
        out["coverage"] = coverage_sum / n_cov
    if out.get("R_deg", 0) > 0 and "R_deg_identity" in out:
        out["R_skill"] = out["R_deg_identity"] / out["R_deg"]
    if label:
        keys = [k for k in ("R_deg", "R_deg_identity", "R_skill", "t_deg", "d_reproj",
                            "d_reproj_conf", "AbsRel", "delta_1.25", "coverage")
                if k in out]
        print(f"[eval] {label:12s} " + "  ".join(f"{k}={out[k]:.4f}" for k in keys)
              + (f"   (no GT pose in {n_no_pose} pairs)" if n_no_pose else ""))
    return out


def _make_runner(method: str, backbone, camera, args, adapter_state=None):
    """Build a callable for one method, and the teardown handles it needs.

    ``adapter_state`` is whatever ``--adapter`` held, and it is only ever
    meaningful for the one method it was fitted for -- the caller is responsible
    for passing ``None`` to the others. ``raytun3r``, ``lora`` and ``caltok``
    each store a differently-shaped state dict, so feeding one to another is a
    hard error rather than a silently untrained run.
    """
    handles = []
    if method in ("vanilla",):
        backbone.install(None, camera, (args._h, args._w),
                         patch_undistort=False, border_token=False, dpt_grid=False,
                         depth_convention=args.convention)
        return (lambda imgs: backbone.forward(imgs[None])), handles

    if method == "param_free":
        backbone.install(None, camera, (args._h, args._w), grid_mode=args.grid_mode,
                         depth_convention=args.convention)
        return (lambda imgs: backbone.forward(imgs[None])), handles

    if method == "raytun3r":
        adapter = backbone.make_adapter(n_radial=args.n_radial, n_angular=args.n_angular,
                                        n_rope_radial=args.n_rope_radial).to(args.device)
        backbone.install(adapter, camera, (args._h, args._w),
                         patch_undistort=not args.no_patch_undistort,
                         border_token=not args.no_border_token,
                         dpt_grid=not args.no_dpt_grid, grid_mode=args.grid_mode,
                         depth_convention=args.convention)
        if adapter_state:
            adapter.load_state_dict(adapter_state)
        else:
            print("[eval] WARNING: no --adapter given; raytun3r is running zero-init, "
                  "which is identical to param_free")
        return (lambda imgs: backbone.forward(imgs[None])), handles

    if method in ("center_ph", "multi_ph"):
        backbone.install(None, camera, (args._h, args._w),
                         patch_undistort=False, border_token=False, dpt_grid=False,
                         depth_convention=args.convention)
        ctor = CenterPH if method == "center_ph" else MultiPH
        base = ctor(backbone, camera, fov_deg=args.ph_fov, depth_convention=args.convention)
        return base, handles

    if method in ("lora", "caltok"):
        backbone.install(None, camera, (args._h, args._w),
                         patch_undistort=False, border_token=False, dpt_grid=False,
                         depth_convention=args.convention)
        if method == "lora":
            mods, handles = attach_lora(backbone, r=args.lora_r, alpha=args.lora_alpha)
        else:
            mods, handles = attach_caltok(backbone, n_tokens=args.caltok_t)
        if adapter_state:
            for i, m in enumerate(mods):
                m.load_state_dict(adapter_state[f"m{i}"])
        else:
            print(f"[eval] WARNING: no fitted checkpoint for {method!r}; it is running "
                  f"randomly/zero-initialised. Fit one with "
                  f"'python -m raytun3r.train --method {method}'.")
        return (lambda imgs: backbone.forward(imgs[None])), handles

    raise ValueError(f"unknown method {method!r}")


from .backbones import BACKBONE_NAMES  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("raytun3r.eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="vggt", choices=BACKBONE_NAMES)
    p.add_argument("--weights", default="pretrained")
    p.add_argument("--variant", default="small",
                   choices=["small", "base", "large", "giant"])
    p.add_argument("--dataset", default="scannetpp", choices=["scannetpp", "adt"])
    p.add_argument("--path", required=True)
    p.add_argument("--adapter", default=None, help="adapter.pt from raytun3r.train")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--methods", default="vanilla,center_ph,raytun3r",
                   help=f"comma-separated subset of {METHODS}")
    p.add_argument("--n-radial", type=int, default=20)
    p.add_argument("--n-angular", type=int, default=8)
    p.add_argument("--n-rope-radial", type=int, default=20)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--caltok-t", type=int, default=4)
    p.add_argument("--ph-fov", type=float, default=110.0)

    p.add_argument("--no-patch-undistort", action="store_true")
    p.add_argument("--no-border-token", action="store_true")
    p.add_argument("--no-dpt-grid", action="store_true")
    p.add_argument("--grid-mode", default="auto", choices=["auto", "tan", "angular"])

    p.add_argument("--matcher", default="auto", choices=["auto", "ufm", "raft", "sift"])
    p.add_argument("--windows", type=int, default=100)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-flow-px", type=float, default=2.0)
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--extrinsics-json", default=None)
    p.add_argument("--max-fov", type=float, default=None,
                   help="restrict Omega to this total FOV in degrees (never widens); "
                        "the knob for the paper's stated 115 deg vs ScanNet++'s real ~170")
    p.add_argument("--eval-min-flow", type=float, default=0.0,
                   help="static-pair filter at EVALUATION time. The paper applies "
                        "its 2 px filter to the adaptation set only and evaluates "
                        "on the full sequence, so this defaults to 0 (no filtering). "
                        "--min-flow-px still governs the adaptation set in train.py")
    p.add_argument("--keep-bad", action="store_true",
                   help="keep ScanNet++ frames the dataset flags is_bad; they are "
                        "dropped by default (143 of 896 on 3f15a9266d)")
    p.add_argument("--convention", default="range", choices=["range", "z"])
    p.add_argument("--max-depth", type=float, default=None,
                   help="cap ground-truth depth before AbsRel/delta_1.25 "
                        "(Tab. 3). The paper states no cap, so the default is "
                        "none; set it to measure the sensitivity.")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)

    backbone = build_backbone(args.backbone, weights=args.weights, device=args.device,
                              **({"variant": args.variant} if args.backbone == "da3" else {}))
    source = load_sequence(args.dataset, args.path, max_size=args.max_size,
                           patch=backbone.patch_size, max_frames=args.max_frames,
                           depth_convention=args.convention,
                           **({"extrinsics_json": args.extrinsics_json}
                              if args.dataset == "adt" else
                              {"keep_bad": args.keep_bad}))
    if args.max_fov is not None:
        before = 2 * math.degrees(source.camera.theta_max)
        source.camera = source.camera.with_max_fov(args.max_fov)
        after = 2 * math.degrees(source.camera.theta_max)
        print(f"[data] Omega restricted: {before:.0f} deg -> {after:.0f} deg FOV "
              f"(images unchanged; this narrows where the method is scored)")

    args._h, args._w = source.h, source.w

    matcher = build_matcher(args.matcher, device=args.device)
    # The paper's 2 px static filter applies to the *adaptation set* only -- "we
    # build a short adaptation set of 30 three-frame windows, filtering out nearly
    # static windows [...] and evaluate on the full sequence". Filtering at
    # evaluation time silently drops the easiest pairs and biases every metric
    # toward high-motion ones, so it is off here by default.
    windows = build_windows(source, matcher, n_windows=args.windows, seq_len=2,
                            stride=args.stride, min_flow_px=args.eval_min_flow,
                            with_pose_targets=False, with_gt=True, seed=args.seed,
                            device=args.device)

    state, state_method = None, None
    if args.adapter:
        blob = torch.load(args.adapter, map_location=args.device)
        state = blob.get("adapter") or blob.get("baseline_state")
        state_method = blob.get("method")
        print(f"[eval] loaded {args.adapter} (method={state_method}, "
              f"backbone={blob.get('backbone')})")
        if blob.get("backbone") != args.backbone:
            print(f"[eval] WARNING: adapter was fitted on {blob.get('backbone')!r} but "
                  f"evaluating {args.backbone!r}")

    results = {"_meta": {"matcher": matcher.name, "backbone": args.backbone,
                         "dataset": args.dataset, "scene": source.name,
                         "n_windows": len(windows), "convention": args.convention,
                         "gt_pose": windows[0].gt_R is not None,
                         "gt_depth": windows[0].gt_depth is not None}}
    if not results["_meta"]["gt_pose"]:
        print("[eval] no ground-truth camera pose available -> R/t/d_reproj skipped "
              "(depth metrics still reported)")

    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        # A checkpoint only belongs to the method it was fitted for. Handing a
        # raytun3r adapter to --methods lora used to raise a bare KeyError.
        own = state if (state is not None and state_method == method) else None
        runner, handles = _make_runner(method, backbone, source.camera, args, own)
        try:
            results[method] = evaluate(runner, windows, source.camera,
                                       convention=args.convention, device=args.device,
                                       label=method, max_depth=args.max_depth)
        finally:
            for h in handles:
                h.remove()
            backbone.remove()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[out] wrote {args.out}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
