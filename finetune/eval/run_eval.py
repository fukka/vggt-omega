# Copyright (c) 2026.
"""CLI runner for depth evaluation: VGGT-Omega and DAv2, pretrained vs finetuned.

Evaluates up to four model variants side-by-side:

  vggt_pretrained   VGGT-Omega base weights (metric depth; 'none' alignment meaningful)
  vggt_finetuned    base + LoRA from --finetune-checkpoint's "vggt" key
  dav2_pretrained   Depth-Anything-V2 base weights (affine-invariant; scale_shift)
  dav2_finetuned    DAv2 weights from --finetune-checkpoint's "dav2" key

Finetuned variants are only added when --finetune-checkpoint is given.

Alignment modes per model type
-------------------------------
  VGGT  → none, scale_only, scale_shift      (none = metric, the key diagnostic)
  DAv2  → scale_shift, disparity_scale_shift  (relative model; 'none' is meaningless)

Eval sources
------------
  ADT (dense GT) :  --eval-adt-root  /path/to/ADT
                    auto-picks Apartment_release_clean_seq131_M1292
  MPS (sparse GT):  --eval-mps-frame-dir  /path/to/aria01_214-1
                    --eval-mps-traj-csv   .../mps/slam/closed_loop_trajectory.csv
                    --eval-mps-points-gz  .../mps/slam/semidense_points.csv.gz

Usage examples
--------------
  # ADT only (dense GT, all 4 variants):
  python -m finetune.eval.run_eval \\
    --vggt-checkpoint /path/to/model.pt \\
    --finetune-checkpoint finetune_outputs/checkpoint_best.pt \\
    --eval-adt-root /path/to/ADT \\
    --out-dir eval_out/

  # MPS only (sparse, in-domain Ego-Exo4D), pretrained only:
  python -m finetune.eval.run_eval \\
    --vggt-checkpoint /path/to/model.pt \\
    --eval-mps-frame-dir /path/to/take/aria01_214-1 \\
    --eval-mps-traj-csv .../closed_loop_trajectory.csv \\
    --eval-mps-points-gz .../semidense_points.csv.gz \\
    --out-dir eval_out/

Outputs
-------
  eval_out/
    eval_results.json      full nested metrics dict
    eval_summary.txt       human-readable comparison table (key output)
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    __package__ = "finetune.eval"

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# predict_fn factories
# Normalise VGGT (dict output) and DAv2 (tensor output) to the same interface:
#   predict_fn(images [B,S,3,H,W] on device) → (depth_np [B,S,H,W], pose_np or None)
# --------------------------------------------------------------------------- #

def make_vggt_predict(model: torch.nn.Module, device: torch.device) -> Callable:
    """Return a predict_fn for a VGGT-Omega model."""
    model.eval()

    @torch.no_grad()
    def predict(images: torch.Tensor):
        images = images.to(device)
        preds = model(images)
        depth = preds["depth"]
        if depth.ndim == 5:
            depth = depth.squeeze(-1)          # [B,S,H,W,1] → [B,S,H,W]
        depth_np = depth.float().cpu().numpy()
        pose_np = (preds["pose_enc"].float().cpu().numpy()
                   if "pose_enc" in preds else None)
        return depth_np, pose_np

    return predict


def make_dav2_predict(model: torch.nn.Module, device: torch.device) -> Callable:
    """Return a predict_fn for a DAv2 model (no pose output)."""
    model.eval()

    @torch.no_grad()
    def predict(images: torch.Tensor):
        images = images.to(device)
        depth = model(images)                   # [B,S,H,W]
        return depth.float().cpu().numpy(), None

    return predict


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def _load_vggt_base(vggt_checkpoint: str, device: torch.device) -> torch.nn.Module:
    from vggt_omega.models import VGGTOmega
    print(f"[eval] loading VGGT-Omega from {vggt_checkpoint}")
    model = VGGTOmega()
    sd = torch.load(vggt_checkpoint, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eval]   base weights loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    return model.to(device).eval()


def _apply_vggt_finetune(
    base_model: torch.nn.Module,
    finetune_checkpoint: str,
    lora_rank: int,
    lora_alpha: int,
) -> torch.nn.Module:
    """Return a NEW model (separate copy) with LoRA + finetuned weights applied."""
    import copy
    from finetune.models import apply_lora

    # Deep-copy the base so pretrained and finetuned variants are independent.
    model = copy.deepcopy(base_model)
    n = apply_lora(model, r=lora_rank, alpha=lora_alpha, dropout=0.0)
    print(f"[eval]   LoRA applied to {n} VGGT layers")
    ckpt = torch.load(finetune_checkpoint, map_location="cpu")
    sd = ckpt.get("vggt", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eval]   VGGT finetune weights loaded "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model.eval()


def _load_dav2_base(model_name: str, device: torch.device) -> torch.nn.Module:
    from finetune.models.depth_anything import build_depth_anything
    print(f"[eval] loading DAv2 base: {model_name}")
    model = build_depth_anything(use_dummy=False, model_name=model_name)
    return model.to(device).eval()


def _apply_dav2_finetune(
    base_model: torch.nn.Module,
    finetune_checkpoint: str,
    lora_rank: int,
    lora_alpha: int,
    finetune_dav2_lora_only: bool,
) -> torch.nn.Module:
    """Return a NEW DAv2 model with finetuned weights applied."""
    import copy
    from finetune.models import apply_lora

    model = copy.deepcopy(base_model)
    ckpt = torch.load(finetune_checkpoint, map_location="cpu")
    dav2_sd = ckpt.get("dav2", {})

    if not dav2_sd:
        print("[eval]   WARNING: 'dav2' key missing from checkpoint — "
              "DAv2 finetuned variant will equal pretrained")
        return model.eval()

    if finetune_dav2_lora_only:
        n = apply_lora(model, r=lora_rank, alpha=lora_alpha, dropout=0.0)
        print(f"[eval]   LoRA applied to {n} DAv2 layers")

    missing, unexpected = model.load_state_dict(dav2_sd, strict=False)
    print(f"[eval]   DAv2 finetune weights loaded "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model.eval()


# --------------------------------------------------------------------------- #
# Result serialisation and comparison table
# --------------------------------------------------------------------------- #

# Metrics shown in the comparison table, in order.
_TABLE_METRICS = ["AbsRel", "SqRel", "RMSE", "delta1", "delta2", "scale_ratio", "n_frames"]
_PERCENT_KEYS  = {"delta1", "delta2", "delta3"}


def _fmt(v, key: str) -> str:
    if not isinstance(v, (int, float)) or v != v:  # NaN
        return "  N/A  "
    if key in _PERCENT_KEYS:
        return f"{v*100:5.1f}%"
    if key == "n_frames":
        return f"{int(v):6d}"
    return f"{v:7.4f}"


def _print_comparison_table(
    results: dict,
    source: str,
    align_modes: List[str],
    variant_order: List[str],
) -> str:
    """Return a formatted comparison table string and print it."""
    lines = []
    lines.append(f"\n{'='*78}")
    lines.append(f"  {source.upper()} — depth evaluation")
    lines.append(f"{'='*78}")

    for mode in align_modes:
        lines.append(f"\n  alignment = {mode}")
        header = f"  {'Model':22s}" + "".join(f"  {k:>10s}" for k in _TABLE_METRICS)
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for key in variant_order:
            if key not in results:
                continue
            mode_data = results[key].get(mode, {})
            row = f"  {key:22s}"
            for k in _TABLE_METRICS:
                row += f"  {_fmt(mode_data.get(k, float('nan')), k):>10s}"
            lines.append(row)

    # Pose ATE (only for VGGT variants that returned pose_enc)
    pose_rows = [(k, results[k]["pose"]) for k in variant_order
                 if k in results and "pose" in results[k]]
    if pose_rows:
        lines.append(f"\n  Pose ATE (Sim3-aligned, metres)")
        lines.append(f"  {'Model':22s}  {'ATE':>8s}  {'RPE_t':>8s}  {'scale':>8s}  {'n':>6s}")
        lines.append("  " + "-" * 60)
        for key, pr in pose_rows:
            lines.append(
                f"  {key:22s}  {pr.get('ATE', float('nan')):8.4f}  "
                f"{pr.get('RPE_t', float('nan')):8.4f}  "
                f"{pr.get('scale', float('nan')):8.4f}  "
                f"{pr.get('n', 0):6d}"
            )

    text = "\n".join(lines)
    print(text)
    return text


def _save_results(results: dict, out_dir: str, table_text: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] full results → {json_path}")

    txt_path = os.path.join(out_dir, "eval_summary.txt")
    with open(txt_path, "w") as f:
        f.write(table_text)
    print(f"[eval] summary table → {txt_path}")


# --------------------------------------------------------------------------- #
# ADT helpers
# --------------------------------------------------------------------------- #

_DEFAULT_ADT_SEQ = "Apartment_release_clean_seq131_M1292"


def _find_adt_seq_dirs(adt_root: str) -> List[str]:
    if not os.path.isdir(adt_root):
        return []
    default_seq = os.path.join(adt_root, _DEFAULT_ADT_SEQ)
    if (os.path.isdir(os.path.join(default_seq, "videos_rgb")) and
            os.path.isdir(os.path.join(default_seq, "depth_npy"))):
        return [default_seq]
    seq_dirs = []
    for name in sorted(os.listdir(adt_root)):
        seq_dir = os.path.join(adt_root, name)
        if (os.path.isdir(os.path.join(seq_dir, "videos_rgb")) and
                os.path.isdir(os.path.join(seq_dir, "depth_npy"))):
            seq_dirs.append(seq_dir)
    return seq_dirs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate VGGT-Omega and DAv2 depth quality (pretrained vs finetuned)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Checkpoints ────────────────────────────────────────────────────────
    p.add_argument("--vggt-checkpoint", required=True,
                   help="Base VGGT-Omega model.pt")
    p.add_argument("--finetune-checkpoint", default=None,
                   help="Finetune checkpoint (contains 'vggt' and 'dav2' keys). "
                        "If omitted, only pretrained variants are evaluated.")
    p.add_argument("--dav2-model-name",
                   default="depth-anything/Depth-Anything-V2-Small-hf",
                   help="HuggingFace model name for DAv2 base weights")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--finetune-dav2-lora-only", action="store_true",
                   help="Set if DAv2 was finetuned with LoRA only (not full finetune)")
    p.add_argument("--no-dav2", action="store_true",
                   help="Skip DAv2 evaluation entirely")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ── Inference ──────────────────────────────────────────────────────────
    p.add_argument("--seq-len", type=int, default=8,
                   help="VGGT window size (DAv2 uses the same window, scores center frame)")
    p.add_argument("--image-resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)

    # ── ADT eval ───────────────────────────────────────────────────────────
    p.add_argument("--eval-adt-root", default=None,
                   help=f"ADT root; auto-selects {_DEFAULT_ADT_SEQ}")
    p.add_argument("--eval-adt-seq-dirs", nargs="*", default=None,
                   help="Explicit list of ADT sequence dirs (overrides --eval-adt-root)")
    p.add_argument("--adt-depth-max", type=float, default=10.0)
    p.add_argument("--adt-gt-traj-csv", default=None,
                   help="ADT groundtruth/aria_trajectory.csv for pose ATE (VGGT only)")

    # ── MPS eval ───────────────────────────────────────────────────────────
    p.add_argument("--eval-mps-frame-dir", default=None)
    p.add_argument("--eval-mps-traj-csv", default=None)
    p.add_argument("--eval-mps-points-gz", default=None)
    p.add_argument("--eval-mps-calib", default=None)
    p.add_argument("--eval-mps-fx", type=float, default=None)
    p.add_argument("--eval-mps-fy", type=float, default=None)
    p.add_argument("--eval-mps-cx", type=float, default=None)
    p.add_argument("--eval-mps-cy", type=float, default=None)
    p.add_argument("--eval-mps-T-device-cam", default=None,
                   help="4×4 .npy with T_device_cam (inverted internally)")
    p.add_argument("--eval-mps-quality-min", type=float, default=0.5)

    # ── Output ─────────────────────────────────────────────────────────────
    p.add_argument("--out-dir", default="eval_out")

    a = p.parse_args()
    device = torch.device(a.device)
    os.makedirs(a.out_dir, exist_ok=True)

    run_adt = bool(a.eval_adt_root or a.eval_adt_seq_dirs)
    run_mps = bool(a.eval_mps_frame_dir and a.eval_mps_traj_csv)
    if not run_adt and not run_mps:
        p.error("Provide --eval-adt-root or --eval-mps-frame-dir + --eval-mps-traj-csv")

    # ── Build model variants ────────────────────────────────────────────────
    # variant_order controls table row order; only populated keys are shown.
    variant_order = ["vggt_pretrained", "vggt_finetuned",
                     "dav2_pretrained",  "dav2_finetuned"]

    # Each entry: (label, predict_fn, vggt_align_modes, dav2_align_modes)
    # We use a single dict keyed by variant name.
    variants: Dict[str, dict] = {}

    # VGGT pretrained
    vggt_base = _load_vggt_base(a.vggt_checkpoint, device)
    variants["vggt_pretrained"] = {
        "label":       "VGGT pretrained",
        "predict_fn":  make_vggt_predict(vggt_base, device),
        "align_modes": ("none", "scale_only", "scale_shift"),
        "with_pose":   True,
    }

    # VGGT finetuned
    if a.finetune_checkpoint:
        vggt_ft = _apply_vggt_finetune(
            vggt_base, a.finetune_checkpoint, a.lora_rank, a.lora_alpha
        )
        variants["vggt_finetuned"] = {
            "label":       "VGGT finetuned",
            "predict_fn":  make_vggt_predict(vggt_ft, device),
            "align_modes": ("none", "scale_only", "scale_shift"),
            "with_pose":   True,
        }

    if not a.no_dav2:
        # DAv2 pretrained
        dav2_base = _load_dav2_base(a.dav2_model_name, device)
        variants["dav2_pretrained"] = {
            "label":       "DAv2 pretrained",
            "predict_fn":  make_dav2_predict(dav2_base, device),
            # 'none' is meaningless for DAv2 (affine-invariant)
            "align_modes": ("scale_shift", "disparity_scale_shift"),
            "with_pose":   False,
        }

        # DAv2 finetuned
        if a.finetune_checkpoint:
            dav2_ft = _apply_dav2_finetune(
                dav2_base, a.finetune_checkpoint,
                a.lora_rank, a.lora_alpha, a.finetune_dav2_lora_only,
            )
            variants["dav2_finetuned"] = {
                "label":       "DAv2 finetuned",
                "predict_fn":  make_dav2_predict(dav2_ft, device),
                "align_modes": ("scale_shift", "disparity_scale_shift"),
                "with_pose":   False,
            }

    print(f"\n[eval] Evaluating {len(variants)} variant(s): {list(variants)}")

    all_results: dict = {}
    all_table_lines: List[str] = []

    # ── ADT eval ─────────────────────────────────────────────────────────────
    if run_adt:
        from .adt_depth import run_adt_eval

        seq_dirs = list(a.eval_adt_seq_dirs or [])
        if a.eval_adt_root:
            seq_dirs += _find_adt_seq_dirs(a.eval_adt_root)
        if not seq_dirs:
            print("[eval] WARNING: no ADT sequence dirs found; skipping ADT eval.")
        else:
            print(f"\n[eval] === ADT evaluation ({len(seq_dirs)} sequence(s)) ===")
            adt_results: dict = {}
            for var_key, var in variants.items():
                print(f"\n[eval] --- {var['label']} ---")
                adt_results[var_key] = run_adt_eval(
                    predict_fn=var["predict_fn"],
                    label=var["label"],
                    seq_dirs=seq_dirs,
                    device=device,
                    seq_len=a.seq_len,
                    image_resolution=a.image_resolution,
                    batch_size=a.batch_size,
                    depth_max_m=a.adt_depth_max,
                    align_modes=var["align_modes"],
                    gt_traj_csv=a.adt_gt_traj_csv if var["with_pose"] else None,
                )
            all_results["adt"] = adt_results

            # Collect all alignment modes that appear in any variant
            all_modes = []
            for var in variants.values():
                for m in var["align_modes"]:
                    if m not in all_modes:
                        all_modes.append(m)
            table = _print_comparison_table(adt_results, "ADT (dense GT)", all_modes, variant_order)
            all_table_lines.append(table)

    # ── MPS eval ─────────────────────────────────────────────────────────────
    if run_mps:
        from .mps_depth import MPSBundle, run_mps_eval
        from .mps_depth import _invert_se3

        print(f"\n[eval] === MPS evaluation ===")
        bundle = MPSBundle(
            traj_csv=a.eval_mps_traj_csv,
            points_csv_gz=a.eval_mps_points_gz,
            calib_jsonl=a.eval_mps_calib,
            quality_min=a.eval_mps_quality_min,
        )

        K: Optional[np.ndarray] = None
        if all(x is not None for x in [a.eval_mps_fx, a.eval_mps_fy,
                                         a.eval_mps_cx, a.eval_mps_cy]):
            K = np.array([
                [a.eval_mps_fx, 0.0, a.eval_mps_cx],
                [0.0, a.eval_mps_fy, a.eval_mps_cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

        T_cam_device: Optional[np.ndarray] = None
        if a.eval_mps_T_device_cam is not None:
            T_cam_device = _invert_se3(np.load(a.eval_mps_T_device_cam).astype(np.float64))

        mps_results: dict = {}
        for var_key, var in variants.items():
            print(f"\n[eval] --- {var['label']} ---")
            # MPS eval: only metric-adjacent modes (skip disparity for sparse sparse GT)
            mps_modes = tuple(m for m in var["align_modes"]
                              if m in ("none", "scale_only", "scale_shift"))
            mps_results[var_key] = run_mps_eval(
                predict_fn=var["predict_fn"],
                label=var["label"],
                frame_dir=a.eval_mps_frame_dir,
                bundle=bundle,
                device=device,
                K=K,
                T_cam_device=T_cam_device,
                seq_len=a.seq_len,
                image_resolution=a.image_resolution,
                align_modes=mps_modes,
                gt_traj_for_ate=var["with_pose"],
            )
        all_results["mps"] = mps_results

        all_mps_modes = []
        for var in variants.values():
            for m in var["align_modes"]:
                if m not in all_mps_modes and m in ("none", "scale_only", "scale_shift"):
                    all_mps_modes.append(m)
        table = _print_comparison_table(mps_results, "MPS (sparse GT)", all_mps_modes, variant_order)
        all_table_lines.append(table)

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_results(all_results, a.out_dir, "\n".join(all_table_lines))
    print(f"\n[eval] Done. Results in {a.out_dir}/")


if __name__ == "__main__":
    main()
