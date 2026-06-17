# Copyright (c) 2026.
"""CLI runner for VGGT-Omega depth evaluation.

Supports two evaluation modes that can run independently or together:

ADT (dense GT, absolute accuracy)
----------------------------------
  python -m finetune.eval.run_eval \\
      --vggt-checkpoint /path/to/base/model.pt \\
      --finetune-checkpoint /path/to/checkpoint_best.pt \\
      --eval-adt-root /path/to/ADT \\
      --out-dir eval_out/

  Evaluates depth on ADT Blender-rendered pinhole frames with dense GT.
  Reports AbsRel/δ<1.25 under alignment modes: none (metric), scale_only,
  scale_shift. Also computes ATE if groundtruth/aria_trajectory.csv exists.

MPS (sparse GT, in-domain Ego-Exo4D)
--------------------------------------
  python -m finetune.eval.run_eval \\
      --vggt-checkpoint /path/to/base/model.pt \\
      --finetune-checkpoint /path/to/checkpoint_best.pt \\
      --eval-mps-frame-dir /path/to/take/aria01_214-1 \\
      --eval-mps-traj-csv  /path/to/take/mps/slam/closed_loop_trajectory.csv \\
      --eval-mps-points-gz /path/to/take/mps/slam/semidense_points.csv.gz \\
      --eval-mps-calib     /path/to/take/mps/online_calibration.jsonl \\
      --out-dir eval_out/

  Alternatively, pass explicit pinhole intrinsics instead of calib JSONL:
      --eval-mps-fx 282 --eval-mps-fy 282 --eval-mps-cx 256 --eval-mps-cy 256

Outputs
-------
  eval_out/
    eval_results.json          full metrics dict
    eval_summary.txt           human-readable table
    adt_comparison/<frame>.png 4-panel comparison figures (if --save-figures)
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
from typing import List, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def _load_vggt(
    vggt_checkpoint: str,
    finetune_checkpoint: Optional[str],
    lora_rank: int,
    lora_alpha: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load VGGT-Omega with optional finetuning LoRA weights."""
    # Import here so the module is importable without the vggt_omega package
    from vggt_omega.models import VGGTOmega

    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from finetune.models import apply_lora  # noqa: E402

    print(f"[eval] loading VGGT-Omega base from {vggt_checkpoint}")
    model = VGGTOmega()
    sd = torch.load(vggt_checkpoint, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eval]   base loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    if finetune_checkpoint is not None:
        print(f"[eval] applying finetune checkpoint: {finetune_checkpoint}")
        n = apply_lora(model, r=lora_rank, alpha=lora_alpha, dropout=0.0)
        print(f"[eval]   LoRA applied to {n} layers")
        ckpt = torch.load(finetune_checkpoint, map_location="cpu")
        vggt_sd = ckpt.get("vggt", ckpt)  # saved as {"vggt": ..., "dav2": ...}
        missing2, unexpected2 = model.load_state_dict(vggt_sd, strict=False)
        print(f"[eval]   finetune weights loaded "
              f"(missing={len(missing2)}, unexpected={len(unexpected2)})")

    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[eval] model ready on {device} ({n_params:.1f}M params)")
    return model


# --------------------------------------------------------------------------- #
# Result serialisation
# --------------------------------------------------------------------------- #

def _save_results(results: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # JSON (full)
    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] results saved → {json_path}")

    # Human-readable summary table
    txt_path = os.path.join(out_dir, "eval_summary.txt")
    with open(txt_path, "w") as f:
        for section, section_data in results.items():
            f.write(f"\n{'='*70}\n")
            f.write(f"  {section}\n")
            f.write(f"{'='*70}\n")
            if isinstance(section_data, dict):
                for mode, metrics in section_data.items():
                    if isinstance(metrics, dict):
                        f.write(f"\n  alignment={mode}\n")
                        for k, v in metrics.items():
                            if isinstance(v, float):
                                f.write(f"    {k:15s}: {v:.6f}\n")
                            else:
                                f.write(f"    {k:15s}: {v}\n")
    print(f"[eval] summary saved → {txt_path}")


# --------------------------------------------------------------------------- #
# ADT eval helpers
# --------------------------------------------------------------------------- #

# Default ADT sequence used for evaluation.
_DEFAULT_ADT_SEQ = "Apartment_release_clean_seq131_M1292"


def _find_adt_seq_dirs(adt_root: str) -> List[str]:
    """Return ADT sequence dirs that have real sensor data (videos_rgb/ + depth_npy/).

    Defaults to the single sequence Apartment_release_clean_seq131_M1292 when
    it exists; falls back to scanning all subdirectories.
    """
    if not os.path.isdir(adt_root):
        return []

    # Prefer the canonical eval sequence
    default_seq = os.path.join(adt_root, _DEFAULT_ADT_SEQ)
    if os.path.isdir(os.path.join(default_seq, "videos_rgb")) and \
       os.path.isdir(os.path.join(default_seq, "depth_npy")):
        return [default_seq]

    # Fall back: scan for any subdirectory with both videos_rgb and depth_npy
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
        description="Evaluate VGGT-Omega depth quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    p.add_argument("--vggt-checkpoint", required=True,
                   help="Path to base VGGT-Omega model.pt")
    p.add_argument("--finetune-checkpoint", default=None,
                   help="Path to finetune checkpoint_best.pt (optional; "
                        "omit to eval the unmodified base model)")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ── Inference ──────────────────────────────────────────────────────────
    p.add_argument("--seq-len", type=int, default=8, help="VGGT window size")
    p.add_argument("--image-resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)

    # ── ADT eval ───────────────────────────────────────────────────────────
    p.add_argument("--eval-adt-root", default=None,
                   help="Root of ADT download; auto-selects "
                        f"{_DEFAULT_ADT_SEQ} (videos_rgb/ + depth_npy/)")
    p.add_argument("--eval-adt-seq-dirs", nargs="*", default=None,
                   help="Explicit list of ADT sequence dirs (overrides --eval-adt-root)")
    p.add_argument("--adt-depth-max", type=float, default=10.0,
                   help="Max valid GT depth in metres for ADT")
    p.add_argument("--adt-gt-traj-csv", default=None,
                   help="Optional: ADT groundtruth/aria_trajectory.csv for pose ATE")

    # ── MPS eval ───────────────────────────────────────────────────────────
    p.add_argument("--eval-mps-frame-dir", default=None,
                   help="Directory of extracted egocentric RGB frames (e.g. aria01_214-1/)")
    p.add_argument("--eval-mps-traj-csv", default=None,
                   help="closed_loop_trajectory.csv from Aria MPS")
    p.add_argument("--eval-mps-points-gz", default=None,
                   help="semidense_points.csv.gz from Aria MPS (optional)")
    p.add_argument("--eval-mps-calib", default=None,
                   help="online_calibration.jsonl (optional; use --eval-mps-fx etc. instead)")
    p.add_argument("--eval-mps-fx", type=float, default=None,
                   help="Pinhole fx (pixels) — override calib JSONL")
    p.add_argument("--eval-mps-fy", type=float, default=None)
    p.add_argument("--eval-mps-cx", type=float, default=None)
    p.add_argument("--eval-mps-cy", type=float, default=None)
    p.add_argument("--eval-mps-T-device-cam", default=None,
                   help="Path to a 4×4 .npy with the device-to-RGB-camera transform "
                        "(T_device_cam; inverted internally). "
                        "Leave None to use identity (approx. when cam ≈ device frame).")
    p.add_argument("--eval-mps-quality-min", type=float, default=0.5,
                   help="Min MPS trajectory quality score (0–1)")
    p.add_argument("--eval-mps-timestamp-csv", default=None,
                   help="CSV with columns 'filename,timestamp_us' for frames whose "
                        "filenames don't embed timestamps")

    # ── Output ─────────────────────────────────────────────────────────────
    p.add_argument("--out-dir", default="eval_out")
    p.add_argument("--save-figures", action="store_true",
                   help="Save 4-panel comparison figures (requires matplotlib)")
    p.add_argument("--align-modes", nargs="*",
                   default=["none", "scale_only", "scale_shift"],
                   help="Alignment modes to report: none scale_only scale_shift "
                        "disparity_scale_shift")

    a = p.parse_args()
    device = torch.device(a.device)
    os.makedirs(a.out_dir, exist_ok=True)

    # ── Validate at least one eval target ──────────────────────────────────
    run_adt = bool(a.eval_adt_root or a.eval_adt_seq_dirs)
    run_mps = bool(a.eval_mps_frame_dir and a.eval_mps_traj_csv)
    if not run_adt and not run_mps:
        p.error("Provide at least one of: --eval-adt-root / --eval-mps-frame-dir + "
                "--eval-mps-traj-csv")

    # ── Load model ──────────────────────────────────────────────────────────
    vggt = _load_vggt(
        vggt_checkpoint=a.vggt_checkpoint,
        finetune_checkpoint=a.finetune_checkpoint,
        lora_rank=a.lora_rank,
        lora_alpha=a.lora_alpha,
        device=device,
    )

    results = {}

    # ── ADT eval ────────────────────────────────────────────────────────────
    if run_adt:
        from .adt_depth import run_adt_eval

        seq_dirs = list(a.eval_adt_seq_dirs or [])
        if a.eval_adt_root:
            seq_dirs += _find_adt_seq_dirs(a.eval_adt_root)
        if not seq_dirs:
            print("[eval] WARNING: --eval-adt-root found no sequence dirs with "
                  "videos_rgb/ + depth_npy/; skipping ADT.")
        else:
            print(f"\n[eval] === ADT evaluation ({len(seq_dirs)} sequences) ===")
            adt_metrics = run_adt_eval(
                vggt=vggt,
                seq_dirs=seq_dirs,
                device=device,
                seq_len=a.seq_len,
                image_resolution=a.image_resolution,
                batch_size=a.batch_size,
                depth_max_m=a.adt_depth_max,
                align_modes=tuple(a.align_modes),
                gt_traj_csv=a.adt_gt_traj_csv,
            )
            results["adt"] = adt_metrics

    # ── MPS eval ─────────────────────────────────────────────────────────────
    if run_mps:
        from .mps_depth import MPSBundle, run_mps_eval

        print(f"\n[eval] === MPS evaluation ===")
        bundle = MPSBundle(
            traj_csv=a.eval_mps_traj_csv,
            points_csv_gz=a.eval_mps_points_gz,
            calib_jsonl=a.eval_mps_calib,
            quality_min=a.eval_mps_quality_min,
        )

        # Build K from explicit args if provided
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
            from finetune.eval.mps_depth import _invert_se3
            T_device_cam = np.load(a.eval_mps_T_device_cam).astype(np.float64)
            T_cam_device = _invert_se3(T_device_cam)

        mps_metrics = run_mps_eval(
            vggt=vggt,
            frame_dir=a.eval_mps_frame_dir,
            bundle=bundle,
            device=device,
            K=K,
            T_cam_device=T_cam_device,
            seq_len=a.seq_len,
            image_resolution=a.image_resolution,
            align_modes=tuple(m for m in a.align_modes if m in ("none", "scale_only")),
        )
        results["mps"] = mps_metrics

    # ── Save ──────────────────────────────────────────────────────────────
    _save_results(results, a.out_dir)
    print(f"\n[eval] Done. Results in {a.out_dir}/")


if __name__ == "__main__":
    main()
