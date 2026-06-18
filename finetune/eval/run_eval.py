# Copyright (c) 2026.
"""ADT depth evaluation for a finetuning run — driven by the run NAME alone.

Point this at an experiment under ``runs/`` and it reconstructs everything from
that run's ``config.yaml`` (the VGGT base checkpoint, LoRA rank/alpha, DAv2 model
name, image/seq sizing, and the ADT eval root), evaluates the **pretrained**
models plus the run's **best** and **last** finetune checkpoints, prints a
comparison table, and writes the results under ``eval_out/<name>/``.

    python -m finetune.eval.run_eval ssi_r8

is equivalent to the old long form (vggt-checkpoint + finetune-checkpoint + ADT
root + lora args), with both ``checkpoint_best.pt`` and ``checkpoint_last.pt``
evaluated in one pass.

Variants (pretrained loaded once; finetuned per available checkpoint)::

    vggt_pretrained   VGGT-Omega base weights         (metric; 'none' meaningful)
    vggt_best/last    base + LoRA from checkpoint_{best,last}.pt's "vggt" key
    dav2_pretrained   Depth-Anything-V2 base weights  (relative; scale_shift)
    dav2_best/last    DAv2 weights from checkpoint_{best,last}.pt's "dav2" key

Alignment modes: VGGT → none, scale_shift; DAv2 → scale_shift (relative model).

This file handles **ADT (dense GT)** only. The **MPS (sparse GT)** evaluation
lives in its own runner, :mod:`finetune.eval.mps_depth` (``python -m
finetune.eval.mps_depth <name> --mps-frame-dir ...``), because MPS needs paths
that are not stored in a run's config.

Outputs (``eval_out/<name>/``)::

    eval_results.json      full nested metrics dict
    eval_summary.txt       human-readable comparison table
    qual/<variant>/*.png   qualitative depth panels (when --n-qual > 0)
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    __package__ = "finetune.eval"

import argparse
import dataclasses
import json
import os
from typing import Callable, Dict, List, Optional

import torch


# --------------------------------------------------------------------------- #
# predict_fn factories
# Normalise VGGT (dict output) and DAv2 (tensor output) to the same interface:
#   predict_fn(images [B,S,3,H,W] on device) → (depth_np [B,S,H,W], pose_np or None)
# (Imported by the in-training eval in finetune/trainers/base.py — keep stable.)
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
    device: torch.device,
) -> torch.nn.Module:
    """Return a NEW model (separate copy) with LoRA + finetuned weights applied."""
    import copy
    from finetune.models import apply_lora

    # Deep-copy the base so pretrained and finetuned variants are independent.
    model = copy.deepcopy(base_model)
    # Match training: LoRA lives in the aggregator backbone only (the heads were
    # finetuned fully), so the checkpoint's keys line up exactly.
    n = apply_lora(model.aggregator, r=lora_rank, alpha=lora_alpha, dropout=0.0)
    print(f"[eval]   LoRA applied to {n} VGGT aggregator layers")
    ckpt = torch.load(finetune_checkpoint, map_location="cpu")
    sd = ckpt.get("vggt", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eval]   VGGT finetune weights loaded "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model.to(device).eval()


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
    device: torch.device,
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
        return model.to(device).eval()

    if finetune_dav2_lora_only:
        n = apply_lora(model, r=lora_rank, alpha=lora_alpha, dropout=0.0)
        print(f"[eval]   LoRA applied to {n} DAv2 layers")

    missing, unexpected = model.load_state_dict(dav2_sd, strict=False)
    print(f"[eval]   DAv2 finetune weights loaded "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Run resolution: name under runs/ → run dir + resolved config + checkpoints
# --------------------------------------------------------------------------- #

def resolve_run(name: str, runs_root: str = "runs") -> str:
    """Resolve a run NAME (or a path to a run dir) to its directory."""
    cand = name if os.path.isdir(name) else os.path.join(runs_root, name)
    if not os.path.isdir(cand):
        raise FileNotFoundError(
            f"run {name!r} not found (looked for {cand!r}). Pass the experiment "
            f"name under {runs_root}/ (e.g. 'ssi_r8'), or a path to a run dir."
        )
    return cand


def load_run_config(run_dir: str) -> Dict:
    """Return the run's resolved config as a dict (FinetuneConfig defaults filled in)."""
    from ..config import FinetuneConfig

    cfg = dataclasses.asdict(FinetuneConfig())
    path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(path):
        import yaml

        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in loaded.items() if k in cfg})
        print(f"[eval] loaded run config from {path}")
    else:
        print(f"[eval] WARNING: no config.yaml under {run_dir!r}; using FinetuneConfig defaults.")
    return cfg


def find_checkpoint(run_dir: str, tag: str) -> Optional[str]:
    """Resolve a checkpoint tag to a file. 'best'→checkpoint_best.pt;
    'last'→checkpoint_last.pt, falling back to checkpoint_final.pt."""
    candidates = {
        "best": ["checkpoint_best.pt"],
        "last": ["checkpoint_last.pt", "checkpoint_final.pt"],
        "final": ["checkpoint_final.pt"],
    }.get(tag, [f"checkpoint_{tag}.pt"])
    for fn in candidates:
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            return p
    return None


def build_variants(
    cfg: Dict,
    run_dir: str,
    device: torch.device,
    checkpoints=("best", "last"),
    include_dav2: bool = True,
) -> Dict[str, dict]:
    """Build the evaluation variants from a run's config + checkpoints.

    Returns an ordered dict ``variant_key -> {label, predict_fn, align_modes,
    with_pose}``. The pretrained base models are loaded once; one finetuned
    variant is added per checkpoint tag that exists in ``run_dir``.
    """
    lora_rank = int(cfg.get("lora_rank", 8))
    lora_alpha = int(cfg.get("lora_alpha", 16))
    dav2_name = cfg.get("dav2_model_name", "depth-anything/Depth-Anything-V2-Small-hf")
    dav2_lora_only = bool(cfg.get("finetune_dav2_lora_only", True))

    variants: Dict[str, dict] = {}

    # VGGT: pretrained (once) + one variant per checkpoint tag.
    vggt_base = _load_vggt_base(cfg.get("vggt_checkpoint", ""), device)
    variants["vggt_pretrained"] = {
        "label": "VGGT pretrained", "predict_fn": make_vggt_predict(vggt_base, device),
        "align_modes": ("none", "scale_shift"), "with_pose": True,
    }
    for tag in checkpoints:
        ckpt = find_checkpoint(run_dir, tag)
        if ckpt is None:
            print(f"[eval] no {tag} checkpoint in {run_dir!r}; skipping vggt_{tag}.")
            continue
        print(f"[eval] VGGT {tag}: {ckpt}")
        m = _apply_vggt_finetune(vggt_base, ckpt, lora_rank, lora_alpha, device)
        variants[f"vggt_{tag}"] = {
            "label": f"VGGT ({tag})", "predict_fn": make_vggt_predict(m, device),
            "align_modes": ("none", "scale_shift"), "with_pose": True,
        }

    if include_dav2:
        dav2_base = _load_dav2_base(dav2_name, device)
        variants["dav2_pretrained"] = {
            "label": "DAv2 pretrained", "predict_fn": make_dav2_predict(dav2_base, device),
            "align_modes": ("scale_shift",), "with_pose": False,
        }
        for tag in checkpoints:
            ckpt = find_checkpoint(run_dir, tag)
            if ckpt is None:
                continue
            print(f"[eval] DAv2 {tag}: {ckpt}")
            m = _apply_dav2_finetune(dav2_base, ckpt, lora_rank, lora_alpha, dav2_lora_only, device)
            variants[f"dav2_{tag}"] = {
                "label": f"DAv2 ({tag})", "predict_fn": make_dav2_predict(m, device),
                "align_modes": ("scale_shift",), "with_pose": False,
            }

    return variants


def collect_align_modes(variants: Dict[str, dict], allowed=None) -> List[str]:
    """Ordered union of the align modes across variants (optionally filtered)."""
    modes: List[str] = []
    for var in variants.values():
        for m in var["align_modes"]:
            if (allowed is None or m in allowed) and m not in modes:
                modes.append(m)
    return modes


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
    lines.append(f"  {source}")
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


def _save_results(
    results: dict, out_dir: str, table_text: str,
    json_name: str = "eval_results.json", txt_name: str = "eval_summary.txt",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] full results → {json_path}")

    txt_path = os.path.join(out_dir, txt_name)
    with open(txt_path, "w") as f:
        f.write(table_text)
    print(f"[eval] summary table → {txt_path}")


# --------------------------------------------------------------------------- #
# ADT helpers
# --------------------------------------------------------------------------- #

_DEFAULT_ADT_SEQ = "Apartment_release_clean_seq131_M1292"


def _find_adt_seq_dirs(adt_root: str) -> List[str]:
    if not adt_root or not os.path.isdir(adt_root):
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
# Main — single positional arg: the run name under runs/
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="ADT depth eval for a finetuning run (pretrained vs best/last)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run", help="experiment name under runs/ (e.g. ssi_r8), or a run-dir path")
    p.add_argument("--runs-root", default="runs", help="parent dir holding run folders")
    p.add_argument("--eval-out", default="eval_out", help="results go to <eval-out>/<run>/")
    p.add_argument("--checkpoints", nargs="+", default=["best", "last"],
                   choices=["best", "last", "final"],
                   help="which finetune checkpoints to evaluate (pretrained always included)")
    p.add_argument("--no-dav2", action="store_true", help="skip DAv2 variants")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-qual", type=int, default=4,
                   help="qualitative depth panels saved per variant (0 = skip)")
    # ADT params — default to the run's config; override here if needed.
    p.add_argument("--adt-root", default=None, help="override cfg.eval_adt_root")
    p.add_argument("--adt-max-frames", type=int, default=None,
                   help="cap frames per ADT seq (-1 = all); default cfg.eval_adt_max_frames")
    p.add_argument("--adt-depth-max", type=float, default=10.0)
    p.add_argument("--adt-gt-traj-csv", default=None,
                   help="ADT groundtruth/aria_trajectory.csv for pose ATE (VGGT only)")
    p.add_argument("--seq-len", type=int, default=None, help="default: cfg.seq_len")
    p.add_argument("--image-resolution", type=int, default=None, help="default: cfg.image_resolution")
    p.add_argument("--batch-size", type=int, default=1)
    a = p.parse_args()

    device = torch.device(a.device)
    run_dir = resolve_run(a.run, a.runs_root)
    run_name = os.path.basename(os.path.normpath(run_dir))
    cfg = load_run_config(run_dir)
    out_dir = os.path.join(a.eval_out, run_name)

    # Resolve eval params: CLI override → run config → default.
    adt_root = a.adt_root or cfg.get("eval_adt_root", "")
    seq_len = a.seq_len or int(cfg.get("seq_len", 8))
    image_resolution = a.image_resolution or int(cfg.get("image_resolution", 512))
    adt_max_frames = (a.adt_max_frames if a.adt_max_frames is not None
                      else int(cfg.get("eval_adt_max_frames", 100)))

    seq_dirs = _find_adt_seq_dirs(adt_root)
    if not seq_dirs:
        p.error(
            f"no ADT sequences under {adt_root!r} (from "
            f"{'--adt-root' if a.adt_root else 'cfg.eval_adt_root'}). "
            f"Pass --adt-root, or use `python -m finetune.eval.mps_depth {a.run} ...` for MPS."
        )

    print(f"[eval] run={run_name!r}  run_dir={run_dir}")
    print(f"[eval] ADT: {len(seq_dirs)} seq dir(s), <= {adt_max_frames} frames, "
          f"seq_len={seq_len}, res={image_resolution}")

    variants = build_variants(
        cfg, run_dir, device, checkpoints=tuple(a.checkpoints), include_dav2=not a.no_dav2
    )
    print(f"[eval] variants: {list(variants)}")

    from .adt_depth import run_adt_eval

    adt_results: dict = {}
    for var_key, var in variants.items():
        print(f"\n[eval] --- {var['label']} ---")
        qual_dir = os.path.join(out_dir, "qual", var_key) if a.n_qual > 0 else None
        adt_results[var_key] = run_adt_eval(
            predict_fn=var["predict_fn"],
            label=var["label"],
            seq_dirs=seq_dirs,
            device=device,
            seq_len=seq_len,
            image_resolution=image_resolution,
            batch_size=a.batch_size,
            depth_max_m=a.adt_depth_max,
            align_modes=var["align_modes"],
            gt_traj_csv=a.adt_gt_traj_csv if var["with_pose"] else None,
            qual_dir=qual_dir,
            n_qual=a.n_qual,
            max_frames=(None if adt_max_frames < 0 else adt_max_frames),
        )

    modes = collect_align_modes(variants)
    table = _print_comparison_table(
        adt_results, f"ADT (dense GT) — {run_name}", modes, list(variants)
    )
    _save_results({"adt": adt_results}, out_dir, table)
    print(f"\n[eval] Done. Results in {out_dir}/")


if __name__ == "__main__":
    main()
