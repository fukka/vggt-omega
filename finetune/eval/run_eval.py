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

Eval modes (combinable; each writes its own ``eval_out/<name>_<mode>/`` folder)::

    real-rectify / real-norectify        real-sensor RGB (videos_rgb)     vs depth_npy
    blender-rectify / blender-norectify  synthetic RGB (videos_synthetic) vs depth_npy
    blender-pinhole                      native pinhole render (blender_rendered_maps/
                                         pinhole/) vs depth_maps (float32 m), no rectify

All sources live under the (fixed) ADT root, so only a mode flag is needed — not a
path. real-*/blender-* fisheye GT is uint16 mm; *-rectify remaps fisheye→pinhole.
With no mode flag the legacy cfg-driven ADT eval runs into ``eval_out/<name>/``.

Outputs (per mode, under its folder)::

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
    # missing is expected (the frozen backbone comes from the base ckpt); but
    # unexpected != 0 means saved finetune weights did NOT match the model and were
    # silently dropped (e.g. a lora_rank/architecture mismatch) -> fail loudly.
    if unexpected:
        raise RuntimeError(
            f"VGGT finetune checkpoint {finetune_checkpoint!r} has {len(unexpected)} key(s) "
            f"that don't match the model — those finetune weights were NOT applied. "
            f"Likely a config mismatch (lora_rank/alpha or architecture). "
            f"unexpected[:5]={unexpected[:5]}"
        )
    print(f"[eval]   VGGT finetune weights loaded "
          f"({len(sd)} tensors; {len(missing)} frozen-backbone keys from base, unexpected=0)")
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
    if unexpected:
        raise RuntimeError(
            f"DAv2 finetune checkpoint {finetune_checkpoint!r} has {len(unexpected)} key(s) "
            f"that don't match the model — those finetune weights were NOT applied. "
            f"Likely the run used a different dav2_model_name (e.g. Small vs Large) or "
            f"lora setting. unexpected[:5]={unexpected[:5]}"
        )
    print(f"[eval]   DAv2 finetune weights loaded "
          f"({len(dav2_sd)} tensors; {len(missing)} frozen keys from base, unexpected=0)")
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
            "align_modes": ("disparity_scale_shift",), "with_pose": False,
        }
        for tag in checkpoints:
            ckpt = find_checkpoint(run_dir, tag)
            if ckpt is None:
                continue
            print(f"[eval] DAv2 {tag}: {ckpt}")
            m = _apply_dav2_finetune(dav2_base, ckpt, lora_rank, lora_alpha, dav2_lora_only, device)
            variants[f"dav2_{tag}"] = {
                "label": f"DAv2 ({tag})", "predict_fn": make_dav2_predict(m, device),
                "align_modes": ("disparity_scale_shift",), "with_pose": False,
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
# RGB source: the rendered stream that is pixel-aligned with the rendered GT depth
# (see adt_depth.ADT_RGB_SUBDIR / ADTWindowDataset).
_ADT_RGB_SUBDIR = "videos_synthetic"


def _is_adt_seq(seq_dir: str) -> bool:
    return (os.path.isdir(os.path.join(seq_dir, _ADT_RGB_SUBDIR)) and
            os.path.isdir(os.path.join(seq_dir, "depth_npy")))


def _find_seq_dirs(adt_root: str, rgb_subdir: str, depth_subdir: str) -> List[str]:
    """Sequence dirs under adt_root that contain <rgb_subdir>/ and <depth_subdir>/.

    Handles three layouts: adt_root itself laid out as a single sequence; the
    single default sequence (_DEFAULT_ADT_SEQ); or adt_root holding many sequence
    dirs.  rgb_subdir/depth_subdir may be nested (e.g. the blender pinhole renders
    at blender_rendered_maps/pinhole/{videos_rgb,depth_maps}).
    """
    if not adt_root or not os.path.isdir(adt_root):
        return []

    def _has(seq_dir: str) -> bool:
        return (os.path.isdir(os.path.join(seq_dir, rgb_subdir)) and
                os.path.isdir(os.path.join(seq_dir, depth_subdir)))

    if _has(adt_root):
        return [adt_root]
    default_seq = os.path.join(adt_root, _DEFAULT_ADT_SEQ)
    if _has(default_seq):
        return [default_seq]
    return [os.path.join(adt_root, name) for name in sorted(os.listdir(adt_root))
            if _has(os.path.join(adt_root, name))]


def _find_adt_seq_dirs(adt_root: str) -> List[str]:
    """Back-compat wrapper (imported by trainers/base.py + diagnose_dav2_rectify.py):
    sequence dirs with videos_synthetic/ + depth_npy/."""
    return _find_seq_dirs(adt_root, _ADT_RGB_SUBDIR, "depth_npy")


# --------------------------------------------------------------------------- #
# Eval modes — what RGB to feed the model and which GT to score against.
# --------------------------------------------------------------------------- #
# Every source lives under the (fixed) ADT root, one set per sequence dir, so the
# user only picks a mode; --adt-root stays a rare override.
#
#   real-*     real-sensor RGB (videos_rgb)        vs depth_npy  (fisheye, uint16 mm)
#   blender-*  synthetic RGB   (videos_synthetic)  vs depth_npy  (fisheye, uint16 mm)
#   *-rectify  remap fisheye→pinhole (RGB+GT);  *-norectify  score raw fisheye
#   blender-pinhole   native pinhole render at blender_rendered_maps/pinhole/
#                     (videos_rgb + depth_maps, float32 metres) — no rectification
#
# The Aria 270° CCW rotation is applied in every mode (ADTWindowDataset default).

_PINHOLE_RGB   = os.path.join("blender_rendered_maps", "pinhole", "videos_rgb")
_PINHOLE_DEPTH = os.path.join("blender_rendered_maps", "pinhole", "depth_maps")

# mode name -> dataset spec; dict order is the report order.
_MODE_SPECS: Dict[str, dict] = {
    "real-rectify":      dict(rgb_subdir="videos_rgb",       depth_subdir="depth_npy",    depth_scale=0.001, rectify=True),
    "real-norectify":    dict(rgb_subdir="videos_rgb",       depth_subdir="depth_npy",    depth_scale=0.001, rectify=False),
    "blender-rectify":   dict(rgb_subdir="videos_synthetic", depth_subdir="depth_npy",    depth_scale=0.001, rectify=True),
    "blender-norectify": dict(rgb_subdir="videos_synthetic", depth_subdir="depth_npy",    depth_scale=0.001, rectify=False),
    "blender-pinhole":   dict(rgb_subdir=_PINHOLE_RGB,       depth_subdir=_PINHOLE_DEPTH, depth_scale=1.0,   rectify=False),
}


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
    # Eval modes — combinable; each is scored separately into its own output
    # folder. Sources are fixed under the ADT root, so no per-source paths needed.
    g = p.add_argument_group("eval modes (combinable; each scored separately)")
    g.add_argument("--real-rectify", action="store_true",
                   help="real-sensor RGB (videos_rgb), fisheye→pinhole, vs depth_npy GT")
    g.add_argument("--real-norectify", action="store_true",
                   help="real-sensor RGB (videos_rgb), raw fisheye, vs depth_npy GT")
    g.add_argument("--blender-rectify", action="store_true",
                   help="synthetic RGB (videos_synthetic), fisheye→pinhole, vs depth_npy GT")
    g.add_argument("--blender-norectify", action="store_true",
                   help="synthetic RGB (videos_synthetic), raw fisheye, vs depth_npy GT")
    g.add_argument("--blender-pinhole", action="store_true",
                   help="native pinhole render (blender_rendered_maps/pinhole) vs depth_maps (float32 m)")
    p.add_argument("--seq-len", type=int, default=None, help="default: cfg.seq_len")
    p.add_argument("--image-resolution", type=int, default=None, help="default: cfg.image_resolution")
    p.add_argument("--batch-size", type=int, default=1)
    # Legacy ADT input + fisheye rectification — used only when NO mode flag above
    # is given (each mode fully specifies its own RGB stream and rectification).
    p.add_argument("--rgb-subdir", default="videos_synthetic",
                   help="[no-mode only] ADT RGB source (videos_synthetic GT-aligned; videos_rgb real)")
    p.add_argument("--rectify", dest="rectify", action="store_true",
                   help="[no-mode only] force fisheye→pinhole rectify RGB+GT (default: follow cfg.rectify)")
    p.add_argument("--no-rectify", dest="rectify", action="store_false",
                   help="[no-mode only] force raw fisheye (no rectification)")
    p.set_defaults(rectify=None)
    p.add_argument("--camera-preset", default=None, help="default: cfg.camera_preset (aria-214-1)")
    a = p.parse_args()

    device = torch.device(a.device)
    run_dir = resolve_run(a.run, a.runs_root)
    run_name = os.path.basename(os.path.normpath(run_dir))
    cfg = load_run_config(run_dir)

    # Resolve eval params: CLI override → run config → default.
    adt_root = a.adt_root or cfg.get("eval_adt_root", "")
    seq_len = a.seq_len or int(cfg.get("seq_len", 8))
    image_resolution = a.image_resolution or int(cfg.get("image_resolution", 512))
    adt_max_frames = (a.adt_max_frames if a.adt_max_frames is not None
                      else int(cfg.get("eval_adt_max_frames", 100)))
    camera_preset = a.camera_preset or cfg.get("camera_preset", "aria-214-1")
    fisheye_k = cfg.get("fisheye_k", "")
    fisheye_d = cfg.get("fisheye_d", "")

    # Build the eval jobs. Each selected mode fully specifies its RGB stream, GT
    # depth, depth scale and rectification (_MODE_SPECS). With no mode flag we fall
    # back to the legacy cfg-driven ADT eval (videos_synthetic + cfg.rectify) so
    # existing invocations keep working unchanged.
    selected = [m for m in _MODE_SPECS if getattr(a, m.replace("-", "_"))]
    if selected:
        jobs = [(m, _MODE_SPECS[m]) for m in selected]
    else:
        rectify = bool(cfg.get("rectify", True)) if a.rectify is None else a.rectify
        jobs = [("default", dict(rgb_subdir=a.rgb_subdir, depth_subdir="depth_npy",
                                 depth_scale=0.001, rectify=rectify))]

    # Resolve sequence dirs per job up front so we fail fast (before loading models)
    # if nothing matches the requested mode(s).
    resolved: List[tuple] = []
    for mode_name, spec in jobs:
        sd = _find_seq_dirs(adt_root, spec["rgb_subdir"], spec["depth_subdir"])
        if not sd:
            print(f"[eval] WARN no sequences for mode {mode_name!r} under {adt_root!r} "
                  f"(need {spec['rgb_subdir']}/ + {spec['depth_subdir']}/); skipping.")
            continue
        resolved.append((mode_name, spec, sd))

    if not resolved:
        p.error(
            f"no sequences found under {adt_root!r} (from "
            f"{'--adt-root' if a.adt_root else 'cfg.eval_adt_root'}) for the "
            f"requested mode(s) {[name for name, _ in jobs]}. Check --adt-root, or "
            f"use `python -m finetune.eval.mps_depth {a.run} ...` for MPS sparse GT."
        )

    # Models are loaded once and reused across every mode.
    variants = build_variants(
        cfg, run_dir, device, checkpoints=tuple(a.checkpoints), include_dav2=not a.no_dav2
    )
    print(f"[eval] run={run_name!r}  run_dir={run_dir}")
    print(f"[eval] variants: {list(variants)}")
    print(f"[eval] modes: {[name for name, _, _ in resolved]}")

    from .adt_depth import run_adt_eval
    align_modes = collect_align_modes(variants)

    out_dirs: List[str] = []
    for mode_name, spec, seq_dirs in resolved:
        # Each mode → its own output folder so scores never overwrite each other.
        # Keep the legacy "_norectify" suffix for the no-mode path.
        if mode_name == "default":
            suffix = "_norectify" if a.rectify is False else ""
        else:
            suffix = f"_{mode_name}"
        out_dir = os.path.join(a.eval_out, run_name + suffix)
        out_dirs.append(out_dir)

        print(f"\n[eval] ===== mode={mode_name} =====")
        print(f"[eval] {len(seq_dirs)} seq dir(s), <= {adt_max_frames} frames, "
              f"seq_len={seq_len}, res={image_resolution}, rgb={spec['rgb_subdir']}, "
              f"depth={spec['depth_subdir']} (scale={spec['depth_scale']}), "
              f"rectify={spec['rectify']}"
              f"{f' ({camera_preset})' if spec['rectify'] else ''}")

        mode_results: dict = {}
        for var_key, var in variants.items():
            print(f"\n[eval] --- {var['label']} [{mode_name}] ---")
            qual_dir = os.path.join(out_dir, "qual", var_key) if a.n_qual > 0 else None
            mode_results[var_key] = run_adt_eval(
                predict_fn=var["predict_fn"],
                label=var["label"],
                seq_dirs=seq_dirs,
                device=device,
                seq_len=seq_len,
                image_resolution=image_resolution,
                batch_size=a.batch_size,
                depth_scale=spec["depth_scale"],
                depth_max_m=a.adt_depth_max,
                align_modes=var["align_modes"],
                gt_traj_csv=a.adt_gt_traj_csv if var["with_pose"] else None,
                qual_dir=qual_dir,
                n_qual=a.n_qual,
                max_frames=(None if adt_max_frames < 0 else adt_max_frames),
                rgb_subdir=spec["rgb_subdir"],
                depth_subdir=spec["depth_subdir"],
                rectify=spec["rectify"],
                camera_preset=camera_preset,
                fisheye_k=fisheye_k,
                fisheye_d=fisheye_d,
            )

        # JSON key "adt" for the legacy path (back-compat), else the mode name.
        json_key = "adt" if mode_name == "default" else mode_name
        title = "ADT" if mode_name == "default" else mode_name
        table = _print_comparison_table(
            mode_results, f"{title} (dense GT) — {run_name}", align_modes, list(variants)
        )
        _save_results({json_key: mode_results}, out_dir, table)

    print(f"\n[eval] Done. Results in:")
    for d in out_dirs:
        print(f"[eval]   {d}/")


if __name__ == "__main__":
    main()
