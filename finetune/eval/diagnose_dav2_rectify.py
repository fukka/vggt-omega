# Copyright (c) 2026.
"""Diagnostic: DAv2-Small vs DAv2-Large, RECTIFIED vs RAW, on ADT dense GT.

Isolates whether fisheye->pinhole rectification is what degrades DAv2 (and whether
Large is hit harder than Small). For each model it scores the ADT depth twice --
once on raw fisheye frames, once on the rectified (pinhole) frames -- under
scale_shift alignment (the meaningful mode for a relative model), and prints a 2x2
table. Optionally saves qualitative RGB|pred|GT panels so you can eyeball it.

    python -m finetune.eval.diagnose_dav2_rectify \
        --adt-root /group-volume/Fengjia/data/projectaria_tools_adt_data_clean

The GT depth is rectified with the SAME maps as the RGB when rectify=on (so pred
and GT stay aligned in pinhole space), and left raw when rectify=off (both fisheye)
-- so each row is a valid pred-vs-GT comparison; only the input space differs.
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    __package__ = "finetune.eval"

import argparse
import json
import os

import torch

from .adt_depth import run_adt_eval
from .run_eval import _find_adt_seq_dirs, make_dav2_predict

_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
_COLS = ["AbsRel", "SqRel", "RMSE", "delta1", "delta2", "scale_ratio", "n_frames"]
_PCT = {"delta1", "delta2", "delta3"}


def _fmt(v, key):
    if not isinstance(v, (int, float)) or v != v:
        return "  N/A "
    if key in _PCT:
        return f"{v*100:5.1f}%"
    if key == "n_frames":
        return f"{int(v):6d}"
    return f"{v:7.4f}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="DAv2 Small vs Large, rectified vs raw, on ADT depth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adt-root", default="/group-volume/Fengjia/data/projectaria_tools_adt_data_clean")
    p.add_argument("--models", nargs="+", default=["small", "large"], choices=list(_MODELS),
                   help="which DAv2 sizes to test")
    p.add_argument("--rgb-subdir", default="videos_synthetic",
                   help="ADT RGB source (videos_synthetic is GT-aligned)")
    p.add_argument("--camera-preset", default="aria-214-1")
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--image-resolution", type=int, default=512)
    p.add_argument("--depth-max", type=float, default=10.0)
    p.add_argument("--n-qual", type=int, default=2, help="qual panels saved per (model,rectify); 0 = none")
    p.add_argument("--out-dir", default="eval_out/dav2_rectify_diag")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    device = torch.device(a.device)
    seq_dirs = _find_adt_seq_dirs(a.adt_root)
    if not seq_dirs:
        p.error(f"no ADT sequences under {a.adt_root!r} "
                f"(need <seq>/{a.rgb_subdir}/ + <seq>/depth_npy/)")
    print(f"[diag] {len(seq_dirs)} ADT seq dir(s), <= {a.max_frames} frames, "
          f"rgb={a.rgb_subdir}, preset={a.camera_preset}, res={a.image_resolution}")

    from finetune.models.depth_anything import build_depth_anything

    results: dict = {}      # tag -> scale_shift metrics
    for mkey in a.models:
        name = _MODELS[mkey]
        print(f"\n[diag] === DAv2-{mkey} ({name}) ===")
        model = build_depth_anything(use_dummy=False, model_name=name).to(device).eval()
        predict = make_dav2_predict(model, device)
        for rectify in (False, True):
            tag = f"dav2_{mkey}_{'rectified' if rectify else 'raw'}"
            qual = os.path.join(a.out_dir, tag) if a.n_qual > 0 else None
            res = run_adt_eval(
                predict_fn=predict, label=tag, seq_dirs=seq_dirs, device=device,
                seq_len=a.seq_len, image_resolution=a.image_resolution,
                depth_max_m=a.depth_max, align_modes=("scale_shift",),
                qual_dir=qual, n_qual=a.n_qual, max_frames=a.max_frames,
                rgb_subdir=a.rgb_subdir, rectify=rectify, camera_preset=a.camera_preset,
            )
            results[tag] = res.get("scale_shift", {})
        del model, predict
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------- comparison table --------
    lines = []
    lines.append("\n" + "=" * 86)
    lines.append("  DAv2 — rectified vs raw (ADT dense GT, scale_shift alignment)")
    lines.append("=" * 86)
    header = f"  {'Model / input':22s}" + "".join(f"  {c:>10s}" for c in _COLS)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for mkey in a.models:
        for rectify in (False, True):
            tag = f"dav2_{mkey}_{'rectified' if rectify else 'raw'}"
            m = results.get(tag, {})
            label = f"DAv2-{mkey} {'rectified' if rectify else 'raw'}"
            row = f"  {label:22s}" + "".join(f"  {_fmt(m.get(c, float('nan')), c):>10s}" for c in _COLS)
            lines.append(row)
        # per-model raw->rectified AbsRel delta
        raw = results.get(f"dav2_{mkey}_raw", {}).get("AbsRel", float("nan"))
        rec = results.get(f"dav2_{mkey}_rectified", {}).get("AbsRel", float("nan"))
        if raw == raw and rec == rec:
            lines.append(f"  {'  -> rectify AbsRel ' + ('WORSE' if rec > raw else 'better'):22s}"
                         f"  {rec - raw:+.4f}  ({raw:.4f} -> {rec:.4f})")
    text = "\n".join(lines)
    print(text)

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "dav2_rectify_diag.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(a.out_dir, "dav2_rectify_diag.txt"), "w") as f:
        f.write(text)
    print(f"\n[diag] saved -> {a.out_dir}/ (dav2_rectify_diag.json/.txt"
          + (f", qual panels per model/input" if a.n_qual > 0 else "") + ")")


if __name__ == "__main__":
    main()
