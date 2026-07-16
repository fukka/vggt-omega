# Copyright (c) 2026.
"""Depth metrics + alignment for the ADT fisheye evaluation.

The upstream evaluator (``utils/metric.py``, from PanoFormer) median-aligns
and crops 68 ERP rows top/bottom to discard the panorama poles — meaningless
on a fisheye frame.  Here the "crop" is the analytic validity mask instead
(GT validity AND the imaged cone), and alignment matches this repo's baseline
protocol (``finetune/eval/metrics.py``) so a "VGGT-360-fisheye" row can drop
straight into the existing ADT benchmark table next to DAC / UniK3D:

  * ``scale_shift`` (default): per-frame least-squares affine ``a*pred + b``
    in depth space — the affine-invariant protocol, appropriate because
    VGGT-360's fused output is up-to-scale.
  * ``median``: per-frame median-ratio scaling (upstream's alignment).
  * ``none``: raw comparison (only meaningful for metric-scale predictors).

numpy-only, torch-free.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def align_depth(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                mode: str = "scale_shift") -> np.ndarray:
    """Align ``pred`` to ``gt`` over ``mask`` and return the aligned map."""
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    if p.size < 10:
        return pred
    if mode == "none":
        return pred
    if mode == "median":
        med_p = np.median(p)
        scale = np.median(g) / med_p if med_p > 1e-9 else 1.0
        return pred * scale
    if mode == "scale_shift":
        # least squares  [p, 1] @ [a, b]^T = g
        A = np.stack([p, np.ones_like(p)], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, g, rcond=None)
        return pred * a + b
    raise ValueError(f"unknown align mode: {mode!r}")


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                  max_depth: float = 10.0) -> Dict[str, float]:
    """Standard depth error/accuracy metrics over the masked pixels.

    ``pred`` must already be aligned (see ``align_depth``); it is clamped to
    ``[1e-3, max_depth]`` first so an affine shift cannot produce negatives.
    """
    p = np.clip(pred[mask].astype(np.float64), 1e-3, max_depth)
    g = np.clip(gt[mask].astype(np.float64), 1e-3, max_depth)
    thresh = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "log10": float(np.mean(np.abs(np.log10(p) - np.log10(g)))),
        "d1": float(np.mean(thresh < 1.25)),
        "d2": float(np.mean(thresh < 1.25 ** 2)),
        "d3": float(np.mean(thresh < 1.25 ** 3)),
        "n_px": float(p.size),
    }


def aggregate_metrics(per_frame: List[Dict[str, float]]) -> Dict[str, float]:
    """Unweighted mean over frames (each frame counts once, like the repo eval)."""
    if not per_frame:
        return {}
    keys = [k for k in per_frame[0] if k != "n_px"]
    out = {k: float(np.mean([m[k] for m in per_frame])) for k in keys}
    out["n_frames"] = float(len(per_frame))
    return out


def print_summary(agg: Dict[str, float], label: str, align: str) -> None:
    if not agg:
        print(f"[{label}] ({align}) — no frames evaluated")
        return
    print(f"[{label}] align={align}  frames={int(agg.get('n_frames', 0))}")
    print("  abs_rel {abs_rel:.4f}  sq_rel {sq_rel:.4f}  rmse {rmse:.4f}  "
          "rmse_log {rmse_log:.4f}  log10 {log10:.4f}".format(**agg))
    print("  d1 {d1:.4f}  d2 {d2:.4f}  d3 {d3:.4f}".format(**agg))
