# Copyright (c) 2026.
"""Depth evaluation metrics and trajectory metrics (pure numpy).

Depth
-----
  align_depth(pred, gt, mask, mode)  -- four alignment modes
  depth_metrics(pred_aligned, gt, mask)  -- AbsRel / SqRel / RMSE / RMSElog / δ1/2/3
  aggregate_metrics(list_of_dicts)  -- mean across frames

Pose
----
  pose_ate_rpe(pred_t, gt_t)  -- ATE + RPE from N×3 translation arrays
                                  (Sim3-aligned by default)

All functions operate on numpy arrays so they can be called from any eval loop
without touching the GPU.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Depth alignment
# --------------------------------------------------------------------------- #

def align_depth(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    mode: str = "scale_shift",
) -> np.ndarray:
    """Align predicted depth to GT scale.

    Parameters
    ----------
    pred, gt : float32 (H, W) arrays; pred may be in arbitrary scale.
    mask     : bool (H, W); True where GT is valid.
    mode     : one of
               'none'              -- no alignment (metric prediction)
               'scale_only'        -- median ratio alignment: s = median(gt/pred)
               'scale_shift'       -- least-squares affine in depth space
               'disparity_scale_shift' -- LS affine in 1/depth (MiDaS protocol)

    Returns
    -------
    pred_aligned : float32 (H, W), in GT's metric units.
    """
    if mode == "none":
        return pred.astype(np.float32)

    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)

    if mode == "scale_only":
        s = np.median(g / (p + 1e-8))
        return (s * pred).astype(np.float32)

    if mode == "scale_shift":
        A = np.stack([p, np.ones_like(p)], axis=1)
        x, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
        s, t = x
        return (s * pred + t).astype(np.float32)

    if mode == "disparity_scale_shift":
        # ``pred`` is DEPTH (our pipeline convention). The correct alignment for an
        # affine-invariant DISPARITY model (MiDaS / Depth-Anything) is in disparity
        # space: recover disparity = 1/depth, fit s*pred_disp + shift ~= gt_disp,
        # then invert back. Depth-space 'scale_shift' cannot undo the disparity SHIFT
        # and scores even a perfect disparity model poorly (the eval_depth_anything_v2
        # reference uses exactly this disparity-space protocol).
        p_disp = 1.0 / (p + 1e-8)               # pred disparity (masked)
        g_disp = 1.0 / (g + 1e-8)               # gt disparity
        A = np.stack([p_disp, np.ones_like(p_disp)], axis=1)
        x, _, _, _ = np.linalg.lstsq(A, g_disp, rcond=None)
        s, t = x
        pred_disp = s * (1.0 / (pred.astype(np.float64) + 1e-8)) + t
        out = np.where(pred_disp > 1e-8, 1.0 / pred_disp, 0.0)
        return out.astype(np.float32)

    raise ValueError(f"Unknown alignment mode: {mode!r}; "
                     "choose 'none', 'scale_only', 'scale_shift', or 'disparity_scale_shift'")


# --------------------------------------------------------------------------- #
# Depth metrics
# --------------------------------------------------------------------------- #

def depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    min_depth: float = 0.01,
    max_depth: float = 100.0,
) -> dict:
    """Standard depth evaluation metrics over valid (masked) pixels.

    Parameters
    ----------
    pred  : float32 (H, W) — predicted depth, already aligned to GT scale.
    gt    : float32 (H, W) — ground-truth depth in metres.
    mask  : bool   (H, W) — True where GT is valid.
    min_depth, max_depth : clamp range for valid-pixel selection.

    Returns
    -------
    dict with keys:
        AbsRel, SqRel, RMSE, RMSElog, delta1, delta2, delta3,
        scale_ratio (median(gt/pred) before alignment — 1.0 = perfect metric scale),
        n_valid (number of valid pixels used)
    """
    # Valid = GT in range AND prediction in range. Excluding out-of-range PRED is
    # what the official/reference DAv2 eval does: disparity alignment can push a few
    # pixels' aligned disparity ~0 -> depth -> ~1e6, and without this exclusion those
    # blow up SqRel/RMSE (e.g. DAv2-small raw measured SqRel 5e6, RMSE 918 m).
    combined = (mask & np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)
                & np.isfinite(pred) & (pred > 0) & (pred <= max_depth))
    n_valid = int(combined.sum())
    if n_valid == 0:
        return {k: float("nan") for k in
                ["AbsRel", "SqRel", "RMSE", "RMSElog",
                 "delta1", "delta2", "delta3", "scale_ratio", "n_valid"]}

    p = np.clip(pred[combined].astype(np.float64), 1e-6, None)
    g = np.clip(gt[combined].astype(np.float64), 1e-6, None)

    diff = np.abs(p - g)
    diff2 = (p - g) ** 2

    ratio = np.maximum(p / g, g / p)
    scale_ratio = float(np.median(g / (pred[combined].astype(np.float64) + 1e-8)))

    return {
        "AbsRel":      float(np.mean(diff / g)),
        "SqRel":       float(np.mean(diff2 / g)),
        "RMSE":        float(np.sqrt(np.mean(diff2))),
        "RMSElog":     float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "delta1":      float(np.mean(ratio < 1.25)),
        "delta2":      float(np.mean(ratio < 1.25 ** 2)),
        "delta3":      float(np.mean(ratio < 1.25 ** 3)),
        "scale_ratio": scale_ratio,
        "n_valid":     n_valid,
    }


def aggregate_metrics(frames: list) -> dict:
    """Mean of depth_metrics dicts across frames, skipping NaN frames."""
    if not frames:
        return {}
    keys = [k for k in frames[0] if k != "n_valid"]
    out = {}
    for k in keys:
        vals = [f[k] for f in frames if np.isfinite(f.get(k, float("nan")))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_frames"] = int(sum(1 for f in frames if f.get("n_valid", 0) > 0))
    out["n_valid_total"] = int(sum(f.get("n_valid", 0) for f in frames))
    return out


def print_depth_summary(summary: dict, label: str = "", align: str = "") -> None:
    tag = label + (f"  [{align}]" if align else "")
    print(f"\n{'='*60}")
    print(f"  {tag or 'depth metrics'}  (n_frames={summary.get('n_frames', '?')})")
    print(f"{'='*60}")
    for k in ["AbsRel", "SqRel", "RMSE", "RMSElog"]:
        print(f"  {k:10s}: {summary.get(k, float('nan')):.4f}")
    for k in ["delta1", "delta2", "delta3"]:
        print(f"  {k:10s}: {summary.get(k, float('nan'))*100:.2f}%")
    print(f"  scale_ratio: {summary.get('scale_ratio', float('nan')):.4f}  (1.0 = perfect metric)")
    print(f"{'='*60}\n")


# --------------------------------------------------------------------------- #
# Pose metrics: ATE and RPE
# --------------------------------------------------------------------------- #

def _umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Sim3 alignment: dst ≈ s * R @ src + t  (minimise L2 error).

    Returns (s, R [3,3], t [3]) and the aligned src.
    """
    n = src.shape[0]
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = float(np.mean(np.sum(src_c ** 2, axis=1)))
    H = (src_c.T @ dst_c) / n
    U, D, Vt = np.linalg.svd(H)
    V = Vt.T
    # Handle reflection
    diag = np.ones(3)
    diag[-1] = np.sign(np.linalg.det(V @ U.T))
    R = V @ np.diag(diag) @ U.T
    s = float(np.dot(D, diag)) / max(var_s, 1e-12)
    t = mu_d - s * R @ mu_s
    aligned = (s * (R @ src_c.T).T) + mu_d
    return s, R, t, aligned


def pose_ate_rpe(
    pred_t: np.ndarray,
    gt_t: np.ndarray,
    align_sim3: bool = True,
) -> dict:
    """Absolute Trajectory Error and Relative Pose Error from translation sequences.

    Parameters
    ----------
    pred_t   : (N, 3) predicted camera positions (translation component of T_cam_world).
    gt_t     : (N, 3) ground-truth camera positions (same convention).
    align_sim3 : if True, Sim3-align pred to gt before computing ATE (removes
                 global scale/rotation/translation ambiguity — standard SLAM eval).

    Returns
    -------
    dict with keys:
        ATE      -- Absolute Trajectory Error (RMSE over aligned positions), metres
        RPE_t    -- RPE translation component (mean over consecutive pairs)
        RPE_r    -- RPE rotation error is not computed from translations alone;
                    this key is always NaN — supply SE3 poses for rotation RPE
        scale    -- Sim3 scale factor (1.0 = correct metric scale); NaN if align=False
        n        -- number of frames used
    """
    pred_t = np.array(pred_t, dtype=np.float64)
    gt_t = np.array(gt_t, dtype=np.float64)
    assert pred_t.shape == gt_t.shape and pred_t.ndim == 2 and pred_t.shape[1] == 3

    scale = float("nan")
    if align_sim3:
        s, R, t, aligned = _umeyama_sim3(pred_t, gt_t)
        scale = s
    else:
        aligned = pred_t

    diff = aligned - gt_t
    ate = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    # RPE translation: magnitude of relative translation error per step
    if len(pred_t) > 1:
        rpe_t_vals = []
        for i in range(len(pred_t) - 1):
            gt_rel = gt_t[i + 1] - gt_t[i]
            if align_sim3:
                pred_rel = s * (R @ (pred_t[i + 1] - pred_t[i]))
            else:
                pred_rel = pred_t[i + 1] - pred_t[i]
            rpe_t_vals.append(float(np.linalg.norm(pred_rel - gt_rel)))
        rpe_t = float(np.mean(rpe_t_vals))
    else:
        rpe_t = float("nan")

    return {
        "ATE":   ate,
        "RPE_t": rpe_t,
        "RPE_r": float("nan"),
        "scale": scale,
        "n":     len(pred_t),
    }


def print_pose_summary(summary: dict, label: str = "") -> None:
    print(f"\n{'='*60}")
    print(f"  {label or 'pose metrics'}  (n={summary.get('n', '?')})")
    print(f"{'='*60}")
    print(f"  ATE     : {summary.get('ATE', float('nan')):.4f} m")
    print(f"  RPE_t   : {summary.get('RPE_t', float('nan')):.4f} m/step")
    if not np.isnan(summary.get("scale", float("nan"))):
        print(f"  scale   : {summary.get('scale'):.4f}  (1.0 = metric)")
    print(f"{'='*60}\n")
