"""Evaluation metrics, Appendix A (Eq. 14-18).

Three families, all reported per image pair or per frame and then averaged:

* ``R_err`` / ``t_err`` (Eq. 14) -- geodesic rotation error and
  translation-*direction* error, both in degrees. Short baselines amplify
  ``t_err`` even when depth is fine, which is why the paper reports it alongside
  a depth metric rather than instead of one.
* ``d_reproj`` (Eq. 16) -- reprojection error in pixels using *ground-truth*
  pose, so it scores depth independently of predicted pose. The global scale
  ambiguity is resolved by a single scalar per pair.
* ``AbsRel`` / ``delta_1.25`` (Eq. 17-18) -- scale-aligned dense depth error,
  where reliable ground-truth depth exists.

Eq. 17 in the paper is printed with an L2 norm and no division by the ground
truth, which does not match the name "AbsRel" or Eigen et al. [53] that it
cites. This module implements the standard Eigen definition,
``mean(|s*D - D*| / D*)``, since that is what the reported values
(ETH3D 0.107, ScanNet++ 0.108) are comparable against.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .cameras import Camera
from .losses import backproject

__all__ = ["rotation_error_deg", "translation_error_deg", "pose_errors",
           "reprojection_depth_error", "align_scale", "depth_metrics",
           "aggregate"]


def rotation_error_deg(R_hat: Tensor, R_gt: Tensor) -> float:
    """Eq. 14, left."""
    tr = (R_gt.transpose(-1, -2) @ R_hat).diagonal(dim1=-2, dim2=-1).sum(-1)
    return float(torch.rad2deg(torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))).mean())


def translation_error_deg(t_hat: Tensor, t_gt: Tensor) -> float:
    """Eq. 14, right -- angle between translation *directions*."""
    a = t_hat.reshape(-1, 3)
    b = t_gt.reshape(-1, 3).to(a.dtype)
    na, nb = a.norm(dim=-1).clamp_min(1e-12), b.norm(dim=-1).clamp_min(1e-12)
    cos = ((a * b).sum(-1) / (na * nb)).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos)).mean())


def pose_errors(R_hat: Tensor, t_hat: Tensor, R_gt: Tensor, t_gt: Tensor) -> Dict[str, float]:
    return {"R_deg": rotation_error_deg(R_hat, R_gt),
            "t_deg": translation_error_deg(t_hat, t_gt)}


def align_scale(pred: Tensor, gt: Tensor, weight: Optional[Tensor] = None,
                *, robust: bool = True) -> float:
    """Single positive scalar aligning ``pred`` to ``gt``.

    ``robust`` uses the weighted median of the per-element ratio (the repo's
    ``scale_only`` convention in ``finetune/eval/metrics.py``); otherwise a
    weighted least-squares scale.
    """
    p = pred.reshape(-1)
    g = gt.reshape(-1)
    m = torch.isfinite(p) & torch.isfinite(g) & (p > 1e-8) & (g > 1e-8)
    if weight is not None:
        m = m & (weight.reshape(-1) > 0)
    if not bool(m.any()):
        return 1.0
    p, g = p[m], g[m]
    if robust:
        return float((g / p).median())
    return float((p * g).sum() / (p * p).sum().clamp_min(1e-12))


#: How the per-pixel reprojection residual is averaged. See
#: :func:`reprojection_depth_error`.
REPROJ_WEIGHTINGS = ("omega", "confidence")


def reprojection_depth_error(depth_i: Tensor, camera: Camera, matches,
                             R_gt: Tensor, t_gt: Tensor, *,
                             convention: str = "range",
                             valid: Optional[Tensor] = None,
                             weighting: str = "omega") -> Optional[float]:
    """Eq. 16 -- ``d_reproj`` in pixels, under ground-truth relative pose.

    The scale ``s`` is solved for once per pair by a 1-D search over the
    closed-form candidates, rather than by gradient descent: ``min_s`` in Eq. 16
    is over a single positive scalar, and reprojection is not convex in it, so a
    coarse-to-fine scan is both cheaper and more reliable than an optimiser.

    ``weighting`` selects how the per-pixel residual is averaged, and the two
    options are **not** small variations on each other:

    * ``"omega"`` (default) is Eq. 16 as written: an *unweighted* mean over every
      pixel of ``Omega``. Note that Eq. 16 carries no ``w_ij`` at all -- unlike
      Eq. 8, which does. Pixels the matcher considers non-covisible are still
      scored, against whatever dense flow it emitted there.
    * ``"confidence"`` is a confidence-weighted mean, ``sum(w e) / sum(w)``, i.e.
      the mean over the *confidently matched* subset.

    The gap between them is large, not cosmetic. UFM's covisibility mask goes to
    zero exactly where reprojection error is worst -- the fisheye periphery and
    content that left the frame -- so ``"confidence"`` drops the hardest pixels
    *and* renormalises by their absence. Comparing a ``"confidence"`` number
    against the paper's tables understates our error by whatever fraction of
    ``Omega`` the matcher gave up on. ``"omega"`` is the only one comparable to
    Tab. 1/2/5.

    Returns ``None`` when there is nothing to score.
    """
    if weighting not in REPROJ_WEIGHTINGS:
        raise ValueError(f"weighting must be one of {REPROJ_WEIGHTINGS}, got {weighting!r}")

    w = matches.weight
    if valid is not None:
        w = w * valid.to(w.dtype)
    if float(w.sum()) <= 0:
        return None

    if weighting == "omega":
        # Omega is geometric -- the valid fisheye disc -- not "where the matcher
        # succeeded". Fall back to the lens' own mask when the caller gave none.
        omega = (valid.to(w.dtype) if valid is not None
                 else camera.valid_mask(*w.shape[-2:]).to(w.dtype))
        norm = float(omega.sum())
        if norm <= 0:
            return None
    else:
        omega, norm = w, float(w.sum())

    X = backproject(depth_i, camera, convention=convention)
    R_gt = R_gt.to(X.dtype)
    t_gt = t_gt.to(X.dtype)

    def err_at(s: float) -> float:
        Xj = (s * X) @ R_gt.transpose(-1, -2) + t_gt
        uv = camera.project(Xj)
        e = (uv - matches.target).norm(dim=-1)
        e = torch.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
        return float((omega * e).sum() / norm)

    # Coarse geometric sweep, then a local refinement around the best decade.
    best_s, best = 1.0, err_at(1.0)
    for k in range(-12, 13):
        s = float(2.0 ** k)
        e = err_at(s)
        if e < best:
            best, best_s = e, s
    lo, hi = best_s / 2.0, best_s * 2.0
    for _ in range(24):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if err_at(m1) < err_at(m2):
            hi = m2
        else:
            lo = m1
    return min(best, err_at(0.5 * (lo + hi)))


def depth_metrics(pred: Tensor, gt: Tensor, *, valid: Optional[Tensor] = None,
                  max_depth: Optional[float] = None) -> Dict[str, float]:
    """Eq. 17-18 after a single scale alignment: ``AbsRel`` and ``delta_1.25``."""
    m = torch.isfinite(pred) & torch.isfinite(gt) & (gt > 1e-6)
    if valid is not None:
        m = m & valid
    if max_depth is not None:
        m = m & (gt <= max_depth)
    if not bool(m.any()):
        return {"AbsRel": float("nan"), "delta_1.25": float("nan"), "n": 0}

    p, g = pred[m], gt[m]
    s = align_scale(p, g)
    p = (s * p).clamp_min(1e-6)

    absrel = float(((p - g).abs() / g).mean())
    ratio = torch.maximum(p / g, g / p)
    return {"AbsRel": absrel, "delta_1.25": float((ratio < 1.25).float().mean()),
            "scale": s, "n": int(m.sum())}


def aggregate(rows) -> Dict[str, float]:
    """Mean over rows, ignoring NaNs, keeping the paper's reported keys."""
    out: Dict[str, float] = {}
    keys = set()
    for r in rows:
        keys.update(r.keys())
    for k in sorted(keys):
        vals = [r[k] for r in rows
                if k in r and r[k] is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            out[k] = float(sum(vals) / len(vals))
    out["n_rows"] = len(rows)
    return out
