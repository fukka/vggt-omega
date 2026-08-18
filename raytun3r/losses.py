"""The adaptation objective, Eq. 8-13.

    L = L_reproj + w_smooth L_smooth + w_L2 L_L2 + w_TV L_TV + w_pose L_pose

with ``w_pose = 1, w_smooth = 10, w_L2 = 2, w_TV = 20`` in all the paper's
experiments. No photometric term: the paper avoids photometric warping because
it is weak in low-texture regions and sensitive to initialisation, and the
initial geometry is far from the optimum under pinhole bias.

**Depth convention.** Eq. 7 writes ``X_i(u,v) = D_i(u,v) kappa^-1(u,v)`` without
saying how ``kappa^-1`` is normalised, and the two readings are not
interchangeable on a wide lens (see the repo's CONTEXT.md):

* ``"range"`` (default) -- ``kappa^-1`` is the *unit* ray, so ``D`` is euclidean
  range. Well defined for every incidence angle, including past 90 deg.
* ``"z"`` -- ``kappa^-1`` is the ``z = 1`` ray of Eq. 1, so ``D`` is planar z.
  This matches what the VGGT-family depth heads natively emit, but it diverges
  as ``theta -> 90 deg`` and is unusable on the 185-200 deg datasets.

The two agree exactly where both are defined (``range = z / cos theta``). Pick
``"z"`` only for lenses well inside 180 deg, and keep the choice consistent
between training and evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .cameras import Camera, pixel_grid
from .matching import Matches

__all__ = ["LossWeights", "backproject", "reprojection_loss", "pose_loss",
           "smoothness_loss", "l2_penalty", "tv_penalty", "total_loss"]


@dataclass
class LossWeights:
    """Eq. 13 weights; defaults are the paper's."""

    pose: float = 1.0
    smooth: float = 10.0
    l2: float = 2.0
    tv: float = 20.0


def backproject(depth: Tensor, camera: Camera, *, convention: str = "range") -> Tensor:
    """Eq. 7: ``X(u,v) = D(u,v) kappa^-1(u,v)`` -> ``(H, W, 3)``."""
    h, w = depth.shape[-2:]
    rays = camera.ray_grid(h, w, device=depth.device, dtype=torch.float32).to(depth.dtype)
    if convention == "range":
        pass
    elif convention == "z":
        rays = rays / rays[..., 2:3].clamp_min(1e-6)
    else:
        raise ValueError(f"unknown depth convention {convention!r}; expected 'range' or 'z'")
    return depth.unsqueeze(-1) * rays


def _transform(points: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """Apply ``(R, t)`` to ``(..., 3)`` points."""
    return points @ R.transpose(-1, -2) + t


def reprojection_loss(depth_i: Tensor, R_rel: Tensor, t_rel: Tensor,
                      matches: Matches, camera: Camera, *,
                      convention: str = "range",
                      valid: Optional[Tensor] = None,
                      max_err_px: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    """Eq. 8 -- confidence-weighted L1 reprojection error, in pixels.

    Eq. 8 normalises by ``|Omega|`` -- the number of pixels inside the valid
    fisheye disc -- with ``w_ij`` *inside* the sum::

        L_reproj(i,j) = 1/|Omega| sum_{(u,v) in Omega} w_ij(u,v) ||...||_1

    so the confidence weights are a genuine down-weighting, not a re-normalising
    average: a pair where UFM is confident about only a third of the disc
    contributes about a third as much. Dividing by ``sum(w)`` instead would undo
    exactly that and inflate the term by ``|Omega|/sum(w)``, which on a fisheye
    with heavy occlusion is a factor of 2-3. Since ``L_reproj`` carries weight 1
    against ``w_smooth=10``, ``w_L2=2`` and ``w_TV=20`` (Eq. 13), that factor
    silently re-tunes the whole objective.

    Returns ``(loss, weight_sum)``; ``weight_sum`` lets the caller tell a pair
    with no confident matches from one that merely scored zero.
    """
    X = backproject(depth_i, camera, convention=convention)
    Xj = _transform(X, R_rel, t_rel)
    uv = camera.project(Xj)

    err = (uv - matches.target).abs().sum(-1)
    w = matches.weight
    if valid is not None:
        w = w * valid.to(w.dtype)

    # Clamp, do not drop. A pixel whose predicted reprojection lands outside the
    # imaged cone still has a real correspondence in image j, so it carries a
    # genuine (large) error; zeroing it would hand the optimiser a free escape --
    # push geometry out of frame and the penalty disappears. What has to be
    # bounded is the *magnitude*: past theta_max the projection polynomial is
    # extrapolating and its coordinates mean nothing, so one wild pixel could
    # dominate an L1 sum whose useful signal is a few pixels wide.
    # Upstream VGGT does the same thing in its own losses
    # (``check_and_fix_inf_nan`` with ``hard_max``, and ``loss_T.clamp(max=100)``).
    #
    # The predecessor of this clamp was ``Xj[..., 2] > -1e6``, which reads as a
    # behind-the-camera test but excludes nothing except -inf.
    #
    # What this does NOT do is rescue the gradient. If ``depth`` itself is NaN the
    # derivative through ``camera.project`` is already NaN, and no substitution
    # downstream of it changes that -- ``nan_to_num``'s zero meets a non-finite
    # local derivative one node up and becomes ``0 * inf``. That case is caught
    # where it is detectable, on the gradient norm in ``train.fit_adapter``, not
    # here. The clamp's job is only to stop a large-but-finite error from
    # dominating an L1 sum whose useful signal is a few pixels wide.
    #
    # ``err`` is the L1 (Manhattan) distance, so the largest value two in-frame
    # points can produce is ``(w-1) + (h-1)``, not the Euclidean diagonal.
    if max_err_px is None:
        max_err_px = float(camera.width + camera.height)
    err = torch.nan_to_num(err, nan=max_err_px, posinf=max_err_px,
                           neginf=max_err_px).clamp(max=max_err_px)

    wsum = w.sum()
    if float(wsum) <= 0:
        return depth_i.sum() * 0.0, wsum
    omega = float(valid.sum()) if valid is not None else float(err.numel())
    omega = max(omega, 1.0)
    return (w * err).sum() / omega, wsum


def _safe_norm(x: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    """``||x||`` whose gradient is finite at the origin.

    ``sqrt(sum x^2 + eps)`` has derivative ``x / safe_norm``, bounded by 1 -- unlike
    plain ``norm``, whose gradient is NaN at zero.
    """
    return torch.sqrt(x.pow(2).sum(dim=dim) + eps)


def _angle_between(a: Tensor, b: Tensor) -> Tensor:
    """Angle between two vectors via ``atan2``: exact at 0 and pi, no clamping."""
    cross = torch.cross(a, b, dim=-1)
    return torch.atan2(_safe_norm(cross), (a * b).sum(-1))


def pose_loss(R_hat: Tensor, t_hat: Tensor, R_tilde: Tensor, t_tilde: Tensor) -> Tensor:
    """Eq. 9 -- geodesic rotation error plus translation-direction angle.

    ``(R~, t~)`` is the fixed MAGSAC++ target: computed once, before adaptation,
    from an external matcher, so the adapter cannot influence its own label.

    Both terms use ``atan2`` rather than the literal ``arccos`` of Eq. 9.
    ``arccos`` has an infinite derivative at its endpoints, so it needs its input
    clamped away from +-1 -- and that clamp puts a floor under the loss (at
    ``1e-6`` of slack, ~2.8e-3 rad) which the optimiser can never get below. The
    ``atan2`` forms are algebraically identical, reach exactly zero at the
    target, and have bounded gradients everywhere.
    """
    R_err = R_tilde.transpose(-1, -2) @ R_hat
    # For a rotation, ||vee(R - R^T)/2|| = sin(theta) and (tr(R) - 1)/2 = cos(theta).
    v = 0.5 * torch.stack((R_err[..., 2, 1] - R_err[..., 1, 2],
                           R_err[..., 0, 2] - R_err[..., 2, 0],
                           R_err[..., 1, 0] - R_err[..., 0, 1]), dim=-1)
    tr = R_err.diagonal(dim1=-2, dim2=-1).sum(-1)
    rot = torch.atan2(_safe_norm(v), (tr - 1.0) / 2.0)

    a = t_hat.reshape(-1, 3)
    b = t_tilde.reshape(-1, 3).to(a.dtype)
    trans = _angle_between(a, b)
    return rot.mean() + trans.mean()


def smoothness_loss(depth: Tensor, image: Tensor, *,
                    valid: Optional[Tensor] = None) -> Tensor:
    """Eq. 10 -- edge-aware smoothness on mean-normalised depth ``D* = D/mean(D)``.

    Eq. 10 sums the x and y terms per pixel and divides by ``|Omega|``. The
    image-gradient factors ``e^{-|dI|}`` are weights *inside* that sum, so a
    strongly textured frame -- where every ``e^{-|dI|}`` is small -- is meant to
    contribute little smoothness pressure. Normalising by ``sum(e^{-|dI|})``
    instead would rescale that away and make the term as strong on a textured
    frame as on a blank wall.
    """
    if valid is not None and bool(valid.any()):
        mean = depth[valid].mean().clamp_min(1e-6)
    else:
        mean = depth.mean().clamp_min(1e-6)
    d = depth / mean

    dx = (d[:, 1:] - d[:, :-1]).abs()
    dy = (d[1:, :] - d[:-1, :]).abs()

    gray = image.mean(0) if image.dim() == 3 else image
    ix = (gray[:, 1:] - gray[:, :-1]).abs()
    iy = (gray[1:, :] - gray[:-1, :]).abs()

    wx = torch.exp(-ix)
    wy = torch.exp(-iy)
    if valid is not None:
        wx = wx * (valid[:, 1:] & valid[:, :-1]).to(wx.dtype)
        wy = wy * (valid[1:, :] & valid[:-1, :]).to(wy.dtype)

    omega = float(valid.sum()) if valid is not None else float(depth.numel())
    if omega <= 0:
        return depth.sum() * 0.0
    return ((wx * dx).sum() + (wy * dy).sum()) / omega


def l2_penalty(residual: Tensor) -> Tensor:
    """Eq. 11 -- keep ``P'`` close to ``P_A``.

    ``P' - P_A`` is exactly the adapter residual, so this is its mean squared
    norm over token-grid locations.
    """
    if residual.numel() == 0:
        return residual.sum() * 0.0
    return residual.pow(2).sum(-1).mean()


def tv_penalty(residual: Tensor, grid_hw: Tuple[int, int],
               pe_table: Optional[Tensor] = None) -> Tensor:
    """Eq. 12 -- total variation of the *adapted* table ``P'``.

    Eq. 12 is written on ``P'``, not on the residual, so the pretrained table
    ``P_A`` enters through the cross term. ``pe_table=None`` (a RoPE-only
    backbone, where no ``P_A`` exists) falls back to the TV of the residual
    alone, which is the same penalty up to that cross term.
    """
    if residual.numel() == 0:
        return residual.sum() * 0.0
    gh, gw = grid_hw
    p = residual if pe_table is None else residual + pe_table.to(residual.dtype)
    p = p.reshape(gh, gw, -1)
    dx = (p[:, 1:] - p[:, :-1]).pow(2).sum(-1).sum()
    dy = (p[1:, :] - p[:-1, :]).pow(2).sum(-1).sum()
    return (dx + dy) / (gh * gw)


def total_loss(pred, batch, camera: Camera, adapter, *,
               weights: LossWeights = None, convention: str = "range",
               valid: Optional[Tensor] = None,
               pe_table: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, float]]:
    """Assemble Eq. 13 for one three-frame window.

    ``batch`` supplies ``images`` ``(S, 3, H, W)``, ``matches[(i, j)]`` for every
    ordered pair, and optional ``pose_targets[(i, j)] = (R~, t~)``. Pairs with no
    pose target (MAGSAC++ failed or too few matches) simply skip the pose term.
    """
    weights = weights or LossWeights()
    # Eq. 7 is only correct when kappa^-1's normalisation matches what D means.
    pred.require_convention(convention)
    s = pred.depth.shape[0]
    dev = pred.depth.device

    total = torch.zeros((), device=dev, dtype=pred.depth.dtype)
    parts: Dict[str, float] = {}

    # -- Eq. 8, over all ordered pairs
    reproj, n_pairs = torch.zeros((), device=dev, dtype=pred.depth.dtype), 0
    for (i, j), m in batch["matches"].items():
        R_rel, t_rel = pred.relative(i, j)
        val, wsum = reprojection_loss(pred.depth[i], R_rel, t_rel, m, camera,
                                      convention=convention, valid=valid)
        if float(wsum) > 0:
            reproj = reproj + val
            n_pairs += 1
    if n_pairs:
        reproj = reproj / n_pairs
    total = total + reproj
    parts["reproj"] = float(reproj)

    # -- Eq. 9, over pairs with a MAGSAC++ target
    targets = batch.get("pose_targets") or {}
    pose, n_pose = torch.zeros((), device=dev, dtype=pred.depth.dtype), 0
    for (i, j), tgt in targets.items():
        if tgt is None:
            continue
        R_t, t_t = tgt
        R_rel, t_rel = pred.relative(i, j)
        pose = pose + pose_loss(R_rel, t_rel, R_t.to(dev), t_t.to(dev))
        n_pose += 1
    if n_pose:
        pose = pose / n_pose
        total = total + weights.pose * pose
    parts["pose"] = float(pose)

    # -- Eq. 10, per frame
    smooth = torch.zeros((), device=dev, dtype=pred.depth.dtype)
    for i in range(s):
        smooth = smooth + smoothness_loss(pred.depth[i], batch["images"][i], valid=valid)
    smooth = smooth / max(s, 1)
    total = total + weights.smooth * smooth
    parts["smooth"] = float(smooth)

    # -- Eq. 11 and 12, on the PE residual only
    if adapter is not None and adapter.pe is not None:
        res = adapter.pe_residual()
        l2 = l2_penalty(res)
        tv = tv_penalty(res, adapter._grid, pe_table)
        total = total + weights.l2 * l2 + weights.tv * tv
        parts["l2"], parts["tv"] = float(l2), float(tv)
    else:
        parts["l2"] = parts["tv"] = 0.0

    parts["total"] = float(total)
    return total, parts
