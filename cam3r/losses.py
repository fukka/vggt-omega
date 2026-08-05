# Copyright (c) 2026.
"""CAM3R training objectives -- paper Eq. 5-11.

Three terms, combined in :func:`cam3r_loss`:

* **Asymmetric angular loss** (Eq. 5-6) on the Ray Module's spherical angles.
  Quantile regression with ``alpha = 0.7`` on the incidence angle ``theta``
  makes *under*-prediction cost 0.7 and over-prediction 0.3, which is the
  paper's stated defence against the model collapsing onto the narrow-FoV
  pinhole mode its pretraining is biased toward.
* **Local regression loss** (Eq. 7-8) on pointmaps, each side divided by its own
  mean point norm so the term is invariant to scene scale.
* **Relative pose loss** (Eq. 9-10): geodesic rotation error plus a
  scale-anchored translation term.

Unspecified in the paper, chosen here (see cam3r/README.md "Unspecified"):
``beta = 0.5``, ``lambda_A = lambda_regr = lambda_pose = 1.0``, azimuth
differences are wrapped into ``(-pi, pi]``, and the confidence term follows
DUSt3R.  Each is a keyword argument, so none of them is baked in.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from cam3r.geometry import geodesic_angle, rays_to_spherical, wrap_angle


# --------------------------------------------------------------------------- #
# Eq. 5-6 -- asymmetric angular loss
# --------------------------------------------------------------------------- #

def asymmetric_angular_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    valid: Optional[torch.Tensor] = None,
    wrapped: bool = False,
) -> torch.Tensor:
    """Eq. 5: pinball / quantile loss with asymmetry ``alpha``.

    Residuals where ``pred < target`` are weighted ``alpha``; the rest
    ``1 - alpha``.  ``alpha = 0.5`` reduces to half the mean absolute error.
    Set ``wrapped`` for angles that live on a circle (azimuth).
    """
    diff = pred - target
    if wrapped:
        diff = wrap_angle(diff)
    weight = torch.where(diff < 0, torch.full_like(diff, alpha), torch.full_like(diff, 1.0 - alpha))
    per_elem = weight * diff.abs()
    if valid is None:
        return per_elem.mean()
    if not bool(valid.any()):
        return per_elem.sum() * 0.0
    return per_elem[valid].mean()


def angular_loss(
    pred_rays: torch.Tensor,
    target_rays: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    alpha_theta: float = 0.7,
    alpha_phi: float = 0.5,
) -> torch.Tensor:
    """Eq. 6: ``beta * L^0.7(theta) + (1 - beta) * L^0.5(phi)``.

    Rays are (B, 3, H, W); ``valid`` is (B, H, W).  Azimuth residuals are
    wrapped -- the paper writes a plain absolute difference, which would report
    a ~2pi error for two rays a hundredth of a radian apart across the seam.
    """
    pred = pred_rays.permute(0, 2, 3, 1)
    target = target_rays.permute(0, 2, 3, 1)
    theta_p, phi_p = rays_to_spherical(pred)
    theta_t, phi_t = rays_to_spherical(target)
    l_theta = asymmetric_angular_loss(theta_p, theta_t, alpha_theta, valid)
    l_phi = asymmetric_angular_loss(phi_p, phi_t, alpha_phi, valid, wrapped=True)
    return beta * l_theta + (1.0 - beta) * l_phi


# --------------------------------------------------------------------------- #
# Eq. 7-8 -- local regression loss
# --------------------------------------------------------------------------- #

def _normalize_by_mean_norm(
    pts: torch.Tensor, valid: Optional[torch.Tensor], eps: float = 1e-6
) -> torch.Tensor:
    """Eq. 7: divide each view's pointmap by its own mean point norm ``eta``."""
    norms = pts.norm(dim=1)                                  # (B, H, W)
    if valid is None:
        eta = norms.flatten(1).mean(dim=1)
    else:
        m = valid.flatten(1).float()
        eta = (norms.flatten(1) * m).sum(1) / m.sum(1).clamp(min=1.0)
    return pts / eta.clamp(min=eps).view(-1, 1, 1, 1)


def local_regression_loss(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    conf: Optional[torch.Tensor] = None,
    conf_mode: str = "none",
    conf_alpha: float = 0.2,
) -> torch.Tensor:
    """Eq. 8: squared error between scale-normalized pointmaps.

    ``conf_mode="dust3r"`` additionally weights each pixel by its predicted
    confidence and subtracts ``conf_alpha * log(conf)``.  The paper uses sigma
    only in global alignment (Eq. 12) and never says what trains it; without a
    term like this the confidence head receives no gradient at all, which would
    make the confidence weighting in Ray-Aware Global Alignment meaningless.
    """
    pred = _normalize_by_mean_norm(pred_points, valid)
    target = _normalize_by_mean_norm(target_points, valid)
    per_pix = ((pred - target) ** 2).sum(dim=1)              # (B, H, W)

    if conf_mode == "dust3r":
        if conf is None:
            raise ValueError("conf_mode='dust3r' needs a confidence map")
        c = conf.clamp(min=1e-6)
        per_pix = c * per_pix - conf_alpha * torch.log(c)
    elif conf_mode != "none":
        raise ValueError(f"unknown conf_mode {conf_mode!r}")

    if valid is None:
        return per_pix.mean()
    if not bool(valid.any()):
        return per_pix.sum() * 0.0
    return per_pix[valid].mean()


# --------------------------------------------------------------------------- #
# Eq. 9-10 -- relative pose loss
# --------------------------------------------------------------------------- #

def relative_pose_loss(
    R_pred: torch.Tensor,
    t_dir_pred: torch.Tensor,
    scale_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    w_rot: float = 1.0,
    w_trans: float = 1.0,
    w_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Eq. 9-10: geodesic rotation error, translation direction, and scale.

    Eq. 10 is ambiguous about how ``s`` enters, and the obvious reading is a
    trap.  Writing the prediction as ``s*u_p`` and the target as ``sg(s)*u_g``
    gives ``L = s^2 ||u_p - u_g||^2``, whose derivative with respect to ``s`` is
    ``2s(1 - cos)`` -- positive for *every* non-zero direction error.  Nothing
    else in Eq. 11 constrains ``s`` (Eq. 8 divides pointmaps by their own mean
    norm, so it is scale-free), so that formulation is minimized by driving
    ``s`` to zero rather than by fixing the direction.

    Splitting the two removes the degeneracy:

    * ``L_trans`` is a pure **direction** term, ``||u_p - u_g||^2``, with no
      dependence on ``s`` at all;
    * ``L_scale`` supervises magnitude in log space against ``||t_gt||``, which
      is well-posed whenever the GT translation is metric (it is on ADT).
      Pass ``w_scale=0`` for a corpus that is only defined up to scale.
    """
    l_rot = geodesic_angle(R_pred, R_gt).mean()

    u_pred = torch.nn.functional.normalize(t_dir_pred, dim=-1, eps=1e-8)
    u_gt = torch.nn.functional.normalize(t_gt, dim=-1, eps=1e-8)
    l_trans = ((u_pred - u_gt) ** 2).sum(-1).mean()

    gt_norm = t_gt.norm(dim=-1).clamp(min=1e-8)
    l_scale = ((torch.log(scale_pred.clamp(min=1e-8)) - torch.log(gt_norm)) ** 2).mean()

    loss = w_rot * l_rot + w_trans * l_trans + w_scale * l_scale
    return loss, {
        "pose/rot_rad": float(l_rot.detach()),
        "pose/trans_sq": float(l_trans.detach()),
        "pose/scale_log_sq": float(l_scale.detach()),
    }


# --------------------------------------------------------------------------- #
# Eq. 11 -- total
# --------------------------------------------------------------------------- #

def cam3r_loss(
    preds: Dict,
    targets: Dict,
    lambda_a: float = 1.0,
    lambda_regr: float = 1.0,
    lambda_pose: float = 1.0,
    beta: float = 0.5,
    conf_mode: str = "none",
    conf_alpha: float = 0.2,
    w_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Eq. 11: ``lambda_A L_A + lambda_regr L_regr + lambda_pose L_pose``.

    ``preds`` carries ``rays``/``points``/``conf`` as 2-element lists (one per
    view) plus ``R``, ``t_dir``, ``scale``.  ``targets`` carries ``rays``,
    ``points``, ``valid`` (2-element lists) plus ``R``, ``t``.  Any target may
    be ``None`` to drop that term -- ADT has ray and pose GT, MegaDepth does not
    have rays.
    """
    device = preds["R"].device
    zero = torch.zeros((), device=device)
    logs: Dict[str, float] = {}

    valid: List[Optional[torch.Tensor]] = targets.get("valid") or [None, None]

    l_ang = zero
    if targets.get("rays") is not None:
        terms = [
            angular_loss(preds["rays"][v], targets["rays"][v], valid[v], beta=beta)
            for v in range(2)
        ]
        l_ang = torch.stack(terms).mean()
    logs["loss/angular"] = float(l_ang.detach())

    l_regr = zero
    if targets.get("points") is not None:
        conf = preds.get("conf") or [None, None]
        terms = [
            local_regression_loss(
                preds["points"][v], targets["points"][v], valid[v],
                conf=conf[v], conf_mode=conf_mode, conf_alpha=conf_alpha,
            )
            for v in range(2)
        ]
        l_regr = torch.stack(terms).mean()
    logs["loss/regr"] = float(l_regr.detach())

    l_pose = zero
    if targets.get("R") is not None:
        l_pose, pose_logs = relative_pose_loss(
            preds["R"], preds["t_dir"], preds["scale"], targets["R"], targets["t"],
            w_scale=w_scale,
        )
        logs.update(pose_logs)
    logs["loss/pose"] = float(l_pose.detach())

    total = lambda_a * l_ang + lambda_regr * l_regr + lambda_pose * l_pose
    logs["loss/total"] = float(total.detach())
    return total, logs
