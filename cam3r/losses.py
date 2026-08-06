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
* **Relative pose loss** (Eq. 9-10): geodesic rotation error plus a translation
  term weighted by the detached pointmap-magnitude ratio.

Unspecified in the paper, chosen here (see cam3r/README.md "Unspecified"):
``beta = 0.5``, ``lambda_A = lambda_regr = lambda_pose = 1.0``, azimuth
differences are wrapped into ``(-pi, pi]``, and the confidence term follows
DUSt3R.  Each is a keyword argument, so none of them is baked in.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from cam3r.geometry import geodesic_angle, rays_to_spherical, wrap_angle
from cam3r.model import pointmap_scale


# --------------------------------------------------------------------------- #
# Eq. 5-6 -- asymmetric angular loss
# --------------------------------------------------------------------------- #

def _reduce(per_elem: torch.Tensor, valid: Optional[torch.Tensor], reduction: str) -> torch.Tensor:
    """Collapse a per-element loss over the valid set.

    ``"sum"`` is what Eq. 5 and Eq. 8 literally write -- neither carries a
    ``1/|D^v|`` -- and ``"mean"`` is the default here.  The difference is not
    cosmetic: a sum scales with resolution and with how much of the frame is
    inside the lens cone, so at fixed ``lambda_A = lambda_regr = lambda_pose = 1``
    (values the paper never gives) the *balance* between the three terms of
    Eq. 11 would depend on image size, and a 512 px fisheye with a 55 deg cone
    would weight ``L_regr`` differently from a 512 px panorama. See
    ``reproduction.md``.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum' (got {reduction!r})")
    if valid is None:
        return per_elem.sum() if reduction == "sum" else per_elem.mean()
    if not bool(valid.any()):
        return per_elem.sum() * 0.0
    picked = per_elem[valid]
    return picked.sum() if reduction == "sum" else picked.mean()


def asymmetric_angular_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    valid: Optional[torch.Tensor] = None,
    wrapped: bool = False,
    reduction: str = "mean",
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
    return _reduce(weight * diff.abs(), valid, reduction)


def angular_loss(
    pred_rays: torch.Tensor,
    target_rays: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    alpha_theta: float = 0.7,
    alpha_phi: float = 0.5,
    reduction: str = "mean",
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
    l_theta = asymmetric_angular_loss(theta_p, theta_t, alpha_theta, valid, reduction=reduction)
    l_phi = asymmetric_angular_loss(phi_p, phi_t, alpha_phi, valid, wrapped=True, reduction=reduction)
    return beta * l_theta + (1.0 - beta) * l_phi


# --------------------------------------------------------------------------- #
# Eq. 7-8 -- local regression loss
# --------------------------------------------------------------------------- #

def _normalize_by_mean_norm(
    pts: torch.Tensor, valid: Optional[torch.Tensor], eps: float = 1e-6
) -> torch.Tensor:
    """Eq. 7: divide each view's pointmap by its own mean point norm ``eta``."""
    return pts / pointmap_scale(pts, valid, eps).view(-1, 1, 1, 1)


def local_regression_loss(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    conf: Optional[torch.Tensor] = None,
    conf_mode: str = "none",
    conf_alpha: float = 0.2,
    reduction: str = "mean",
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

    return _reduce(per_pix, valid, reduction)


# --------------------------------------------------------------------------- #
# Eq. 9-10 -- relative pose loss
# --------------------------------------------------------------------------- #

def relative_pose_loss(
    R_pred: torch.Tensor,
    t_dir_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    w_rot: float = 1.0,
    w_trans: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Eq. 9-10: ``L_pose = lambda (L_rot + L_trans)``.

    Eq. 9 is the geodesic error on SO(3).  Eq. 10 is
    ``||t_{2->1} - t*_{2->1}||^2`` with ``t = s t_hat`` on both sides, where the
    paper takes ``s`` from "the ratio of the predicted pointmap magnitudes to
    the ground-truth magnitudes" and **detaches** it "to serve as a static
    target".  So the same constant multiplies prediction and target, and the
    term is a direction loss weighted by ``s^2``:

        L_trans = s^2 ||u_pred - u_gt||^2 ,   s = sg(eta_pred / eta_gt)

    Two consequences worth being explicit about.  ``s`` carries no gradient, so
    nothing in Eq. 11 supervises a scale *head* -- which is why
    :class:`cam3r.model.PoseHead` does not have one and the magnitude of ``t``
    is read off the pointmap instead.  And because ``s`` is a ratio, a corpus
    that is only defined up to scale still trains: it drops out of both sides.

    ``scale`` may be omitted, which reduces the term to a plain unit-direction
    loss (``s = 1``) -- the right thing when no pointmap GT accompanies the pose.
    """
    l_rot = geodesic_angle(R_pred, R_gt).mean()

    u_pred = torch.nn.functional.normalize(t_dir_pred, dim=-1, eps=1e-8)
    u_gt = torch.nn.functional.normalize(t_gt, dim=-1, eps=1e-8)
    s = torch.ones_like(u_pred[..., 0]) if scale is None else scale.detach()
    l_trans = (s**2 * ((u_pred - u_gt) ** 2).sum(-1)).mean()

    loss = w_rot * l_rot + w_trans * l_trans
    return loss, {
        "pose/rot_rad": float(l_rot.detach()),
        "pose/trans_sq": float(l_trans.detach()),
        "pose/scale": float(s.mean().detach()),
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
    reduction: str = "mean",
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

    # Eq. 8 sums over v in {1, 2}; averaging keeps the term independent of how
    # many views a sample carries.  Tied to `reduction` so the two cannot
    # disagree about whether this objective sums or means.
    def over_views(terms: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(terms)
        return stacked.sum() if reduction == "sum" else stacked.mean()

    l_ang = zero
    if targets.get("rays") is not None:
        l_ang = over_views([
            angular_loss(preds["rays"][v], targets["rays"][v], valid[v],
                         beta=beta, reduction=reduction)
            for v in range(2)
        ])
    logs["loss/angular"] = float(l_ang.detach())

    l_regr = zero
    if targets.get("points") is not None:
        conf = preds.get("conf") or [None, None]
        l_regr = over_views([
            local_regression_loss(
                preds["points"][v], targets["points"][v], valid[v],
                conf=conf[v], conf_mode=conf_mode, conf_alpha=conf_alpha,
                reduction=reduction,
            )
            for v in range(2)
        ])
    logs["loss/regr"] = float(l_regr.detach())

    l_pose = zero
    if targets.get("R") is not None:
        # Eq. 10's s = eta_pred / eta_gt, both measured on the reference view.
        s = None
        if preds.get("points") is not None and targets.get("points") is not None:
            s = pointmap_scale(preds["points"][0], valid[0]) / pointmap_scale(
                targets["points"][0], valid[0]
            )
        l_pose, pose_logs = relative_pose_loss(
            preds["R"], preds["t_dir"], targets["R"], targets["t"], scale=s,
        )
        logs.update(pose_logs)
    logs["loss/pose"] = float(l_pose.detach())

    total = lambda_a * l_ang + lambda_regr * l_regr + lambda_pose * l_pose
    logs["loss/total"] = float(total.detach())
    return total, logs
