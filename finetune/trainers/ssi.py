# Copyright (c) 2026.
"""SSI strategy: scale-shift-invariant co-distillation.

Phase B (improve VGGT) keeps the original label-free stack — photometric +
geometric + smoothness + scale-shift-invariant distillation from DAv2. Every
term here is invariant to VGGT's *absolute* metric scale, so this variant
accepts up-to-scale depth and is judged on structure (``scale_shift`` eval). Use
``trainer: metric_anchor`` if you need the released checkpoint's metric scale
preserved (``none`` eval) — see ``trainers/metric_anchor.py``.

Phase A (improve DAv2) is the fixed version of the original: DAv2 now gets a real
**photometric appearance anchor** on the raw images (under VGGT's poses, after a
detached scale alignment), and the VGGT->DAv2 distillation is **gated** to
regions where VGGT is confident and static — so DAv2 is no longer dragged
wholesale toward a possibly-worse teacher.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from ..geometry import EPS
from ..losses import (
    affine_invariant_l1,
    compute_self_supervised_losses,
    gradient_matching_loss,
    ssi_loss,
    to_disparity,
)
from ..registry import TRAINER_REGISTRY
from .base import BaseAlternatingTrainer


def _conf_weight(conf: torch.Tensor) -> torch.Tensor:
    """Detached per-pixel weight from VGGT depth confidence (>=1): log-compressed."""
    return conf.detach().clamp(1.0, 50.0).log() + EPS


def _distill_gate(mode: str, conf, images, eps: float = 1e-3):
    """Per-pixel gate ``[B*S,H,W]`` in [0,1] for the Phase-B DAv2->VGGT distillation.

    Directs the (globally weaker) DAv2 teacher to where it actually helps:
      "conf" — where VGGT is UNcertain (low depth_conf): it needs correcting.
      "edge" — image edges / depth discontinuities: where DAv2 is sharp.
      "conf_edge" — a soft OR of the two (gate high if EITHER is high).
    Returns None for "none" (ungated). All inputs detached (gate is a weight).
    """
    if not mode or mode == "none":
        return None
    B, S = images.shape[:2]
    H, W = images.shape[-2:]
    gates = []
    if "conf" in mode and conf is not None:
        c = conf.detach().clamp(1.0, 50.0)
        gates.append((1.0 - (c - 1.0) / 49.0).reshape(B * S, H, W))  # 1=uncertain, 0=confident
    if "edge" in mode:
        g = images.detach().mean(2).reshape(B * S, H, W)             # grayscale
        gx = torch.zeros_like(g); gy = torch.zeros_like(g)
        gx[:, :, 1:] = (g[:, :, 1:] - g[:, :, :-1]).abs()
        gy[:, 1:, :] = (g[:, 1:, :] - g[:, :-1, :]).abs()
        gates.append(1.0 - torch.exp(-10.0 * (gx + gy)))             # 1=strong edge, 0=flat
    if not gates:
        return None
    gate = gates[0]
    for g in gates[1:]:                                              # soft OR: a + b - a*b
        gate = gate + g - gate * g
    return gate.clamp(eps, 1.0)


def _metric_align(dav2_depth: torch.Tensor, vggt_depth: torch.Tensor) -> torch.Tensor:
    """Scale DAv2 (relative) depth to VGGT's metric scale per frame.

    Uses a per-frame median ratio, **detached** so gradient flows only into
    DAv2's structure, not the (gauge-free) global scale. Needed so the
    photometric warp uses a depth that is metrically consistent with VGGT poses.
    """
    md = dav2_depth.flatten(2).median(dim=2).values        # [B,S]
    mv = vggt_depth.flatten(2).median(dim=2).values        # [B,S]
    s = (mv / md.clamp_min(EPS)).detach()[..., None, None]  # [B,S,1,1]
    return dav2_depth * s


@TRAINER_REGISTRY.register(name="ssi")
class SSITrainer(BaseAlternatingTrainer):
    def phase_b_losses(self, images, depth, pose_enc, conf, dav2_depth):
        cfg = self.cfg
        ss, dyn = compute_self_supervised_losses(
            images, depth, pose_enc, conf=conf,
            offsets=getattr(cfg, "offsets", (-1, 1)), alpha=cfg.ssim_alpha,
            dynamic_mask=cfg.use_dynamic_mask, dynamic_pow=cfg.dynamic_mask_pow,
        )
        B, S, _, H, W = images.shape
        v_disp = to_disparity(depth).reshape(B * S, H, W)
        d_disp = to_disparity(dav2_depth).reshape(B * S, H, W)
        # Gate the distillation to where DAv2 helps (VGGT is the better model
        # globally, so an ungated transfer drags it down). "none" = ungated.
        gate = _distill_gate(getattr(cfg, "b_distill_gate", "none"), conf, images)
        distill_ssi = ssi_loss(v_disp, d_disp, mask=gate)
        distill_grad = gradient_matching_loss(v_disp, d_disp, mask=gate)
        total = (
            cfg.w_photometric * ss["photometric"]
            + cfg.w_geometric * ss["geometric"]
            + cfg.w_smoothness * ss["smoothness"]
            + cfg.w_distill_ssi * distill_ssi
            + cfg.w_distill_grad * distill_grad
        )
        logs = {
            "total": float(total.detach()),
            "photometric": float(ss["photometric"].detach()),
            "geometric": float(ss["geometric"].detach()),
            "smoothness": float(ss["smoothness"].detach()),
            "distill_ssi": float(distill_ssi.detach()),
            "distill_grad": float(distill_grad.detach()),
        }
        return total, logs, dyn

    def phase_a_losses(self, images, dav2_depth, depth_v, pose_enc, conf_v):
        cfg = self.cfg
        B, S, _, H, W = images.shape

        # --- appearance anchor: photometric reprojection of DAv2 depth under
        #     VGGT poses (DAv2 metric-aligned, scale detached) ---------------- #
        dav2_metric = _metric_align(dav2_depth, depth_v)
        ss, dyn = compute_self_supervised_losses(
            images, dav2_metric, pose_enc, conf=None,
            offsets=getattr(cfg, "offsets", (-1, 1)), alpha=cfg.ssim_alpha,
            dynamic_mask=cfg.use_dynamic_mask, dynamic_pow=cfg.dynamic_mask_pow,
        )

        # --- distill VGGT structure into DAv2 (disparity space), gated by VGGT
        #     (teacher) CONFIDENCE so a low-confidence teacher can't drag DAv2.
        #     Note: we deliberately do NOT gate by DAv2's own consistency — that
        #     would suppress distillation exactly where DAv2 is wrong, i.e. where
        #     it most needs correcting. --------------------------------------- #
        d_disp = to_disparity(dav2_depth).reshape(B * S, H, W)
        v_disp = to_disparity(depth_v).reshape(B * S, H, W)
        gate = None
        if cfg.a_distill_gate and conf_v is not None:
            gate = _conf_weight(conf_v).reshape(B * S, H, W)
        distill = affine_invariant_l1(d_disp, v_disp, mask=gate)

        total = (
            cfg.w_a_distill * distill
            + cfg.w_a_multiview * ss["geometric"]
            + cfg.w_a_photometric * ss["photometric"]
            + cfg.w_a_smoothness * ss["smoothness"]
        )
        logs = {
            "total": float(total.detach()),
            "distill": float(distill.detach()),
            "multiview": float(ss["geometric"].detach()),
            "photometric": float(ss["photometric"].detach()),
            "smoothness": float(ss["smoothness"].detach()),
        }
        return total, logs, dyn
