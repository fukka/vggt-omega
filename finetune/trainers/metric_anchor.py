# Copyright (c) 2026.
"""Metric-anchor strategy: like SSI, but preserve VGGT's metric scale.

Diagnosis this addresses: every SSI Phase-B term is invariant to a global
rescaling of VGGT depth (with a compensating pose-translation scale), so the
released checkpoint's *metric* calibration is free to drift during finetuning —
which is exactly what the ``none`` (metric) eval measures. This variant adds a
light anchor that ties the live depth to a **frozen copy of the pretrained
VGGT** evaluated on the same frames, in log-depth space and weighted by VGGT's
own confidence. It pins the otherwise-free scale gauge (and nudges local metric
structure back toward pretrained) while the self-supervised + distillation terms
still refine structure.

Cost: this keeps a second, frozen VGGT in memory (no grad, runs under autocast).
It is only built when ``w_metric_anchor > 0``; otherwise this behaves exactly
like the SSI trainer. Run ``trainer: ssi`` (no reference model) and
``trainer: metric_anchor`` side by side to compare metric vs up-to-scale.
"""
from __future__ import annotations

import copy

import torch

from ..geometry import EPS, masked_weighted_mean
from ..registry import TRAINER_REGISTRY
from .base import _squeeze_depth, _unwrap
from .ssi import SSITrainer, _conf_weight


@TRAINER_REGISTRY.register(name="metric_anchor")
class MetricAnchorTrainer(SSITrainer):
    def __init__(self, vggt, dav2, cfg, device="cpu") -> None:
        super().__init__(vggt, dav2, cfg, device=device)
        self.ref_vggt = None
        if cfg.w_metric_anchor > 0:
            # Snapshot the pretrained model: at construction time LoRA deltas are
            # zero and the heads are still pretrained, so this copy reproduces the
            # released checkpoint's metric depth and never updates.
            self.ref_vggt = copy.deepcopy(_unwrap(self.vggt)).to(self.device).eval()
            for p in self.ref_vggt.parameters():
                p.requires_grad_(False)
            self.logger.text(
                "[finetune] metric_anchor: holding a frozen reference VGGT "
                f"(w_metric_anchor={cfg.w_metric_anchor}); extra GPU memory in use."
            )

    @torch.no_grad()
    def _reference_depth(self, images: torch.Tensor) -> torch.Tensor:
        with self._autocast():
            preds = self.ref_vggt(images)
        return _squeeze_depth(preds["depth"]).float()

    def _metric_anchor_loss(self, images, depth, conf) -> torch.Tensor:
        ref = self._reference_depth(images)                      # detached (no_grad)
        ld = depth.clamp_min(1e-3).log()
        lr = ref.clamp_min(1e-3).log()
        if self.cfg.metric_anchor_mode == "scale":
            # Pin only the per-frame global (geometric-mean) depth scale — this
            # is exactly the depth<->pose-translation gauge that the SSI terms
            # leave free — and let structure adapt to the distillation freely.
            diff = ld.mean(dim=(-1, -2)) - lr.mean(dim=(-1, -2))  # [B,S]
            return (diff ** 2).mean()
        # "full": proximal anchor — keep depth close to pretrained per pixel
        # (preserves scale AND structure; more conservative finetuning).
        l = (ld - lr).abs()
        w = _conf_weight(conf) if conf is not None else torch.ones_like(l)
        return masked_weighted_mean(l, w, torch.ones_like(l, dtype=torch.bool))

    def phase_b_losses(self, images, depth, pose_enc, conf, dav2_depth):
        total, logs, dyn = super().phase_b_losses(images, depth, pose_enc, conf, dav2_depth)
        if self.ref_vggt is not None:
            anchor = self._metric_anchor_loss(images, depth, conf)
            total = total + self.cfg.w_metric_anchor * anchor
            logs["metric_anchor"] = float(anchor.detach())
            logs["total"] = float(total.detach())
        return total, logs, dyn
