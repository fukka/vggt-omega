# Copyright (c) 2026.
"""DAv2-focused trainer: freeze VGGT, train ONLY DAv2.

The alternating co-distillation is overkill — and, per the experiments, harmful —
when the goal is just to improve DAv2: VGGT no longer needs DAv2 as a teacher, and
running Phase B both wastes compute and (via the VGGT->DAv2 chain) drags DAv2 down.
This trainer drops Phase B entirely: VGGT is a FROZEN teacher and every step trains
DAv2, so the Phase-A losses can be ablated cleanly — turn them on one at a time via
the ``dav2_a*.yaml`` ladder to read each loss's marginal effect on DAv2.

It also adds one loss the alternating trainers lack: an optional scale-shift-
invariant SELF-ANCHOR to the frozen pretrained DAv2 (``w_a_anchor``) — the DAv2
mirror of the VGGT structure anchor that turned long VGGT finetuning from a
regression into a gain. Judge runs under ``scale_shift``.
"""
from __future__ import annotations

import copy
import math

import torch

from ..losses.distillation import ssi_loss, to_disparity
from ..models import count_parameters
from ..registry import TRAINER_REGISTRY
from .base import _cycle, _is_main, _unwrap, _warmup_cosine
from .ssi import SSITrainer


@TRAINER_REGISTRY.register(name="dav2")
class DAv2Trainer(SSITrainer):
    def __init__(self, vggt, dav2, cfg, device="cpu") -> None:
        super().__init__(vggt, dav2, cfg, device=device)
        # DAv2-only: every micro-step trains DAv2 for the FULL per-phase length
        # (no Phase B). Configs set dav2_steps_mult: 0 so epoch-sizing routes the
        # whole budget here; then rebuild the DAv2 LR schedule to span it.
        self.steps_a = max(1, cfg.steps_per_phase)
        self._total_steps = cfg.rounds * self.steps_a   # Phase A only (no Phase B)
        accum = max(1, cfg.grad_accum)
        opt_steps_d = cfg.rounds * math.ceil(self.steps_a / accum)
        self.sched_dav2 = torch.optim.lr_scheduler.LambdaLR(
            self.opt_dav2,
            _warmup_cosine(cfg.warmup_steps, opt_steps_d, cfg.min_lr_ratio, cfg.lr_schedule),
        )
        # optional SSI self-anchor to the frozen pretrained DAv2
        self.ref_dav2 = None
        if cfg.w_a_anchor > 0:
            self.ref_dav2 = copy.deepcopy(_unwrap(self.dav2)).to(self.device).eval()
            for p in self.ref_dav2.parameters():
                p.requires_grad_(False)
            self.logger.text(
                f"[finetune] dav2: SSI self-anchor to a frozen pretrained DAv2 "
                f"(w_a_anchor={cfg.w_a_anchor}); extra GPU memory in use."
            )

    # --- loss: Phase A (+ optional self-anchor) ----------------------------- #
    def phase_a_losses(self, images, dav2_depth, depth_v, pose_enc, conf_v):
        total, logs, dyn = super().phase_a_losses(images, dav2_depth, depth_v, pose_enc, conf_v)
        if self.ref_dav2 is not None:
            with torch.no_grad():
                ref = self._dav2_depth(self.ref_dav2, images)   # frozen pretrained DAv2
            B, S, H, W = dav2_depth.shape
            anchor = ssi_loss(
                to_disparity(dav2_depth).reshape(B * S, H, W),
                to_disparity(ref).reshape(B * S, H, W),
            )
            total = total + self.cfg.w_a_anchor * anchor
            logs["anchor"] = float(anchor.detach())
            logs["total"] = float(total.detach())
        return total, logs, dyn

    # --- validation: DAv2 only; best checkpoint by Phase-A val total -------- #
    @torch.no_grad()
    def validate(self, val_loader):
        self.vggt.eval()
        self.dav2.eval()
        agg, n = {}, 0
        for i, batch in enumerate(val_loader):
            if i >= self.cfg.val_steps:
                break
            images = batch["images"].to(self.device)
            depth_v, pose_enc, conf_v = self._vggt_forward(images, train=False)
            dav2_live = self._dav2_depth(self.dav2, images)
            _, la, _ = self.phase_a_losses(images, dav2_live, depth_v, pose_enc, conf_v)
            for k, v in la.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        if n == 0:
            return {}
        return {"A": {k: self._reduce_scalar(v / n) for k, v in agg.items()}}

    def _do_validation(self, val_loader, tag):
        results = self.validate(val_loader)
        if not results:
            return
        self.logger.log_scalars(self.global_step, "val", "A", results["A"])
        a_tot = results["A"].get("total", float("nan"))
        self.logger.text(f"[{tag}] VAL gstep {self.global_step}  DAv2 A/total={a_tot:.4f}")
        if a_tot < self.best_val:                      # best = best DAv2 val (not VGGT)
            self.best_val = a_tot
            self.save_checkpoint("best")

    # --- loop: Phase A only ------------------------------------------------- #
    def train(self, loader, val_loader=None):
        cfg = self.cfg
        if _is_main():
            td, total_d = count_parameters(_unwrap(self.dav2))
            self.logger.text(f"[finetune] trainer={cfg.trainer!r} run={cfg.name!r} -> {cfg.out_dir}")
            self.logger.text(
                f"[finetune] DAv2-only: VGGT FROZEN (teacher) | DAv2 trainable "
                f"{td/1e6:.2f}M / {total_d/1e6:.2f}M | {cfg.rounds} x {self.steps_a} DAv2 steps"
            )
        if cfg.eval_every > 0:
            self._periodic_eval(val_loader)
        data = _cycle(loader)
        for rnd in range(cfg.rounds):
            self._run_phase("A", data, self._phase_a, self.steps_a, self.opt_dav2,
                            f"round {rnd} / DAv2", val_loader)
        self.save_checkpoint("final")
        self.logger.close()
