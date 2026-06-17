# Copyright (c) 2026.
"""Alternating co-distillation trainer for Depth-Anything-V2 <-> VGGT-Omega.

Phase A (improve DAv2): freeze VGGT-Omega, use its multi-view-consistent depth +
poses to (i) affine-align-and-distill into DAv2 and (ii) make DAv2 multi-view
consistent.

Phase B (improve VGGT-Omega): freeze DAv2, train VGGT-Omega with the
self-supervised photometric + geometric stack plus scale-shift-invariant
structure distillation from DAv2 (finite-depth prior + sharp edges).

Both directions are label-free. One model is always frozen and acts as a clean
teacher (alternating / EM style), which is more stable than joint co-training.
"""
from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import FinetuneConfig
from ..losses import (
    affine_invariant_l1,
    compute_self_supervised_losses,
    gradient_matching_loss,
    multiview_consistency_loss,
    ssi_loss,
    to_disparity,
)
from ..models import EmaTeacher, count_parameters, trainable_parameters


def _unwrap(m: nn.Module) -> nn.Module:
    """Strip DDP wrapper; no-op on plain modules."""
    return m.module if isinstance(m, DDP) else m


def _is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _cycle(loader: Iterable):
    while True:
        for batch in loader:
            yield batch


def _squeeze_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    return depth


class AlternatingTrainer:
    def __init__(
        self,
        vggt: nn.Module,
        dav2: nn.Module,
        cfg: FinetuneConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        # models may already be DDP-wrapped and on device; .to() is a no-op if so
        self.vggt = vggt.to(self.device)
        self.dav2 = dav2.to(self.device)

        # Unwrap DDP to access raw parameters for optimizer + EMA
        vggt_raw = _unwrap(self.vggt)
        dav2_raw = _unwrap(self.dav2)

        self.vggt_params = trainable_parameters(vggt_raw)
        self.dav2_params = trainable_parameters(dav2_raw)
        self.opt_vggt = torch.optim.AdamW(
            self.vggt_params or list(vggt_raw.parameters()),
            lr=cfg.lr_vggt,
            weight_decay=cfg.weight_decay,
        )
        self.opt_dav2 = torch.optim.AdamW(
            self.dav2_params or list(dav2_raw.parameters()),
            lr=cfg.lr_dav2,
            weight_decay=cfg.weight_decay,
        )

        self.use_amp = bool(cfg.amp) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # EMA is built from the raw module so it stays a plain nn.Module
        self.ema = EmaTeacher(dav2_raw, cfg.ema_decay) if cfg.ema_teacher else None

    # ------------------------------------------------------------------ #
    def _vggt_forward(self, images: torch.Tensor, train: bool):
        self.vggt.train(train)
        preds = self.vggt(images)
        depth = _squeeze_depth(preds["depth"])
        pose_enc = preds["pose_enc"]
        conf = preds.get("depth_conf")
        return depth, pose_enc, conf

    def _backward_step(self, loss, optimizer, params):
        optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            optimizer.step()

    # ------------------------------------------------------------------ #
    def phase_b_step(self, images: torch.Tensor) -> Dict[str, float]:
        """Improve VGGT-Omega (DAv2 frozen)."""
        cfg = self.cfg
        images = images.to(self.device)
        self.dav2.eval()

        depth, pose_enc, conf = self._vggt_forward(images, train=True)
        with torch.no_grad():
            dav2_depth = _squeeze_depth(self.dav2(images))

        ss, _dyn = compute_self_supervised_losses(
            images, depth, pose_enc, conf=conf, offsets=cfg.offsets, alpha=cfg.ssim_alpha
        )

        B, S, _, H, W = images.shape
        v_disp = to_disparity(depth).reshape(B * S, H, W)
        d_disp = to_disparity(dav2_depth).reshape(B * S, H, W)
        distill_ssi = ssi_loss(v_disp, d_disp)
        distill_grad = gradient_matching_loss(v_disp, d_disp)

        total = (
            cfg.w_photometric * ss["photometric"]
            + cfg.w_geometric * ss["geometric"]
            + cfg.w_smoothness * ss["smoothness"]
            + cfg.w_distill_ssi * distill_ssi
            + cfg.w_distill_grad * distill_grad
        )
        self._backward_step(total, self.opt_vggt, self.vggt_params)
        return {
            "phaseB/total": float(total.detach()),
            "phaseB/photometric": float(ss["photometric"].detach()),
            "phaseB/geometric": float(ss["geometric"].detach()),
            "phaseB/smoothness": float(ss["smoothness"].detach()),
            "phaseB/distill_ssi": float(distill_ssi.detach()),
            "phaseB/distill_grad": float(distill_grad.detach()),
        }

    def phase_a_step(self, images: torch.Tensor) -> Dict[str, float]:
        """Improve DAv2 (VGGT-Omega frozen)."""
        cfg = self.cfg
        images = images.to(self.device)

        with torch.no_grad():
            depth_v, pose_enc, _ = self._vggt_forward(images, train=False)

        self.dav2.train()
        dav2_depth = _squeeze_depth(self.dav2(images))

        B, S, _, H, W = images.shape
        d_disp = to_disparity(dav2_depth).reshape(B * S, H, W)
        v_disp = to_disparity(depth_v).reshape(B * S, H, W)
        distill = affine_invariant_l1(d_disp, v_disp)  # align DAv2 -> VGGT
        multiview = multiview_consistency_loss(dav2_depth, pose_enc, offsets=cfg.offsets)

        total = cfg.w_a_distill * distill + cfg.w_a_multiview * multiview
        self._backward_step(total, self.opt_dav2, self.dav2_params)
        if self.ema is not None:
            self.ema.update(self.dav2)
        return {
            "phaseA/total": float(total.detach()),
            "phaseA/distill": float(distill.detach()),
            "phaseA/multiview": float(multiview.detach()),
        }

    # ------------------------------------------------------------------ #
    def train(self, loader: Iterable) -> None:
        cfg = self.cfg
        if _is_main():
            tv, total_v = count_parameters(_unwrap(self.vggt))
            td, total_d = count_parameters(_unwrap(self.dav2))
            print(f"[finetune] VGGT trainable {tv/1e6:.2f}M / {total_v/1e6:.2f}M | "
                  f"DAv2 trainable {td/1e6:.2f}M / {total_d/1e6:.2f}M")
        data = _cycle(loader)
        for rnd in range(cfg.rounds):
            self._run_phase(data, self.phase_a_step, f"round {rnd} / phase A (DAv2)")
            self._run_phase(data, self.phase_b_step, f"round {rnd} / phase B (VGGT)")

    def _run_phase(self, data, step_fn, tag: str) -> None:
        running: Dict[str, float] = {}
        for it in range(self.cfg.steps_per_phase):
            batch = next(data)
            logs = step_fn(batch["images"])
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            if _is_main() and (it + 1) % self.cfg.log_every == 0:
                msg = " ".join(f"{k.split('/')[-1]}={running[k]/self.cfg.log_every:.4f}" for k in logs)
                print(f"[{tag}] step {it+1}/{self.cfg.steps_per_phase} {msg}")
                running = {}
