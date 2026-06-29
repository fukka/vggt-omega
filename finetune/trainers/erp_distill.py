# Copyright (c) 2026.
"""Unsupervised DAC-style ERP finetuning of DAv2 (single-model, no VGGT).

This trainer is deliberately *not* part of the alternating VGGT↔DAv2 machinery in
``base.py`` — there is no second model and no GT. It finetunes DAv2 to reason in
DAC's ERP canonical space on EgoExo4D fisheye, using only the label-free terms in
:mod:`finetune.losses.erp_consistency`:

    total = w_erp_consistency · equivariance(DAv2(T·I), T·teacher(I))
          + w_erp_anchor      · ssi(DAv2(I), frozen_pretrained_DAv2(I))
          + w_erp_smooth      · edge_aware_smoothness(DAv2(I), I)

``teacher`` is the EMA copy (default), the frozen pretrained model, or the live
student (pure self-consistency) — see ``erp_teacher``. Outputs mirror the other
runs: ``runs/<name>/`` with metrics, viz, and trainable-only checkpoints.
"""
from __future__ import annotations

import copy
import math
import os
from contextlib import nullcontext
from typing import Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..config import FinetuneConfig
from ..engine.logger import TrainLogger
from ..losses.erp_consistency import (erp_anchor_loss, erp_equivariance_loss,
                                       random_similarity, warp_similarity)
from ..losses.photometric import edge_aware_smoothness
from ..models import EmaTeacher, count_parameters, trainable_parameters
from ..registry import TRAINER_REGISTRY
from ..viz import training_montage
from .base import _is_main, _param_groups, _unwrap, _warmup_cosine


@TRAINER_REGISTRY.register(name="erp_distill")
class ErpDistillTrainer:
    """Single-model unsupervised ERP finetuning of DAv2."""

    def __init__(self, dav2: nn.Module, cfg: FinetuneConfig, device="cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.dav2 = dav2.to(self.device)
        dav2_raw = _unwrap(self.dav2)
        self.params = trainable_parameters(dav2_raw)

        groups = _param_groups(dav2_raw, cfg.lr_dav2, cfg.lr_dav2, cfg.weight_decay,
                               no_decay_substrings=("head",))
        self.opt = torch.optim.AdamW(
            groups or [{"params": list(dav2_raw.parameters()), "lr": cfg.lr_dav2}],
            lr=cfg.lr_dav2, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps)

        self.total_steps = max(1, cfg.rounds * cfg.steps_per_phase)
        accum = max(1, cfg.grad_accum)
        opt_steps = math.ceil(self.total_steps / accum)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, _warmup_cosine(cfg.warmup_steps, opt_steps, cfg.min_lr_ratio, cfg.lr_schedule))

        self._amp_dtype = None
        if cfg.amp and cfg.amp_dtype != "fp32" and self.device.type == "cuda":
            self._amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
        self._use_scaler = self._amp_dtype == torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_scaler)

        # EMA teacher for the equivariance target (mean-teacher; stable structure).
        self.ema = EmaTeacher(dav2_raw, cfg.ema_decay) if cfg.ema_teacher else None
        # frozen pretrained DAv2 for the SSI anchor (protect the structure prior).
        self.ref = None
        if cfg.w_erp_anchor > 0:
            self.ref = copy.deepcopy(dav2_raw).to(self.device).eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)

        self._rng = torch.Generator(device=self.device).manual_seed(cfg.seed + 1234)
        self.global_step = 0
        self.best_val = float("inf")
        self.logger = TrainLogger(cfg.out_dir, enabled=_is_main(), use_tensorboard=cfg.tensorboard)

    # ------------------------------------------------------------------ #
    def _autocast(self):
        return nullcontext() if self._amp_dtype is None else torch.autocast(
            device_type=self.device.type, dtype=self._amp_dtype)

    def _depth(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """``images [B,1,3,H,W]`` → depth ``[B,1,H,W]`` (fp32)."""
        with self._autocast():
            d = model(images)
        if d.dim() == 5 and d.shape[-1] == 1:
            d = d[..., 0]
        return d.float()

    def _teacher(self) -> nn.Module:
        mode = getattr(self.cfg, "erp_teacher", "ema")
        if mode == "ema" and self.ema is not None:
            return self.ema.model
        if mode == "frozen" and self.ref is not None:
            return self.ref
        return self.dav2  # "self": live student (detached at the loss)

    def _reduce(self, x: float) -> float:
        if not dist.is_initialized():
            return x
        t = torch.tensor([x], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item() / dist.get_world_size())

    # ------------------------------------------------------------------ #
    def _losses(self, images: torch.Tensor, active: torch.Tensor, train: bool):
        """Compute the unsupervised ERP losses for one batch. Returns (total, logs, viz_pack)."""
        cfg = self.cfg
        B = images.shape[0]
        H, W = images.shape[-2:]

        # equivariance: build T and warp the input
        theta = random_similarity(
            B, cfg.erp_roll_deg, cfg.erp_scale_lo, cfg.erp_scale_hi,
            cfg.erp_trans_frac, self.device, generator=self._rng)
        images_T = warp_similarity(images.reshape(B, 3, H, W), theta).reshape(B, 1, 3, H, W)
        # ONE student forward over [I ; T·I] (a single backward → DDP-safe, no
        # double gradient-hook firing on shared params).
        both = torch.cat([images, images_T], dim=0)              # [2B,1,3,H,W]
        d_both = self._depth(self.dav2, both)                    # [2B,1,H,W], grad
        d_I, d_TI = d_both[:B], d_both[B:]
        with torch.no_grad():
            teacher = self._teacher()
            teacher.eval()
            d_teacher = self._depth(teacher, images)             # canonical target
        equiv, _ = erp_equivariance_loss(d_TI, d_teacher, theta, active)

        # anchor to frozen pretrained DAv2
        anchor = d_I.new_zeros(())
        if self.ref is not None and cfg.w_erp_anchor > 0:
            with torch.no_grad():
                d_ref = self._depth(self.ref, images)
            anchor = erp_anchor_loss(d_I, d_ref, active)

        # edge-aware smoothness on the canonical depth, inside the active cone
        smooth = d_I.new_zeros(())
        if cfg.w_erp_smooth > 0:
            a = (active.reshape(B, 1, H, W) > 0.5).float()
            smooth = edge_aware_smoothness(
                _squeeze_bhw(d_I) * a[:, 0], images.reshape(B, 3, H, W))

        total = (cfg.w_erp_consistency * equiv
                 + cfg.w_erp_anchor * anchor
                 + cfg.w_erp_smooth * smooth)
        logs = {
            "total": float(total.detach()),
            "equiv": float(equiv.detach()),
            "anchor": float(anchor.detach()) if torch.is_tensor(anchor) else float(anchor),
            "smooth": float(smooth.detach()) if torch.is_tensor(smooth) else float(smooth),
        }
        viz = None
        if not train:
            return total, logs, None
        viz = {"images": images[0].detach().float().cpu(),
               "student": d_I[0].detach().float().cpu(),
               "teacher": d_teacher[0].detach().float().cpu()}
        return total, logs, viz

    # ------------------------------------------------------------------ #
    def _step(self, batch, do_step: bool, collect_viz: bool):
        images = batch["images"].to(self.device)
        active = batch["active"].to(self.device)
        self.dav2.train()
        total, logs, viz = self._losses(images, active, train=True)
        loss = total / max(1, self.cfg.grad_accum)
        if self._use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if do_step:
            if self._use_scaler:
                if self.cfg.grad_clip:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                if self.cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
                self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()
            if self.ema is not None:
                self.ema.update(_unwrap(self.dav2))
        return logs, (viz if collect_viz else None)

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        if val_loader is None:
            return {}
        self.dav2.eval()
        agg, n = {}, 0
        for i, batch in enumerate(val_loader):
            if i >= self.cfg.val_steps:
                break
            images = batch["images"].to(self.device)
            active = batch["active"].to(self.device)
            _, logs, _ = self._losses(images, active, train=False)
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        return {k: self._reduce(v / n) for k, v in agg.items()} if n else {}

    def _do_val(self, val_loader, tag: str) -> None:
        res = self.validate(val_loader)
        if not res:
            return
        self.logger.log_scalars(self.global_step, "val", "erp", res)
        tot = res.get("total", float("nan"))
        self.logger.text(f"[{tag}] VAL gstep {self.global_step} total={tot:.4f}")
        if tot < self.best_val:
            self.best_val = tot
            self.save_checkpoint("best")

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, tag: str = "last") -> None:
        if not _is_main():
            return
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        raw = _unwrap(self.dav2)
        keep = {n for n, p in raw.named_parameters() if p.requires_grad}
        sd = raw.state_dict()
        torch.save(
            {"dav2": {k: sd[k].detach().cpu() for k in sd if k in keep},
             "global_step": self.global_step, "best_val": self.best_val,
             "cfg": vars(self.cfg)},
            os.path.join(self.cfg.out_dir, f"checkpoint_{tag}.pt"))
        self.logger.text(f"[finetune] saved checkpoint_{tag}.pt")

    def _save_viz(self, viz) -> None:
        if not _is_main() or viz is None:
            return
        import cv2

        out = os.path.join(self.cfg.out_dir, "viz")
        os.makedirs(out, exist_ok=True)
        montage = training_montage(
            viz["images"], viz["teacher"], viz["student"], None,
            num_frames=1, caption=f"ERP gstep {self.global_step} (in|teacher|student)")
        cv2.imwrite(os.path.join(out, f"erp_step{self.global_step:07d}.jpg"), montage)

    # ------------------------------------------------------------------ #
    def train(self, loader: Iterable, val_loader: Optional[Iterable] = None) -> None:
        cfg = self.cfg
        if _is_main():
            td, total_d = count_parameters(_unwrap(self.dav2))
            self.logger.text(f"[finetune] trainer={cfg.trainer!r} run={cfg.name!r} -> {cfg.out_dir}")
            self.logger.text(
                f"[finetune] ERP-DAv2 (unsupervised): teacher={getattr(cfg,'erp_teacher','ema')} "
                f"| DAv2 trainable {td/1e6:.2f}M / {total_d/1e6:.2f}M "
                f"| {self.total_steps} steps | w(equiv/anchor/smooth)="
                f"{cfg.w_erp_consistency}/{cfg.w_erp_anchor}/{cfg.w_erp_smooth}")

        from .base import _cycle

        data = _cycle(loader)
        accum = max(1, cfg.grad_accum)
        running, count = {}, 0
        for it in range(self.total_steps):
            batch = next(data)
            do_step = ((it + 1) % accum == 0) or (it + 1 == self.total_steps)
            do_viz = _is_main() and cfg.viz_every > 0 and (self.global_step + 1) % cfg.viz_every == 0
            logs, viz = self._step(batch, do_step=do_step, collect_viz=do_viz)
            self.global_step += 1
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            count += 1

            if self.global_step % cfg.log_every == 0:
                avg = {k: self._reduce(running[k] / count) for k in running}
                avg["lr"] = self.opt.param_groups[-1]["lr"]
                self.logger.log_scalars(self.global_step, "train", "erp", avg)
                if _is_main():
                    self.logger.text(f"[erp] step {it+1}/{self.total_steps} (gstep "
                                     f"{self.global_step}) " + " ".join(f"{k}={avg[k]:.4g}" for k in avg))
                running, count = {}, 0
            if viz is not None:
                self._save_viz(viz)
            if val_loader is not None and cfg.val_every > 0 and self.global_step % cfg.val_every == 0:
                self._do_val(val_loader, "erp")
            if cfg.save_every > 0 and self.global_step % cfg.save_every == 0:
                self.save_checkpoint("last")

        if val_loader is not None:
            self._do_val(val_loader, "erp-final")
        self.save_checkpoint("final")
        self.logger.close()


def _squeeze_bhw(d: torch.Tensor) -> torch.Tensor:
    """``[B,1,H,W]`` → ``[B,H,W]``."""
    return d[:, 0] if d.dim() == 4 else d
