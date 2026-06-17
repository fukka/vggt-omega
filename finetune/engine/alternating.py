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

import os
from typing import Dict, Iterable, Optional, Tuple

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
from ..viz import training_montage
from .logger import TrainLogger


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

        # monitoring state
        self.global_step = 0
        self.best_val = float("inf")  # tracks Phase-B val total (VGGT is the target model)
        self.logger = TrainLogger(
            cfg.out_dir, enabled=_is_main(), use_tensorboard=cfg.tensorboard
        )

    # ------------------------------------------------------------------ #
    def _reduce_scalar(self, x: float) -> float:
        """Average a python scalar across DDP ranks (no-op single-GPU)."""
        if not dist.is_initialized():
            return x
        t = torch.tensor([x], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item() / dist.get_world_size())

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

    # --- pure loss math (no backward); reused by training and validation ----- #
    def _phase_b_losses(self, images, depth, pose_enc, conf, dav2_depth):
        """Phase B (improve VGGT) loss. Returns (total, logs, dyn_mask)."""
        cfg = self.cfg
        ss, dyn = compute_self_supervised_losses(
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
        logs = {
            "total": float(total.detach()),
            "photometric": float(ss["photometric"].detach()),
            "geometric": float(ss["geometric"].detach()),
            "smoothness": float(ss["smoothness"].detach()),
            "distill_ssi": float(distill_ssi.detach()),
            "distill_grad": float(distill_grad.detach()),
        }
        return total, logs, dyn

    def _phase_a_losses(self, images, dav2_depth, depth_v, pose_enc):
        """Phase A (improve DAv2) loss. Returns (total, logs)."""
        cfg = self.cfg
        B, S, _, H, W = images.shape
        d_disp = to_disparity(dav2_depth).reshape(B * S, H, W)
        v_disp = to_disparity(depth_v).reshape(B * S, H, W)
        distill = affine_invariant_l1(d_disp, v_disp)  # align DAv2 -> VGGT
        multiview = multiview_consistency_loss(dav2_depth, pose_enc, offsets=cfg.offsets)
        total = cfg.w_a_distill * distill + cfg.w_a_multiview * multiview
        logs = {
            "total": float(total.detach()),
            "distill": float(distill.detach()),
            "multiview": float(multiview.detach()),
        }
        return total, logs

    @staticmethod
    def _viz_pack(images, vggt_depth, dav2_depth, dyn_mask=None):
        """Detach + move to CPU the tensors needed for a montage (batch item 0)."""
        pack = {
            "images": images[0].detach().float().cpu(),
            "vggt_depth": vggt_depth[0].detach().float().cpu(),
            "dav2_depth": dav2_depth[0].detach().float().cpu(),
        }
        if dyn_mask is not None:
            pack["dyn_mask"] = dyn_mask[0].detach().float().cpu()
        return pack

    # --- training steps (forward + backward), optionally collecting viz ------ #
    def _phase_b(self, images, collect_viz=False):
        images = images.to(self.device)
        self.dav2.eval()
        depth, pose_enc, conf = self._vggt_forward(images, train=True)
        with torch.no_grad():
            dav2_depth = _squeeze_depth(self.dav2(images))
        total, logs, dyn = self._phase_b_losses(images, depth, pose_enc, conf, dav2_depth)
        self._backward_step(total, self.opt_vggt, self.vggt_params)
        viz = self._viz_pack(images, depth, dav2_depth, dyn) if collect_viz else None
        return logs, viz

    def _phase_a(self, images, collect_viz=False):
        images = images.to(self.device)
        with torch.no_grad():
            depth_v, pose_enc, _ = self._vggt_forward(images, train=False)
        self.dav2.train()
        dav2_depth = _squeeze_depth(self.dav2(images))
        total, logs = self._phase_a_losses(images, dav2_depth, depth_v, pose_enc)
        self._backward_step(total, self.opt_dav2, self.dav2_params)
        if self.ema is not None:
            self.ema.update(_unwrap(self.dav2))
        viz = self._viz_pack(images, depth_v, dav2_depth) if collect_viz else None
        return logs, viz

    # backward-compatible thin wrappers (used by smoke_test)
    def phase_b_step(self, images: torch.Tensor) -> Dict[str, float]:
        """Improve VGGT-Omega (DAv2 frozen)."""
        return self._phase_b(images)[0]

    def phase_a_step(self, images: torch.Tensor) -> Dict[str, float]:
        """Improve DAv2 (VGGT-Omega frozen)."""
        return self._phase_a(images)[0]

    # ------------------------- validation ----------------------------- #
    @torch.no_grad()
    def validate(self, val_loader: Iterable) -> Dict[str, Dict[str, float]]:
        """Run both phases' losses on the held-out split (no grad/update).

        Returns ``{"A": {...}, "B": {...}}`` of rank-averaged scalars, or ``{}``
        if the loader yielded nothing.
        """
        self.vggt.eval()
        self.dav2.eval()
        agg_a: Dict[str, float] = {}
        agg_b: Dict[str, float] = {}
        n = 0
        for i, batch in enumerate(val_loader):
            if i >= self.cfg.val_steps:
                break
            images = batch["images"].to(self.device)
            depth, pose_enc, conf = self._vggt_forward(images, train=False)
            dav2_depth = _squeeze_depth(self.dav2(images))
            _, lb, _ = self._phase_b_losses(images, depth, pose_enc, conf, dav2_depth)
            _, la = self._phase_a_losses(images, dav2_depth, depth, pose_enc)
            for k, v in la.items():
                agg_a[k] = agg_a.get(k, 0.0) + v
            for k, v in lb.items():
                agg_b[k] = agg_b.get(k, 0.0) + v
            n += 1
        if n == 0:
            return {}
        va = {k: self._reduce_scalar(v / n) for k, v in agg_a.items()}
        vb = {k: self._reduce_scalar(v / n) for k, v in agg_b.items()}
        return {"A": va, "B": vb}

    def _do_validation(self, val_loader: Iterable, tag: str) -> None:
        results = self.validate(val_loader)
        if not results:
            return
        self.logger.log_scalars(self.global_step, "val", "A", results["A"])
        self.logger.log_scalars(self.global_step, "val", "B", results["B"])
        a_tot = results["A"].get("total", float("nan"))
        b_tot = results["B"].get("total", float("nan"))
        self.logger.text(
            f"[{tag}] VAL gstep {self.global_step}  A/total={a_tot:.4f}  B/total={b_tot:.4f}"
        )
        # best checkpoint tracked by Phase-B val total (VGGT-Omega is the target)
        if b_tot < self.best_val:
            self.best_val = b_tot
            self.save_checkpoint("best")

    # ------------------------- checkpoint / viz ----------------------- #
    def save_checkpoint(self, tag: str = "last") -> None:
        """Save only trainable params (LoRA deltas + heads, or full DAv2)."""
        if not _is_main():
            return
        os.makedirs(self.cfg.out_dir, exist_ok=True)

        def trainable_state(m: nn.Module):
            raw = _unwrap(m)
            keep = {name for name, p in raw.named_parameters() if p.requires_grad}
            sd = raw.state_dict()
            return {k: sd[k].detach().cpu() for k in sd if k in keep}

        path = os.path.join(self.cfg.out_dir, f"checkpoint_{tag}.pt")
        torch.save(
            {
                "vggt": trainable_state(self.vggt),
                "dav2": trainable_state(self.dav2),
                "global_step": self.global_step,
                "best_val": self.best_val,
                "cfg": vars(self.cfg),
            },
            path,
        )
        self.logger.text(f"[finetune] saved checkpoint -> {path}")

    def _save_viz(self, phase: str, viz: Optional[dict]) -> None:
        if not _is_main() or viz is None:
            return
        import cv2

        out = os.path.join(self.cfg.out_dir, "viz")
        os.makedirs(out, exist_ok=True)
        montage = training_montage(
            viz["images"], viz["vggt_depth"], viz["dav2_depth"],
            viz.get("dyn_mask"), num_frames=self.cfg.num_viz_frames,
            caption=f"phase {phase}  gstep {self.global_step}",
        )
        path = os.path.join(out, f"phase{phase}_step{self.global_step:07d}.jpg")
        cv2.imwrite(path, montage)

    # ------------------------------ loop ------------------------------ #
    def train(self, loader: Iterable, val_loader: Optional[Iterable] = None) -> None:
        cfg = self.cfg
        if _is_main():
            tv, total_v = count_parameters(_unwrap(self.vggt))
            td, total_d = count_parameters(_unwrap(self.dav2))
            self.logger.text(
                f"[finetune] VGGT trainable {tv/1e6:.2f}M / {total_v/1e6:.2f}M | "
                f"DAv2 trainable {td/1e6:.2f}M / {total_d/1e6:.2f}M"
            )
            sched = f"every {cfg.val_every} steps" if cfg.val_every > 0 else "end of each phase"
            self.logger.text(
                f"[finetune] outputs -> {cfg.out_dir} | metrics.jsonl/csv, viz/ "
                f"(every {cfg.viz_every}), checkpoints (every {cfg.save_every}), "
                f"val {sched if val_loader is not None else 'disabled'}, tb={cfg.tensorboard}"
            )
        data = _cycle(loader)
        for rnd in range(cfg.rounds):
            self._run_phase("A", data, self._phase_a, f"round {rnd} / phase A (DAv2)", val_loader)
            self._run_phase("B", data, self._phase_b, f"round {rnd} / phase B (VGGT)", val_loader)
        self.save_checkpoint("final")
        self.logger.close()

    def _run_phase(self, phase: str, data, step_fn, tag: str, val_loader) -> None:
        cfg = self.cfg
        running: Dict[str, float] = {}
        count = 0
        for it in range(cfg.steps_per_phase):
            batch = next(data)
            do_viz = _is_main() and cfg.viz_every > 0 and (self.global_step + 1) % cfg.viz_every == 0
            logs, viz = step_fn(batch["images"], collect_viz=do_viz)
            self.global_step += 1

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            count += 1

            if self.global_step % cfg.log_every == 0:
                avg = {k: self._reduce_scalar(running[k] / count) for k in running}
                self.logger.log_scalars(self.global_step, "train", phase, avg)
                if _is_main():
                    msg = " ".join(f"{k}={avg[k]:.4f}" for k in avg)
                    self.logger.text(
                        f"[{tag}] step {it+1}/{cfg.steps_per_phase} (gstep {self.global_step}) {msg}"
                    )
                running, count = {}, 0

            if viz is not None:
                self._save_viz(phase, viz)

            if val_loader is not None and cfg.val_every > 0 and self.global_step % cfg.val_every == 0:
                self._do_validation(val_loader, tag)

            if cfg.save_every > 0 and self.global_step % cfg.save_every == 0:
                self.save_checkpoint("last")

        # flush any partial accumulation as a final train point for this phase
        if count > 0:
            avg = {k: self._reduce_scalar(running[k] / count) for k in running}
            self.logger.log_scalars(self.global_step, "train", phase, avg)

        # validate once at phase end when no periodic schedule was requested
        if val_loader is not None and cfg.val_every == 0:
            self._do_validation(val_loader, tag)
