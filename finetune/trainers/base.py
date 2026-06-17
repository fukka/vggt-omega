# Copyright (c) 2026.
"""Base alternating co-distillation trainer (machinery only).

This class owns everything that does NOT change between training strategies:
optimizers + layer-group LRs, warmup/cosine schedules, AMP, grad-accum, DDP
scalar reduction, logging, checkpointing, viz, validation, and the Phase A / B
alternation loop.

What DOES change between strategies (which loss terms compose Phase A / Phase B,
and e.g. whether VGGT's metric scale is anchored) is delegated to two hooks:

    phase_a_losses(images, dav2_depth, depth_v, pose_enc, conf_v) -> (total, logs, dyn)
    phase_b_losses(images, depth, pose_enc, conf, dav2_depth)     -> (total, logs, dyn)

Concrete strategies subclass this and implement the hooks; they register
themselves in ``finetune.registry.TRAINER_REGISTRY`` so a run can pick one by
name from its YAML (``trainer: ...``). See ``trainers/ssi.py`` and
``trainers/metric_anchor.py``.
"""
from __future__ import annotations

import math
import os
from contextlib import nullcontext
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from ..config import FinetuneConfig
from ..engine.logger import TrainLogger
from ..models import EmaTeacher, count_parameters, trainable_parameters
from ..viz import training_montage


def _unwrap(m: nn.Module) -> nn.Module:
    """Strip DDP wrapper; no-op on plain modules."""
    return m.module if isinstance(m, DDP) else m


def _is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _cycle(loader: Iterable):
    """Infinite stream over ``loader``; advances DistributedSampler epochs so
    each pass reshuffles (DDP correctness for multi-epoch training)."""
    epoch = 0
    sampler = getattr(loader, "sampler", None)
    while True:
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def _param_groups(
    model: nn.Module,
    base_lr: float,
    lora_lr: float,
    weight_decay: float,
    no_decay_substrings: Tuple[str, ...] = (),
) -> List[dict]:
    """Split trainable params into (weight-decay matrices @base_lr),
    (no-decay norms/biases/heads @base_lr), (LoRA adapters @lora_lr, no decay).

    ``no_decay_substrings`` routes matching params (e.g. pretrained prediction
    heads) to the no-decay group: decaying carefully-pretrained head weights
    toward zero slowly erodes them, which we do not want when finetuning.
    """
    decay, no_decay, lora = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora.append(p)
        elif (p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower()
              or any(s in name for s in no_decay_substrings)):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": base_lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": base_lr, "weight_decay": 0.0})
    if lora:
        groups.append({"params": lora, "lr": lora_lr, "weight_decay": 0.0})
    return groups


def _warmup_cosine(warmup: int, total: int, min_ratio: float, schedule: str):
    """LambdaLR multiplier: linear warmup then cosine decay (or constant)."""
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if schedule != "cosine" or total <= warmup:
            return 1.0
        prog = (step - warmup) / max(1, total - warmup)
        prog = min(1.0, max(0.0, prog))
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * prog))

    return fn


def _squeeze_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    return depth


class BaseAlternatingTrainer:
    # VGGT prediction-head name fragments that should be fully trainable but NOT
    # weight-decayed (they are pretrained, not fresh).
    VGGT_HEAD_SUBSTRINGS: Tuple[str, ...] = ("dense_head", "camera_head")

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
        betas, eps = (cfg.adam_beta1, cfg.adam_beta2), cfg.adam_eps
        vggt_groups = _param_groups(
            vggt_raw, cfg.lr_vggt_head, cfg.lr_vggt_lora, cfg.weight_decay,
            no_decay_substrings=self.VGGT_HEAD_SUBSTRINGS,
        )
        # DAv2's prediction head is pretrained too -> keep it out of weight decay.
        dav2_groups = _param_groups(dav2_raw, cfg.lr_dav2, cfg.lr_dav2, cfg.weight_decay,
                                    no_decay_substrings=("head",))
        self.opt_vggt = torch.optim.AdamW(
            vggt_groups or [{"params": list(vggt_raw.parameters()), "lr": cfg.lr_vggt_head}],
            lr=cfg.lr_vggt_head, betas=betas, eps=eps,
        )
        self.opt_dav2 = torch.optim.AdamW(
            dav2_groups or [{"params": list(dav2_raw.parameters()), "lr": cfg.lr_dav2}],
            lr=cfg.lr_dav2, betas=betas, eps=eps,
        )

        # per-phase micro-iteration counts (Phase A can be shortened via dav2_steps_mult)
        self.steps_b = max(1, cfg.steps_per_phase)
        self.steps_a = max(1, int(round(cfg.steps_per_phase * cfg.dav2_steps_mult)))
        accum = max(1, cfg.grad_accum)
        opt_steps_v = cfg.rounds * math.ceil(self.steps_b / accum)
        opt_steps_d = cfg.rounds * math.ceil(self.steps_a / accum)
        self.sched_vggt = torch.optim.lr_scheduler.LambdaLR(
            self.opt_vggt, _warmup_cosine(cfg.warmup_steps, opt_steps_v, cfg.min_lr_ratio, cfg.lr_schedule))
        self.sched_dav2 = torch.optim.lr_scheduler.LambdaLR(
            self.opt_dav2, _warmup_cosine(cfg.warmup_steps, opt_steps_d, cfg.min_lr_ratio, cfg.lr_schedule))

        # mixed precision: autocast dtype + (fp16-only) gradient scaler
        self._amp_dtype = None
        if cfg.amp and cfg.amp_dtype != "fp32" and self.device.type == "cuda":
            self._amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
        self._use_scaler = self._amp_dtype == torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_scaler)

        # EMA is built from the raw module so it stays a plain nn.Module
        self.ema = EmaTeacher(dav2_raw, cfg.ema_decay) if cfg.ema_teacher else None

        # monitoring state
        self.global_step = 0
        self.best_val = float("inf")  # tracks Phase-B val total (VGGT is the target model)
        self.logger = TrainLogger(
            cfg.out_dir, enabled=_is_main(), use_tensorboard=cfg.tensorboard
        )

    # ------------------------------------------------------------------ #
    # loss hooks — implemented by strategy subclasses
    # ------------------------------------------------------------------ #
    def phase_b_losses(self, images, depth, pose_enc, conf, dav2_depth):
        """Phase B (improve VGGT). Return (total, logs_dict, dyn_mask)."""
        raise NotImplementedError

    def phase_a_losses(self, images, dav2_depth, depth_v, pose_enc, conf_v):
        """Phase A (improve DAv2). Return (total, logs_dict, dyn_mask_or_None)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    def _reduce_scalar(self, x: float) -> float:
        """Average a python scalar across DDP ranks (no-op single-GPU)."""
        if not dist.is_initialized():
            return x
        t = torch.tensor([x], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item() / dist.get_world_size())

    def _autocast(self):
        if self._amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)

    def _vggt_forward(self, images: torch.Tensor, train: bool):
        self.vggt.train(train)
        with self._autocast():
            preds = self.vggt(images)
        # losses run in fp32 for numerical stability of the geometry math
        depth = _squeeze_depth(preds["depth"]).float()
        pose_enc = preds["pose_enc"].float()
        conf = preds.get("depth_conf")
        conf = conf.float() if conf is not None else None
        return depth, pose_enc, conf

    def _dav2_depth(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        with self._autocast():
            d = model(images)
        return _squeeze_depth(d).float()

    def _backward_step(self, loss, optimizer, params, scheduler, do_step: bool):
        """Accumulation-aware step: scales loss by 1/grad_accum, only steps the
        optimizer + scheduler (and zeroes grads) on ``do_step``."""
        loss = loss / max(1, self.cfg.grad_accum)
        if self._use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if not do_step:
            return
        if self._use_scaler:
            if self.cfg.grad_clip:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

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

    def _dav2_teacher(self) -> nn.Module:
        """Frozen DAv2 used as the Phase-B distillation teacher: the EMA copy
        when enabled (stable mean-teacher target), else the live DAv2."""
        if self.ema is not None and self.cfg.use_ema_teacher_in_distill:
            return self.ema.model
        return self.dav2

    # --- training steps (forward + backward), optionally collecting viz ------ #
    def _phase_b(self, images, collect_viz=False, do_step=True):
        images = images.to(self.device)
        self.dav2.eval()
        depth, pose_enc, conf = self._vggt_forward(images, train=True)
        with torch.no_grad():
            teacher = self._dav2_teacher()
            teacher.eval()
            dav2_depth = self._dav2_depth(teacher, images)
        total, logs, dyn = self.phase_b_losses(images, depth, pose_enc, conf, dav2_depth)
        self._backward_step(total, self.opt_vggt, self.vggt_params, self.sched_vggt, do_step)
        viz = self._viz_pack(images, depth, dav2_depth, dyn) if collect_viz else None
        return logs, viz

    def _phase_a(self, images, collect_viz=False, do_step=True):
        images = images.to(self.device)
        with torch.no_grad():
            depth_v, pose_enc, conf_v = self._vggt_forward(images, train=False)
        self.dav2.train()
        dav2_depth = self._dav2_depth(self.dav2, images)
        total, logs, dyn = self.phase_a_losses(images, dav2_depth, depth_v, pose_enc, conf_v)
        self._backward_step(total, self.opt_dav2, self.dav2_params, self.sched_dav2, do_step)
        if self.ema is not None and do_step:
            self.ema.update(_unwrap(self.dav2))
        viz = self._viz_pack(images, depth_v, dav2_depth, dyn) if collect_viz else None
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

        Mirrors training: Phase B scores against the same DAv2 teacher used in
        training (EMA when enabled), Phase A against the live DAv2. Returns
        ``{"A": {...}, "B": {...}}`` of rank-averaged scalars, or ``{}`` if empty.
        """
        self.vggt.eval()
        self.dav2.eval()
        teacher = self._dav2_teacher()
        teacher.eval()
        agg_a: Dict[str, float] = {}
        agg_b: Dict[str, float] = {}
        n = 0
        for i, batch in enumerate(val_loader):
            if i >= self.cfg.val_steps:
                break
            images = batch["images"].to(self.device)
            depth, pose_enc, conf = self._vggt_forward(images, train=False)
            dav2_live = self._dav2_depth(self.dav2, images)
            dav2_teach = self._dav2_depth(teacher, images)
            _, lb, _ = self.phase_b_losses(images, depth, pose_enc, conf, dav2_teach)
            _, la, _ = self.phase_a_losses(images, dav2_live, depth, pose_enc, conf)
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
                f"[finetune] trainer={cfg.trainer!r} run={cfg.name!r} -> {cfg.out_dir}"
            )
            self.logger.text(
                f"[finetune] VGGT trainable {tv/1e6:.2f}M / {total_v/1e6:.2f}M | "
                f"DAv2 trainable {td/1e6:.2f}M / {total_d/1e6:.2f}M"
            )
            sched = f"every {cfg.val_every} steps" if cfg.val_every > 0 else "end of each phase"
            self.logger.text(
                f"[finetune] metrics.jsonl/csv, viz/ (every {cfg.viz_every}), "
                f"checkpoints (every {cfg.save_every}), "
                f"val {sched if val_loader is not None else 'disabled'}, tb={cfg.tensorboard}"
            )
        data = _cycle(loader)
        for rnd in range(cfg.rounds):
            self._run_phase("A", data, self._phase_a, self.steps_a, self.opt_dav2,
                            f"round {rnd} / phase A (DAv2)", val_loader)
            self._run_phase("B", data, self._phase_b, self.steps_b, self.opt_vggt,
                            f"round {rnd} / phase B (VGGT)", val_loader)
        self.save_checkpoint("final")
        self.logger.close()

    def _run_phase(self, phase, data, step_fn, num_steps, optimizer, tag, val_loader) -> None:
        cfg = self.cfg
        accum = max(1, cfg.grad_accum)
        running: Dict[str, float] = {}
        count = 0
        for it in range(num_steps):
            batch = next(data)
            do_step = ((it + 1) % accum == 0) or (it + 1 == num_steps)
            do_viz = _is_main() and cfg.viz_every > 0 and (self.global_step + 1) % cfg.viz_every == 0
            logs, viz = step_fn(batch["images"], collect_viz=do_viz, do_step=do_step)
            self.global_step += 1

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            count += 1

            if self.global_step % cfg.log_every == 0:
                avg = {k: self._reduce_scalar(running[k] / count) for k in running}
                avg["lr"] = optimizer.param_groups[-1]["lr"]  # LoRA group LR (schedule visible)
                self.logger.log_scalars(self.global_step, "train", phase, avg)
                if _is_main():
                    msg = " ".join(f"{k}={avg[k]:.4g}" for k in avg)
                    self.logger.text(
                        f"[{tag}] step {it+1}/{num_steps} (gstep {self.global_step}) {msg}"
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
