# Copyright (c) 2026.
"""Configuration for alternating egocentric finetuning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FinetuneConfig:
    # data
    data_root: str = ""
    val_data_root: str = ""        # held-out split for periodic validation (e.g. <root>/val)
    seq_len: int = 8
    stride: int = 2
    window_stride: int = 1         # frames between window starts (1 = max overlap; seq_len*stride = none)
    clip_pattern: str = "*214-1"   # keep only egocentric RGB (aria*_214-1); "" loads every camera
    image_resolution: int = 512
    patch_size: int = 16
    batch_size: int = 1
    num_workers: int = 4

    # models / checkpoints
    vggt_checkpoint: str = ""
    dav2_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"
    dav2_dummy: bool = False
    vggt_dummy: bool = False

    # LoRA / trainable
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    finetune_dav2_lora_only: bool = False  # else finetune DAv2 fully

    # pairing
    offsets: Tuple[int, ...] = (-1, 1)
    ssim_alpha: float = 0.85

    # loss weights (Phase B: improve VGGT-Omega)
    w_photometric: float = 1.0
    w_geometric: float = 0.5
    w_smoothness: float = 0.05
    w_distill_ssi: float = 0.5     # structure transfer from DAv2 (scale-shift invariant)
    w_distill_grad: float = 0.25   # gradient matching from DAv2

    # loss weights (Phase A: improve DAv2)
    w_a_distill: float = 1.0       # affine-aligned distill from VGGT depth
    w_a_multiview: float = 0.5     # multi-view consistency under VGGT poses

    # optimization — AdamW with layer-group LRs. LoRA adapters start at 0 and
    # need a higher LR than the pretrained heads they augment; betas (0.9, 0.95)
    # follow large-model finetuning practice (VGGT/DUSt3R/LLM-style).
    lr_vggt_head: float = 2e-5     # VGGT dense + camera heads (continue from pretrained)
    lr_vggt_lora: float = 2e-4     # VGGT LoRA adapters (fresh; ~10x the head LR)
    lr_dav2: float = 5e-5          # DAv2 (full finetune, or LoRA if finetune_dav2_lora_only)
    weight_decay: float = 0.05     # applied to weight matrices only (not norms/biases/LoRA)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    grad_accum: int = 1            # micro-batches per optimizer step (multiplies effective batch)

    # LR schedule (per optimizer, over its own total optimizer steps)
    lr_schedule: str = "cosine"    # "cosine" | "constant"
    warmup_steps: int = 200        # linear warmup (optimizer steps) at the start of each model's training
    min_lr_ratio: float = 0.05     # cosine floor as a fraction of peak LR

    # mixed precision
    amp: bool = True               # master switch; False = fp32
    amp_dtype: str = "bf16"        # "bf16" (recommended on Ampere+) | "fp16" | "fp32"

    # alternating schedule
    epochs: float = 0.0            # if >0, auto-set steps_per_phase to cover this many passes over the train set
    rounds: int = 6                # alternation rounds (each = one phase A + one phase B)
    steps_per_phase: int = 2000    # micro-iterations per phase (auto-overridden when epochs>0)
    dav2_steps_mult: float = 1.0   # phase A length = steps_per_phase * this (shorten DAv2 phase if <1)

    # EMA teacher (mean-teacher style; provides a stable distillation target)
    ema_teacher: bool = True
    ema_decay: float = 0.999
    use_ema_teacher_in_distill: bool = True  # phase B distills from EMA(DAv2) instead of the raw DAv2

    # validation
    val_every: int = 0             # steps between validation; 0 = once at the end of each phase
    val_steps: int = 50            # number of val batches averaged per validation pass

    # io / monitoring
    out_dir: str = "finetune_outputs"
    log_every: int = 20            # steps between train-loss console/JSONL logs
    save_every: int = 500          # steps between checkpoint saves (0 disables periodic saves)
    viz_every: int = 200           # steps between saved depth-montage images (0 disables)
    num_viz_frames: int = 4        # frames per saved montage
    tensorboard: bool = False      # also log to <out_dir>/tb/
    seed: int = 0
