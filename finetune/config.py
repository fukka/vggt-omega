# Copyright (c) 2026.
"""Configuration for alternating egocentric finetuning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FinetuneConfig:
    # data
    data_root: str = ""
    seq_len: int = 8
    stride: int = 2
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

    # optimization
    lr_vggt: float = 1e-4
    lr_dav2: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True

    # alternating schedule
    rounds: int = 3
    steps_per_phase: int = 500
    ema_teacher: bool = False
    ema_decay: float = 0.999

    # io
    out_dir: str = "finetune_outputs"
    log_every: int = 20
    save_every: int = 500
    seed: int = 0
