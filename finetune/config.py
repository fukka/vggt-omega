# Copyright (c) 2026.
"""Configuration for alternating egocentric finetuning.

This is the single source of truth for every tunable. Runs are driven by YAML
files (``finetune/configs/*.yaml``) that are merged onto these defaults; the
``trainer`` field selects which training strategy class to instantiate from
``finetune.registry.TRAINER_REGISTRY`` (e.g. ``ssi`` vs ``metric_anchor``).
See ``finetune/options.py`` for the loader and ``finetune/README.md`` for the
run workflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FinetuneConfig:
    # ---- experiment identity / strategy ----------------------------------- #
    name: str = "default"          # run name; outputs go to <runs_root>/<name>/
    trainer: str = "ssi"           # TRAINER_REGISTRY key: "ssi" | "metric_anchor"
    runs_root: str = "runs"        # parent dir holding one folder per run
    notes: str = ""                # free-form description, stored in the run's config.yaml

    # data
    data_root: str = ""
    val_data_root: str = ""        # held-out split for periodic validation (e.g. <root>/val)
    seq_len: int = 8
    stride: int = 2
    window_stride: int = 1         # frames between window starts (1 = max overlap; seq_len*stride = none)
    clip_pattern: str = "*214-1"   # keep only egocentric RGB (aria*_214-1); "" loads every camera
    image_resolution: int = 512
    patch_size: int = 16
    batch_size: int = 1            # >1 works for uniform-resolution clips (e.g. square
                                   # Aria 214-1 -> 512x512); mixed aspect ratios cannot
                                   # be stacked into one batch (see collate_windows).
    num_workers: int = 4

    # fisheye rectification (the geometric losses assume a PINHOLE camera).
    # If the extracted frames are raw Aria fisheye, enable this or the
    # photometric/geometric terms are wrong toward the image periphery.
    rectify: bool = False          # rectify each frame to pinhole before training
    camera_preset: str = "none"    # "none" | "aria-214-1"
    fisheye_k: str = ""            # "fx,fy,cx,cy" override (else from preset)
    fisheye_d: str = ""            # "k1,k2,k3,k4" KB4 distortion override (else from preset)
    warn_unrectified: bool = True  # loudly warn if data looks like fisheye and rectify is off

    # models / checkpoints
    vggt_checkpoint: str = ""
    dav2_model_name: str = "depth-anything/Depth-Anything-V2-Large-hf"
    dav2_dummy: bool = False
    vggt_dummy: bool = False

    # LoRA / trainable
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    finetune_dav2_lora_only: bool = True   # LoRA on DAv2 (safer); False = full finetune

    # pairing
    offsets: Tuple[int, ...] = (-1, 1)
    ssim_alpha: float = 0.85

    # loss weights (Phase B: improve VGGT-Omega)
    w_photometric: float = 1.0
    w_geometric: float = 0.5
    w_smoothness: float = 0.05
    w_distill_ssi: float = 0.5     # structure transfer from DAv2 (scale-shift invariant)
    w_distill_grad: float = 0.25   # gradient matching from DAv2
    b_distill_gate: str = "none"   # gate the Phase-B DAv2->VGGT distillation to where DAv2
                                   # actually helps, so a globally-weaker teacher doesn't drag
                                   # VGGT down: "none" | "conf" (where VGGT is UNcertain) |
                                   # "edge" (image edges/discontinuities) | "conf_edge" (either).
    w_metric_anchor: float = 0.1   # ONLY used by trainer="metric_anchor": tie depth to the
                                   # frozen pretrained VGGT (preserves metric scale). 0 disables.
    metric_anchor_mode: str = "full"  # "full" = per-pixel log-depth (proximal: scale+structure)
                                      # | "scale" = pin per-frame global scale only (let structure adapt)
                                      # | "ssi" = scale-shift-invariant STRUCTURE anchor to pretrained
                                      #   (protect structure, leave scale free; for scale_shift eval)

    # loss weights (Phase A: improve DAv2)
    w_a_distill: float = 1.0       # affine-aligned distill from VGGT depth (gated by conf*dyn)
    w_a_multiview: float = 0.5     # multi-view consistency under VGGT poses
    w_a_photometric: float = 0.5   # NEW: photometric appearance anchor for DAv2 (real-image signal)
    w_a_smoothness: float = 0.05   # edge-aware smoothness on DAv2 depth
    a_distill_gate: str = "conf"   # WHERE to trust the VGGT teacher in the A-distill:
                                   # "none"        — ungated (every pixel equal)
                                   # "conf" / true — VGGT depth-head confidence (log) [legacy bool]
                                   # "static"      — VGGT multi-view geometric consistency (1-r)^pow:
                                   #                 trust where VGGT depth is cross-view consistent
                                   #                 (static / non-occluded) — a DIRECT reliability
                                   #                 signal, not the learned conf head
                                   # "conf_static" — product of conf AND static
                                   # "edge"        — image edges (DAv2 already sharp there)
                                   # legacy booleans still parse: true->"conf", false->"none".
    a_gate_pow: float = 1.0        # sharpness of the "static" gate ((1-r)^pow); >1 = more aggressive
    # ordinal (ranking) distillation — match the TEACHER's depth ORDERING instead of
    # its values (invariant to any positive affine map, so it transfers structure
    # without forcing the teacher's scale/shift). FisheyeDistill-style ordinal
    # distillation (arXiv 2205.02930). 0 disables (default -> no change).
    w_a_rank: float = 0.0
    a_rank_pairs: int = 4096       # random pixel pairs per image for the ranking loss
    # teacher-weight decay — linearly fade the VGGT->DAv2 distillation weight from
    # full to (1 - distill_decay) over training, so the model leans on the teacher
    # early and on its other terms (photometric/anchor) late. Mitigates the
    # teacher-overfit / homogenization failure mode of long co-distillation
    # (Decoupled Knowledge for Online KD, arXiv 2312.11218). 0 disables.
    distill_decay: float = 0.0
    w_a_anchor: float = 0.0        # ONLY used by trainer="dav2": scale-shift-invariant SELF-anchor
                                   # to the frozen pretrained DAv2 (protect its strong structure
                                   # prior while the other terms refine it). 0 disables.

    # dynamic / occlusion robustness — down-weight the photometric + geometric
    # terms on pixels whose cross-view depth is inconsistent (moving / occluded),
    # so the model isn't forced to explain motion as depth. The weight is the
    # (detached) geometric residual mask M = (1 - r)^pow, an IRLS-style robust
    # reweighting. Egocentric clips are hand/body-heavy, so this matters here.
    use_dynamic_mask: bool = True
    dynamic_mask_pow: float = 1.0  # >1 = sharper (more aggressive masking of inconsistent pixels)

    # optimization — AdamW with layer-group LRs. LoRA adapters start at 0 and
    # need a higher LR than the pretrained heads they augment; betas (0.9, 0.95)
    # follow large-model finetuning practice (VGGT/DUSt3R/LLM-style).
    lr_vggt_head: float = 2e-5     # VGGT dense + camera heads (continue from pretrained)
    lr_vggt_lora: float = 2e-4     # VGGT LoRA adapters (fresh; ~10x the head LR)
    lr_dav2: float = 5e-5          # DAv2 (full finetune, or LoRA if finetune_dav2_lora_only)
    weight_decay: float = 0.05     # applied to weight matrices only (not norms/biases/LoRA/heads)
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

    # validation — cheap loss-based pass on val_data_root
    val_every: int = 0             # steps between validation; 0 = once at the end of each phase
    val_steps: int = 50            # number of val batches averaged per validation pass

    # in-training EVALUATION (the "when does it start to fail?" signal). Runs two
    # arms every eval_every steps (0 disables both):
    #   - qualitative: depth montages on val_data_root (no GT), val_qual_n of them
    #   - quantitative: ADT dense-GT depth metrics (Aria), <= eval_adt_max_frames
    # All four variants (VGGT/DAv2 x pretrained/finetuned) are reported, but the
    # pretrained ones are computed ONCE at step 0 (model == pretrained then) and
    # cached as a constant baseline thereafter.
    eval_every: int = 500          # steps between full eval (quant ADT + qual val); 0 disables
    eval_adt_root: str = "/group-volume/Fengjia/data/projectaria_tools_adt_data_clean"
    eval_adt_max_frames: int = 100  # cap ADT depth frames per quantitative eval (-1 = all)
    val_qual_n: int = 2            # qualitative val montages saved per eval (~"sequences")

    # ---- DAC-style ERP finetuning of DAv2 (trainer="erp_distill") --------- #
    # Unsupervised finetuning of DAv2 in DAC's ERP canonical space on EgoExo4D
    # fisheye (no depth GT). See trainers/erp_distill.py + data/erp_egoexo.py +
    # losses/erp_consistency.py. These fields are ignored by the other trainers.
    egoexo_root: str = "../EgoX/example/egoexo4D"   # dir with videos/<clip>/<stream>.mp4
    egoexo_stream: str = "ego_GT"                   # "ego_GT" | "ego_Prior"
    egoexo_frames_per_clip: int = 16                # evenly-sampled frames per clip
    erp_cano: int = 1400                            # ERP canonical height (DAC cano_sz)
    erp_fwd_h: int = 500                            # ERP patch fed to DAv2 (rounded to /14 inside)
    erp_fwd_w: int = 750
    erp_crop_wfov: float = 180.0                    # ERP crop FOV (cone mask trims to the real lens)
    erp_focal_scale: float = 1.0                    # tune modelled ego FOV (<1 widens) — intrinsics
                                                    # are a centred Aria-KB4 approx (see erp_egoexo.py)
    erp_input_scale_jitter: float = 0.0             # DAC scale_fac jitter at warp time (input variety)
    # equivariance augmentation (2-D similarity on the ERP patch, applied on-GPU)
    erp_roll_deg: float = 20.0                      # in-plane roll range (±)
    erp_scale_lo: float = 0.8                       # zoom (≈FOV) range for the consistency T
    erp_scale_hi: float = 1.25
    erp_trans_frac: float = 0.05                    # small normalized translation (±)
    erp_pitch_deg: float = 0.0                      # reserved: sphere-pitch aug (dataset-side; 0=off)
    # teacher for the equivariance target: "ema" (mean-teacher) | "frozen"
    # (pretrained) | "self" (live student, pure consistency)
    erp_teacher: str = "ema"
    # unsupervised loss weights
    w_erp_consistency: float = 1.0                  # DAv2(T·I) ≈ T·teacher(I), SSI
    w_erp_anchor: float = 0.5                       # SSI anchor to frozen pretrained DAv2 (anti-drift)
    w_erp_smooth: float = 0.05                      # edge-aware smoothness on ERP depth

    # io / monitoring
    out_dir: str = ""              # resolved to <runs_root>/<name> by options.py (leave "" to auto-set)
    log_every: int = 50            # steps between train-loss console/JSONL logs
    save_every: int = 2000         # steps between checkpoint saves (0 disables periodic saves)
    viz_every: int = 500           # steps between saved depth-montage images (0 disables)
    num_viz_frames: int = 4        # frames per saved montage
    tensorboard: bool = True       # also log to <out_dir>/tb/ (scalars: train, val, eval)
    seed: int = 0
