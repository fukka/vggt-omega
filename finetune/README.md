# Self-supervised egocentric finetuning for VGGT-Omega

Finetune VGGT-Omega on **real egocentric video with no depth ground truth**, by
**alternating co-distillation** between Depth-Anything-V2 (DAv2) and VGGT-Omega.
The two models are complementary — VGGT-Omega contributes multi-view geometry,
camera pose and scale; DAv2 contributes sharp per-frame structure and a
finite-depth prior — and each supplies what the other lacks.

> Label-free throughout: the only external signal is **raw pixels** (photometric
> reconstruction) and **multi-view geometric consistency** of the model's own
> predictions. No depth labels, no SfM preprocessing.

## Method

VGGT-Omega jointly predicts depth **and** camera pose, which closes the
self-supervision loop via view synthesis: warp neighbouring frames using
predicted depth + predicted pose and minimize photometric error.

**Phase B — improve VGGT-Omega** (DAv2 frozen):
- `photometric` — min-reprojection view synthesis over temporal neighbours with
  static-pixel auto-masking (`losses/photometric.py`, `losses/self_supervised.py`)
- `geometric` — scale-normalized cross-view depth-consistency residual
  `|d_proj − d_sampled| / (d_proj + d_sampled)`, which also yields a **free
  dynamic/occlusion mask** `M = 1 − r`
- `smoothness` — edge-aware disparity smoothness
- `distill_ssi` + `distill_grad` — scale-shift-invariant + gradient-matching
  structure transfer from DAv2 (sharp edges, finite depth on hands / carried
  objects) **without** overwriting VGGT-Omega's metric scale

**Phase A — improve DAv2** (VGGT-Omega frozen):
- `distill` — affine-align DAv2 to VGGT-Omega's consistent depth and match it
  (resolves DAv2's per-frame scale/shift ambiguity)
- `multiview` — make DAv2 multi-view consistent under VGGT-Omega's predicted poses
  (injects the cross-view coherence DAv2 structurally lacks)

One model is always frozen and acts as a clean teacher (alternating / EM style),
which is more stable than joint co-training. Anchoring to the photometric +
geometric terms prevents the two models from forming an echo chamber.

## Install

```bash
pip install -r requirements.txt        # core VGGT-Omega
pip install -e .
pip install -r finetune/requirements-finetune.txt
```

## Quick start

Offline dry run (no checkpoint, no data — stand-in models on CPU/GPU):

```bash
python -m finetune.train --dummy --rounds 1 --steps-per-phase 20
python -m finetune.smoke_test          # self-checking unit/integration tests
```

Real run (needs the gated VGGT-Omega checkpoint and extracted egocentric frames):

```bash
python -m finetune.train \
  --data-root /path/to/egocentric_frames \
  --vggt-checkpoint checkpoints/vggt_omega_1b_512.pt \
  --image-resolution 512 --seq-len 8 --batch-size 1 \
  --rounds 3 --steps-per-phase 500
```

`--data-root` is a directory of clips, each a folder of ordered frames
(`clip_xxxx/frame_000000.jpg ...`); a flat folder of frames is treated as one clip.

## Data conventions

The losses assume the **pinhole** intrinsics VGGT-Omega predicts. Aria / fisheye
captures should be **rectified to a pinhole model upstream** before training.
Pose/depth conventions mirror the released checkpoints exactly
(`finetune/geometry.py` re-derives them from `vggt_omega/utils/pose_enc.py`):
9D pose encoding, quaternion XYZW scalar-last, camera-from-world extrinsics,
metric positive depth.

## Layout

```
finetune/
  geometry.py              # decode pose enc, backproject/transform/project, warp
  losses/
    self_supervised.py     # photometric + geometric consistency + dynamic mask
    photometric.py ssim.py # appearance terms
    distillation.py        # affine / SSI / gradient-matching / multiview
    dynamic.py             # rigid-flow vs optical-flow motion segmentation helpers
  models/
    lora.py                # LoRA on nn.Linear + freeze utils
    depth_anything.py      # DAv2 wrapper (+ dummy)  |  dummy.py: DummyVGGT
    teacher.py             # EMA teacher
  data/egocentric_video.py # sliding-window video dataset (+ random batch)
  engine/alternating.py    # Phase A / Phase B alternating trainer
  config.py train.py smoke_test.py
```

## Notes & next steps

- **Dynamic scenes:** the geometric residual auto-masks moving pixels; for
  hand-heavy clips, fuse a semantic hand mask (`losses/dynamic.combine_masks`)
  and pass it as `extra_static_mask`. Per-object SE(3) and per-frame 4D handling
  are natural extensions.
- **No metric anchor here:** monocular self-supervision is up-to-scale. Add an
  IMU-gravity or mono-metric anchor if absolute scale matters.
- **Evaluation (no GT):** EPIC-Fields pose, ADT, novel-view-synthesis PSNR, and
  ordinal depth are recommended egocentric probes (not included).
- The real path requires Python ≥3.10 / torch ≥2.3 (per the core package) and a
  CUDA GPU; the dummy path and smoke test run on CPU under older versions.
```
