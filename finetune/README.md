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
- `metric_anchor` *(metric_anchor strategy only)* — log-depth L1 to a **frozen
  pretrained VGGT** on the same frames. Every other Phase-B term is invariant to
  the absolute depth scale (the depth↔pose-translation gauge), so without this
  the released checkpoint's *metric* scale drifts and the `none`-alignment eval
  degrades. See **Strategies** below.

**Phase A — improve DAv2** (VGGT-Omega frozen):
- `photometric` + `smoothness` — DAv2's depth (scale-aligned to VGGT, scale
  detached) must photo-reproject under VGGT's poses. This is DAv2's only direct
  **real-image** anchor; without it Phase A merely imitates VGGT and can only
  echo/degrade a teacher that is often better than VGGT per-frame on this domain.
- `distill` — affine-align DAv2 to VGGT's depth and match it, **gated** to
  regions where VGGT is confident and static (`conf * dynamic_mask`) so a
  weak/wrong teacher cannot drag DAv2 wholesale.
- `multiview` — make DAv2 multi-view consistent under VGGT's predicted poses.

DAv2 is **LoRA-only by default** (`finetune_dav2_lora_only: true`) to limit
drift. One model is always frozen and acts as a clean teacher (alternating / EM
style); the photometric anchor on *both* phases keeps them from forming an echo
chamber.

## Install

```bash
pip install -r requirements.txt        # core VGGT-Omega
pip install -e .
pip install -r finetune/requirements-finetune.txt
```

## Strategies (run them in parallel)

A run is described entirely by a **YAML config**; its `trainer:` field selects a
strategy class from the registry (`finetune/registry.py`). Two ship today and
exist to answer "should VGGT keep its metric scale, or go up-to-scale?":

| config | `trainer` | Phase B scale handling | eval under |
|---|---|---|---|
| `configs/ssi.yaml` | `ssi` | scale-shift-invariant (scale floats) | `scale_shift` |
| `configs/metric_anchor.yaml` | `metric_anchor` | anchored to frozen pretrained VGGT (metric preserved) | `none` |

Both share everything else via `__base__: base.yaml`. Add a strategy by
subclassing `BaseAlternatingTrainer` and `@TRAINER_REGISTRY.register()`-ing it
(`finetune/trainers/`).

**Ablations** (each isolates one diagnosed risk; all pure-config off `base.yaml`):

| config | what it changes | question it answers |
|---|---|---|
| `no_distill.yaml` | Phase B drops DAv2→VGGT distill | does DAv2 distillation help VGGT or just bias it? |
| `dav2_selfsup.yaml` | Phase A drops VGGT→DAv2 distill | does dropping imitation stop DAv2 degrading? |
| `metric_anchor_strong.yaml` | `w_metric_anchor 0.1→0.3` | scale-preservation vs plasticity trade-off |
| `metric_anchor_scale.yaml` | anchor pins global scale only, not structure | is the structural constraint needed to hold metric scale? |
| `no_rectify.yaml` | feed raw fisheye to pinhole losses | how much does in-loader rectification buy? |

## Quick start

Offline dry run (no checkpoint, no data — stand-in models on CPU):

```bash
python -m finetune.train --dummy --device cpu --name smoke \
  --set rounds=1 --set steps_per_phase=20 --set warmup_steps=4
python -m finetune.smoke_test          # self-checking unit/integration tests
```

Real runs (needs the gated VGGT-Omega checkpoint + extracted egocentric frames;
edit paths in `configs/base.yaml`). The two strategies write to separate folders
and can run **at the same time**:

```bash
python -m finetune.train --config finetune/configs/ssi.yaml
python -m finetune.train --config finetune/configs/metric_anchor.yaml
# 8 GPUs + a quick override and a custom run name:
torchrun --nproc_per_node=8 -m finetune.train \
  --config finetune/configs/metric_anchor.yaml \
  --name metric_anchor_lr1e4 --set lr_vggt_lora=1.0e-4
```

`--set field=value` overrides any config field (repeatable). `data_root` (in the
config) is a directory of clips, each a folder of ordered frames
(`clip_xxxx/frame_000000.jpg ...`); a flat folder of frames is treated as one clip.

### Outputs & records

Each run lands in `runs/<name>/` and **refuses to overwrite** an existing run
(pass `--resume`/`--overwrite`). Alongside the usual metrics/checkpoints/viz it
writes `config.yaml` (the fully-resolved config) and `provenance.json` (git SHA,
argv, host, time). `runs/index.csv` gets one row per run — the at-a-glance log of
what has been trained.

### Catching when it starts to fail

Every `eval_every` steps (default 500) the trainer runs two validation arms and
logs to TensorBoard (`<out_dir>/tb/`, on by default) + `metrics.jsonl`:

- **Qualitative** — depth montages (input | VGGT | DAv2) on `val_data_root`,
  no GT, saved to `runs/<name>/val_qual/step*.jpg` (`val_qual_n` per eval).
- **Quantitative** — ADT dense-GT depth metrics on `eval_adt_root`
  (`eval_adt_max_frames`, default 100). All four variants are reported
  (VGGT/DAv2 × pretrained/finetuned); the **pretrained baseline is computed once**
  at step 0 (the model is still pretrained then) and held constant. Watch
  `eval/adt/vggt_finetuned/none/AbsRel` (metric) and `.../scale_shift/AbsRel`
  (structure) cross their pretrained baselines to see exactly when finetuning
  starts to hurt.

  ADT eval reads the **`videos_synthetic/`** RGB (pixel-aligned with the rendered
  GT depth; the real `videos_rgb/` is slightly off-registered) and **rectifies both
  RGB and GT fisheye→pinhole** to match training (`cfg.rectify`), so the
  pinhole-trained model is fed pinhole input and scored against pinhole GT.

The standalone report takes just the **run name** — it reads everything else
(VGGT base checkpoint, LoRA rank, ADT root, sizing) from `runs/<name>/config.yaml`
and evaluates the pretrained models plus the run's **best** and **last**
checkpoints, writing to `eval_out/<name>/`:
```bash
python -m finetune.eval.run_eval ssi_r8                 # ADT (dense GT)
# MPS (sparse, in-domain) lives in its own runner (needs paths not in config):
python -m finetune.eval.mps_depth ssi_r8 \
  --mps-frame-dir <take>/aria01_214-1 \
  --mps-traj-csv  <take>/mps/slam/closed_loop_trajectory.csv \
  --mps-points-gz <take>/mps/slam/semidense_points.csv.gz
```
Overrides are optional (`--adt-root`, `--checkpoints best`, `--no-dav2`, `--device`).

## Data conventions

The losses assume the **pinhole** intrinsics VGGT-Omega predicts. Aria / fisheye
captures must be rectified — either **upstream** (`data/prepare_egoexo4d.py
--undistort`) or **in the loader** by setting `rectify: true` +
`camera_preset: aria-214-1` in the config (the default in `base.yaml`). Training
on raw fisheye with a pinhole model corrupts the photometric/geometric terms at
the periphery; the loader warns if it detects this.
Pose/depth conventions mirror the released checkpoints exactly
(`finetune/geometry.py` re-derives them from `vggt_omega/utils/pose_enc.py`):
9D pose encoding, quaternion XYZW scalar-last, camera-from-world extrinsics,
metric positive depth.

## Layout

```
finetune/
  geometry.py              # decode pose enc, backproject/transform/project, warp
  registry.py              # name->class registry (TRAINER_REGISTRY)
  config.py                # FinetuneConfig dataclass (single source of truth)
  options.py               # YAML load/merge + run-dir setup + provenance + index.csv
  configs/                 # base.yaml + ssi.yaml + metric_anchor.yaml (run recipes)
  losses/
    self_supervised.py     # photometric + geometric consistency + dynamic mask
    photometric.py ssim.py # appearance terms
    distillation.py        # affine / SSI / gradient-matching / multiview
    dynamic.py             # rigid-flow vs optical-flow motion segmentation helpers
  models/
    lora.py                # LoRA on nn.Linear + freeze utils
    depth_anything.py      # DAv2 wrapper (+ dummy)  |  dummy.py: DummyVGGT
    teacher.py             # EMA teacher
  data/
    egocentric_video.py    # sliding-window video dataset (+ random batch)
    rectify.py             # fisheye -> pinhole (KB4) for the training loader
  trainers/
    base.py                # BaseAlternatingTrainer: loop/optim/log/ckpt/val (no loss policy)
    ssi.py                 # SSITrainer (scale-shift-invariant)
    metric_anchor.py       # MetricAnchorTrainer (preserves metric scale)
  engine/logger.py         # JSONL/CSV/TensorBoard logger + loss curves
  train.py smoke_test.py
```

## Notes & next steps

- **Dynamic scenes:** the geometric residual auto-masks moving pixels; for
  hand-heavy clips, fuse a semantic hand mask (`losses/dynamic.combine_masks`)
  and pass it as `extra_static_mask`. Per-object SE(3) and per-frame 4D handling
  are natural extensions.
- **Metric scale:** monocular self-supervision is up-to-scale, so the `ssi`
  strategy lets VGGT's scale float (judge it with `scale_shift`). Use the
  `metric_anchor` strategy to hold the released checkpoint's metric scale (it
  anchors to a frozen pretrained VGGT; judge it with `none`). For a stronger
  absolute anchor add an IMU-gravity or mono-metric term.
- **Evaluation (no GT):** EPIC-Fields pose, ADT, novel-view-synthesis PSNR, and
  ordinal depth are recommended egocentric probes (not included).
- The real path requires Python ≥3.10 / torch ≥2.3 (per the core package) and a
  CUDA GPU; the dummy path and smoke test run on CPU under older versions.
```
