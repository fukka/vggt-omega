# CAM3R (reproduction)

Re-implementation of **"CAM3R: Camera-Agnostic Model for 3D Reconstruction"**
(Guruprasad, Yadav, Peng, Chellappa — [arXiv:2603.22631](https://arxiv.org/abs/2603.22631),
23 Mar 2026).

The official repo ([nam1410/cam3r](https://github.com/nam1410/cam3r)) is the
project website only — its README says `CODE - TBD` and there are no weights, so
everything here is reconstructed from the paper and supplementary material.
Same situation, and same house style, as [`fisheye3r/`](../fisheye3r/README.md).

CAM3R matters to this repo because it reports on **ADT Aria fisheye** — 99.0 /
95.0 RRA@15 / RTA@15 (Table 1) — the exact data
[`VGGT-360-fisheye/`](../VGGT-360-fisheye/README.md) works on.

## Paper → code map

| Paper | Where |
|---|---|
| Eq. 1, `X(u) = d(u) · r(u)` | `model.py::CAM3R.forward` |
| Eq. 2, SH ray expansion (deg ≤ 3, DC dropped → 15 coeffs) | `rays.py::decode_rays`, `sht.py` |
| Ray Module, UniK3D angular module | `model.py::RayModule`, `AngularHead`, `AngularAttentionBlock` |
| Ray Module read-out layers {6, 12, 18, 24} | `model.py::default_readout_layers` |
| Table S3 `Linear projection (1024→512)`, one per depth | `model.py::RayModule.camera_token_adapter` |
| Eq. 3, cross-attending dual decoder | `model.py::DecoderBlock`, `CrossViewModule` |
| Table S3 DPT multi-scale fusion {0, 6, 9, 12}, D_feat 256 | `model.py::CrossViewModule.dpt_hooks`, `DPTHead` |
| Eq. 4, `X^{2,1} = R X^{2,2} + t` | `alignment.py::PairwiseEdge.from_pointmaps`, `eval_adt.py::_multi_view` |
| Eq. 5–6, asymmetric angular loss (α=0.7 on θ, 0.5 on φ) | `losses.py::asymmetric_angular_loss`, `angular_loss` |
| Eq. 7–8, scale-normalized local regression loss | `losses.py::local_regression_loss` |
| Eq. 9–10, geodesic rotation + `s²`-weighted translation direction | `losses.py::relative_pose_loss` |
| Eq. 11, total objective | `losses.py::cam3r_loss` |
| Eq. 12, Ray-Aware Global Alignment | `alignment.py::ray_aware_global_alignment` |
| Eq. S4, ray conditioning (points slide along frozen rays) | `alignment.py::ray_conditioned_points` |
| Scene-graph pruning (symmetry + adaptive 20th-percentile MNN gate) | `alignment.py::prune_scene_graph` |
| Two-phase homogeneous → heterogeneous curriculum | `data.py::CurriculumSampler` |
| Sec. D.1, synthetic fisheye (equidistant) from panoramas | `data.py::synthesize_view`, `_random_kb4` |
| Init: UniK3D ViT backbone + DUSt3R | `pretrained.py` |
| Table S2: AdamW β=(0.9,0.95), lr 5e-5, 10-epoch warmup, cosine → 1e-6, accum 2 | `train.py` |
| RRA/RTA, mAA@30, ATE post-Umeyama | `metrics.py` |
| Depth-domain diagnostic (scale_only align + AbsRel/δ1) | `finetune/eval/metrics.py` (repo's single protocol) |
| ADT pair window (0.35–1.75 m, 25–65°) | `adt.py::select_pairs` |

## Usage

```bash
# unit tests (138 of them)
python -m pytest cam3r/tests/ -q

# CPU end-to-end wiring check, incl. the real ADT sample if present
python cam3r/smoke_test.py

export CAM3R_ADT_ROOT=/path/to/projectaria_tools_adt_data   # or pass --adt-root

# evaluate on ADT — paper's two-view protocol, at the paper's ViT-L / 512 px
python -m cam3r.eval_adt --dust3r /path/to/dust3r.pth --unik3d /path/to/unik3d.pt

# ...with multi-view Ray-Aware Global Alignment and ATE
python -m cam3r.eval_adt --multi-view

# train (paper: 300–500 epochs on 4×H200; this script is single-process).
# --seq is spelled the same way here and in eval_adt, so a held-out sequence
# can be named on both sides.
python -m cam3r.train --out runs/cam3r --epochs 10 \
    --dust3r /path/to/dust3r.pth --unik3d /path/to/unik3d.pt
```

CLI defaults are the paper's (ViT-L 1024/24 trunk, 512-wide angular module,
768-wide/12-head decoder, DPT D_feat 256, 512 px). The CPU runs recorded below
used explicit small overrides:

```bash
python -m cam3r.eval_adt --resolution 128 --width 128 --dec-width 96 --depth 4 --heads 4 --max-frames 28
```

Both entry points honour `$CAM3R_ADT_ROOT`, as does the test suite — one export
and nothing reads a different copy of ADT. Frames are consecutive, so
`--max-frames` has to span enough motion for the 0.35 m baseline: ~28 frames is
enough across six sequences, ~24 within one is not.

Without `--weights` the evaluator applies whichever of `--dust3r`/`--unik3d` you
pass and labels its output untrained: the backbones are pretrained but CAM3R
itself has not been trained, so the pose numbers are a pipeline check, not a
result.

### Initialization coverage

Against the real checkpoints (`DUSt3R_ViTLarge_BaseDecoder_512_dpt`,
`lpiccinelli/unik3d-vitl`), at the paper's config:

| Module | Tensors | Params | Left at init |
|---|---|---|---|
| Ray Module ← UniK3D | 506 / 506 | 318.6M (100%) | nothing |
| Cross-view ← DUSt3R | 1016 / 1128 | 530.7M (98%) | `dec_blocks.*.norm_ctx` (CroCo has no such norm) |
| **whole model** | | **98.6%** | pose head, `norm_ctx` |

Zero shape mismatches on either side. `pretrained.py` still refuses a checkpoint
that matches nothing rather than silently staying random.

## What was measured here

* **A degree-3 SH expansion represents the Aria KB4 fisheye to 0.055° mean /
  0.32° max** (96×96, over the imaged cone). Comfortably enough that the ray
  parameterization is not a bottleneck on this lens. (`tests/test_rays.py`)
* **The imaged cone is 54.83°, not the 62.33° fold-back turnover** — and
  conflating them matters. An earlier version of this code masked to the
  turnover and measured 0.155° / 2.82°; the entire tail was dead vignette ring
  that the lens never illuminates. Fixing the cone to
  `min(imaged, turnover)` (per `CONTEXT.md` and `VGGT-360-fisheye`) cut mean
  error 3× and max error 9×, and now reproduces that module's validity mask
  **exactly**, 0 pixels different, at every resolution.
* **The base-grid FoV is weakly identified.** Any `hfov` from 10° to 60° fits
  the Aria lens to well under a tenth of a degree, because the SH coefficients
  absorb the angular scale. A network predicting a "wrong" base FoV is not
  thereby broken.
* **The local ADT sequence yields 44 pairs** inside the paper's selection
  window from 28 frames.
* **`arccos(z)` NaNs the Ray Module's backward pass on the first step.** The
  on-axis pixel has `z == 1` and `x == y == 0` *exactly*, where `arccos` has an
  infinite derivative and `atan2` an undefined one. A random ray field never
  lands on those values, so the bug was invisible until the UniK3D
  initialization made the rays accurate — then 1634 parameters went non-finite
  on step 1. `rays_to_spherical` now uses `atan2(hypot(x,y), z)`, which is exact
  and has `|∂θ/∂ρ|, |∂θ/∂z| ≤ 1` everywhere on the sphere, and zeroes φ's
  gradient where φ carries no information. (`tests/test_geometry.py`)
* **A UniK3D-initialized Ray Module already predicts the Aria fisheye to
  3.12° mean, untrained.** Cross-dataset, no CAM3R training at all — the
  angular prior transfers to a lens UniK3D did not see.

## Unspecified in the paper — chosen here

Each is a keyword argument, not a baked-in constant.

1. **λ_A = λ_regr = λ_pose = 1.0** and **β = 0.5** (Eq. 6, 11) — no values given.
2. **Azimuth residuals are wrapped** into (−π, π]. Eq. 5 writes a plain absolute
   difference, which would score two rays 0.02 rad apart as ~2π apart across the
   ±π seam.
3. **Confidence supervision.** The paper uses σ only in Eq. 12 and never says
   what trains it — as written, the confidence head gets no gradient and the
   confidence weighting in global alignment would be meaningless. Default is
   the DUSt3R term (`σ·L − α log σ`); `--conf-mode none` gives the literal Eq. 8.
4. **Pose parameterization** — 6D rotation (Zhou et al.) and translation
   L2-normalized onto S². Paper says only `R ∈ SO(3)`, `t̂ ∈ S²`.
5. **Which learning rate.** Sec. 4.2 says "an initial learning rate of 5×10⁻⁵";
   Table S2 says "Base Learning Rate (blr) 1.5×10⁻⁴". These disagree, and the
   MAE/DUSt3R `blr × batch/256` convention reconciles them to neither
   (1.5e-4 × 32/256 = 1.9e-5). `--lr` follows the main text at 5e-5.
6. **τ_rot = 15°, τ_tra = 30°** for pruning — the paper names the thresholds but
   gives no numbers.
7. **`geodesic_angle` uses `atan2`**, not the literal `arccos((tr−1)/2)` of
   Eq. 9. Analytically identical; arccos loses precision and has a divergent
   gradient at 0 and π, and the same function serves as both loss and metric.

## Known deviations

1. **The pose head has no scale output.** Eq. 10 applies the *same* detached
   `s` to prediction and target, so nothing in Eq. 11 gives a scale head any
   gradient. `t = s·t̂` instead takes its magnitude from the predicted pointmap
   (`model.py::pointmap_scale`). Same class of gap as the confidence head
   above, but here the paper does say where `s` comes from.
2. **Only ADT is wired up, and the heterogeneous phase never runs.** 2D3DS,
   360Loc and MegaDepth plug into `data.TwoViewSource`. A cross-lens pair must
   be one scene rendered two ways (the paper takes "simultaneous fisheye and
   pinhole renders per frame") — pairing two *independent* datasets would give a
   pair with no defined relative pose — so a source must declare
   `supports_heterogeneous`. None here does, so phase 2 degenerates to phase 1
   and says so at runtime. The paper's ablation says this costs a lot (65.4 vs
   97.7 RRA@15 on 2D3DS).
3. **RAGA is optimized with Adam**, seeded by a spanning tree of closed-form
   weighted-Umeyama fits, and returns its **best** iterate. The paper says only
   "alternating optimization: poses→scales→joint". The seed matters: from an
   identity start Adam converges slowly and imprecisely, and on clean input the
   seed is already optimal, so a last-iterate return would be worse than not
   refining at all.
4. **Overlap pruning follows the supplementary, not the main text.** Sec. 3.3
   reads like a fixed "<20% of the pixel count" cut, but supp. C.1 says
   *"Rather than using a fixed threshold, we apply an adaptive quantile gate"*
   at the 20th percentile of the scene-wide match distribution. The default is
   the quantile gate; `gate="fixed"` gives the main-text reading. The
   difference is real — absolute match counts depend on texture and resolution,
   so a fixed cut tends to keep everything or nothing.
5. **DPT head is a compact reassemble/fuse**, not DUSt3R's exact head.
6. **Ray Module patch size (14) need not divide the image size**; inputs are
   resized to the nearest multiple rather than rejected.
7. **Camera extrinsics on ADT — resolved, not assumed.** The relative *camera*
   pose needs `T_device_camera`, which does not cancel out.
   `adt.py::resolve_extrinsics` tries an explicit JSON, then MPS
   `online_calibration.jsonl`, then `projectaria_tools` over the sequence's VRS
   (`main_recording.vrs`, `video.vrs`, then any other — the public ADT download
   ships `video.vrs`, which an earlier version of this code did not look for).
   On the GPU box this resolves exactly: `T_device_camera` for `camera-rgb` is a
   13.6 mm lever arm **and a ~38° rotation**, so the device-frame fallback is
   worse than the "~8° lever-arm bias" an earlier draft of this note claimed —
   the conjugation preserves rotation *angle*, so RRA is unaffected, but it
   rotates the translation direction by the full 38°. `extrinsics_source` /
   `extrinsics_exact` record which path was taken; **do not report RTA from a
   fallback run**.
8. **Pairs never cross recordings.** ADT sequences of one apartment share a
   world frame, so two frames from *different* sessions can sit 0.5 m and 40°
   apart, pass the Sec. D.3 window, and show different people and object
   placement — a pair with a well-defined relative pose and no valid
   correspondence. `select_pairs(groups=...)` restricts to same-recording pairs.
   Regression-tested.
9. **Single-process training.** The paper uses 4×H200, batch 4/GPU with 2-step
   accumulation (effective 32); `--accum` approximates this on one device.

## Status on ADT

The pipeline runs end to end on the real local sequence
(`Apartment_release_clean_seq131_M1292`): 28 frames → 44 pairs in the paper's
window, two-view forward, pose/ray/depth metrics, scene-graph pruning, global
alignment, ATE. Training reduces the objective on real data (5.94 → 2.98 over
6 short epochs; angular 0.431 → 0.144, regression 1.63 → 0.40, pose 3.88 → 2.44).

Pose accuracy is **chance level**, as it must be for an untrained network — the
median rotation error sits near the ~120° expected of a random rotation.
Reproducing 99.0 / 95.0 needs the paper's four corpora, its initializations, and
its 300–500 epochs on real hardware.
