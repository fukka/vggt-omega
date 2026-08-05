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
| Ray Module, UniK3D-compatible angular head | `model.py::RayModule`, `AngularHead` |
| Eq. 3, cross-attending dual decoder | `model.py::DecoderBlock`, `CrossViewModule` |
| Eq. 4, `X^{2,1} = R X^{2,2} + t` | `alignment.py::PairwiseEdge.from_pointmaps`, `eval_adt.py::_multi_view` |
| Eq. 5–6, asymmetric angular loss (α=0.7 on θ, 0.5 on φ) | `losses.py::asymmetric_angular_loss`, `angular_loss` |
| Eq. 7–8, scale-normalized local regression loss | `losses.py::local_regression_loss` |
| Eq. 9–10, geodesic rotation + scale-anchored translation | `losses.py::relative_pose_loss` |
| Eq. 11, total objective | `losses.py::cam3r_loss` |
| Eq. 12, Ray-Aware Global Alignment | `alignment.py::ray_aware_global_alignment` |
| Eq. S4, ray conditioning (points slide along frozen rays) | `alignment.py::ray_conditioned_points` |
| Scene-graph pruning (symmetry + adaptive 20th-percentile MNN gate) | `alignment.py::prune_scene_graph` |
| Two-phase homogeneous → heterogeneous curriculum | `data.py::CurriculumSampler` |
| Synthetic fisheye/perspective from panoramas | `data.py::synthesize_view` |
| Init: UniK3D angular module + DUSt3R | `pretrained.py` |
| AdamW β=(0.9,0.95), lr 5e-5, warmup+cosine, accum | `train.py` |
| RRA/RTA, mAA@30, ATE post-Umeyama | `metrics.py` |
| Depth-domain diagnostic (scale_only align + AbsRel/δ1) | `finetune/eval/metrics.py` (repo's single protocol) |
| ADT pair window (0.35–1.75 m, 25–65°) | `adt.py::select_pairs` |

## Usage

```bash
# unit tests (130 of them)
python -m pytest cam3r/tests/ -q

# CPU end-to-end wiring check, incl. the real ADT sample if present
python cam3r/smoke_test.py

export CAM3R_ADT_ROOT=/path/to/projectaria_tools_adt_data   # or pass --adt-root

# evaluate on ADT — paper's two-view protocol, at the paper's ViT-L / 512 px
python -m cam3r.eval_adt

# ...with multi-view Ray-Aware Global Alignment and ATE
python -m cam3r.eval_adt --multi-view

# train (paper: 300–500 epochs on 4×H200; this script is single-process)
python -m cam3r.train --out runs/cam3r --epochs 10 \
    --dust3r /path/to/dust3r.pth --unik3d /path/to/unik3d.pt
```

CLI defaults are the paper's (ViT-L 1024/24, 768-wide decoder, 512 px). The CPU
runs recorded below used explicit small overrides:

```bash
python -m cam3r.eval_adt --resolution 128 --width 128 --dec-width 96 --depth 4 --heads 4 --max-frames 28
```

Without `--weights` the evaluator builds a **randomly initialized** network and
labels its own output as untrained; it is a pipeline check, not a result.

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
4. **Pose parameterization** — 6D rotation (Zhou et al.), translation
   L2-normalized onto S², scale via `exp`. Paper says only `R ∈ SO(3)`, `t̂ ∈ S²`.
5. **Eq. 10 is split into a direction term and a scale term.** The natural
   reading — prediction `s·û_p` against target `sg(s)·û_g` — is a trap: it
   equals `s²‖û_p−û_g‖²`, whose derivative in `s` is `2s(1−cos) > 0` for *any*
   direction error, and nothing else in Eq. 11 pins `s` down (Eq. 8 divides by
   η, so it is scale-free). That objective is minimized by collapsing the scale
   rather than fixing the direction. So `L_trans` is direction-only and a
   separate log-space `L_scale` supervises magnitude against ‖t_gt‖; pass
   `--w-scale 0` for an up-to-scale corpus. Regression-tested.
6. **τ_rot = 15°, τ_tra = 30°** for pruning — the paper names the thresholds but
   gives no numbers.
7. **`geodesic_angle` uses `atan2`**, not the literal `arccos((tr−1)/2)` of
   Eq. 9. Analytically identical; arccos loses precision and has a divergent
   gradient at 0 and π, and the same function serves as both loss and metric.

## Known deviations

1. **No pretrained initialization was available locally**, so every number
   produced here comes from a randomly initialized network. `pretrained.py`
   implements both loaders the paper prescribes (splitting DUSt3R's fused `qkv`,
   mapping UniK3D's per-degree names) and refuses a checkpoint that matches zero
   tensors rather than silently staying random — but neither has been run
   against a real checkpoint.
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
7. **Camera extrinsics on ADT.** The relative *camera* pose needs
   `T_device_camera`, which does not cancel out. `adt.py::resolve_extrinsics`
   tries an explicit JSON, then MPS `online_calibration.jsonl`, then
   `projectaria_tools`. The local sample has none of these, so runs here fall
   back to the **device frame** — rotation error is unaffected, but translation
   direction carries a lever-arm bias of up to ~8° at a 0.35 m baseline. The
   fallback is recorded in `extrinsics_source` / `extrinsics_exact` and printed
   as a warning; **do not report RTA from a fallback run**. On the GPU box,
   install `projectaria_tools` or pass `--extrinsics-json`.
8. **Single-process training.** The paper uses 4×H200, batch 4/GPU with 2-step
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
