# RayTun3R (reproduction)

Re-implementation of **"RayTun3R: Online Camera Adaptation in 3D Foundation
Models"** (Sinitsyn, Araslanov, Cremers; [arXiv:2607.02711](https://arxiv.org/abs/2607.02711),
TU Munich / MCML / Oxford) on the backbones this repo carries.

The paper says "our code will be made publicly available"; nothing is published
as of 2026-08, so everything here is reconstructed from the paper text. Where the
paper is ambiguous, the choice made is listed under
[Interpretation decisions](#interpretation-decisions) — read that section before
comparing any number against a table.

> **[PAPER.md](PAPER.md) is the condensed paper, and only the paper.** Every
> hyperparameter, the full evaluation protocol, all six result tables, all four
> ablation tables, the sequences it actually names, the twelve things it leaves
> unspecified, and its six errata. **Read it instead of the PDF** — it is written
> so no run needs the paper open.
>
> **[reproduction.md](reproduction.md) is what is ours**: how we resolved each of
> the paper's gaps, which numbers we are chasing, and how to read a run without
> mistaking a paper-consistent result for a failure.

**The idea.** Pretrained 3D foundation models carry a *pinhole bias* in their
positional encodings: the local Jacobian of the pretrained PE is flat in image
radius, which is only correct for a pinhole camera. RayTun3R freezes the whole
backbone and learns tiny radial/angular residuals on the positional encodings —
10,752 parameters — from a short unlabelled fisheye segment, plus three
parameter-free corrections. It claims 2–12× lower rotation error than the
unadapted model at no inference cost.

---

## Quick start

Needs a modern PyTorch (≥2.0), plus `opencv-python` for MAGSAC++ and
`numpy<2`. Run everything from the repo root, not from inside `raytun3r/`.

Neither of these needs weights, data, or a GPU — they are the fastest way to
confirm the install works:

```bash
python raytun3r/smoke_test.py
```

```bash
python -m pytest raytun3r/tests -q
```

Fit an adapter on one sequence. This is the whole method: ~10k trainable
parameters, ~3 minutes on one GPU. `da3 --variant small` is the paper's primary
backbone and needs `pip install depth-anything-3`:

```bash
python -m raytun3r.train --backbone da3 --variant small --weights pretrained --dataset scannetpp --path /data/scannetpp/data/<scene> --stride 10 --out runs/rt3r/<scene>
```

Then evaluate. `--adapter` is applied **only** to the method it was fitted for,
so this command compares the fitted adapter against the training-free baselines:

```bash
python -m raytun3r.eval --backbone vggt --weights pretrained --dataset scannetpp --path /data/scannetpp/data/<scene> --adapter runs/rt3r/<scene>/adapter.pt --methods vanilla,param_free,center_ph,multi_ph,raytun3r --out runs/rt3r/<scene>/results.json
```

`lora` and `caltok` are *learned* baselines, so each needs its own fit before it
means anything (they share the objective, so only `--method` changes). Evaluate
each against its own checkpoint:

```bash
python -m raytun3r.train --backbone vggt --dataset scannetpp --path /data/scannetpp/data/<scene> --method lora --out runs/rt3r/<scene>-lora
```

```bash
python -m raytun3r.eval --backbone vggt --dataset scannetpp --path /data/scannetpp/data/<scene> --adapter runs/rt3r/<scene>-lora/adapter.pt --methods lora
```

Evaluating a learned method with no matching checkpoint is allowed but warns
loudly — it is an untrained model, not a baseline result.

On ADT / Aria fisheye:

```bash
python -m raytun3r.train --backbone vggt --dataset adt --path /group-volume/Fengjia/data/projectaria_tools_adt_data_clean/<seq> --out runs/rt3r/adt
```

Useful flags: `--backbone {da3,vggt,vggt_omega}` (see
[Backbone support](#backbone-support) before choosing), `--matcher ufm` to
refuse the fallbacks, `--stride` (see decision 11 — it dominates the result),
`--max-fov` to restrict Ω, `--max-frames` to shorten a debug run, and `--iters` /
`--windows` to trade fit quality for time. `--help` lists all of them.

### Full-dataset and ablation drivers

Whole of ScanNet++, one scene per GPU, per-scene fit, aggregated with a standard
error over scenes:

```bash
python -m raytun3r.experiments.scannetpp_all --backbone da3 --variant small --weights pretrained --root /netapp/datasets/scannetpp/data --out runs/rt3r/snpp-all-da3s --workers 4
```

The FOV sweep that tests why the virtual-pinhole baselines currently win:

```bash
python -m raytun3r.experiments.fov_sweep --backbone da3 --variant small --weights pretrained --dataset scannetpp --path /data/scannetpp/data/<scene> --out runs/fov-sweep/<scene>
```

The adaptation-length sweep, which tests the one hyperparameter the paper never
states (see [PAPER.md §10](PAPER.md#10--what-the-paper-does-not-specify)):

```bash
python -m raytun3r.experiments.iters_sweep --backbone da3 --variant small --weights pretrained --dataset scannetpp --path /data/scannetpp/data/<scene> --out runs/iters-sweep/<scene>
```

All accept `--dry-run` / `--limit` to print or shorten the plan first.

---

## Paper → code map

| Paper | Where |
|---|---|
| Central camera model, `κ` / `κ⁻¹`, KB4 [38] and EUCM [39] (Eq. 1, 3) | `cameras.py` |
| Backprojection Jacobian, pinhole-bias diagnostic (Eq. 2, 4; Fig. 2) | `cameras.py::Camera.backproject_jacobian` |
| Absolute-PE residual `P' = P_A + t_r(ρ) + ρ·δ_θ(θ)` (Eq. 5) | `adapter.py::RadialAngularPE` |
| RoPE residual `ω' = ω + Δ_r(ρ)` (Eq. 6) | `adapter.py::RadialRoPE` |
| Patch tokenization: local undistortion + border tokens (Sec. 4.2) | `corrections.py::patch_undistort_grid`, `fill_border_tokens` |
| Prediction-grid coordinate correction (Sec. 4.2) | `corrections.py::camera_aware_uv_grid` |
| Hook points on a frozen backbone | `backbones.py::Backbone.install` |
| `X = D·κ⁻¹` (Eq. 7) | `losses.py::backproject` |
| Reprojection loss (Eq. 8) | `losses.py::reprojection_loss` |
| Fixed MAGSAC++ pose target (Eq. 9) | `losses.py::pose_loss`, `matching.py::relative_pose_magsac` |
| Edge-aware smoothness (Eq. 10) | `losses.py::smoothness_loss` |
| L2 and TV regularisers (Eq. 11, 12) | `losses.py::l2_penalty`, `tv_penalty` |
| Total objective, `w_pose=1, w_smooth=10, w_L2=2, w_TV=20` (Eq. 13) | `losses.py::LossWeights`, `total_loss` |

### Run-quality flags and what lands in the log

Three things now travel with every fit, because each one silently changes what
the objective means:

| Flag / field | Why it exists |
|---|---|
| `--allow-sparse-matcher` | Eq. 8 divides by `\|Omega\|`, not by `sum(w)`, so `L_reproj`'s weight against `w_smooth=10 / w_L2=2 / w_TV=20` is set by how much of the disc the matcher is confident about. Below 5% coverage the fit refuses to run; this overrides that. |
| `--no-grad-checkpointing` | The adapter sits at the first layer, so the whole trunk is on the gradient path. Checkpointing is on by default and is numerically inert on VGGT (pinned by `test_grad_checkpointing_is_bit_identical_on_identical_windows`); turn it off to trade memory back for speed. |
| `--windows-cache PATH` | Saves the built windows (images, matches, MAGSAC pose targets) on first run, loads them afterwards. The matcher's GPU forward is not run-to-run reproducible and MAGSAC turns that into a *discretely* different Eq. 9 target, so any A/B over something fit-side must pin the windows or it measures pipeline variance instead (issue #26). |
| `match_coverage`, `matcher` in `train_log.json` and in `results.json`'s `_meta` | The measured coverage and the matcher that produced it. A number without these two cannot be compared against the paper, which assumes UFM. |

A run on OpenCV older than 4.5 also warns that the Eq. 9 pose target fell back
from MAGSAC++ to RANSAC. Record that too.
| Adam, lr 1e-3, grad clip 1.0, 504×504, zero-init (Impl. details) | `train.py::fit_adapter` |
| UFM [44] correspondences | `matching.py::UFMMatcher` |
| 30 three-frame windows, drop flow < 2 px (Sec. 5) | `data.py::build_windows` |
| `R°`, `t°`, `d_reproj`, AbsRel, `δ₁.₂₅` (Eq. 14–18) | `metrics.py` |
| Center-PH, Multi-PH (Fig. 3) | `baselines.py::CenterPH`, `MultiPH` |
| LoRA `r=8, α=16`; CalTok `t=4` (Tab. 3, right) | `baselines.py::attach_lora`, `attach_caltok` |
| Component ablations (Tab. 4a, 7b, 8) | `--no-patch-undistort`, `--no-border-token`, `--no-dpt-grid`, `--n-angular 0`, `--n-radial`, `--methods param_free` |

---

## Backbone support

The method's dominant term is the **absolute-PE** residual. The paper's own
Tab. 7(b), on ETH3D terrains, measures:

| Configuration | R° | t° | d_reproj |
|---|---|---|---|
| Absolute PE only (no RoPE) | 0.68 | 0.9 | 1.6 |
| **RoPE only (no absolute PE)** | **19.52** | **7.8** | **9.6** |
| Both (full) | 0.48 | 0.9 | 1.6 |

That single row decides what each backbone here can show:

| `--backbone` | ViT | Absolute PE | RoPE | Adapter params (C) | Status |
|---|---|---|---|---|---|
| `da3 --variant small` | DINOv2 ViT-S | ✅ `pos_embed` | ✅ `RotaryPositionEmbedding2D` | **10,772** (C=384) | **The paper's primary**, and the target for reproduction. Hooks verified against `depth_anything_3` 0.1.1. |
| `vggt` | DINOv2 (vendored `vggt_visfeat`) | ✅ `pos_embed` | ✅ `RotaryPositionEmbedding2D` | 28,692 (C=1024) | Also one of the paper's three. The only backbone run on real data so far. |
| `vggt_omega` | DINOv3 | ❌ none | ✅ `RopePositionEmbedding` | 20 | **Expected to barely help.** Reproduces the "RoPE only" row, not Tab. 1. |

DA3-Small is where the paper's headline parameter count comes from, and the
adapter reproduces it exactly on the real model: `20·384 + 8·384 = 10,752` for
the two PE tables, `+20` for the radial RoPE table.

Three things about `depth_anything_3` 0.1.1 that the code has to work around,
each confirmed by building `da3-small` and running it:

* `DepthAnything3.forward` wraps the call in `torch.no_grad()` and queries
  `torch.cuda.is_bf16_supported()`. Adaptation needs gradients through the frozen
  model, so `DA3Backbone` wraps the **inner** `DepthAnything3Net`.
* The DPT grid is added by `DualDPT._add_pos_embed`, not VGGT's `_apply_pos_embed`,
  and the helpers live in `model.utils.head_utils`, not `model.heads.utils`.
* `create_uv_grid`'s docstring claims `(width, height, 2)` while its caller
  consumes `(height, width, 2)` — the same trap as in VGGT.

`vggt_omega` is wired up so the claim can be *tested* on this repo's own model,
not because it is expected to match the headline result. If it does improve
substantially, that is a finding about DINOv3, and it contradicts Tab. 7(b).

---

## Interpretation decisions

Each of these is a place the paper does not fully specify. All are options; the
default is listed first.

1. **Depth convention in Eq. 7.** `X = D·κ⁻¹` does not say how `κ⁻¹` is
   normalised, and the two readings differ by a *per-pixel* `1/cos θ` — 1.74× at
   the Aria rim, ~11× at a 170° frame corner. Being radially varying, no global
   scale alignment can absorb it, so Eq. 16–18's `min_s` does not rescue a
   mismatch.

   Default `--convention range` treats `κ⁻¹` as the unit ray, which is defined at
   every incidence angle; `--convention z` uses Eq. 1's `z=1` ray, which is what
   every depth head here natively emits but diverges at 90°.

   **The pairing is now enforced, not assumed.** Backbones declare
   `native_depth`, `install(depth_convention=...)` converts once at the boundary,
   `Prediction` carries the tag, and consumers call `require_convention`. This
   was a live bug: `baselines.py` divided by `cos` to get range while the direct
   fisheye path handed raw planar z to `backproject(convention="range")`, and
   both were scored against ADT ground truth that *had* been converted. The
   pinhole baselines were therefore the only methods in the right convention —
   worth ~0.99 px of `d_reproj` on ScanNet++ geometry and ~0.66 px on Aria,
   against a measured method-to-method spread of 0.10 px.

2. **Prediction-grid radial coordinate.** "Undistorting through the calibrated
   fisheye-to-pinhole map" means `tan θ`, which saturates past the clamp: on a
   180° lens the outer grid cells all collapse to the same coordinate — exactly
   the resolution loss the paper's own Appendix C describes for naive PE
   remapping. `--grid-mode auto` (default) uses `tan` below 80° half-angle and
   the angular radius `θ` beyond; `tan` and `angular` force either.

3. **Scale of the patch linearisation.** A raw linearisation magnifies rim
   patches enormously on a 185–200° lens, so the resampler would read a handful
   of source pixels and upsample. `preserve_scale=True` (default) divides the
   Jacobian by `sqrt|det|`, correcting patch *shape* (anisotropy, shear) while
   keeping its footprint.

4. **Tangent frame for the patch linearisation.** Taken in a *local* gnomonic
   frame about each patch's own ray, using the shortest-arc rotation from `+z`.
   A single global pinhole is undefined at 90° incidence and cannot cover the
   wide lenses in the benchmark. This frame is the identity on-axis by
   construction (verified in the tests).

5. **RoPE residual semantics.** Eq. 6 adds one scalar per radial bin to the
   rotary angle, "shared across RoPE frequencies". Implemented literally: the
   same `Δ_r(ρ)` is added to every frequency and both axial halves. For
   DINOv3-style modules that return `(sin, cos)` this uses the angle-addition
   identities; for VGGT's module, which returns already-rotated tokens, the
   equivalent extra rotation is composed onto the output (exact, since rotations
   compose).

6. **Eq. 12 is written on `P'`, not on the residual.** The pretrained table
   `P_A` therefore enters through a cross term, so it is captured from the PE
   hook on the first forward. RoPE-only backbones have no `P_A` and fall back to
   the TV of the residual.

7. **Eq. 9 uses `arccos`, which cannot reach zero numerically.** `arccos` has an
   infinite derivative at ±1, so it needs its argument clamped — and the clamp
   puts a floor under the loss (~2.8e-3 rad at 1e-6 of slack) that the optimiser
   can never get below. `losses.py` uses the algebraically identical `atan2`
   forms, which hit exactly zero at the target with bounded gradients. The
   *reported metric* in `metrics.py` keeps plain `arccos` — no gradients there.

8. **Multi-PH pose fusion** is unspecified. All views of a frame share an optical
   centre and differ by a known rotation, so each view's predicted pose is mapped
   back through that rotation and averaged (chordal SO(3) mean). Depth is fused
   by preferring the view whose axis is closest to each ray.

9. **Matcher.** `--matcher auto` prefers UFM, then RAFT (with a
   forward–backward consistency confidence), then SIFT. **Only UFM is
   comparable to the paper**; the matcher actually used is recorded in the eval
   JSON and warned about at startup. SIFT in particular is sparse on fisheye and
   gives the reprojection loss a weak signal.

10. **LoRA/CalTok scope.** Applied to the last 12 encoder blocks, prediction
    heads excluded. That reproduces the paper's quoted 147.5K LoRA parameters on
    DA3-Small exactly (`12 × (8·384 + 1152·8) = 147,456`).

11. **Evaluation stride.** The paper says "evaluate on the full sequence" and
    filters below 2 px of flow, but never states a stride, and on this data the
    choice dominates the method — see [Finding 1](#finding-1--the-evaluation-stride-matters-more-than-the-method).
    `--stride 10` is used throughout; `--stride 1` is retained only to reproduce
    the degenerate runs on the `results` branch.

12. **Restricting Ω for an FOV ablation.** `--max-fov` narrows `theta_max`, which
    defines `valid_mask`, which is Ω. Images are untouched, so the model still
    sees the whole frame — this isolates *where a method is scored* from what it
    is shown. It only ever narrows.

---

## Discrepancies found in the paper

* **The 10,752 parameter count excludes the RoPE table.** `20×384 + 8×384 =
  10,752` is exactly the two PE tables at DA3-Small's width. The paper also
  describes a 20-parameter radial RoPE table and ablates it on DA3-Small
  (Tab. 7b), so the full adapter is 10,772. `param_breakdown()` reports both;
  the smoke test asserts both. The 20-parameter difference is immaterial to any
  result, but it means "10,752" and "the adapter" are not quite the same object.

* **Eq. 17 is not AbsRel.** It is printed as
  `mean(‖s·D − D*‖₂)` — an absolute error with no division by the ground truth,
  which is neither *relative* nor consistent with Eigen et al. [53] that it
  cites, nor with the reported magnitudes (0.107 on ETH3D). `metrics.py`
  implements the standard `mean(|s·D − D*| / D*)`.

* **Tab. 4(a) `RayTun3R (ours)` is not the best row in its own ablation.** The
  full model reports R° 1.183 while six of the seven ablated variants score lower
  (0.942 for "w/o RoPE adapter", 0.810 for "Naive remap of PE"); only "Patch
  undistortion" at 1.397 is worse. The paper acknowledges
  "some ablations achieve slightly lower pose error" and selects on `d_reproj`,
  where the full model does win. Worth knowing before treating Tab. 4(a) as
  evidence that every component helps pose.

---

## Measured results

All numbers: `lambda_63`, ScanNet++ `3f15a9266d`, UFM matcher, frozen
checkpoints (`facebook/VGGT-1B @ 860abec`, `yyfz233/Pi3`, `depth-anything/DA3-SMALL`).
Artifacts on the `results` branch; the run directory is named next to each table.
Everything produced before `organized@979fc3a` is void per issue #25 (the Eq. 6
and Eq. 12 fixes both change every number a fit produces) and is not shown here.

### Vanilla backbones — the training-free reference numbers

No adapter, no training, no matcher, no randomness: a frozen backbone on posed
pairs. `identity` is the median GT rotation of the pair set — fixed by the frame
span alone, and the x-axis everything else is read against.
From `results/protocol-identify-3f15a9266d/` (ticket 10, code `7b7034f`;
n=300 pairs for VGGT/π³, n=100 for DA3; `square=False`, `seq_len=2`,
`is_bad` honoured):

| stride | identity | VGGT `R°` | π³ `R°` | DA3-S `R°` |
|---|---|---|---|---|
| 1 | 0.84 | 0.416 | 0.444 | 0.458 |
| 2 | 1.60 | 0.643 | 0.829 | 0.674 |
| 5 | 3.91 | 1.191 | 1.617 | 1.189 |
| 10 | 7.56 | 1.759 | 2.158 | 2.342 |
| 20 | 15.32 | 2.912 | 2.958 | 4.138 |
| 40 | 29.42 | 5.312 | 4.627 | 6.686 |
| **60** | **43.71** | **7.242** | **6.392** | 8.096 |
| 80 | 57.25 | 8.867 | 7.594 | 9.936 |

**Stride 60 is the paper's protocol, and it is identified, not fitted.** Tab. 2
gives vanilla `R°` on this named scene for two backbones: VGGT **7.21** and π³
**6.17**. One span has to satisfy both independently, and stride 60 does — 7.242
(off by 0.03) and 6.392 (off by 0.22) — while strides 40 and 80 miss both. A
one-backbone match would be a curve crossing (ticket 9 demonstrated exactly
that); the two-backbone agreement is what breaks the degeneracy. DA3-Small has
no per-scene paper number (Tab. 1 is a mean over unnamed scenes), so its column
is our reference measurement, not a comparison: **8.10° at the paper's span**.

`t°` at stride 60: VGGT 20.3 (paper 16.6), π³ 16.6 (paper 19.7) — same ballpark,
neither tight; `t°` never corroborated the span the way `R°` did, and the
paper's Center-PH rows are not reproduced by any span (see
`table-*-p300.txt` in the run directory for the full blocks).

### Post-audit adaptation results — stride 10

From `results/rt3r/*-222d4a3/` (ticket 003 / issue #4, code `222d4a3`): fit on
30 three-frame windows, 3000 iters; eval on 100 windows, stride 10, identity
10.94°. `gain = 1 − R°/identity`. UFM coverage 0.82–0.95 everywhere.

| backbone | method | R° | gain | t° | AbsRel |
|---|---|---|---|---|---|
| VGGT | vanilla | 2.35 | 0.785 | 27.1 | 0.254 |
| VGGT | **raytun3r** | 1.65 | 0.849 | 22.5 | 0.198 |
| VGGT | center_ph | **0.36** | **0.967** | 4.8 | 0.172 |
| π³ | vanilla | 2.68 | 0.755 | 20.8 | 0.255 |
| π³ | **raytun3r** | 1.11 | 0.898 | 17.1 | 0.240 |
| π³ | center_ph | **0.18** | **0.983** | 3.2 | 0.171 |
| DA3-S | vanilla | 2.63 | 0.760 | 40.7 | 0.295 |
| DA3-S | raytun3r | — | — | — | — |
| DA3-S | center_ph | 1.79 | 0.837 | 39.6 | 0.232 |

The `harness_verify` reference on the same pairs: classical geometry (UFM +
MAGSAC++) reaches 0.76° / gain 0.935. So the adapter moves both measurable
backbones in the right direction — VGGT 0.785 → 0.849, π³ 0.755 → 0.898 — but
stalls well short of both the classical reference and Center-PH, which is the
"underperforms a pinhole crop" outcome under the ticket's own reading guide,
not the paper's claimed ordering. Caveats that keep this from being final: one
scene; Center-PH answers only 66% of Ω (it discards the rim); DA3's raytun3r
row was blocked by the RoPE hook (issue #26, since fixed — its global attention
is deliberately position-free and the hook now skips it rather than refusing);
and stride 10 is our conditioned choice, not the paper's protocol — the
stride-60 comparison against Tab. 2's adapted numbers has not been run yet.

### Finding 1 — the evaluation stride matters more than the method

The paper says "evaluate on the full sequence" and filters windows below 2 px of
optical flow, but never specifies a stride. Taken as consecutive frames on this
data, that admits pairs whose baseline is ~1.1 cm against ~3 m of scene depth. At
that ratio translation direction is unobservable: MAGSAC++ on UFM matches — as
good a geometric reference as exists here — is itself **11.1°** off the
ground-truth translation, and `d_reproj` stops depending on depth at all. At
stride 10 the baseline is ~9 cm, inter-frame rotation ~6.6°, and MAGSAC++ agrees
with ground truth to **3.2°**. Stride-1 numbers measure the protocol, not the
method; `--stride 10` is the default in both experiment drivers.

### Finding 2 — the ScanNet++ pose convention is correct

The nerfstudio→OpenCV conversion in `data.py` was checked against MAGSAC++, which
is independent of it: median rotation disagreement 0.17°. Recorded so it is not
re-litigated.

---

## What is verified, and what is not

**Verified on CPU** (`smoke_test.py`, 35 checks; `tests/`, 46 tests):

* KB4/EUCM/pinhole `project ∘ unproject` round-trips to ~1e-5 px.
* **The paper's central premise (Sec. 3).** The pinhole backprojection Jacobian
  is flat in radius to 1.0000; the KB4 one varies by 10–30× over the same radii.
* Zero-initialised adapters are *exact* no-ops on both in-tree backbones, so
  adaptation provably starts from the pretrained model.
* Gradients reach every adapter table — including through the `torch.no_grad()`
  block in `vggt_omega`'s aggregator — while every backbone parameter stays
  frozen and `remove()` restores the model bit-exactly.
* Eq. 8 and 9 are zero at ground-truth geometry and positive off it; `d_reproj`,
  AbsRel and `δ₁.₂₅` are scale-invariant as Eq. 16–18 require.
* MAGSAC++ recovers the true relative pose from exact matches to <1e-4°.
* Train → checkpoint → eval runs end-to-end on the real ADT sequence.
* **DA3-Small against the real package** (`depth_anything_3` 0.1.1): the adapter
  is a no-op at zero init, the parameter-free corrections demonstrably change the
  output, gradients reach every table while the model stays frozen, and `remove()`
  restores bit-exactly. Its `embed_dim` is read from the ViT (384), which is where
  the paper's 10,752 comes from: `20·384 + 8·384`, confirmed rather than assumed.
* **Every method now reports depth in the same convention**, enforced by
  `Prediction.require_convention` rather than by comment.

**Not verified — needs the GPU box:**

* **Any number in the paper.** Every check above is behavioural. Reproducing
  Tab. 1/2/3 needs real weights, real data, and UFM.
* Whether the reproduction recovers the claimed 2–12× rotation improvement on the
  paper's own backbone and dataset — the open question this code exists to answer.
  The one run so far (VGGT, one scene) says 1.28×.
* Whether the FOV gap explains the inverted baseline ordering
  (`experiments/fov_sweep.py`).
* DA3-Small has run only on random weights, at 70×70, on CPU. Nothing about
  *pretrained* DA3 behaviour is established.

---

## Data

**ScanNet++** (paper dataset) expects the official layout:

```
<scene>/dslr/nerfstudio/transforms.json    OPENCV_FISHEYE intrinsics + poses
<scene>/dslr/resized_images/<file_path>    RGB
<scene>/dslr/render_depth/<stem>.png       rendered depth, uint16 mm (optional)
```

Poses are converted from nerfstudio/OpenGL to the OpenCV camera-from-world
convention used everywhere else here (validated against MAGSAC++ to 0.17°).

> **The DSLR is a full-frame ~170° fisheye, not the 115° the paper states.** Its
> released calibration puts the frame corner at 84.84° incidence (diagonal FOV
> 169.68° on `3f15a9266d`, 174.20° on a second scene), and the corners carry real
> image content — grey ≈ 80 ± 11, where a vignetted circular fisheye would read
> ≈ 0 ± 0. This matters twice: Ω is the whole rectangle rather than the inscribed
> circle (which would discard 47% of a 504×336 frame), and 170° is far outside
> what any of these backbones saw in training. `--max-fov 115` scores on the
> paper's stated cone; `experiments/fov_sweep.py` exists to find out whether that
> is the explanation for the inverted baseline ordering.
>
> The paper's 115° most likely refers to ScanNet++'s **undistorted** image set,
> whose diagonal measures 132.3° / 118.6° on those same two scenes — the only
> family the number falls inside.
>
> **Full camera reference, including six more traps that have each cost time
> here (planar-z depth, OpenCV's broken fisheye inverse, the anonymisation masks,
> `is_bad` frames): [docs/research/scannetpp-camera-reference.md](../docs/research/scannetpp-camera-reference.md).**

`render_depth/` is optional and was **absent** in the run on the `results`
branch, so `AbsRel`/`δ₁.₂₅` (Tab. 3) have never been measured on ScanNet++.

**ADT / Aria** reuses this repo's existing helpers — `cam3r/adt.py` for the
trajectory and `T_device_camera`, `finetune/eval/baselines/aria_fisheye.py` for
the KB4 calibration and the usable-cone limit — rather than restating them.

> **ADT pose caveat.** `groundtruth/aria_trajectory.csv` gives world-from-*device*
> poses; camera poses need `T_device_camera`. With `projectaria_tools` installed
> this resolves exactly from `video.vrs` and ADT pose metrics work — that is the
> case on the GPU box. Where it cannot be resolved,
> `cam3r.adt.resolve_extrinsics` reports `exact=False`, and this loader then
> leaves ground-truth poses **unset** rather than emit poses wrong by the sensor
> lever arm and mounting rotation; the evaluator says so and skips
> `R°`/`t°`/`d_reproj`. Depth metrics still work, and adaptation is unaffected —
> it never reads GT pose. `--extrinsics-json` supplies it manually.

ADT `depth_npy` is **planar z** (`CONTEXT.md`); it is converted to euclidean
range on load to match `--convention range`, and pixels outside the imaged cone
are masked *before* the conversion, since `1/cos θ` explodes there.

---

## Limitations carried over from the paper

1. The correction is camera-specific: a different lens needs a new fit.
2. It assumes a principal point and mostly radial distortion — no strong
   tangential or non-radial optics.
3. It needs camera parameters (Appendix B shows AnyCalib estimates suffice).
4. Fisheye only; panoramic/ERP input is future work.
5. **It needs inter-frame motion.** With degenerate motion the self-supervised
   constraints go weak, because large depth or translation-direction errors
   induce only small reprojection errors. `build_windows` enforces the paper's
   2 px flow threshold and raises if no window qualifies.
