# RayTun3R — paper reference

Condensed from **arXiv:2607.02711v1**, *RayTun3R: Online Camera Adaptation in 3D
Foundation Models* — Daniil Sinitsyn¹˒², Nikita Araslanov¹˒²˒³, Daniel Cremers¹˒²
(¹ TU Munich, ² Munich Center for Machine Learning, ³ University of Oxford),
submitted 2 Jul 2026. 15 pages: 9 pages of main paper + references, then
appendices A–I. There is no separate supplement — the appendix is inside the same
PDF and arXiv lists no ancillary files. Marked "Preprint"; code "will be made
publicly available".

**This file is the paper and only the paper.** Every number, hyperparameter and
protocol detail below is transcribed from the PDF. Nothing here is a result,
choice, or opinion of the reproduction — those live in
[reproduction.md](reproduction.md).

---

## 1. Method in one paragraph

Freeze a pinhole-pretrained 3D foundation model; adapt **only** the components
mapping token location → camera ray. The learned parts are small radial/angular
lookup tables on the absolute PE and (when present) on RoPE. Everything else —
attention, MLPs, prediction heads, DPT weights — stays frozen. Three further
corrections (prediction-grid coordinates, patch undistortion, border tokens) are
parameter-free. Fit online on a short unlabelled fisheye segment with geometric
losses, then run the rest of the sequence at zero extra inference cost.

**Why it should work (Sec. 3).** For a pinhole camera the backprojection Jacobian
`J_{κ⁻¹} = ∂κ⁻¹/∂(u,v)` is *constant over the image*: a one-pixel displacement
induces the same change `1/f_{x,y}` in viewing direction everywhere. For a
radially symmetric fisheye it is not. Measuring the pretrained PE's own Jacobian
`J_PE = ∂P_A/∂(u,v) ∈ ℝ^{C×2}` and summarising it by its largest singular value
σ₁ and its local area element `√det(J_PEᵀ J_PE)` shows both are nearly **flat in
normalised radius** for every frozen DA3 size — the position-independent structure
of a pinhole camera. After fitting the adapter **on KITTI-360**, the same curves
bend toward the analytic fisheye reference (Fig. 2).

**Camera model.** Central model, projection `κ`, inverse `κ⁻¹`, over the valid
fisheye disc `Ω ⊂ ℝ²`. A ray at angle α to the optical axis lands at image radius
ρ(α); ρ(α) is a polynomial for **Kannala–Brandt (KB)** [38] or a rational function
for the **Enhanced Unified Camera Model (EUCM)** [39]. The paper names these two
and does not commit to either per dataset.

**Backbone assumptions (Sec. 3).** Transformer-based 3D foundation models "often
initialized from DINOv2" [40]. A ViT splits `H_img × W_img` into patches of size
`p`, giving an `H × W` token grid, each patch embedded to `C` dimensions, plus a
learned absolute table `P_A ∈ ℝ^{H×W×C}` added to the image tokens. **When the
input resolution changes, ViTs usually resize `P_A` by bilinear interpolation** —
but its spatial structure still reflects the camera geometry seen at training
time. Some ViTs additionally use 2D axial RoPE [41] in self-attention, denoted
`P_R`: for a token at position n, channel pair (2j, 2j+1) is rotated by an angle
proportional to `n·ω_j` with fixed frequencies `{ω_j}`, applied separately along
the vertical and horizontal grid coordinates.

---

## 2. What the paper adapts (Sec. 4.2)

Four components, in the paper's own order. Only the first two are learned.

1. **Absolute positional embedding.** Parameterised in polar coordinates about the
   *calibrated principal point*. For each patch, `ρ ∈ [0,1]` is its normalised
   radius and `θ ∈ [0,2π)` its angle. Radius is discretised into `N_r` bins and
   angle into `N_θ` bins; a radial update `t_r(ρ)` and an angular update `δ_θ(θ)`
   are learned as lookup tables and **evaluated at continuous ρ, θ by linear
   interpolation over the radial and angular bins**. See Eq. 5.

2. **Rotary positional encoding**, for backbones that have it. One learnable
   scalar per radial bin, **shared across RoPE frequencies**, added to the rotary
   angle. See Eq. 6.

3. **Prediction-grid coordinates** (parameter-free). Some depth heads use a
   DPT-style [42] head with a 2D sinusoidal prediction grid. The regular grid
   coordinates are replaced with camera-aware coordinates obtained by
   **undistorting each fisheye grid location through the calibrated
   fisheye-to-pinhole map**. No model weights change.

4. **Patch tokenization** (parameter-free), two parts:
   * **Border tokens** — patches outside the valid fisheye lens circle are
     replaced by the **mean valid token**, avoiding artifacts from the invalid
     black region.
   * **Patch undistortion** — each patch is locally undistorted before
     tokenization, resampled using the **local linearization of the
     fisheye-to-pinhole map at the patch centre**, following Qin and Li [43], so
     each tokenized patch is closer to a pinhole crop from the same viewing
     direction.

**No photometric loss, deliberately (Sec. 4.3).** "Direct photometric alignment
can be weak in low-texture regions and sensitive to initialization. The initial
depth and pose can be far from optimum due to the model's pinhole bias."
Following AnyCam [47], the paper optimises geometric consistency on UFM
correspondences instead.

---

## 3. Equations

| Eq. | Content |
|---|---|
| 1 | Pinhole `κ⁻¹(u,v) = ((u−cₓ)/fₓ, (v−c_y)/f_y, 1)ᵀ` |
| 2 | `J_{κ⁻¹} = ∂κ⁻¹/∂(u,v)` — constant for pinhole |
| 3 | Fisheye `κ⁻¹(u,v) = (g(u,v)(u−cₓ)/fₓ, g(u,v)(v−c_y)/f_y, 1)ᵀ`, `g` non-linear in image radius (pinhole = `g ≡ 1`) |
| 4 | `J_{κ⁻¹} = g·diag(1/fₓ, 1/f_y, 0) + ((u−cₓ)/fₓ, (v−c_y)/f_y, 0)ᵀ (∂_u g, ∂_v g)` — depends on `(u,v)` |
| **5** | **Absolute PE**: `P'(u,v) = P_A(u,v) + t_r(ρ_{u,v}) + ρ_{u,v}·δ_θ(θ_{u,v})` |
| **6** | **RoPE**: `ω'(u,v) = ω(u,v) + Δ_r(ρ_{u,v})` — one parameter per radial bin, shared across frequencies |
| 7 | `Xᵢ(u,v) = Dᵢ(u,v)·κ⁻¹(u,v)` |
| **8** | `L_reproj(i,j) = (1/\|Ω\|) Σ_{(u,v)∈Ω} w_ij(u,v)·‖κ(T_j T_i⁻¹ Xᵢ(u,v)) − m_ij(u,v)‖₁` |
| **9** | `L_pose(i,j) = arccos((tr(R̃ᵀR̂)−1)/2) + arccos(⟨t̃,t̂⟩/(‖t̃‖‖t̂‖))` |
| **10** | `L_smooth(i) = (1/\|Ω\|) Σ e^{−\|∂ₓI\|}\|∂ₓD*\| + e^{−\|∂_yI\|}\|∂_yD*\|`, `D* = D/mean(D)` |
| **11** | `L_L2 = (1/H_pW_p) Σ_{x,y} ‖P'(x,y) − P_A(x,y)‖²₂` |
| **12** | `L_TV = (1/H_pW_p) Σ_{x,y} (‖P'(x+1,y)−P'(x,y)‖²₂ + ‖P'(x,y+1)−P'(x,y)‖²₂)`, valid neighbours only |
| **13** | `L = L_reproj + w_smooth·L_smooth + w_L2·L_L2 + w_TV·L_TV + w_pose·L_pose` |
| 14 | `R_err = arccos((tr(R*ᵀR̂)−1)/2)`, `t_err = arccos(⟨t*,t̂⟩/(‖t*‖‖t̂‖))`, both in degrees |
| 15 | `Xᵢ(u) = Dᵢ(u)κ⁻¹(u)` (same as 7) |
| **16** | `d_reproj(i,j) = min_{s>0} (1/\|Ω\|) Σ_{u∈Ω} ‖κ(R*_ij(s·Xᵢ(u)) + t*_ij) − u_ij(u)‖₂` — **ground-truth pose**, one global scale per pair |
| 17 | `d_AbsRel(i) = (1/\|Ωᵢ\|) Σ ‖s*ᵢDᵢ(u) − D*ᵢ(u)‖₂`, `s*ᵢ` fitted to ground-truth depth |
| 18 | `δ₁.₂₅(i) = (1/\|Ωᵢ\|) Σ 1(max(s*ᵢDᵢ/D*ᵢ, D*ᵢ/s*ᵢDᵢ) < 1.25)` |

Notes that bite:

* **Eq. 8 is L1, Eq. 16 is L2.** The loss and the metric use different norms.
* **Eq. 8/10 divide by `|Ω|` with the confidence weights *inside* the sum** —
  down-weighting, not re-normalising. There is no division by `Σw`.
* **`T̃_ij` (Eq. 9) is computed once, before adaptation**, from UFM matches +
  MAGSAC++. It is a frozen pseudo-label: "the adapter cannot influence its own
  pose pseudo-label."
* **`ρ` is normalised to [0,1] by the grid boundary**, computed as a patch radius;
  the `ρ·δ_θ` factor "suppresses the angular term at the image center, where the
  angle is ill-defined, and lets it grow toward the periphery."
* **All adapter parameters are initialised to zero** → adaptation starts exactly
  at the pretrained positional table.
* **Eq. 16 uses ground-truth pose**, so it "evaluates depth independently of
  predicted pose."

---

## 4. Hyperparameters (Sec. 4.3, "Implementation details")

| Item | Value |
|---|---|
| Resolution | maximum patch-aligned **504 × 504** |
| Optimiser | **Adam**, learning rate **1 × 10⁻³** |
| Gradient clipping | **norm 1.0** |
| Batch | batches of three-frame windows |
| PE bins | **N_r = 20 radial, N_θ = 8 angular** |
| RoPE bins | **20 radial** (RoPE backbones only) |
| Init | all residual adapter parameters **zero** |
| `w_pose, w_smooth, w_L2, w_TV` | **1, 10, 2, 20** — "in all experiments" |
| Adaptation set | **30 three-frame windows** per sequence |
| Static filter | drop windows with average optical-flow displacement **< 2 px** |
| Matcher | **UFM** [44] correspondences + confidences; **MAGSAC++** [45] for `T̃` |
| Inference after adaptation | **a single forward pass per fisheye frame** |
| LoRA baseline | QKV adapters, **r = 8, α = 16** (swept `r ∈ {4,8,16}`, `α ∈ {8,16,32}`) |
| CalTok baseline | **t = 4** auxiliary tokens added to attention layers (swept `t ∈ {4,8}`) |
| Center-PH | one forward-looking **110°** virtual pinhole crop — 110° "to avoid severe center compression and peripheral stretching from wider rectilinear projections" |
| Multi-PH | Center-PH + **four additional virtual pinhole views = 5 total**, fused back into the fisheye frame |
| Default backbone | **DA3-Small**, "unless specified otherwise" |

**Trainable count.** `N_r·C + N_θ·C` for the PE tables, plus `20` for RoPE. Tab. 4
states `C = 384` for DA3-Small, giving `20·384 + 8·384 = 7680 + 3072 = 10,752` —
the abstract's and Sec. 1's headline. Tab. 4b lists **10.8K**. Compare LoRA
147.5K and CalTok 18.4K; the abstract's "∼14× fewer than LoRA" matches
`147.5/10.752 = 13.7`.

---

## 5. Evaluation protocol and metrics (Sec. 5, App. A)

* Adaptation set = **30 three-frame windows** from the sequence, nearly static
  windows (mean flow < 2 px) filtered out.
* **Fit on that set, evaluate on the *full* sequence.** Adapting unsupervised on
  the test sequence is the method, not leakage.
* **Relative pose on consecutive image pairs.** "All methods use the same pair
  sampling."
* Metrics: `R°` and `t°` (Eq. 14, degrees); `d_reproj` (Eq. 16) as the main depth
  metric because "ground-truth dense depth is not available for all fisheye
  datasets and is often sparse or sensor-projected"; `AbsRel` following Eigen et
  al. [53] and `δ₁.₂₅` where reliable dense depth exists.
* `s` in Eq. 16 is "estimated per pair in closed form by robustly minimizing the
  distance between high-confidence matched 3D points."
* Ground-truth calibration by default ("the calibration provided with each
  dataset"); **Sec. B** repeats one sequence per dataset with AnyCalib-predicted
  calibration.
* ⚠️ The paper warns explicitly: *"short baselines can amplify
  translation-direction errors (`t°`) even when reprojection error (`d_reproj`)
  remains moderate."* Sequences with little motion are hostile to `t°`.

---

## 6. Datasets (Sec. 5)

| Dataset | Capture type | FOV | Dense depth | Sequence named in Tab. 2/5 |
|---|---|---|---|---|
| ETH3D [51] | multi-camera captures | 110° | ✅ | `terrains` |
| ScanNet++ [50] | posed DSLR fisheye scans | **115°** | ✅ | `3f15…` |
| KITTI-360 [48] | outdoor driving | 185° | ❌ | drive `0009`; Tab. 4a uses drive `0000`, cam02 |
| TUM-VI [49] | handheld **dual-fisheye** video | 195° | ❌ | `room6` |
| FIORD [52] | **dual-fisheye** indoor/outdoor | 200° | ❌ | `Kitchen` |

"All datasets provide reference poses; ETH3D and ScanNet++ also provide dense
depth."

⚠️ **Named sequences are the tightest reproduction targets the paper offers.**
Tab. 2 and Tab. 5 are single-sequence. Tab. 1, Tab. 3 and Tab. 6 aggregate — Tab. 3
says "the mean of per-scene means", Tab. 6 is "mean ETH3D results across scenes".
Match the sequence before comparing to a number.

---

## 7. Results

All pose/reprojection entries are `R° ↓ / t° ↓ / d_reproj ↓`. **Bold marks the
best in a column group**, following the paper's own marking.

### Tab. 1 — DA3-Small across five datasets

| Dataset | Vanilla | Center-PH | Multi-PH | LoRA | CalTok | **RayTun3R** |
|---|---|---|---|---|---|---|
| ETH3D | 8.59 15.16 15.98 | 3.46 13.70 10.92 | 3.31 13.68 13.48 | 2.18 10.74 9.02 | 2.48 13.21 11.94 | **0.70 4.48 5.82** |
| KITTI-360 | 1.69 12.81 11.64 | **0.79** 4.17 **3.10** | 1.71 9.75 4.72 | 1.37 8.49 5.56 | 1.66 10.05 5.83 | 0.84 **2.92** 3.88 |
| TUM-VI | 10.41 23.23 57.01 | 3.33 29.24 **3.22** | 2.99 25.60 4.92 | 3.38 13.63 3.83 | 3.84 16.17 9.61 | **2.41 13.23** 3.81 |
| ScanNet++ | 10.21 30.26 23.82 | 3.27 22.77 2.21 | 1.66 10.43 **1.63** | 3.68 17.66 4.98 | 4.51 23.20 7.02 | **1.11 5.78** 4.16 |
| FIORD | 18.20 29.50 75.30 | 6.92 23.40 **7.20** | 6.30 18.90 15.60 | 7.75 12.20 12.10 | 20.40 22.20 25.20 | **4.10 5.40** 9.00 |

The paper's own summary: RayTun3R "gives the lowest translation-direction error on
all datasets and the lowest rotation error on ETH3D, TUM-VI, ScanNet++, and
FIORD" — i.e. it loses `R°` only on KITTI-360, to Center-PH.

⚠️ **RayTun3R does not win `d_reproj`.** It loses to Center-PH or Multi-PH on 4 of
5 datasets (all but ETH3D). The claim of the paper is about **pose**; the text
concedes "Center-PH remains strong for depth on some moderate-FOV datasets, while
Multi-PH improves coverage at a higher cost."

### Tab. 2 — π³ and VGGT backbones, single named sequences, GT calibration

| BB | Method | ETH3D terrains | KITTI 0009 | TUM-VI room6 | ScanNet++ 3f15 | FIORD Kitchen |
|---|---|---|---|---|---|---|
| π³ | LoRA | 1.40 4.9 1.4 | 2.22 3.4 2.7 | 9.01 22.4 6.8 | 3.11 16.1 2.1 | 4.59 4.62 **5.5** |
| π³ | CalTok | 1.49 7.5 1.3 | 0.93 11.5 4.6 | 12.3 24.7 10.0 | 38.1 17.2 9.1 | 27.24 11.96 7.7 |
| π³ | Center-PH | 0.97 5.9 1.1 | 0.88 4.8 **1.5** | 1.67 13.5 **1.2** | 2.28 25.7 5.2 | **1.11** 3.24 5.6 |
| π³ | Vanilla | 4.65 4.4 3.9 | 3.24 14.2 5.4 | 9.12 27.4 31.6 | 6.17 19.7 38.6 | 15.36 11.63 28.7 |
| π³ | **RayTun3R** | **0.60 0.7 0.3** | **0.76 1.9** 1.9 | **1.14 6.4** 2.0 | **0.78 1.9 0.9** | 3.49 **2.24** 5.6 |
| VGGT | LoRA | 1.40 6.4 1.4 | 0.60 25.3 3.2 | 13.0 24.7 7.9 | 8.95 17.2 4.4 | 6.48 3.81 **6.1** |
| VGGT | CalTok | 2.45 15.6 1.9 | **0.52** 28.5 4.0 | 23.4 32.2 9.4 | 16.7 26.3 6.9 | 7.83 36.39 7.0 |
| VGGT | Center-PH | 0.97 7.4 2.9 | 0.82 4.3 3.0 | 1.13 10.5 **2.4** | 2.45 27.3 6.1 | 6.43 10.11 7.58 |
| VGGT | Vanilla | 5.98 6.9 12.4 | 2.72 20.9 14.5 | 8.54 22.1 88.6 | 7.21 16.6 39.4 | 25.31 27.02 33.9 |
| VGGT | **RayTun3R** | **0.53 1.0 1.2** | 0.70 **3.9 2.2** | **1.12 7.7** 3.6 | **0.93 6.0 3.2** | **6.19 3.72 6.1** |

Paper's summary: "Across both backbones, RayTun3R gives the lowest translation
error on every sequence and consistently ranks among the best methods for
rotation." It loses `R°` twice — π³/FIORD Kitchen to Center-PH, VGGT/KITTI 0009 to
CalTok.

### Tab. 3 left — dense depth (AbsRel ↓ / δ₁.₂₅ ↑), DA3-Small

| Method | ETH3D | ScanNet++ |
|---|---|---|
| Vanilla | 0.178 / 0.751 | 0.282 / 0.601 |
| Center-PH | 0.111 / 0.867 | **0.066 / 0.961** |
| LoRA | 0.166 / 0.814 | 0.175 / 0.760 |
| CalTok (t=4) | 0.175 / 0.793 | 0.168 / 0.769 |
| **RayTun3R** | **0.107 / 0.884** | 0.108 / 0.886 |

⚠️ Center-PH **wins depth on ScanNet++** by a wide margin (0.066 vs 0.108). The
paper concedes it: "Center-PH remains strong on depth because it produces
perspective images close to the backbone's pretraining distribution. However,
Center-PH discards the peripheral fisheye content, yielding less accurate pose
estimates: On ScanNet++, RayTun3R reduces Center-PH rotation from 3.27° to 1.11°
and translation from 22.77° to 5.78°."

### Tab. 3 right — baseline sweeps on ETH3D (selects the main-table settings)

| Method | R° | t° | d_reproj |
|---|---|---|---|
| Center-PH | 3.46 | 13.70 | 10.92 |
| CalTok t=4 (default) | 2.48 | 13.21 | 11.94 |
| CalTok t=8 | 2.63 | 12.83 | 15.77 |
| LoRA r=4, α=8 | 2.47 | 12.18 | 10.85 |
| **LoRA r=8, α=16 (default)** | **2.18** | **10.74** | **9.02** |
| LoRA r=16, α=32 | 3.01 | 11.99 | 10.68 |
| RayTun3R | 0.70 | 4.48 | 5.82 |

### Tab. 5 (Sec. B) — GT vs AnyCalib calibration, DA3-Small, one sequence per dataset

The only place the paper gives **per-sequence DA3-Small numbers**.

| Method | ETH3D terrains | KITTI 0009 | TUM-VI room6 | ScanNet++ 3f15 | FIORD Kitchen |
|---|---|---|---|---|---|
| **RayTun3R (GT)** | 0.48 0.9 1.7 | 0.69 2.4 2.9 | 1.94 11.2 3.6 | 0.40 2.2 1.7 | 3.1 2.5 5.5 |
| RayTun3R (AnyCalib) | 1.02 2.3 2.6 | 1.14 2.5 3.1 | 1.16 7.0 4.4 | 0.75 4.6 3.1 | 4.9 3.0 5.2 |
| LoRA (GT) | 3.62 3.4 4.0 | 0.61 2.8 4.6 | 2.96 12.6 2.8 | 4.22 23.0 2.9 | 7.7 14.6 10.1 |
| LoRA (AnyCalib) | 2.30 4.7 3.2 | 2.32 2.8 4.8 | 6.60 20.4 16.6 | 3.52 17.5 4.9 | 10.2 14.2 10.8 |
| CalTok (GT) | 3.41 4.8 4.5 | 1.09 3.1 4.4 | 3.79 14.9 4.6 | 3.09 20.0 4.4 | 15.8 15.5 9.0 |
| CalTok (AnyCalib) | 2.15 4.4 3.4 | 1.90 8.5 5.8 | 7.21 42.0 38.1 | 4.44 24.7 5.5 | 30.3 59.9 10.6 |

"The errors vary across sequences; however, RayTun3R exhibits higher robustness
compared to the larger failures observed for LoRA and CalTok on TUM-VI,
ScanNet++, and FIORD." Note that AnyCalib calibration *helps* RayTun3R on TUM-VI
room6 (1.94 → 1.16 `R°`) and FIORD `d_reproj` (5.5 → 5.2).

### Tab. 6 (Sec. E) — ETH3D mean across five frozen backbones (`R°`, `t°`, AbsRel, δ₁.₂₅)

| Method | DA3-Small | DA3-Base | DA3-Large | π³ | VGGT |
|---|---|---|---|---|---|
| Baseline | 8.59 15.16 0.178 0.751 | 8.27 12.24 0.147 0.794 | 6.36 13.94 0.135 0.828 | 2.66 11.30 0.250 0.642 | 3.19 11.52 0.285 0.557 |
| Center-PH | 3.46 13.70 0.111 0.867 | 1.85 9.36 **0.082 0.911** | 1.56 9.32 **0.075 0.941** | 1.08 10.46 **0.156** 0.772 | 1.17 8.98 0.228 0.623 |
| **RayTun3R** | **0.70 4.48 0.107 0.884** | **0.54 3.26** 0.089 0.910 | **0.51 2.96** 0.083 0.925 | **0.66 2.48** 0.175 **0.863** | **0.96 4.82 0.139 0.834** |

"RayTun3R gives the lowest mean rotation and translation error for every backbone.
Center-PH is competitive on depth for DA3-Base, DA3-Large, and π³, but its pose
errors remain consistently higher."

---

## 8. Ablations — what each piece is worth

### Tab. 4a — components, KITTI-360 drive 0000 cam02 (185°), train 30 frames / eval first 500

| Configuration | R° | t° | d_reproj |
|---|---|---|---|
| Patch undistortion (no learnable PE) | 1.397 | 6.66 | 8.96 |
| Naive remap of PE | 0.810 | 12.93 | 11.53 |
| Radial PE only | 1.154 | 5.48 | 3.70 |
| Radial + angular PE | 1.038 | 4.21 | 3.39 |
| RayTun3R w/o border token | 1.061 | 4.45 | 3.17 |
| RayTun3R w/o DPT pos fix | 1.094 | 4.78 | 3.64 |
| RayTun3R w/o RoPE adapter | 0.942 | 5.56 | 3.27 |
| **RayTun3R (full)** | 1.183 | 4.81 | **3.03** |

Paper's reading: "The largest gain comes from the learned PE residual. Radial PE
only already reduces reprojection error substantially, and Radial + angular PE
further improves the pose–depth balance, indicating that the remaining mismatch is
not purely radial. The Naive remap of PE baseline … improves rotation, but gives
poor translation and depth."

⚠️ **The full model is selected on `d_reproj`, not on pose.** Its `R°` of 1.183 is
beaten by **six of the seven** other rows (all but "Patch undistortion" at 1.397),
and its `t°` of 4.81 by "Radial + angular PE" (4.21) and "w/o border token"
(4.45). The paper states this plainly: "the Full model gives the lowest
reprojection error, although some ablations achieve slightly lower pose error."

### Tab. 7b (Sec. G) — PE/RoPE split and bin counts, ETH3D terrains

| Configuration | R° | t° | d_reproj |
|---|---|---|---|
| Absolute PE only (no RoPE) | 0.68 | 0.9 | 1.6 |
| **RoPE only (no absolute PE)** | **19.52** | 7.8 | 9.6 |
| Both (full) | 0.48 | 0.9 | 1.6 |
| N_r=10, N_θ=8 | 0.72 | 0.9 | 1.7 |
| **N_r=20, N_θ=8 (default)** | 0.48 | 0.9 | 1.6 |
| N_r=40, N_θ=8 | 0.47 | 0.9 | 1.5 |
| **N_r=20, N_θ=0 (radial only)** | **2.82** | 3.3 | 3.6 |

"Absolute PE is the main learned component on DA3-Small: using RoPE only remains
close to the unadapted model, while Absolute PE alone nearly matches the full
model." (RoPE-only at 19.52° is indeed close to Tab. 8's Vanilla 19.87° on the
same sequence.) On bins: "increasing the radial resolution beyond 20 bins gives
only marginal gains, while removing angular bins substantially degrades
performance" — 0.48 → 2.82, a 6× cost — "This indicates that the remaining
camera-geometry mismatch contains an orientation-dependent component, not only a
radial one."

### Tab. 7a (Sec. F) — loss components, ETH3D terrains

| Pose | L2 | TV | R° | t° | d_reproj |
|---|---|---|---|---|---|
| ✓ | ✓ | ✓ | **0.48** | **0.9** | **1.7** |
| ✗ | ✓ | ✓ | 0.68 | 1.1 | 1.7 |
| ✓ | ✗ | ✓ | 0.81 | 1.0 | 1.8 |
| ✓ | ✓ | ✗ | 0.58 | 1.3 | 1.9 |
| ✗ | ✗ | ✓ | 0.65 | 1.4 | 1.8 |
| ✗ | ✓ | ✗ | 0.61 | 1.5 | 1.7 |
| ✓ | ✗ | ✗ | 0.78 | 1.5 | 1.8 |
| ✗ | ✗ | ✗ | 0.75 | 1.8 | 1.8 |

"The full objective is best or tied on this sequence. Removing the pose loss
mainly degrades pose accuracy, while removing either the ℓ2 or TV regularizer
generally weakens pose or reprojection performance." The spread is small
(0.48–0.81 on `R°`) next to the PE/RoPE split above.

### Tab. 8 (Sec. G) — learned residuals vs. parameter-free corrections

"Param.-free" = patch undistortion + prediction-grid correction + border token.

| Configuration | ETH3D terrains | FIORD Kitchen |
|---|---|---|
| Vanilla (no PE/RoPE, no param.-free) | 19.87 8.2 10.7 | 28.09 20.7 14.2 |
| Param.-free only | 17.78 7.3 10.9 | 39.04 36.3 19.2 |
| Learned only (PE+RoPE) | 0.67 1.0 1.8 | 4.64 4.1 5.8 |
| **RayTun3R Full** | **0.48 0.9 1.6** | **3.10 2.5 5.5** |

⚠️ **Parameter-free corrections alone make FIORD Kitchen dramatically worse**
(28.09 → 39.04 on `R°`, and worse on all three metrics). "Parameter-free
corrections alone provide only small gains on ETH3D and degrade FIORD Kitchen, but
they improve the learned adapter when combined with it."

### Tab. 4b — inference cost (DA3-Small, 504×504, one RTX A4000, averaged over 1000 frames)

| Method | ms/frame | Overhead | Trainable params |
|---|---|---|---|
| Vanilla DA3 | ~100 | baseline | 0 |
| Center-PH (single 110°) | ~105 | +5% | 0 |
| Multi-PH (5 views) | ~400 | +300% | 0 |
| LoRA (r=8) | ~110 | +10% | 147.5K |
| CalTok (t=4) | ~105 | +5% | 18.4K |
| **RayTun3R** | **~100** | **≈ 0%** | **10.8K** |

---

## 9. Compute (Sec. D)

* Hardware: **NVIDIA RTX A6000 and RTX A4000**, "with independent scenes,
  sequences, and baselines executed in parallel when possible."
* **~2–3 hours** for a full train-and-evaluate run per ETH3D, ScanNet++ or FIORD
  scene; **~3–5 hours** per KITTI-360 drive or TUM-VI sequence, "because of longer
  sequences."
* Reported experiments: **~180–250 total GPU-hours**, excluding preliminary
  development and failed configurations.

---

## 10. ⚠️ What the paper does not specify

Each of these is a real degree of freedom left open by the text.

1. **Number of adaptation iterations / epochs / optimiser steps.** Never stated
   anywhere — the words "iteration", "epoch" and "step" do not appear in a
   training-length sense. Sec. D's 2–3 h per scene is the only indirect handle,
   and it covers training *and* evaluation over the full sequence.
2. **Batch size.** "Batches of three-frame windows" — how many windows per batch is
   not given.
3. **Number of GPUs per run.** Only the GPU models and that independent
   scenes/sequences/baselines run in parallel. Nothing suggests within-scene
   sharding of a ~10k-parameter adapter.
4. **Depth convention in Eq. 7.** Eq. 3 writes the fisheye `κ⁻¹` with third
   component **1**, which makes `D` in Eq. 7 *planar z*. That form cannot
   represent rays past 90°, so it is impossible as written for TUM-VI (195°) and
   FIORD (200°); unit-norm rays (euclidean range) must be used at least there. The
   two readings differ by a per-pixel `1/cos θ`, which is radially varying, so the
   single global `s` in Eq. 16–18 cannot absorb a mismatch.
5. **How `Ω` is determined.** Defined as "the set of pixels inside the valid
   fisheye disc", but not how it is derived for datasets whose frames are
   full-frame rather than circularly vignetted.
6. **Which camera of the dual-fisheye datasets** (TUM-VI, FIORD) is evaluated, and
   whether the two are used jointly.
7. **Multi-PH fusion rule.** "Fuses predictions … back into the fisheye frame" —
   neither the pose fusion nor the depth fusion rule is given, nor the orientation
   of the four extra views (Fig. 3b suggests tilted, but the text does not say).
8. **`d_reproj` scale estimate.** Eq. 16 writes `min_{s>0}` over the 2D
   reprojection; the text says `s` is found "in closed form by robustly minimizing
   the distance between high-confidence matched 3D points" — a 3D fit, not the 2D
   minimisation the equation states. These are different estimators.
9. **Which and how many scenes/sequences** feed the aggregates. Tab. 1, Tab. 3 and
   Tab. 6 are means; only one sequence per dataset is ever named (Tab. 2, Tab. 5).
10. **Optical-flow source for the 2 px static filter.** Plausibly UFM, but not
    stated.
11. **LoRA and CalTok placement.** "QKV adapters" and "auxiliary tokens added to
    attention layers" — which blocks, and how many, is not given, so the quoted
    147.5K / 18.4K counts are the only constraint on the scope.
12. **Backbone of Tab. 8.** Not stated in the caption or Sec. G; Sec. 5's "unless
    specified otherwise, experiments use DA3-Small" is the only basis.

---

## 11. Errata and internal inconsistencies (verified against the PDF)

1. **The 10,752 headline counts only the two PE tables.** `20·384 + 8·384 =
   10,752` is exactly `N_r·C + N_θ·C` at DA3-Small's width. The paper separately
   specifies a 20-parameter radial RoPE table and ablates it on DA3-Small
   (Tab. 4a, Tab. 7b), so the full adapter has 10,772 parameters. Tab. 4b's
   "10.8K" does not disambiguate — both values round to 10.8K — but "10,752
   trainable parameters" in the abstract and Sec. 1 is the PE-only figure.
2. **Eq. 17 as printed is not AbsRel.** It is `mean(‖s*ᵢDᵢ − D*ᵢ‖₂)` — an absolute
   error with no division by the ground truth, so it is neither *relative* nor the
   Eigen et al. [53] definition it cites, nor consistent with the reported
   magnitudes (0.107 on ETH3D).
3. **Fig. 2 panel references in the body text are swapped.** Sec. 3 says the
   pretrained curves are flat "cf. Fig. 2a,b" and the adapted ones bend "cf.
   Fig. 2c,d", but the subfigure labels are (a) pretrained σ₁, (b) adapted σ₁,
   (c) pretrained det, (d) adapted det. The correct references are (a,c) and
   (b,d); the figure caption itself has it right.
4. **`d_reproj` for the full model on ETH3D terrains is reported as both 1.6 and
   1.7.** Same sequence, same configuration: Tab. 7b "Both (full)" = 1.6 and
   Tab. 8 "RayTun3R Full" = 1.6, but Tab. 7a "✓✓✓" = 1.7 and Tab. 5 "RayTun3R
   (GT)" = 1.7. `R°` (0.48) and `t°` (0.9) agree across all four.
5. **Tab. 4a's full model is not the best row in its own ablation on pose** — six
   of seven ablated variants beat it on `R°`. The paper acknowledges this and
   selects on `d_reproj` (see §8 above).
6. Cosmetic: Tab. 3 left labels the row "CalTok4" where Tab. 3 right writes
   "CalTok t=4".

---

## 12. Limitations the authors state (Sec. 6)

1. The correction is **camera-specific** — a different fisheye camera or
   distortion profile requires new adaptation.
2. It assumes a principal point and **mostly radial** distortion; strong
   tangential or non-radial optics are not explicitly covered.
3. It requires camera parameters, "although Sec. B shows that predicted
   calibration from off-the-shelf networks such as AnyCalib [23] remains
   accurate."
4. Fisheye only — "panoramic or equirectangular inputs remain for future work."
5. ⚠️ **"The training set for RayTun3R needs sufficient inter-frame displacement.
   With small or degenerate motion, the self-supervised constraints become weak
   because large depth or translation-direction errors can induce only small
   reprojection errors."** This is the stated reason the 2 px static filter exists.

---

## 13. Related work, and assets

Positioning (Sec. 2):

* `Fisheye3R` (arXiv:2603.28896) — calibration tokens + masked attention, trained
  with supervision from synthetic fisheye distortions and optionally supervised
  fisheye data. Contemporaneous.
* `FishRoPE` (arXiv:2604.10391) — fisheye-aware angular RoPE for supervised
  fisheye detection and BEV segmentation; RoPE-only, and the full model also
  trains LoRA modules and task-specific heads. Contemporaneous.
* `PRoPE` (NeurIPS 2025) — camera intrinsics/extrinsics as relative positional
  encodings for multi-view transformers.
* `CalTok` (ICCV 2025) — "the closest baseline to our setting because it adapts
  mostly frozen depth networks to fisheye cameras."
* `AnyCalib` (ICCV 2025) — the calibration predictor used in Sec. B.
* Projection-based alternatives: cubemap/equirectangular reuse of pinhole
  backbones, adaptive perspective slicing for VGGT-style models, cubemap alignment
  by graph optimisation — all "duplicate computation across virtual views, require
  prediction fusion, and do not let the model represent the original fisheye
  geometry directly."

**Appendix C** rules out the obvious alternative — re-index the pretrained PE table
at undistorted coordinates. Undistortion samples the pinhole grid highly
non-uniformly, so many fisheye patches collapse onto a small central region of the
table and their embeddings become nearly indistinguishable (Fig. 5). The "Naive
remap of PE" row in Tab. 4a is the quantitative version.

**Assets and licences (Sec. I, Tab. 9)** — relevant to anyone re-running this:
KITTI-360 CC BY-NC-SA 3.0 (scripts MIT); TUM-VI data CC BY 4.0 (code BSD-2);
ScanNet++ under its Terms of Use, non-commercial research/education; ETH3D
CC BY-NC-SA 4.0; FIORD CC BY 4.0. Models: DA3 Apache-2.0 but **CC BY-NC 4.0 for
Large/Giant**; π³ code BSD-3, weights CC BY-NC 4.0; VGGT checkpoint-specific,
non-commercial; UFM code BSD-3, checkpoint CC BY-NC-SA 4.0.

**Broader impact (Sec. H)** notes that fisheye cameras are common in vehicles,
robots, doorbells and indoor monitoring, so easier camera adaptation lowers the
barrier to 3D mapping in private or shared spaces; deployment "should include
consent, calibration validation, and application-specific safety checks."
