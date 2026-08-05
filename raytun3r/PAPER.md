# RayTun3R — paper reference

Condensed from **arXiv:2607.02711v1** (Sinitsyn, Araslanov, Cremers; TU Munich / MCML /
Oxford), 15 pages incl. appendices A–I. There is no separate supplementary: the appendix
is inside the same PDF, and arXiv lists no ancillary files.

**Read this instead of the PDF.** It carries every number, hyperparameter and protocol
detail needed to reproduce or extend the work. Sections marked ⚠️ are things the paper
does *not* say — read those before assuming a run is "wrong".

Code is not public ("will be made publicly available"), so `raytun3r/` is a
from-paper reconstruction. Cross-references below are to our flags in
[train.py](train.py) / [eval.py](eval.py).

---

## 1. Method in one paragraph

Freeze a pinhole-pretrained 3D foundation model; adapt **only** the components mapping
token location → camera ray. Learned parts are small radial/angular lookup tables on the
absolute PE and (when present) RoPE. Everything else — attention, MLPs, DPT heads — stays
frozen. Fit online on a short unlabelled fisheye segment with geometric losses, then run
the rest of the sequence at zero extra inference cost.

**Why it should work (Sec. 3).** For a pinhole camera the backprojection Jacobian
`J_{κ⁻¹} = ∂κ⁻¹/∂(u,v)` is *constant over the image*. For a radially symmetric fisheye it
is not. Measuring the pretrained PE's own Jacobian `J_PE = ∂P_A/∂(u,v)` (via σ₁ and
`√det(J_PEᵀJ_PE)`) shows it is nearly **flat in radius** for every DA3 size — i.e. the PE
encodes a pinhole spatial prior. After adaptation the curves bend to match the analytic
fisheye reference (Fig. 2).

---

## 2. Equations

Camera: central model, projection `κ`, inverse `κ⁻¹`, over the valid disc `Ω ⊂ ℝ²`.

| Eq. | Content |
|---|---|
| 1 | Pinhole `κ⁻¹(u,v) = ((u−cₓ)/fₓ, (v−c_y)/f_y, 1)ᵀ` |
| 2 | `J_{κ⁻¹} = ∂κ⁻¹/∂(u,v)` — constant for pinhole |
| 3 | Fisheye `κ⁻¹(u,v) = (g(u,v)(u−cₓ)/fₓ, g(u,v)(v−c_y)/f_y, 1)ᵀ`, `g` non-linear in radius (pinhole = `g ≡ 1`) |
| 4 | `J_{κ⁻¹} = g·diag(1/fₓ, 1/f_y, 0) + ((u−cₓ)/fₓ, (v−c_y)/f_y, 0)ᵀ (∂_u g, ∂_v g)` — radius-dependent |
| **5** | **Absolute PE**: `P'(u,v) = P_A(u,v) + t_r(ρ) + ρ·δ_θ(θ)` |
| **6** | **RoPE**: `ω'(u,v) = ω(u,v) + Δ_r(ρ)` — one param per radial bin, shared across frequencies |
| 7 | `Xᵢ(u,v) = Dᵢ(u,v)·κ⁻¹(u,v)` |
| **8** | `L_reproj(i,j) = (1/\|Ω\|) Σ_{(u,v)∈Ω} w_ij(u,v)·‖κ(T_j T_i⁻¹ Xᵢ(u,v)) − m_ij(u,v)‖₁` |
| **9** | `L_pose(i,j) = arccos((tr(R̃ᵀR̂)−1)/2) + arccos(⟨t̃,t̂⟩/(‖t̃‖‖t̂‖))` |
| **10** | `L_smooth(i) = (1/\|Ω\|) Σ e^{−\|∂ₓI\|}\|∂ₓD*\| + e^{−\|∂_yI\|}\|∂_yD*\|`, `D* = D/mean(D)` |
| **11** | `L_L2 = (1/H_pW_p) Σ ‖P'(x,y) − P_A(x,y)‖²₂` |
| **12** | `L_TV = (1/H_pW_p) Σ (‖P'(x+1,y)−P'(x,y)‖²₂ + ‖P'(x,y+1)−P'(x,y)‖²₂)`, valid neighbours only |
| **13** | `L = L_reproj + w_smooth·L_smooth + w_L2·L_L2 + w_TV·L_TV + w_pose·L_pose` |
| 14 | `R_err = arccos((tr(R*ᵀR̂)−1)/2)`, `t_err = arccos(⟨t*,t̂⟩/(‖t*‖‖t̂‖))` |
| 15 | `Xᵢ(u) = Dᵢ(u)κ⁻¹(u)` (same as 7) |
| **16** | `d_reproj(i,j) = min_{s>0} (1/\|Ω\|) Σ ‖κ(R*_ij(s·Xᵢ(u)) + t*_ij) − u_ij(u)‖₂` — **ground-truth pose**, one global scale per pair |
| 17 | `d_AbsRel(i) = (1/\|Ωᵢ\|) Σ ‖s*Dᵢ(u) − D*ᵢ(u)‖₂` |
| 18 | `δ₁.₂₅(i) = (1/\|Ωᵢ\|) Σ 1(max(s*Dᵢ/D*ᵢ, D*ᵢ/s*Dᵢ) < 1.25)` |

Notes that bite:

* **Eq. 8 is L1, Eq. 16 is L2.** Loss and metric use different norms. Ours matches
  ([losses.py:96](losses.py#L96) `.abs().sum(-1)`, [metrics.py:105](metrics.py#L105) `.norm(dim=-1)`).
* **Eq. 8/10 divide by `|Ω|` with the confidence weights *inside* the sum** — down-weighting,
  not re-normalising. Do not divide by `Σw`.
* **`T̃_ij` (Eq. 9) is computed once, before adaptation**, from UFM matches + MAGSAC++.
  It is a frozen pseudo-label, so the adapter cannot drift its own pose target.
* **ρ is normalised to [0,1] by the grid boundary**; the `ρ·δ_θ` factor suppresses the
  angular term at the centre where θ is ill-defined.
* **All adapter parameters are initialised to zero** → adaptation starts exactly at the
  pretrained model.

---

## 3. Hyperparameters (Sec. 4.3 "Implementation details")

Every one of these is already the default in our CLI.

| Paper | Value | Our flag |
|---|---|---|
| Resolution | max patch-aligned **504 × 504** | `--max-size 504` |
| Optimiser | **Adam**, lr **1e-3** | `--lr 1e-3` |
| Gradient clipping | **norm 1.0** | `--clip 1.0` |
| Batch | three-frame windows | `--seq-len 3` |
| PE bins | **N_r = 20 radial, N_θ = 8 angular** | `--n-radial 20 --n-angular 8` |
| RoPE bins | **20 radial** (RoPE backbones only) | `--n-rope-radial 20` |
| Init | all residuals **zero** | — |
| `w_pose, w_smooth, w_L2, w_TV` | **1, 10, 2, 20** | `--w-pose/-smooth/-l2/-tv` |
| Adaptation set | **30 three-frame windows** | `--windows 30` |
| Static filter | drop windows with mean optical flow **< 2 px** | `--min-flow-px 2.0` |
| Matcher | **UFM** + confidences; **MAGSAC++** for `T̃` | `--matcher ufm` |
| LoRA baseline | QKV, **r=8, α=16** (swept r∈{4,8,16}, α∈{8,16,32}) | `--lora-r 8 --lora-alpha 16` |
| CalTok baseline | **t=4** (swept t∈{4,8}) | `--caltok-t 4` |
| Center-PH | single forward **110°** virtual pinhole | `fov_deg=110.0` |
| Multi-PH | Center-PH + **4 tilted views = 5 total**, fused to fisheye | `n=5` |

**Trainable count.** `20·C + 8·C` PE + `20` RoPE. For DA3-Small (`C = 384`):
`7680 + 3072 = 10,752` — the paper's headline. It excludes the 20 RoPE params it also
describes, so the true total is **10,772**. Tab. 4b rounds to "10.8K", consistent with
including them. Compare LoRA 147.5K, CalTok 18.4K.

---

## 4. Evaluation protocol (Sec. 5)

* Adaptation set = 30 three-frame windows from the sequence, static windows filtered out.
* **Fit on that set, evaluate on the *full* sequence.** Adapting on the test sequence
  unsupervised is the method, not leakage.
* **Relative pose on consecutive image pairs.** All methods use the same pair sampling.
* GT calibration by default; Sec. B repeats with AnyCalib-predicted calibration.
* ⚠️ The paper warns explicitly: *short baselines amplify `t°` even when `d_reproj` stays
  moderate.* Sequences with little motion are hostile to this metric.

---

## 5. Datasets

| Dataset | FOV | Dense depth | Sequence named in Tab. 2 |
|---|---|---|---|
| ETH3D | 110° | ✅ | `terrains` |
| ScanNet++ | **115°** (DSLR) | ✅ | `3f15…` |
| KITTI-360 | 185° | ❌ | drive `0009`; Tab. 4a uses drive `0000`, cam02 |
| TUM-VI | 195° | ❌ | `room6` |
| FIORD | 200° | ❌ | `Kitchen` |

⚠️ **Named sequences are the tightest reproduction targets we have.** Tab. 2 is
single-sequence; Tab. 1 and Tab. 3 aggregate ("mean of per-scene means"). Match the
sequence before comparing to a number.

⚠️ **ScanNet++ FOV conflict.** The paper says 115°. The actual ScanNet++ DSLR frame we
measured is a full-frame **~170°** fisheye. This is a real discrepancy between the paper's
text and the data, not a bug in our loader — see [README.md](README.md) and
`experiments/fov_sweep.py`. 170° is far outside anything these backbones saw in training,
and Ω changes every loss and metric at once, so this is the leading hypothesis for our
inverted baseline ordering.

---

## 6. Reproduction targets

All entries `R° ↓ / t° ↓ / d_reproj ↓`.

### Tab. 1 — DA3-Small across datasets

| Dataset | Vanilla | Center-PH | Multi-PH | LoRA | CalTok | **RayTun3R** |
|---|---|---|---|---|---|---|
| ETH3D | 8.59 15.16 15.98 | 3.46 13.70 10.92 | 3.31 13.68 13.48 | 2.18 10.74 9.02 | 2.48 13.21 11.94 | **0.70 4.48 5.82** |
| KITTI-360 | 1.69 12.81 11.64 | 0.79 4.17 **3.10** | 1.71 9.75 4.72 | 1.37 8.49 5.56 | 1.66 10.05 5.83 | **0.84 2.92** 3.88 |
| TUM-VI | 10.41 23.23 57.01 | 3.33 29.24 **3.22** | 2.99 25.60 4.92 | 3.38 **13.63** 3.83 | 3.84 16.17 9.61 | **2.41** 13.23 3.81 |
| ScanNet++ | 10.21 30.26 23.82 | 3.27 22.77 2.21 | 1.66 10.43 **1.63** | 3.68 17.66 4.98 | 4.51 23.20 7.02 | **1.11 5.78** 4.16 |
| FIORD | 18.20 29.50 75.30 | 6.92 23.40 **7.20** | 6.30 18.90 15.60 | 7.75 12.20 12.10 | 20.40 22.20 25.20 | **4.10 5.40** 9.00 |

⚠️ **RayTun3R does not win `d_reproj`.** It loses to Center-PH or Multi-PH on 4 of 5
datasets. The claim is about **pose**. Do not treat a `d_reproj` loss as a failed
reproduction.

### Tab. 2 — π³ and VGGT backbones, single named sequences

| BB | Method | ETH3D terrains | KITTI 0009 | TUM-VI room6 | ScanNet++ 3f15 | FIORD Kitchen |
|---|---|---|---|---|---|---|
| π³ | LoRA | 1.40 4.9 1.4 | 2.22 3.4 2.7 | 9.01 22.4 6.8 | 3.11 16.1 2.1 | 4.59 4.62 5.5 |
| π³ | CalTok | 1.49 7.5 1.3 | 0.93 11.5 4.6 | 12.3 24.7 10.0 | 38.1 17.2 9.1 | 27.24 11.96 7.7 |
| π³ | Center-PH | 0.97 5.9 1.1 | 0.88 4.8 1.5 | 1.67 13.5 1.2 | 2.28 25.7 5.2 | **1.11** 3.24 5.6 |
| π³ | Vanilla | 4.65 4.4 3.9 | 3.24 14.2 5.4 | 9.12 27.4 31.6 | 6.17 19.7 38.6 | 15.36 11.63 28.7 |
| π³ | **RayTun3R** | **0.60 0.7 0.3** | **0.76 1.9 1.9** | **1.14 6.4 2.0** | **0.78 1.9 0.9** | 3.49 **2.24 5.6** |
| VGGT | LoRA | 1.40 6.4 1.4 | 0.60 25.3 3.2 | 13.0 24.7 7.9 | 8.95 17.2 4.4 | 6.48 3.81 6.1 |
| VGGT | CalTok | 2.45 15.6 1.9 | 0.52 28.5 4.0 | 23.4 32.2 9.4 | 16.7 26.3 6.9 | 7.83 36.39 7.0 |
| VGGT | Center-PH | 0.97 7.4 2.9 | 0.82 4.3 3.0 | 1.13 10.5 2.4 | 2.45 27.3 6.1 | 6.43 10.11 7.58 |
| VGGT | Vanilla | 5.98 6.9 12.4 | 2.72 20.9 14.5 | 8.54 22.1 88.6 | 7.21 16.6 39.4 | 25.31 27.02 33.9 |
| VGGT | **RayTun3R** | **0.53 1.0 1.2** | **0.70** 3.9 2.2 | **1.12 7.7** 3.6 | **0.93 6.0 3.2** | 6.19 **3.72 6.1** |

**Our closest target**: VGGT on ScanNet++ `3f15…` → vanilla 7.21 → RayTun3R **0.93**
(7.8× on `R°`), with Center-PH at 2.45 (RayTun3R **2.6× better**). Our one real run gave
1.28× and Center-PH winning by ~5× — inverted on both counts.

### Tab. 3 left — dense depth (AbsRel ↓ / δ₁.₂₅ ↑), DA3-Small

| Method | ETH3D | ScanNet++ |
|---|---|---|
| Vanilla | 0.178 / 0.751 | 0.282 / 0.601 |
| Center-PH | 0.111 / 0.867 | **0.066 / 0.961** |
| LoRA | 0.166 / 0.814 | 0.175 / 0.760 |
| CalTok t=4 | 0.175 / 0.793 | 0.168 / 0.769 |
| **RayTun3R** | **0.107 / 0.884** | 0.108 / 0.886 |

⚠️ Center-PH **wins depth on ScanNet++** by a wide margin (0.066 vs 0.108); the paper
concedes this — a 110° rectified crop is closest to the backbone's pretraining
distribution. RayTun3R's case is that it keeps the full FOV and still wins pose.

### Tab. 6 — ETH3D mean across five frozen backbones (`R°`, `t°`, AbsRel, δ₁.₂₅)

| Method | DA3-Small | DA3-Base | DA3-Large | π³ | VGGT |
|---|---|---|---|---|---|
| Baseline | 8.59 15.16 0.178 0.751 | 8.27 12.24 0.147 0.794 | 6.36 13.94 0.135 0.828 | 2.66 11.30 0.250 0.642 | 3.19 11.52 0.285 0.557 |
| Center-PH | 3.46 13.70 0.111 0.867 | 1.85 9.36 **0.082 0.911** | 1.56 9.32 **0.075 0.941** | 1.08 10.46 **0.156** 0.772 | 1.17 8.98 0.228 0.623 |
| **RayTun3R** | **0.70 4.48 0.107 0.884** | **0.54 3.26** 0.089 0.910 | **0.51 2.96** 0.083 0.925 | **0.66 2.48** 0.175 **0.863** | **0.96 4.82 0.139 0.834** |

Pose improves for **every** backbone. Depth is a wash against Center-PH on the larger DA3s.

### Tab. 3 right — baseline sweeps on ETH3D (selects the main-table settings)

Center-PH 3.46 13.70 10.92 · CalTok t=4 2.48 13.21 11.94 · CalTok t=8 2.63 12.83 15.77 ·
LoRA r=4 α=8 2.47 12.18 10.85 · **LoRA r=8 α=16 2.18 10.74 9.02** · LoRA r=16 α=32 3.01 11.99 10.68 ·
RayTun3R 0.70 4.48 5.82

---

## 7. Ablations — what each piece is worth

### Tab. 4a — components, KITTI-360 drive 0000 cam02, train 30 frames / eval first 500

| Configuration | R° | t° | d_reproj |
|---|---|---|---|
| Patch undistortion (no learnable PE) | 1.397 | 6.66 | 8.96 |
| Naive remap of PE | 0.810 | 12.93 | 11.53 |
| Radial PE only | 1.154 | 5.48 | 3.70 |
| Radial + angular PE | 1.038 | 4.21 | 3.39 |
| w/o border token | 1.061 | 4.45 | 3.17 |
| w/o DPT pos fix | 1.094 | 4.78 | 3.64 |
| w/o RoPE adapter | 0.942 | 5.56 | 3.27 |
| **Full** | 1.183 | 4.81 | **3.03** |

⚠️ **The full model has the *worst* `R°` in its own ablation table** (1.183 vs 0.942
without the RoPE adapter). The paper states this plainly: the full model minimises
`d_reproj`, and "some ablations achieve slightly lower pose error". Do not tune toward
this table.

### Tab. 7b — PE/RoPE split and bin counts, ETH3D terrains

| Configuration | R° | t° | d_reproj |
|---|---|---|---|
| Absolute PE only (no RoPE) | 0.68 | 0.9 | 1.6 |
| **RoPE only (no absolute PE)** | **19.52** | 7.8 | 9.6 |
| Both (full) | 0.48 | 0.9 | 1.6 |
| N_r=10, N_θ=8 | 0.72 | 0.9 | 1.7 |
| **N_r=20, N_θ=8 (default)** | 0.48 | 0.9 | 1.6 |
| N_r=40, N_θ=8 | 0.47 | 0.9 | 1.5 |
| **N_r=20, N_θ=0 (radial only)** | **2.82** | 3.3 | 3.6 |

**The single most important row for us: RoPE-only = 19.52° vs 0.48° for the full adapter.**
Absolute PE is the method. `vggt_omega` is DINOv3-based — RoPE only, **no `pos_embed`
anywhere** — so `--backbone vggt_omega` reproduces this *negative* row by construction,
not the headline. The faithful targets are `da3` (primary) and `vggt`.

Also: beyond 20 radial bins gains are marginal, but dropping the **angular** term costs
6× (0.48 → 2.82). The residual mismatch is not purely radial.

### Tab. 7a — loss components, ETH3D terrains (Pose / L2 / TV)

`✓✓✓ 0.48 0.9 1.7` · `✗✓✓ 0.68 1.1 1.7` · `✓✗✓ 0.81 1.0 1.8` · `✓✓✗ 0.58 1.3 1.9` ·
`✗✗✓ 0.65 1.4 1.8` · `✗✓✗ 0.61 1.5 1.7` · `✓✗✗ 0.78 1.5 1.8` · `✗✗✗ 0.75 1.8 1.8`

Full objective best or tied. Spread is small (0.48–0.81) — the losses matter far less than
the PE/RoPE split above.

### Tab. 8 — learned vs parameter-free

| Configuration | ETH3D terrains | FIORD Kitchen |
|---|---|---|
| Vanilla | 19.87 8.2 10.7 | 28.09 20.7 14.2 |
| Param.-free only | 17.78 7.3 10.9 | **39.04 36.3 19.2** |
| Learned only | 0.67 1.0 1.8 | 4.64 4.1 5.8 |
| **Full** | **0.48 0.9 1.6** | **3.10 2.5 5.5** |

⚠️ **Parameter-free corrections alone make FIORD dramatically worse** (28.09 → 39.04).
They only help *combined* with the learned residual. If an ablation of ours shows
tokenisation/grid fixes hurting on their own, that reproduces the paper.

### Tab. 4b — inference cost (DA3-Small, 504×504, RTX A4000, 1000 frames)

Vanilla ~100 ms (baseline, 0 params) · Center-PH ~105 ms (+5%, 0) · **Multi-PH ~400 ms
(+300%, 0)** · LoRA r=8 ~110 ms (+10%, 147.5K) · CalTok t=4 ~105 ms (+5%, 18.4K) ·
**RayTun3R ~100 ms (≈0%, 10.8K)**

---

## 8. ⚠️ What the paper does NOT specify

These are real gaps. Each is a degree of freedom we had to choose, and a candidate
explanation for a failed reproduction.

1. **Number of adaptation iterations / epochs.** Never stated. We use `--iters 300`.
   Sec. D says a full train-and-evaluate run is **2–3 h per ScanNet++ scene** on an
   A4000/A6000. Our 300 iters finish in ~3 min. Even allowing that their figure covers
   evaluation over the full sequence and all baselines, the gap suggests **we may be
   training 5–10× too short**. This is testable and cheap — sweep `--iters`.
2. **Number of GPUs.** Not stated, and not a meaningful knob: ~10k parameters, nothing to
   shard within a scene, and a shared adapter across scenes would be a different method.
   Sec. D is explicit that the parallel axis is **independent scenes/sequences/baselines**,
   each on one GPU. "Same setting as the paper" = one GPU per scene, scenes in parallel.
3. **Depth convention.** Eq. 3 writes the fisheye `κ⁻¹` with third component **1**, which
   makes `D` in Eq. 7 *planar z*. But that form cannot represent rays past 90°, so it is
   impossible for TUM-VI (195°) and FIORD (200°). In practice they must use unit-norm rays
   (euclidean range) for those. We use `range` (`--convention range`). The two differ by a
   per-pixel `1/cos θ` — up to ~11× at a 170° frame corner — and **no global scale
   alignment absorbs it**, so `min_s` in Eq. 16 does not rescue a mismatch.
4. **Multi-PH fusion.** "Fuses predictions back into the fisheye frame" — the pose fusion
   rule is unspecified. We average rotations chordally across views.
5. **`d_reproj` scale estimate.** Eq. 16 says `min_{s>0}`; the text says `s` is found "in
   closed form by robustly minimising the distance between high-confidence matched 3D
   points" — i.e. a 3D fit, not the 2D reprojection minimisation the equation writes. We do
   a 1-D search over closed-form candidates on the reprojection.
6. **Which/how many ScanNet++ scenes** feed the Tab. 1 and Tab. 3 aggregates. Only
   Tab. 2's `3f15…` is named.
7. **Optical-flow source for the 2 px static filter.** Presumably UFM, but not stated.

---

## 9. Paper errata (verified against the PDF)

1. **10,752 params counts only the two PE tables** (20·384 + 8·384). The 20 RoPE
   parameters the paper also describes are excluded; true total **10,772**. Tab. 4b's
   "10.8K" is consistent with including them.
2. **Eq. 17 as printed is not AbsRel** — there is no division by ground truth, it is a
   scale-aligned L2. We implement the standard Eigen definition
   ([metrics.py:141](metrics.py#L141)).
3. **ScanNet++ FOV stated as 115°**, measured ~170° in the actual DSLR data (§5 above).

---

## 10. Limitations the authors state (Sec. 6)

* Correction is **camera-specific** — a different lens or distortion profile needs a new fit.
* Assumes a principal point and **mostly radial** distortion; no strong tangential/non-radial optics.
* Needs camera parameters (Sec. B: AnyCalib-predicted calibration is good enough).
* Fisheye only — panoramic / equirectangular left to future work.
* ⚠️ **"The training set needs sufficient inter-frame displacement. With small or
  degenerate motion the self-supervised constraints become weak, because large depth or
  translation-direction errors can induce only small reprojection errors."** This is the
  paper's own account of the failure mode we hit with stride-1 sampling, and the reason
  the 2 px static filter exists.

---

## 11. Related work worth knowing

`Fisheye3R` (arXiv:2603.28896) — calibration tokens + masked attention, supervised on
synthetic fisheye. `FishRoPE` (arXiv:2604.10391) — angular RoPE for supervised fisheye
detection/BEV; RoPE-only, and also trains LoRA + task heads. `PRoPE` (NeurIPS 2025) —
camera intrinsics/extrinsics as relative PE. `CalTok` (ICCV 2025) — the closest baseline.
`AnyCalib` (ICCV 2025) — the calibration predictor used in Sec. B.
Appendix C rules out the obvious alternative (re-index the pretrained PE table at
undistorted coordinates): wide-FOV undistortion collapses many fisheye patches onto a
small central region of the table, destroying positional resolution (Fig. 5).
