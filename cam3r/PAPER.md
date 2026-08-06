# CAM3R — paper reference

Condensed from **arXiv:2603.22631v1**, *CAM3R: Camera-Agnostic Model for 3D
Reconstruction* — Namitha Guruprasad¹, Abhay Yadav¹, Cheng Peng², Rama
Chellappa¹ (¹ Johns Hopkins University, ² University of Virginia), submitted
23 Mar 2026. Project page: <https://nam1410.github.io/cam3r>.

**28 pages in one PDF**: 15 pages of main paper, then "Supplementary Material"
(Figs. S1–S3, appendices A–E), then references. arXiv lists **no separate
supplement and no ancillary files** — everything the paper has is in that one
file. LNCS/ECCV-style two-column-free format, so equation numbers are `(1)…(12)`
in the main paper and `(S1)…(S5)` in the appendix.

**This file is the paper and only the paper.** Every number, hyperparameter and
protocol detail below is transcribed from the PDF ([cam3r.pdf](cam3r.pdf), kept
next to this file but not committed — 20 MB). Nothing here is a result, choice,
or opinion of the reproduction — those live in
[reproduction.md](reproduction.md), with the user-facing guide in
[README.md](README.md).

---

## 1. Method in one paragraph

A **two-view** feed-forward network that **decouples camera geometry from scene
geometry**. Two parallel streams: a **Ray Module (RM)** predicts a per-pixel unit
ray field `dᵢ(u) ∈ S²` from spherical-harmonic coefficients; a **Cross-view
Module (CVM)** — a DUSt3R-style Siamese encoder + dual cross-attending decoder —
predicts per-pixel **radial distance** `rᵢ(u) ∈ ℝ⁺`, a **confidence** `σᵢ(u)`,
and the **relative pose** `P_{2→1}`. The local pointmap is the product
`X^{i,i}(u) = dᵢ(u)·rᵢ(u)` (Eq. 1). The two ViT backbones are initialised from
**UniK3D** (RM) and **DUSt3R** (CVM); everything else is trained from scratch,
end-to-end, under a two-phase homogeneous → heterogeneous curriculum. For
multi-view, an exhaustive pairwise scene graph is **pruned** (cycle consistency +
3D MNN overlap + largest connected component) and then globally aligned by
**Ray-Aware Global Alignment**, which optimises poses/scales/log-depths while
**freezing the ray directions**, so points may only slide along their own optical
rays.

**Why decoupling (Sec. A, the architecture rationale).** Three stages were tried:

1. **Direct fine-tuning of DUSt3R** on the distorted mixture. Because that
   paradigm regresses view 2's geometry *directly into view 1's frame*, the
   network must entangle (a) the non-linear intrinsic projection with (b) scene
   geometry + relative pose. It "struggled to disentangle these conflicting
   signals, yielding severely distorted pointmaps." Tab. S1 quantifies it — see §7.
2. **Decoupled radial prediction with ground-truth rays.** The DPT head regresses
   radial distance strictly in each view's *own local frame*, anchored on GT rays.
   Stabilised pointmap regression, but needs GT intrinsics at inference.
3. **Learned ray estimation via a dedicated camera branch** ← final CAM3R. The
   RM (from UniK3D's angular module) predicts rays; a Relative Pose Network puts
   the two local pointmaps into a shared frame.

The headline claim of Sec. A / Tab. S1: *"simply exposing standard foundation
models to distorted data is mathematically insufficient."*

---

## 2. Architecture (Tab. S3, verbatim)

`N_P = HW/P²` is the token count for patch size `P`.

### Ray Module — camera geometry

| Stage | Configuration | Input | Output |
|---|---|---|---|
| Pixel Encoder | **ViT-L/14, 24 layers**, read out at layers **{6, 12, 18, 24}** | `B×3×H×W` | `{B×N₁₄×1024}×4` |
| Token Projection | Linear **1024 → 512** | `{B×N₁₄×1024}×4` | `{B×N₁₄×512}×4` |
| Angular Regressor | **2 Transformer blocks, 8 heads, D = 512** | `B×18×512` | `B×10` (intrinsics + SH coeffs.) |
| Ray Generation | **FoV projection + SH basis (deg ≤ 3)** | intrinsics + SH + pixel grid | `B×H×W×3` |

### Cross-View Module — scene geometry

| Stage | Configuration | Input | Output |
|---|---|---|---|
| Patch Embedding | Linear, **patch 16×16** | `B×3×H×W` | `B×N₁₆×1024` |
| Transformer Encoder | **24 blocks, 16 heads, D = 1024** | `B×N₁₆×1024` | `B×N₁₆×1024` |
| Decoder Projection | Linear **1024 → 768** | `B×N₁₆×1024` | `B×N₁₆×768` |
| Asymmetric Decoder | **12 blocks** (self + cross attention) | `B×N₁₆×768` | `{B×N₁₆×768}×12` |
| DPT Head | multi-scale fusion **{0, 6, 9, 12}**, `D_feat = 256` | `{B×N₁₆×768}×4` | `B×2×H×W` |
| 3D Point Computation | `P = r·d` | rays & distance map | `B×H×W×3` |
| Relative Pose Head | **global pooling + MLP (D = 256) + SVD** | decoder features | `B×4×4 ∈ SE(3)` |

Notes that bite:

* **The two backbones use different patch sizes** — RM is patch **14** (DINOv2 /
  UniK3D lineage), CVM is patch **16** (DUSt3R lineage). Two token grids, two
  encoders, on the same image. Sec. E calls this out as the model's main cost.
* **DPT output is 2 channels** = radial distance + confidence, at full `H×W`.
* `rᵢ(u) ∈ ℝ⁺` is enforced "via an activation layer" — the activation is not named.
* Sec. 3.1 says the RM's shared geometric encoder extracts **class tokens `Tᵢ`**
  processed by "a pair of Transformer Ray Encoder (T-Enc) layers" — matching the
  2-block/8-head/512-wide Angular Regressor. But Tab. S3's Pixel Encoder emits
  full `N₁₄` token sequences at four depths, and the Angular Regressor's input is
  `B×18×512`. **Where 18 comes from is never explained** (see §11.4) —
  [reproduction.md §3](reproduction.md) shows what it must be.
* **σ is never used in any training loss.** It is predicted by the DPT head and
  consumed only by the global-alignment objective (Eq. 12) and the consensus
  averaging. Eq. 8 has no confidence term.

---

## 3. Equations

| Eq. | Content |
|---|---|
| **1** | `X^{i,i}(u) = dᵢ(u) · rᵢ(u)` — local pointmap = unit ray × radial distance |
| — | `X^{i,j}(u) = Pⱼ Pᵢ⁻¹ X^{i,i}(u)` — frame change (Sec. 3, Notations) |
| **2** | `dᵢ(u) = Σ_{l=1}^{L} Σ_{m=−l}^{l} c^i_{l,m} Y_l^m(ψ(u)) / ‖ Σ Σ c^i_{l,m} Y_l^m(ψ(u)) ‖₂` |
| **3** | `G₁ⁱ = DecoderBlock₁ⁱ(G₁^{i−1}, G₂^{i−1})`, `G₂ⁱ = DecoderBlock₂ⁱ(G₂^{i−1}, G₁^{i−1})` |
| **4** | `X^{2,1}(u) = R_{2→1} X^{2,2}(u) + t_{2→1}`, with `t_{2→1} = s · t̂_{2→1}`, `t̂ ∈ S²`, `s ∈ ℝ⁺` |
| **5** | `L_AA^α(θ̂, θ*) = Σ_{j: θ̂ⱼ<θ*ⱼ} α\|θ̂ⱼ−θ*ⱼ\| + Σ_{j: θ̂ⱼ≥θ*ⱼ} (1−α)\|θ̂ⱼ−θ*ⱼ\|` |
| **6** | `L_A = β · L_AA^{0.7}(θ̂, θ*) + (1−β) · L_AA^{0.5}(φ̂, φ*)` |
| **7** | `η_v = mean_{u∈D^v} ‖X^{v,v}(u)‖₂`, `η̄_v = mean_{u∈D^v} ‖X̄^{v,v}(u)‖₂` |
| **8** | `L_regr = Σ_{v∈{1,2}} Σ_{u∈D^v} ‖ X^{v,v}(u)/η_v − X̄^{v,v}(u)/η̄_v ‖₂²` |
| **9** | `L_rot = arccos( (Tr(R_{2→1}ᵀ R̄_{2→1}) − 1) / 2 )` — geodesic on SO(3), radians |
| **10** | `L_trans = ‖ t_{2→1} − t*_{2→1} ‖₂²`, with `t*_{2→1} = s · t̄_{2→1}`, `s` **detached** |
| — | `L_pose = λ (L_rot + L_trans)` |
| **11** | `L_total = λ_A L_A + λ_regr L_regr + λ_pose L_pose` |
| **12 / S3** | `min_{Pᵢ,sᵢ} Σ_{(i,j)∈E_pruned} Σ_u σ_{i,j}(u) ‖ Pᵢ(sᵢ xᵢ(u)) − Pⱼ(sⱼ xⱼ(u)) ‖₂²` |
| **S1** | `ΔR = R_{ji} R_{ij}`, `θ_rot = arccos((tr(ΔR)−1)/2)·180/π` |
| **S2** | `t_exp = −R_{ij}ᵀ t_{ij}`, `θ_tra = arccos( (t_{ji}·t_exp) / (‖t_{ji}‖₂‖t_exp‖₂) )·180/π` |
| **S4** | `pᵢ = d_{eff,i} rᵢ = exp(log dᵢ + log sᵢ) · rᵢ` — rays `rᵢ` **fixed** |
| **S5** | `X̄^world = R̄ᵢ X̄^{i,i} + t̄ᵢ`, `X̄^{i,j} = R̄_{j←i} X̄^{i,i} + t̄_{j←i}`, with `R̄_{j←i} = R̄ⱼ R̄ᵢᵀ`, `t̄_{j←i} = t̄ⱼ − R̄_{j←i} t̄ᵢ` |

Notes that bite:

* **Eq. 2's sum starts at `l = 1`** — the DC (`l = 0`) term is dropped. With
  `L = 3` (Tab. S3, "deg ≤ 3") that is `3 + 5 + 7 = 15` coefficients, and `c_{l,m}`
  is **bold** (a 3-vector), i.e. 45 scalars. Tab. S3 says the regressor outputs
  **10** numbers total, intrinsics included. Irreconcilable as printed — see §11.4.
* **`ψ: ℝ² ↦ S²`** maps pixel coordinates to `(θ, φ)`. Eq. 2 evaluates the SH basis
  at `ψ(u)`, i.e. the *nominal* spherical coordinate of the pixel; the learned
  coefficients bend that nominal field into the true ray field. `L` is the
  "bandwidth parameter [that] determines the high-frequency detail capturing
  capacity of the reconstructed camera manifold."
* **Eq. 5's asymmetry is on the polar angle only.** `α = 0.7` on `θ` means
  underestimation (`θ̂ < θ*`, a ray pulled *inward* toward the optical axis) is
  penalised 0.7 vs 0.3 for overestimation — the anti-"inward collapse" term, since
  narrow-FoV pinhole images dominate the data. **`α = 0.5` on `φ` makes the
  azimuth loss a plain (half-weight) L1.**
* **Eq. 5 and Eq. 8 are sums, not means** — no `1/|D^v|`, no `1/N_pixels`. As
  written the loss magnitude scales with resolution and valid-pixel count. Eq. 8 is
  called an "MSE formulation" in the prose but has no mean.
* **Eq. 8 normalises each view by its own mean point norm** (Eq. 7), predicted by
  `η_v` and ground truth by `η̄_v` — the scale-invariant DUSt3R-style trick, applied
  **per view in its own local frame**, not to the pair jointly.
* **The `s` in Eq. 10 is not a network output.** Sec. 3.2: "we resolve the
  translation ambiguity by computing scale factor `s` from the predicted
  pointmaps… derived from the ratio of the predicted pointmap magnitudes to the
  ground-truth magnitudes and is detached from the gradient flow to serve as a
  static target", i.e. `s ≈ η/η̄` in Eq. 7's notation. Reading Eq. 4 and Eq. 10
  together with the *same* `s` gives
  `L_trans = s² ‖ t̂_{2→1} − t̄_{2→1} ‖₂²` —
  **a direction-only loss on the pose head, weighted by the squared scene scale.**
  That is the only self-consistent reading, but the paper never says it; it claims
  Eq. 10 supervises "both the direction and the magnitude". See §11.8.
* **Eq. 9 has no `clamp` on the arccos argument.** Standard numerical caveat; the
  paper does not mention it.
* **Eq. 12 minimises over `{Pᵢ}` and `{sᵢ}` with `xᵢ(u) = Rᵢ(u) Dᵢ(u)` frozen** —
  but Eq. S4 also optimises `log dᵢ`. Contradiction, see §11.5.
* **Eq. S5's two lines use opposite pose conventions.** See §11.3 — this one
  changes code.

---

## 4. Training objectives, in words

| Loss | Supervises | Form |
|---|---|---|
| `L_A` (Eq. 5–6) | ray field `dᵢ` | asymmetric quantile regression on `(θ, φ)`, `α = 0.7 / 0.5` |
| `L_regr` (Eq. 7–8) | local pointmaps `X^{v,v}` | scale-normalised squared L2, per view, in its own frame |
| `L_pose` (Eq. 9–10) | `R_{2→1}`, `t̂_{2→1}` | geodesic rotation + scale-anchored translation |
| `L_total` (Eq. 11) | — | `λ_A L_A + λ_regr L_regr + λ_pose L_pose` |

**No loss weight is ever given a number** — `β`, `λ`, `λ_A`, `λ_regr`, `λ_pose` are
all unspecified (§10.1–2). There is no confidence-weighted / DUSt3R-style
`σ·L − α log σ` term anywhere in training.

---

## 5. Ray-Aware Global Alignment (Sec. 3.3 + App. C)

Motivation: "conventional global alignment techniques like DUSt3R implicitly or
explicitly assume pinhole projection", where pixel distances correspond linearly
to 3D distances. Ray-Aware lifts the optimisation into a purely ray-consistent 3D
space.

### 5.1 Scene-graph pruning

Exhaustive directed graph `G(V, E)`; each pair `(i, j)` gives **two directed
edges** `e_ij`, `e_ji`, each from its own CAM3R forward pass.

1. **Symmetric pose consistency** (cycle-consistency heuristic, cites Zach et al.
   [39]). A valid prediction satisfies `R_ji = R_ijᵀ` and `t_ji = −R_ijᵀ t_ij`.
   Measure `θ_rot` (Eq. S1) and `θ_tra` (Eq. S2) in degrees; discard the edge if
   `θ_rot > τ_rot` **or** `θ_tra > τ_tra`. **Neither threshold is given a value.**
2. **Geometric overlap via 3D MNN.** Transform view `j`'s pointmap into frame `i`
   with `P_ij`, build **KD-trees on both point sets**, extract mutual nearest
   neighbours; let `n_e` be the match count. **"Rather than using a fixed
   threshold, we apply an adaptive quantile gate: edges whose `n_e` falls below the
   20th percentile of the scene-wide match distribution are removed."** Then
   **strict symmetry**: if `e_ij` passes but `e_ji` fails, **both** are discarded.
3. **Largest connected component.** Keep only the largest component so global
   optimisation runs on one consistent camera graph.

The main paper (Sec. 3.3) calls this a **"two-stage"** protocol and states the
overlap gate as a fixed **"< 20 % of the pixel count"**. The supplement calls it
three-stage and explicitly *adaptive*. See §11.2 — this is a real discrepancy, not
a rounding of words.

Fig. 4 gives the three qualitative pruning cases: a rejected non-overlapping pair,
a retained pair with dense 3D correspondences, and a rejected **doppelgänger**
(visually identical computer monitors yielding inconsistent relative geometry).

### 5.2 Global consensus initialisation

* **Consensus ray field `Dᵢ`** — for each image, a **confidence-weighted average of
  all normalised rays predicted across its valid incident edges** (the RM runs once
  per edge, so an image has one ray prediction per incident edge).
* **Global radial field `Rᵢ`, in three steps** — (1) align pairwise radial
  distances *along the consensus rays*; (2) resolve relative scale by **robust
  median-based alignment**; (3) fuse by **confidence-weighted averaging**.
* Together they define the frozen per-pixel prior `xᵢ(u) = Rᵢ(u) Dᵢ(u)`.
* **Anchor (Sec. C.2)** — the anchor image `a*` is the **highest-degree node in the
  pruned graph**; its pose is fixed to `T_{a*} = I₄ₓ₄` as the global reference.

### 5.3 Optimisation

* Variables: camera poses `Tᵢ`, per-image `log sᵢ`, and **`log dᵢ`** (log-depths).
  **Unit ray fields `rᵢ` from the Ray Module remain fixed** (Eq. S4).
* *"Freezing the ray directions is critical. Standard point cloud alignment
  optimizes raw (X, Y, Z) coordinates with a pinhole assumption, which can
  implicitly distort the camera projection geometry. By constraining points to
  move only along their optical rays `rᵢ`, the optimizer preserves the
  camera-agnostic ray manifold while refining depth, pose, and scale."*
* **AdamW with a cosine learning-rate schedule.** `σᵢ` down-weights unreliable
  observations.
* **Multi-stage alternating scheme** (cites ADMM [21]): fix scales → optimise
  poses; fix poses → optimise scales; repeat for a fixed number of iterations to
  stabilise the trajectory; **finally a joint optimisation** of poses and scales.
* **No learning rate, iteration count, or stage length is given** (§10.5).

---

## 6. Hyperparameters

### Sec. 4.2 (main paper)

| Item | Value |
|---|---|
| Curriculum | phase 1 **homogeneous pairs only** → `CAM3R-homo`; phase 2 adds **heterogeneous pairs** → `CAM3R` |
| Optimiser | AdamW |
| Initial LR | **5 × 10⁻⁵**, linear warmup + cosine decay |
| Resolution | **512 px on the long edge** |
| Hardware | **4 × NVIDIA H200** |
| Sampling | **balanced dataset sampling** across modalities |
| Init | RM ViT ← **UniK3D**; CVM ViT ← **DUSt3R**; everything else from scratch |

### Tab. S2 (supplement) — full optimisation table

| Hyperparameter | Value |
|---|---|
| Batch size (per GPU) | 4 |
| Gradient accumulation | 2 steps |
| **Effective batch size** | **32** (4 GPUs × 4 × 2) |
| Optimiser | AdamW, **β₁ = 0.9, β₂ = 0.95** |
| **Base LR (blr)** | **1.5 × 10⁻⁴** |
| LR scaling | `lr = blr × (effective_batch_size / 256)` |
| Minimum LR | **1 × 10⁻⁶** (cosine floor) |
| Schedule | **linear warmup (10 epochs)** + half-cycle cosine decay |
| Weight decay | **0.05**, zero on **bias and LayerNorm** |
| **Total epochs** | **300 – 500** |

> ⚠️ **The two learning rates do not agree.** Tab. S2's rule gives
> `1.5e-4 × 32/256 = 1.875 × 10⁻⁵`, while Sec. 4.2 says `5 × 10⁻⁵`. Three
> mutually inconsistent numbers (blr, derived lr, quoted lr). See §11.1.

---

## 7. Datasets (Sec. 4.1, App. D)

**All dataset names in the paper mean the augmented, multi-modal versions.**

| Dataset | Native | Augmentation |
|---|---|---|
| **2D3DS** [1] | panorama, indoor cluttered | synthesise fisheye **and** perspective |
| **360Loc** [12] | panorama, large indoor/outdoor | synthesise fisheye **and** perspective |
| **ADT** [22] | **egocentric fisheye**, indoor | generate synthetic **perspective** counterparts |
| **MegaDepth** [17] | pinhole, internet photos | **none** |
| **CO3Dv2** [25] | pinhole objects | **zero-shot only, never trained**; even frames pinhole, odd frames warped to fisheye |

### Tab. S4 — splits

| Dataset | Training split | Testing split |
|---|---|---|
| 2D3DS | Areas 1, 2, 3, 4 | Areas 5a, 5b, 6 |
| **ADT** | **Apartment: ∼multi-user** | **Apartment: multi-user, Lite** |
| MegaDepth | 0000–4541 (excl. 0015, 0022) | 0015, 0022, 5000–5018 |
| 360Loc | Hall, Piatrium | Atrium, Concourse |

> The ADT row's `∼` is almost certainly a rendered `¬` — "**not** multi-user" for
> training, multi-user + Lite for test. The paper never expands it. This is the
> single most load-bearing ambiguity for anyone reproducing the ADT column (§10.9).

### Sec. D.1 — training-pair curation

| Dataset | Homogeneous pairs | Heterogeneous pairs |
|---|---|---|
| **2D3DS** | pano–pano, baseline `0.1 m ≤ ‖cᵢ−cⱼ‖₂ ≤ 2.2 m`, **top-K = 5** nearest valid neighbours per anchor | pano–pinhole / pano–fisheye: from valid homogeneous neighbours, **pinhole frames at the neighbouring panorama's location** are the targets; highest-overlap frames selected; **synthetic fisheye via equidistant projection** |
| **360Loc** | same logic, baseline widened to **`1.5 m ≤ d ≤ 10.0 m`** (expansive outdoor areas / massive atria) | same logic as 2D3DS |
| **ADT** | **fisheye–fisheye**, baseline **`0.35 m ≤ b ≤ 1.75 m`** **and** relative viewing angle **`25° ≤ θ ≤ 65°`** — "tailored to human motion dynamics" of head-mounted egocentric capture | **fisheye–pinhole**, by "extracting **simultaneous** fisheye and pinhole renders **per frame** using the Aria project toolkit" |
| **MegaDepth** | pairs taken **directly from DUSt3R's precomputed SfM/MVS metadata**; all pinhole–pinhole, **no synthetic heterogeneity injected** | — |
| **CO3Dv2** | — | even-indexed frames pinhole, odd-indexed synthetically warped to fisheye → sequences are inherently heterogeneous |

> ⚠️ **ADT's heterogeneous pairs are same-frame renders.** "Simultaneous … per
> frame" reads as zero baseline, `P_{2→1} ≈ I`, which is a degenerate relative-pose
> target. Whether the pinhole render is then paired *across* the 0.35–1.75 m window
> instead is not stated (§10.10).

### Sec. D.3 — ground-truth rays and pointmaps

GT pointmaps are **explicitly factorised** into a unit ray `dᵢ(u)` and a radial
distance `rᵢ(u)`. Only pixels with `rᵢ > 0` enter the validity mask `V(u)`.

* **Equirectangular panoramas** — deterministic from pixel coordinates, using the
  **pixel-centre convention** `(u,v) = ((u_idx + 0.5)/W, (v_idx + 0.5)/H)`, then
  longitude `λ = (u − 0.5)·2π` and latitude `φ = (0.5 − v)·π`.
* **Pinhole** — inverse projection through `K`. Perspective datasets supply
  **Z-depth** `D_z(u)` along the optical axis; this is **converted to radial
  distance** "to maintain spatial consistency with our spherical formulation".
* **Fisheye** — per-pixel unprojection through **the exact sensor calibration
  routines**, accounting for radial distortion. (No parametric model is named for
  the GT path; "equidistant projection" is only used for *synthesising* fisheye.)

Frame changes for relative supervision follow Eq. S5 — but read §11.3 first.

---

## 8. Evaluation protocol and metrics

**Two-view (Sec. 4.3).** Recover `R ∈ SO(3)` and `t ∈ S²` mapping view 2's frame
into view 1's. Evaluated on the **held-out test splits** of Tab. S4 using
**the exact same pairing logic, baseline thresholds and overlap constraints as
training** (Sec. D.2), restricted to test splits; plus zero-shot CO3Dv2.

* **RRA@15 / RTA@15** — percentage of pairs with angular error below **15°**.
* Baselines: DUSt3R [36], MASt3R [16], Pow3R [13], VGGT [35], π³ [37].

**Multi-view (Sec. 4.4).** Exhaustive pairwise inference → two-stage pruning →
Ray-Aware Global Alignment. Group construction per dataset:

* **MegaDepth** — groups formed around **hub images** with the highest number of
  co-visible connections in the SfM graph; the hub's **first-order neighbourhood**
  ("first circle of matches") is the evaluation set.
* **2D3DS & 360Loc** — **all heterogeneous images** in the test-split
  subscene/scene.
* **ADT & CO3Dv2** — the test-split video sequences loaded with **alternating
  pinhole–fisheye frames**.

* **RRA@30 / RTA@30**, **mAA@30**, and **ATE** = RMSE after **Umeyama alignment**
  (to absorb global scale and pose). Baselines limited to **VGGT and π³** —
  DUSt3R/MASt3R/Pow3R were dropped after their two-view collapse on heterogeneous
  optics.

---

## 9. Results

### Tab. 1 — two-view relative pose, Accuracy@15° (↑)

| Model | 2D3DS RRA / RTA | MegaDepth | CO3Dv2 (zero-shot) | 360Loc | **ADT** |
|---|---|---|---|---|---|
| DUSt3R [36] | 10.6 / 6.0 | 95.6 / 80.8 | 94.7 / 43.1 | **0.0 / 0.0** | 91.0 / 63.6 |
| MASt3R [16] | 18.3 / 9.3 | 69.7 / 56.4 | 98.4 / 33.4 | 39.8 / 5.3 | 96.6 / 63.5 |
| Pow3R [13] | 7.5 / 6.0 | 96.2 / 74.2 | 95.8 / 38.3 | **0.0 / 0.0** | 96.6 / 79.2 |
| VGGT [35] | 11.8 / 11.0 | 98.0 / 88.2 | 90.9 / 29.4 | 37.8 / 11.1 | 92.7 / 82.9 |
| π³ [37] | 16.8 / 11.4 | **99.8** / 93.3 | 90.7 / 22.7 | 38.5 / 13.0 | 97.5 / 93.8 |
| CAM3R-homo | 65.4 / 56.8 | 97.2 / 92.6 | 96.1 / 66.5 | 58.3 / 54.7 | 98.2 / 93.4 |
| **CAM3R** | **97.7 / 94.3** | 96.8 / **94.2** | **97.5 / 88.2** | **96.0 / 91.0** | **99.0 / 95.0** |

Prose notes: DUSt3R and MASt3R **collapse on panoramic data**; π³ and VGGT
"benefit from explicit training on MegaDepth and VGGT additionally on ADT";
CAM3R's largest margin is CO3Dv2 translation (88.2 vs 43.1 best baseline).
**CAM3R's MegaDepth RRA (96.8) is below both CAM3R-homo (97.2) and π³ (99.8)** —
the one place the full model regresses.

### Tab. 2 — multi-view, Accuracy@30° and mAA@30 (↑)

| Model | 2D3DS RRA/RTA/mAA | MegaDepth | CO3Dv2 (zero-shot) | 360Loc | **ADT** |
|---|---|---|---|---|---|
| VGGT | 31.8 / 34.4 / 7.6 | 100.0 / 97.4 / 68.8 | 70.5 / 75.3 / 19.6 | 47.9 / 50.8 / 19.5 | **100.0** / 95.6 / 60.3 |
| π³ | 40.0 / 35.8 / 9.6 | 100.0 / 98.4 / 73.4 | 89.5 / 91.6 / 22.7 | 48.6 / 47.4 / 17.8 | **100.0 / 100.0** / 75.8 |
| CAM3R + DUSt3R GA | 72.3 / 55.1 / 38.8 | 86.5 / 72.3 / 68.5 | 69.7 / 56.3 / 46.1 | 66.7 / 59.2 / 55.5 | 78.3 / 71.1 / 70.8 |
| **CAM3R + Our GA** | **94.0 / 91.5 / 73.5** | 96.6 / 96.3 / **87.4** | **85.0 / 85.2 / 64.9** | **98.7 / 91.2 / 82.6** | 92.2 / 91.5 / **77.3** |

> **On ADT multi-view, CAM3R's RRA/RTA@30 are *below* both baselines** (92.2 / 91.5
> vs 100.0 / 100.0 for π³) while its **mAA@30 is highest** (77.3 vs 75.8). mAA
> integrates accuracy over thresholds, so CAM3R is tighter at small angles and
> loses a few pairs entirely at 30°. Anyone quoting "CAM3R wins on ADT multi-view"
> should quote mAA, not RRA/RTA.

### Tab. 3 — multi-view trajectory error, ATE RMSE (↓)

| Model | 2D3DS | MegaDepth | CO3Dv2 (zero-shot) | 360Loc | **ADT** |
|---|---|---|---|---|---|
| VGGT | 3.8 | 0.7 | 1.3 | 6.3 | 0.5 |
| π³ | 2.9 | **0.6** | **0.7** | 5.8 | **0.4** |
| CAM3R + DUSt3R BA | 2.4 | 1.2 | 1.6 | 4.5 | 0.6 |
| **CAM3R + Our BA** | **1.8** | 0.8 | 1.1 | **2.7** | **0.4** |

(Tab. 3 writes "BA" where Tab. 2 writes "GA" for the same two rows — §11.6.)
**On ADT, CAM3R ties π³ at 0.4**; on MegaDepth and CO3Dv2 it is behind π³.

### Tab. S1 — architecture-stage validation, Accuracy@15° (↑)

| Model iteration | 2D3DS | MegaDepth | CO3Dv2 | 360Loc | ADT |
|---|---|---|---|---|---|
| Vanilla DUSt3R | 10.6 / 6.0 | 95.6 / 80.8 | 94.7 / 43.1 | 0.0 / 0.0 | 91.0 / 63.6 |
| **Fine-tuned DUSt3R (Stage 1)** | 17.8 / 10.9 | 94.2 / 72.5 | 94.3 / 52.7 | **13.0 / 9.2** | 87.5 / 65.4 |
| CAM3R (final, Stage 3) | 97.7 / 94.3 | 96.8 / 94.2 | 97.5 / 88.2 | 96.0 / 91.0 | 99.0 / 95.0 |

The load-bearing row is the middle one: **naive fine-tuning on the distorted
mixture makes MegaDepth and ADT *worse* than vanilla DUSt3R** (95.6→94.2,
91.0→87.5 RRA) and barely moves 360Loc off zero. That is the paper's empirical
case for decoupling.

---

## 10. ⚠️ What the paper does not specify

Real degrees of freedom left open by the text.

1. **All loss weights**: `λ_A`, `λ_regr`, `λ_pose` (Eq. 11) and `λ` in
   `L_pose = λ(L_rot + L_trans)`. No values, no ratio, no ablation.
2. **`β` in Eq. 6** — the blend between the polar and azimuth angular losses.
3. **`L`, the max SH degree in Eq. 2.** Only Tab. S3's "deg ≤ 3" constrains it, and
   that conflicts with the `B×10` output width (§11.4).
4. **`τ_rot` and `τ_tra`** — the symmetric-pose-consistency thresholds.
5. **Everything about the global-alignment optimiser except "AdamW + cosine"**:
   learning rate, total iterations, iterations per alternating stage, number of
   alternating cycles, convergence criterion.
6. **Synthetic fisheye parameters.** "Equidistant projection" is the only
   statement — no focal / FoV range, no sampling distribution, no output
   resolution, no principal-point jitter. Same for the synthetic **perspective**
   renders (FoV, how many per panorama, orientation sampling).
7. **The activation enforcing `r > 0`.**
8. **Curriculum split**: how many of the 300–500 epochs are phase 1 vs phase 2, and
   whether phase 2 keeps homogeneous pairs in the mix.
9. **The ADT training split.** "Apartment: ∼multi-user" is never expanded, and
   "Lite" is never defined against ADT's own release naming.
10. **Whether ADT heterogeneous pairs have any baseline** — "simultaneous …
    per frame" implies none (see §7).
11. **Which UniK3D and DUSt3R checkpoints** (variant, training resolution) seed the
    two backbones.
12. **Resize mechanics.** "512 px on the long edge" with a patch-14 encoder and a
    patch-16 encoder on the same image — aspect handling, padding/cropping, and how
    both grids are made divisible, are not described.
13. **What `σ` is trained by.** It appears in no loss; the paper never says whether
    it is supervised, self-supervised, or free.
14. **How consensus rays are averaged.** "Confidence-weighted average of all
    normalized rays" — but whether the average is re-normalised to `S²`, and which
    confidence weights a *ray* (σ is defined for radial distance) is not said.
15. **`|V|` per multi-view group** — no group sizes, no counts of eval pairs or
    groups for any dataset.
16. **The `18` tokens** in Tab. S3's Angular Regressor input.
17. **`mAA@30` convention** — which of the several mAA definitions in the literature.
18. **Balanced sampling ratios** across the four training datasets.
19. **Runtime / memory / parameter count.** Never reported anywhere, despite Sec. E
    naming architectural overhead and `O(N²)` pairwise cost as the two limitations.
20. **No ablation on** SH degree, the asymmetric quantile `α`, the pruning stages
    individually, or the loss terms. The only ablations are heterogeneous-vs-homo
    training (Tab. 1) and Our-GA-vs-DUSt3R-GA (Tab. 2/3).

---

## 11. Errata and internal inconsistencies (verified against the PDF)

1. **Learning rate: three incompatible numbers.** Sec. 4.2 states `5 × 10⁻⁵`.
   Tab. S2 states `blr = 1.5 × 10⁻⁴` with `lr = blr × effective_batch/256`, which
   at the table's own effective batch of 32 gives `1.875 × 10⁻⁵`. No reading
   reconciles them.
2. **The MNN overlap gate is described two different ways.** Main paper Sec. 3.3:
   "Edges with insufficient geometric overlap (e.g. **< 20 % of the pixel count**)
   are discarded" — a *fixed fraction of pixels*. Supplement C.1: "**Rather than
   using a fixed threshold**, we apply an adaptive quantile gate: edges whose `n_e`
   falls below the **20th percentile of the scene-wide match distribution** are
   removed." Both use "20", but one is a pixel fraction and the other a rank
   statistic over edges; they select different edges and the supplement explicitly
   disowns the main paper's version.
3. **Eq. S5 mixes pose conventions.** The prose says `Pᵢ = (Rᵢ, tᵢ)` is the
   **camera-to-world** pose, and line 1 (`X̄^world = R̄ᵢ X̄^{i,i} + t̄ᵢ`) is correct
   for camera-to-world. But line 2's composition, `R̄_{j←i} = R̄ⱼ R̄ᵢᵀ` and
   `t̄_{j←i} = t̄ⱼ − R̄_{j←i} t̄ᵢ`, is the **world-to-camera** composition. For
   camera-to-world the correct relative transform is
   `R_{j←i} = R̄ⱼᵀ R̄ᵢ`, `t_{j←i} = R̄ⱼᵀ(t̄ᵢ − t̄ⱼ)`. Exactly one of the two lines can
   hold; line 2 is internally self-consistent and is what an implementation would
   use, so the safe reading is that `(R̄ᵢ, t̄ᵢ)` are **world-to-camera** and line 1
   is the slip.
4. **Tab. S3's Angular Regressor output cannot hold the Eq. 2 coefficients.**
   `B × 10` for "intrinsics + SH coeffs.", against Eq. 2's `l = 1…3` with
   vector-valued `c_{l,m}` = 15 coefficient vectors (45 scalars). Relatedly, Eq. 2
   has **no intrinsics or FoV term at all**, yet Tab. S3's Ray Generation stage is
   "**FoV projection** + SH basis" taking "Intrinsics + SH + pixel grid". The ray
   parameterisation actually implemented is therefore underdetermined by the text —
   Eq. 2 and Tab. S3 describe two different things.
5. **Eq. 12 vs Eq. S4 disagree on what is optimised.** Eq. 12 and its prose:
   "**Keeping the fixed geometric priors**", optimising only `{Pᵢ}` and `{sᵢ}`,
   with `xᵢ(u) = Rᵢ(u) Dᵢ(u)` frozen. Eq. S4/Sec. C.3: "The optimization variables
   include camera poses `Tᵢ`, per-image scales `log sᵢ`, **and log-depth values
   `log dᵢ`**", only the *rays* fixed. Radial distance is frozen in one and free in
   the other.
6. **"GA" in Tab. 2, "BA" in Tab. 3** for the identical two method rows
   (`CAM3R + DUSt3R` and `CAM3R + Our`). Cosmetic.
7. **Umeyama alignment is cited to [14] = Kabsch (1976)**, which solves the
   rotation-only Procrustes problem and carries no scale. ATE after "Umeyama
   alignment" needs the similarity variant (Umeyama 1991), which is not in the
   bibliography.
8. **`s` is overloaded three ways**, all in `ℝ⁺` and all called `s`: (i) Sec. 3.1's
   relative scale making `t_{2→1} = s t̂_{2→1}`; (ii) Sec. 3.2's *detached* ratio of
   predicted to ground-truth pointmap magnitudes; (iii) Eq. 12's per-image
   alignment scale `sᵢ`. Under the only consistent reading of (i) = (ii), Eq. 10
   collapses to an `s²`-weighted **direction** loss, contradicting the prose claim
   that it supervises "both the direction and the magnitude".
9. **`Rᵢ` is overloaded**: rotation matrix throughout, *and* the global
   **radial-distance field** `Rᵢ(u)` in Sec. 3.3 / Eq. 12. Likewise `Dᵢ` is the
   consensus **ray** field in Sec. 3.3 but `D_z` is Z-depth in Sec. D.3 and `D^v`
   is the valid-pixel set in Eq. 7–8.
10. **Pruning is "two-stage" in the main paper, three-stage in the supplement** —
    the supplement adds largest-connected-component extraction as step 3.
11. **Sec. C.2 is titled "Anchor-Scale Initialization" but describes no scale
    initialisation** — only anchor selection and pose fixing. The scale
    initialisation (median-based alignment + confidence-weighted fusion) is in the
    main paper's Sec. 3.3 instead.
12. **Tab. S3's Relative Pose Head outputs `B×4×4 ∈ SE(3)`** — a full rigid
    transform including a metric translation — while Sec. 3.1 says the head
    regresses `R_{2→1}` and a **unit** direction `t̂_{2→1} ∈ S²` with the magnitude
    supplied separately by `s`.
13. **"CAM3R remains highly competitive on unseen ADT scenes" (Sec. 4.3)** while
    Tab. S4 gives ADT a training split. "Unseen" can only mean held-out
    scenes/sequences within a trained-on dataset, not an unseen dataset — worth
    keeping straight when comparing against VGGT, which the same sentence flags as
    trained on ADT.
14. Cosmetic: reference [21] (ADMM) lists "Neal, P., Eric, C., Borja, P.,
    Jonathan, E." — Boyd, Parikh, Chu, Peleato and Eckstein with given and family
    names transposed.
15. **The consensus ray field is a no-op on the paper's own architecture.**
    Sec. 3.3 computes `Dᵢ` as "a confidence-weighted average of all normalized
    rays predicted across its valid incident edges". But the Ray Module is
    strictly *per-image* (Sec. 3.1: "For each input image `Iᵢ`, a shared
    geometric encoder…"; Fig. 2 draws it as a **Shared Ray Module** outside the
    two-view path), and every image is processed at the same 512 px. So the `k`
    predictions being averaged for an image on `k` incident edges are `k`
    *identical* vectors, and the average returns the input. The step only does
    work on the **radial** field — which comes from the Cross-view Module, and
    therefore genuinely differs per edge. Averaging identical rays is harmless,
    but Sec. 3.3 presents ray consensus as doing something, and on the
    architecture as described it cannot.

---

## 12. Limitations the authors state (Sec. E)

1. **Architectural overhead.** Two separate ViT backbones (RM and CVM) → high
   memory. Future work: parameter sharing or distillation into a unified encoder
   (cites DUNE [26]).
2. **Pairwise scalability.** `O(N²)` pairwise inferences to build the scene graph.
   Pruning cuts the *alignment* cost but not the inference cost. Future work:
   extend the ray formulation to multi-view transformers (VGGT [35], π³ [37]).

Sec. 5 (conclusion) adds three further directions: unify the geometric and
cross-view encoders for real-time use; explore **ray-based positional encoding**
for high-frequency detail; and build pruning that handles visual ambiguity
(extreme viewpoint / appearance change) better than the current heuristics.

---

## 13. Zero-shot evidence (Figs. S1–S3, qualitative only)

* **Fig. S1** — Matterport [5], **FIORD** [11], CO3Dv2 [25], with a heterogeneous
  mix of panoramic/fisheye/pinhole captures per scene. Caveat the authors state
  themselves: "the Ray Module leverages prior exposure to Matterport's optical
  manifold **via UniK3D initialization**"; only the **Cross-View Module** is truly
  zero-shot there. On FIORD, CAM3R "unwraps extreme 2D fisheye aberrations,
  strictly preserving rectilinear structures".
* **Fig. S2** — in-domain multi-view on 360Loc, ADT, MegaDepth test splits. For
  **ADT** specifically: "enclosed room scenes with dense, short-baseline egocentric
  fisheye captures, our rigorous scene-graph pruning yield a coherent pointcloud"
  — i.e. pruning is what carries ADT, not raw pair quality.
* **Fig. S3** — raw two-view output *before* global alignment, with 3D–3D MNN
  matches overlaid, colour-coded by predicted distance. Covers
  panorama–pinhole and panorama–fisheye heterogeneous pairs.

**There are no quantitative zero-shot numbers for Matterport or FIORD** — CO3Dv2
is the only zero-shot dataset in any table.

---

## 14. Reference map (numbers used above)

| # | Work | Role in CAM3R |
|---|---|---|
| [1] | 2D3DS | training + test, panorama |
| [2] | Bengio et al., curriculum learning | the two-phase curriculum |
| [8] | ViT | both backbones |
| [11] | FIORD | qualitative zero-shot (Fig. S1) |
| [12] | 360Loc | training + test, panorama |
| [13] | Pow3R | baseline |
| [14] | **Kabsch** | cited for "Umeyama alignment" in ATE — see §11.7 |
| [15] | Kannala–Brandt | cited only as prior art for wide-FoV calibration |
| [16] | MASt3R | baseline |
| [17] | MegaDepth | training + test, pinhole |
| [19] | AdamW | training and global-alignment optimiser |
| [21] | ADMM (Boyd et al.) | the alternating optimisation scheme |
| [22] | **ADT** | training + test, egocentric fisheye |
| [23] | **UniK3D** | **Ray Module init**; source of the asymmetric angular loss and the SH ray representation |
| [24] | DPT | the radial-distance / confidence head |
| [25] | CO3Dv2 | zero-shot only |
| [26] | DUNE | proposed remedy for the two-backbone overhead |
| [35] | VGGT | baseline (two-view + multi-view) |
| [36] | **DUSt3R** | **CVM init**; MegaDepth pair metadata; the global-alignment baseline |
| [37] | π³ | baseline (two-view + multi-view) |
| [39] | Zach et al., loop constraints | the cycle-consistency pruning heuristic |
