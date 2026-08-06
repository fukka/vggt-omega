# CAM3R — reproduction notes

Everything on this page is **ours**, not the paper's: the choices we made where
arXiv:2603.22631 is silent, the places we knowingly depart from it, what we have
measured, and what still has to be swept once there is a GPU and real weights.

* [PAPER.md](PAPER.md) is the paper and only the paper. If a claim is not in
  there, the paper does not make it.
* [README.md](README.md) is the user-facing guide — quick start, paper→code map,
  initialization coverage, status.

There is no official code — [nam1410/cam3r](https://github.com/nam1410/cam3r) is
the project website and its README says `CODE - TBD` — and no released weights,
so `cam3r/` is a from-paper reconstruction.

**Status: no paper number is reproduced.** Everything verified so far is
behavioural: 157 pytest + 21 smoke checks on CPU, plus end-to-end runs on the
local ADT sequence with a randomly-initialized network. Reproducing 99.0 / 95.0
needs the four corpora, the DUSt3R/UniK3D checkpoints, and 300–500 epochs on
4×H200.

---

## 1. Open knobs — what to sweep first

These are the paper's silences that actually change a number. Ordered by how
much we expect them to matter.

| # | Paper's gap | Our default | Flag | Risk if wrong |
|---|---|---|---|---|
| 1 | **`λ_A`, `λ_regr`, `λ_pose`** (Eq. 11) — no values, no ratio, no ablation | all **1.0** | `--lambda-a / --lambda-regr / --lambda-pose` | **Highest.** The three terms are in different units — radians, a scale-free squared distance, and radians+squared-direction. There is no reason 1:1:1 balances them, and nothing in the paper pins it. Sweep this before anything else |
| 2 | **Loss reduction** — Eq. 5 and Eq. 8 are written as **sums** with no `1/\|D^v\|`, but the prose calls Eq. 8 an "MSE formulation" | `mean` | `--loss-reduction {mean,sum}` | **High, and coupled to #1.** Under `sum` the loss scales with resolution and with how much of the frame is inside the lens cone, so a 512 px fisheye at a 55° cone weights `L_regr` differently from a 512 px panorama — i.e. the λ balance becomes dataset-dependent. We default to `mean` so #1 is sweepable at all |
| 3 | **`β`** (Eq. 6) — polar/azimuth blend | **0.5** | `--beta` | moderate. θ is the accuracy-relevant angle; φ is degenerate on-axis and wrapped |
| 4 | **Learning rate** — Sec. 4.2 says `5e-5`, Tab. S2 says `blr 1.5e-4` with `lr = blr × batch/256`, which gives `1.875e-5`. Three mutually inconsistent numbers | **5e-5** (main text) | `--lr` | moderate. A 2.7× lr spread across a 300–500 epoch run |
| 5 | **`τ_rot`, `τ_tra`** — named in supp. C.1, never valued | **15° / 30°** | `--tau-rot / --tau-tra` | **Measured to matter.** See §4 |
| 6 | **Curriculum split** — how many of the 300–500 epochs are phase 1 vs 2, and whether phase 2 keeps homogeneous pairs | `--phase both`, 50/50 mix | `--phase`, `--hetero-ratio` | high, but unsweepable until a second corpus lands (§2.1) |
| 7 | **`r > 0` activation** — "via an activation layer", never named | **softplus + 1e-4** | — | low. ReLU would zero the gradient on the wrong side; exp would overflow early |
| 8 | **mAA@30 convention** — several exist in the literature | RelPose/PoseDiffusion: mean over integer thresholds 1…30 of joint rot+trans accuracy | — | moderate for *comparability*, none for ranking |
| 9 | **Synthetic fisheye parameters** — "equidistant projection" is the only statement | FoV ∈ U(140°, 200°), `k1 ∈ U(-0.05, 0.05)`, `k2 ∈ U(-0.02, 0.02)` | `data.py::_random_kb4` | unknown; untested, no panoramic corpus wired |
| 10 | **Which UniK3D / DUSt3R checkpoints** | `lpiccinelli/unik3d-vitl`, `DUSt3R_ViTLarge_BaseDecoder_512_dpt` | `--unik3d / --dust3r` | low — these are the only ViT-L variants that fit Tab. S3 |
| 11 | **Eq. 12 vs Eq. S4** — Eq. 12 freezes the radial prior, Sec. C.3 lists `log dᵢ` as a free variable (PAPER.md erratum 5) | frozen (Eq. 12) | `--refine-log-depth` | moderate; both readings are implemented |
| 12 | **Pose parameterization** — paper says only `R ∈ SO(3)`, `t̂ ∈ S²` | 6D rotation (Zhou et al.), `t̂` L2-normalized | — | low |

---

## 2. Where we knowingly depart from the paper

### 2.1 Only ADT is wired, so the heterogeneous phase never runs

2D3DS, 360Loc and MegaDepth plug into `data.TwoViewSource`, but none is
implemented. A cross-lens pair has to be **one scene rendered two ways** — the
paper extracts "simultaneous fisheye and pinhole renders per frame" — because
pairing two *independent* datasets gives a pair with no defined relative pose.
So a source must declare `supports_heterogeneous`; none here does, phase 2
degenerates to phase 1, and `train.py` says so at runtime instead of pretending.

The paper's own ablation prices this at **65.4 vs 97.7 RRA@15 on 2D3DS**
(CAM3R-homo vs CAM3R, Tab. 1). It is the single largest gap between this code
and the paper's recipe.

### 2.2 Eq. 10 is a direction loss, and there is no scale head

Reading Eq. 4 and Eq. 10 with the same `s`, and `s` **detached** (Sec. 3.2:
"derived from the ratio of the predicted pointmap magnitudes to the ground-truth
magnitudes and is detached from the gradient flow"):

```
L_trans = ‖s·t̂_pred − s·t̄_gt‖²  =  s²‖t̂_pred − t̄_gt‖²
```

`s` is a constant, so this supervises **direction only**, weighted by the squared
scene scale — contradicting the prose claim that Eq. 10 supervises "both the
direction and the magnitude" (PAPER.md erratum 8). Consequently nothing in
Eq. 11 gives a scale *head* any gradient, so `PoseHead` does not have one and
`t = s·t̂` takes its magnitude from the predicted pointmap
(`model.py::pointmap_scale`).

An earlier version of this code instead split Eq. 10 into a direction term plus a
log-space metric-scale term, on the theory that `s²‖·‖²` collapses the scale.
That reasoning was wrong: it only applies if `s` is a *network output*, and the
paper says it is not. The split is gone.

### 2.3 RAGA is Adam over a closed-form seed, returning its best iterate

The paper gives the alternating schedule (poses → scales → joint, citing ADMM)
and "AdamW with a cosine learning rate schedule", and no learning rate, iteration
count, or stage length. We follow the three-phase schedule, but:

* **seed by a spanning tree** of closed-form weighted-Umeyama fits rooted at the
  anchor, not from identity. From identity, Adam discovers an 8× scale spread
  slowly and imprecisely (`test_identity_init_still_converges_but_starts_far_worse`).
* **return the best iterate, not the last.** On clean input the seed is already
  optimal — residual ~1e-29 — and a fixed-step optimizer walks away from it, so a
  last-iterate return would be strictly worse than not refining at all.

### 2.4 Smaller ones

* **DPT head** is a compact reassemble/fuse, not DUSt3R's exact head. Hooks
  `{0, 6, 9, 12}` and `D_feat = 256` do match Tab. S3.
* **Ray Module patch size (14) need not divide the image size**; inputs are
  resized to the nearest multiple rather than rejected. Tab. S3 puts a patch-14
  and a patch-16 encoder on the same image and never says how.
* **`geodesic_angle` uses `atan2`**, not Eq. 9's literal `arccos((tr−1)/2)`.
  Analytically identical; `arccos` loses precision and has a divergent gradient
  at 0 and π, and one function serves as both loss and metric.
* **Azimuth residuals are wrapped** into (−π, π]. Eq. 5 writes a plain absolute
  difference, which scores two rays 0.02 rad apart as ~2π apart across the seam.
* **Confidence supervision.** σ appears in no training loss (PAPER.md §10.13), so
  as written the confidence head takes no gradient and the σ-weighting in Eq. 12
  is meaningless. Default is the DUSt3R term `σ·L − α log σ`; `--conf-mode none`
  is the literal Eq. 8.
* **Single-process training.** `--accum` approximates the paper's 4×H200,
  batch 4/GPU, 2-step accumulation (effective 32).

---

## 3. What the reproduction resolved that the paper does not state

**The `18` in Tab. S3's `B × 18 × 512`.** Never explained in the text
(PAPER.md §10.16). It is UniK3D's latent-slot layout, and it is forced:

```
18  =  3 (pinhole intrinsics)  +  3 + 5 + 7 (SH degrees l = 1, 2, 3)
```

`model.py::AngularHead.num_params` computes exactly this. That also settles the
other half of PAPER.md erratum 4: the Pixel Encoder hands the Angular Regressor
**four read-out class tokens**, not the `{B×N₁₄×1024}×4` full sequences Tab. S3
prints, and learned per-group projections expand those four into the 18 latent
slots. So Sec. 3.1's "class tokens `Tᵢ`" is the correct reading and Tab. S3's
input column is loose notation.

The rest of erratum 4 stands: `B × 10` still cannot hold Eq. 2's coefficients,
which are 15 vector-valued `c_{l,m}` = **45 scalars**, plus intrinsics. Our head
emits 3 intrinsic logits + 45 SH scalars. We have no reading that produces 10.

**The Aria lens is comfortably inside a degree-3 expansion.** 0.055° mean /
0.32° max over the imaged cone at 96×96 — so the ray parameterization is not a
bottleneck on this lens, and `L = 3` (PAPER.md §10.3) is not a knob worth
sweeping for ADT.

**The base-grid FoV is weakly identified.** Any `hfov` from 10° to 60° fits the
Aria lens to well under a tenth of a degree, because the SH coefficients absorb
the angular scale. A network predicting a "wrong" base FoV is not thereby broken.

---

## 4. Bugs this reproduction hit, and what they cost

Each is regression-tested; the test name is the contract.

| Bug | Cost | Test |
|---|---|---|
| **`theta_max` used the fold-back turnover (62.33°) as the imaged cone (54.83°)** | Admitted a ring of dead vignette pixels the lens never illuminates. SH fit error **0.155°/2.82° → 0.055°/0.32°** once fixed — the whole tail was vignette. Now reproduces `VGGT-360-fisheye`'s mask exactly, 0 pixels different, at every resolution | `test_cameras.py` |
| **`arccos(z)` NaN'd the Ray Module's backward pass on step 1** | The on-axis pixel has `z == 1`, `x == y == 0` *exactly*, where `arccos` has infinite derivative. Invisible until the UniK3D init made rays accurate — then **1634 parameters went non-finite** on the first step | `test_geometry.py` |
| **`select_pairs` paired frames across recordings** | ADT sequences of one apartment share a world frame, so two frames from *different* sessions sat 0.5 m / 40° apart, passed the Sec. D.3 window, and showed different people and object layout | `test_adt.py` |
| **`resolve_extrinsics` only looked for `main_recording.vrs`** | The public ADT download ships `video.vrs`, so every run silently used the device frame. The real `T_device_camera` is a 13.6 mm lever arm **and a ~38° rotation** | `test_adt.py` |
| **σ was gathered with the wrong view's indices** | `conf_i` indexed by view *j*'s MNN indices. Same length, so nothing complained — it silently permuted the Eq. 12 weights | `test_confidence_is_gathered_with_each_views_own_indices` |
| **`torch.cdist`'s mm path** reported self-distances up to 1.4e-3 on identical float32 clouds | Broke MNN at a tight radius; overlap of a cloud with itself came back 0.99 instead of 1.0 | `test_overlap_ratio_...` |
| **Pruning compared clouds in different frames** | MNN between `X^{i,i}` and `X^{j,j}` is meaningless; it has to run in the common frame *i* | `test_from_pointmaps_matches_in_the_common_frame_...` |
| **Pruning keyed its verdict by `(i, j)`** | Two edges sharing an ordered pair were judged together, so a junk duplicate rode in on its twin's pass | `test_duplicate_keys_are_judged_per_edge` |
| **The base grid's principal point was `W/2`** | The centre of a `W`-wide integer grid is `(W−1)/2`. Half a pixel out, and against Sec. D.3's ERP pixel-centre convention it is a **constant `180/W`° bias** — 0.35° at 512 px — that no SH degree removes, because a latitude shift is not a rotation | `test_a_half_pixel_base_grid_offset_costs_a_constant_180_over_w` |
| **Weight decay hit bias and LayerNorm** | Tab. S2 says zero on those. One flat group decayed **185** rank≤1 tensors, including the Ray Module's LayerScale gammas — which UniK3D initializes at 1e-4, so decaying them pulls the pretrained residual gates toward zero | `test_weight_decay_skips_bias_and_layernorm` |
| **Three of the four pruning/consensus stages were missing** | See below — this is the big one | `test_alignment.py` |

### The pruning cascade was running on one stage of three

Supp. C.1 is a three-stage cascade and Sec. 3.3 adds a consensus step. Until
2026-08-06 this code had **only the overlap gate**:

1. **Symmetric pose consistency** existed but was unreachable: `eval_adt` built
   one directed edge per pair, so the reciprocal `e_ji` never existed and the
   check silently short-circuited. Both directions are now built, from **their
   own forward pass** — the Cross-view Module is asymmetric, so they are
   genuinely different predictions.
2. **Strict symmetry** ("if `e_ij` passes but `e_ji` fails, both are discarded")
   was absent; each direction was judged alone.
3. **Largest connected component** was absent. This one is not cosmetic: the
   spanning-tree seed leaves an unreachable view at identity and the objective
   has no term touching it, so a fragmented graph reported those cameras **at the
   world origin** and they entered the ATE as if placed. `eval_adt` now reports
   `views_in_component` / `views_dropped`.
4. **Consensus fields** (Sec. 3.3) were absent — per-edge pointmaps went straight
   into Eq. 12. `alignment.py::consensus_fields` now implements the confidence-
   weighted ray average and the three-step radial fusion (align along the
   consensus ray → robust median relative scale → confidence-weighted average).
   Note PAPER.md erratum 15: the *ray* half is provably a no-op on this
   architecture; the radial half is not.

Turning stage 1 on **changes the headline diagnostic**, and correctly so. On the
local ADT sequence with an untrained network, stage 1 now removes **20/20**
edges at `τ_rot = 15°`: `R_{j→i}` and `R_{i→j}` are independent hallucinations
and do not transpose into each other. Before, with stage 1 dead, 18/20 edges
survived on overlap alone and RAGA produced an ATE that looked like a result.
`eval_adt` now attributes drops per stage rather than blaming overlap for
everything.

---

## 5. What we have measured

All on the local sequence `Apartment_release_clean_seq131_M1292`, CPU, with a
**randomly initialized** network unless stated. None of it is a paper number.

* 28 frames → **44 pairs** inside the paper's Sec. D.3 window (0.35–1.75 m,
  25–65°).
* **Pose accuracy is chance level** — median rotation error ~130–157°, against
  the ~120° expected of a random rotation. As it must be.
* **Training reduces the objective on real data**: total 5.94 → 2.98 over 6 short
  epochs; angular 0.431 → 0.144, regression 1.63 → 0.40, pose 3.88 → 2.44.
* **A UniK3D-initialized Ray Module predicts the Aria fisheye to 3.12° mean,
  untrained** — cross-dataset, no CAM3R training at all. The angular prior
  transfers to a lens UniK3D did not see.
* **Initialization coverage against the real checkpoints**: Ray Module 506/506
  tensors (100%), Cross-view 1016/1128 (98%), whole model **98.6%**. Zero shape
  mismatches. Only `dec_blocks.*.norm_ctx` (CroCo has no such norm) and the pose
  head are left random.
* With `--tau-rot 180 --tau-tra 180` to force the path open, the full multi-view
  pipeline runs: 20 edges → 4 kept → largest component **3 of 13 views**, anchor
  = highest-degree node, ATE 0.427 m over the component. The `views_dropped: 10`
  is the point — those were previously scored at the origin.

---

## 6. How to read a run

Three ways to mistake a correct run for a broken one:

1. **`edges_kept: 0` with an untrained network is the cascade working.** Stage 1
   is designed to reject confident-but-inconsistent pose regressions, and an
   untrained regressor produces nothing else. Check
   `edges_after_pose_consistency` before touching `--overlap-radius`.
2. **Never quote RTA from a device-frame run.** If `extrinsics_exact` is `False`,
   no camera calibration was found and GT poses are in the *device* frame.
   Rotation is unaffected (conjugation preserves angle) but translation direction
   is rotated by the full ~38°. The run prints a warning; `extrinsics_source`
   records which path was taken.
3. **`views_dropped > 0` is information, not a bug.** It means the pruned graph
   fragmented and only the largest component was posed — the paper's own
   behaviour. A large value means pruning is too aggressive for the data, which
   is a τ / overlap-radius question.

---

## 7. What needs the GPU box

Nothing in §1–§4 needed it; all of the above is CPU-verifiable and the suite runs
in ~35 s. These do:

* Sweeping §1's knobs — every one of them needs a real training run to score.
* Any claim about Tab. 1's 99.0 / 95.0, or Tab. 2/3's multi-view numbers.
* The heterogeneous curriculum (§2.1), which needs a second corpus built first —
  that part is a `cpu` ticket.
* Confirming the initialization coverage table against freshly downloaded
  checkpoints.
