# RayTun3R — reproduction notes

Everything on this page is **ours**, not the paper's: the choices we made where
arXiv:2607.02711 is silent, the mapping from its text to this code, what we have
measured, and how to read a run without mistaking a paper-consistent result for a
failure.

* [PAPER.md](PAPER.md) is the paper and only the paper. If a claim is not in there,
  the paper does not make it.
* [README.md](README.md) is the user-facing guide — quick start, backbone support,
  the twelve interpretation decisions in full, and the measured results table.

There is no official code ("our code will be made publicly available"; nothing was
published as of 2026-08), so `raytun3r/` is a from-paper reconstruction.

**Status: no paper number is reproduced yet.** Everything verified so far is
behavioural (CPU: 46 pytest + 35 smoke checks) — see
[README § What is verified](README.md#what-is-verified-and-what-is-not).

---

## 1. Hyperparameters → our flags

Every hyperparameter the paper states is already the default in our CLI. Values
are in [PAPER.md §4](PAPER.md#4-hyperparameters-sec-43-implementation-details);
this is only the mapping.

| Paper | Our flag |
|---|---|
| Resolution 504 × 504 | `--max-size 504` |
| Adam, lr 1e-3 | `--lr 1e-3` |
| Gradient clipping, norm 1.0 | `--clip 1.0` |
| Three-frame windows | `--seq-len 3` |
| `N_r = 20`, `N_θ = 8` | `--n-radial 20 --n-angular 8` |
| 20 radial RoPE bins | `--n-rope-radial 20` |
| `w_pose, w_smooth, w_L2, w_TV = 1, 10, 2, 20` | `--w-pose / --w-smooth / --w-l2 / --w-tv` |
| 30 three-frame windows | `--windows 30` |
| Static filter < 2 px flow | `--min-flow-px 2.0` |
| UFM matcher | `--matcher ufm` |
| LoRA r=8, α=16 | `--lora-r 8 --lora-alpha 16` |
| CalTok t=4 | `--caltok-t 4` |
| Center-PH 110° | `fov_deg=110.0` |
| Multi-PH, 5 views | `n=5` |

Two implementation details worth pinning, because the paper uses different norms
in the loss and the metric and it is easy to unify them by accident:

* Eq. 8 is **L1** — [losses.py:96](losses.py#L96) `.abs().sum(-1)`.
* Eq. 16 is **L2** — [metrics.py:105](metrics.py#L105) `.norm(dim=-1)`.
* Eq. 17 as printed is not AbsRel (PAPER.md erratum 2); we implement the standard
  Eigen definition at [metrics.py:141](metrics.py#L141).

The equation-by-equation map from the paper to source files is in
[README § Paper → code map](README.md#paper--code-map).

---

## 2. How we resolved what the paper leaves open

Numbered to match [PAPER.md §10](PAPER.md#10--what-the-paper-does-not-specify).
[README § Interpretation decisions](README.md#interpretation-decisions) carries the
full reasoning for each; this is the short form plus what it costs us.

| # | Paper's gap | Our choice | Risk |
|---|---|---|---|
| 1 | Adaptation iterations | `--iters 300` (~3 min on one GPU) | **Highest.** Sec. D quotes 2–3 h per scene. Even allowing that that covers evaluation of every baseline over the full sequence, we may be training 5–10× too short. `experiments/iters_sweep.py` |
| 2 | Batch size | one window per step | low |
| 3 | GPUs per run | one GPU per scene, scenes in parallel — Sec. D's own parallel axis | none; ~10k parameters have nothing to shard, and a shared adapter across scenes would be a different method |
| 4 | Depth convention | `--convention range` (unit rays), enforced at the backbone boundary | **High.** The two readings differ by a per-pixel `1/cos θ` — 1.74× at the Aria rim, ~11× at a 170° frame corner — and no global `s` absorbs it. This was a live bug; see README decision 1 |
| 5 | How `Ω` is derived | `camera.valid_mask` from `theta_max`; the whole rectangle where the calibration is valid, not the inscribed circle | on ScanNet++ the inscribed circle would discard 47% of a 504×336 frame |
| 6 | Which dual-fisheye camera | first camera of the pair | untested |
| 7 | Multi-PH fusion | chordal SO(3) mean of per-view poses; depth from the view whose axis is closest to each ray | moderate — this is a baseline, and it currently beats us |
| 8 | `d_reproj` scale | 1-D search over closed-form candidates on the reprojection (Eq. 16 as written), not the 3D fit the text describes | moderate |
| 9 | Scenes behind the aggregates | `experiments/scannetpp_all.py` runs all of ScanNet++ and reports a standard error over scenes | aggregation mismatch is unavoidable |
| 10 | Flow source for the 2 px filter | UFM | low |
| 11 | LoRA/CalTok placement | last 12 encoder blocks, heads excluded — reproduces the paper's 147.5K exactly (`12 × (8·384 + 1152·8) = 147,456`) | low, given the count matches |
| 12 | Tab. 8's backbone | assumed DA3-Small | low |

Two more choices where the paper is not ambiguous but is *unimplementable as
written* — full reasoning in README decisions 2, 5, 6, 7:

* **Prediction-grid radial coordinate.** "Undistorting through the calibrated
  fisheye-to-pinhole map" means `tan θ`, which saturates past 90°. `--grid-mode
  auto` uses `tan` below 80° half-angle and angular radius `θ` beyond.
* **Eq. 9's `arccos` cannot reach zero numerically** — it needs its argument
  clamped, and the clamp floors the loss at ~2.8e-3 rad. `losses.py` uses the
  algebraically identical `atan2` forms; `metrics.py` keeps plain `arccos`, where
  there are no gradients.

---

## 3. Which numbers we are actually chasing

The paper's own aggregates (Tab. 1, 3, 6) are means over unnamed scene sets, so
they are not directly comparable. The named-sequence tables are:

* **Tab. 2** — π³ and VGGT on five named sequences. Our closest entry is **VGGT on
  ScanNet++ `3f15…`**: vanilla 7.21 → RayTun3R 0.93 `R°` (7.8×), with Center-PH at
  2.45. This is the one row we have run.
* **Tab. 5** — the only **per-sequence DA3-Small** numbers in the paper, from the
  AnyCalib appendix. `RayTun3R (GT)` on ScanNet++ `3f15…` is **0.40 / 2.2 / 1.7**.
  Since DA3-Small is the paper's primary backbone and `3f15…` is the scene we have
  data for, this is the tightest target available — tighter than Tab. 2 — and it is
  what ticket 003's DA3 rerun should be scored against.
* **Tab. 4a** is on KITTI-360 drive `0000` cam02 with a different protocol (train
  30 frames, evaluate the first 500), so it is not comparable to anything else.

### Backbone choice follows from Tab. 7b

Tab. 7b's RoPE-only row (19.52° vs 0.48° for the full adapter) decides what each
backbone here can show. `vggt_omega` is DINOv3-based — RoPE only, **no `pos_embed`
parameter anywhere** — so `--backbone vggt_omega` reproduces that *negative* row by
construction, not the headline. The faithful targets are `da3` (the paper's
primary) and `vggt`. Per-backbone parameter counts and hook details are in
[README § Backbone support](README.md#backbone-support).

---

## 4. Where our data disagrees with the paper

**ScanNet++ FOV.** The paper states 115° for ScanNet++ DSLR. The actual frames we
have are full-frame **~170°** fisheye: the released calibration puts the frame
corner at ~85° incidence, the corners carry real image content, and
`project ∘ unproject` round-trips there to 1.5e-5 px — so the corners are inside
the lens model, not extrapolation. This is a disagreement between the paper's text
and the data, not a loader bug. It matters because 170° is far outside anything
these backbones saw in training, and because `Ω` changes every loss and metric at
once. `experiments/fov_sweep.py` and `--max-fov 115` exist to test it.

---

## 5. How to read a run

Three ways to mistake a paper-consistent result for a broken one:

1. **A `d_reproj` loss is not a failed reproduction.** RayTun3R loses `d_reproj` to
   Center-PH or Multi-PH on 4 of 5 datasets in the paper's own Tab. 1, and
   Center-PH wins ScanNet++ depth outright in Tab. 3. The claim is about **pose**.
2. **Do not tune toward Tab. 4a.** The full model has the second-worst `R°` in its
   own component ablation; it is selected on `d_reproj`. Six of seven ablated
   variants beat it on rotation.
3. **Parameter-free corrections hurting on their own reproduces the paper.**
   Tab. 8 has them making FIORD Kitchen worse (28.09 → 39.04 `R°`). They only help
   combined with the learned residual.

---

## 6. What we have measured

Full tables, the run configuration, and the two findings are in
[README § Measured results](README.md#measured-results). The summary:

* One real run — VGGT, ScanNet++ `3f15a9266d`, stride 10 — gave **1.28×** rotation
  improvement against the paper's claimed 2–12×, with Center-PH beating RayTun3R
  by ~5× on `R°`: the reverse of Tab. 1's ordering.
* That run is on the exact scene and backbone of Tab. 2, so the rows compare
  one-to-one, and the pattern inverts the obvious reading: our **unadapted**
  numbers are much better than the paper's (vanilla `R°` 2.379 vs 7.21, Center-PH
  0.378 vs 2.45) and only the **adapted** one is worse (1.858 vs 0.93). Every
  `d_reproj` of ours is 14–30× smaller. Only 2.4° of error is available to remove
  where the paper had 7.2°.
* Those numbers predate the depth-convention unification and working DA3 hooks and
  should not be quoted; `experiments/scannetpp_all.py` and
  `experiments/fov_sweep.py` are the reruns.

**Live hypotheses, in the order worth testing:**

1. **Our evaluation is easier than the paper's.** Prime suspect is pair sampling —
   the paper evaluates consecutive pairs of the full sequence, we use
   `--stride 10` on a windowed subset. This also explains the uniformly small
   `d_reproj`, which the convention bug (~1 px) does not.
2. **300 iterations may be far too few** (gap 1 above). `experiments/iters_sweep.py`.
3. **FOV** — §4 above. `experiments/fov_sweep.py`.

The paper's own limitation (v) is the mechanism behind our degenerate stride-1
runs: with small inter-frame displacement the self-supervised constraints go weak,
because large depth or translation-direction errors induce only small reprojection
errors.
