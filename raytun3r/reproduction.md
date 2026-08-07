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

**`d_reproj` weighting — fixed 2026-08-05, and it invalidates the old column.**
Eq. 8 carries `w_ij` inside the sum; **Eq. 16 carries no weights at all**. We had
been computing `sum(w·e)/sum(w)` — a confidence-weighted mean, i.e. the mean over
the *confidently matched* subset — where Eq. 16 is an unweighted mean over all of
Ω. That is not a normalising constant: UFM's covisibility goes to zero exactly
where reprojection error is worst (the fisheye periphery, content that left the
frame), so the old number dropped the hardest pixels *and* renormalised by their
absence.

`reprojection_depth_error(..., weighting=...)` now takes `"omega"` (Eq. 16,
default) or `"confidence"` (the old behaviour), and `eval.py` reports **both** —
`d_reproj` and `d_reproj_conf` — so pre-fix runs stay interpretable and the ratio
gets measured rather than assumed. On synthetic 170° geometry the two differ by
between **1× and 170×**, depending entirely on how much of Ω the matcher gives up
on and how bad its flow is there; with perfect flow everywhere they coincide.
**Only `d_reproj` is comparable to Tab. 1/2/5.**

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

## 6b. Phase −1 data audit — what it settled (2026-08-07)

Run on `3f15a9266d` (896 frames), plus a 1.7 MB local sample of the same scene.

**Stride 10 is the paper's operating point — hypothesis 1 is dead.** `R°` is an
absolute angular error, so the identity predictor ("no rotation") scores exactly
the median GT rotation. Measured:

| stride | 1 | 5 | **10** | 20 | 40 | 60 |
|---|---|---|---|---|---|---|
| median GT rotation = identity `R°` | 0.89 | 4.06 | **7.79** | 14.76 | 28.23 | 41.90 |

The paper's VGGT vanilla on this scene is **7.21°**. That sits on the stride-10
identity score, so our stride-10 pairs are at the paper's rotation scale. The
"easier evaluation" explanation is refuted at stride 10.

**What replaces it: the paper's vanilla is at chance and ours is not.** Skill =
identity `R°` / measured `R°`; ≤ 1.0 means the method carries no information.

| | vanilla | raytun3r | center_ph |
|---|---|---|---|
| paper (VGGT, Tab. 2) @ stride 10 | 7.21 → **1.08×** | 0.93 → **8.38×** | 2.45 → 3.18× |
| ours (VGGT) @ stride 10 | 2.379 → **3.27×** | 1.858 → 4.19× | 0.378 → 20.6× |
| ours (DA3-S) @ stride 10 | 2.496 → **3.12×** | 1.581 → 4.93× | 1.697 → 4.59× |

So there are **two gaps, not one**: our unadapted backbone is ~3× *better* than
the paper's (3.1–3.3× skill vs their 1.08×), leaving far less pinhole bias for the
adapter to remove; and our adapted model is ~2–4× *worse* than theirs (4–5× skill
vs 8.4×). Read at stride 1 the paper is stranger still: its vanilla scores 0.12×
identity — eight times worse than predicting nothing — and its RayTun3R lands at
0.96×, i.e. exactly the no-op baseline.

`eval.py` now reports `R_deg_identity` and `R_skill` on every run, because a bare
`R°` is not interpretable across strides or datasets.

**Bug found and fixed: we were evaluating on frames ScanNet++ flags as unusable.**
`is_bad` is true for **143 of 896 frames (16%)**, in 8 contiguous runs, the longest
132 frames. The loader now drops them (`--keep-bad` restores the old behaviour).
Every number measured before 2026-08-07 includes them.

**FOV settled: our ~170° is right.** Intensity std at 80–85° incidence is 0.148
against 0.230 at 0–30° — real structure, not dead vignette. The frame carries
content to the corner, so Ω is the rectangle. The intrinsics give horizontal
146.3°, vertical 103.7°, diagonal 169.5°; **the paper's 115° matches none of
them.** Poses are metric (camera bbox 1.12 × 1.94 × 0.29 m, 11.2 m path).

**Still unmeasurable on this download:** `dslr/masks/` and `render_depth/` are
absent, so Ω-from-mask and AbsRel (Tab. 3) remain untested. `transforms.json`
does reference masks per frame (`mask_path`), so they exist upstream.

### Local CPU check: vanilla skill across stride

DA3-Small pretrained, run on **this Mac** against the 1.7 MB sample (no GPU). Small
`n` — 2 to 8 pairs per stride — so this is indicative, not a measurement; it exists
to test the shape of the curve, and GPU-Claude's 100-window numbers supersede it.

| stride | pairs | `R°` | identity | **skill** | `t°` |
|---|---|---|---|---|---|
| 1 | 6 | 0.544 | 0.990 | 1.82× | 121.0 |
| 2 | 3 | 0.657 | 1.665 | 2.53× | 97.9 |
| 5 | 4 | 1.803 | 4.498 | 2.49× | 64.6 |
| 10 | 2 | 3.462 | 9.037 | 2.61× | 139.1 |
| 20 | 8 | 4.588 | 16.003 | 3.49× | 26.6 |
| 40 | 5 | 5.941 | 35.224 | 5.93× | 24.0 |
| 60 | 3 | 9.223 | 45.657 | 4.95× | 31.8 |

Two things follow. **Our vanilla never approaches chance** — skill is 1.8–5.9× at
every stride, against the paper's 1.08×. Whatever makes their frozen backbone
useless on this data is not present in ours, at any pair separation. And **`t°` is
at or worse than chance below stride 20**: a random direction in 3D scores 90°, and
we measure 121 / 98 / 65 / 139 at strides 1–10. Our stride-10 `t°` numbers — ours
and GPU-Claude's 40.3 alike — are not measuring anything. Only at stride 20–60
(26.6 / 24.0 / 31.8) does translation direction become observable, which is the
paper's own limitation (v) in Sec. 6.

**Live hypotheses, in the order worth testing:**

0. **Our vanilla is healthier than the paper's** — the newest and most likely to
   explain everything downstream. If their frozen backbone is at chance on this
   data and ours is not, we are not reproducing the *condition* the adapter is
   meant to repair. Prime suspects: the 16% `is_bad` frames they may have dropped
   and we did not, and whatever preprocessing makes their vanilla so much worse.
1. ~~**Our evaluation is easier than the paper's.**~~ **Refuted at stride 10** by
   the table above. Retained below only for the `d_reproj` weighting, which is a
   separate matter:
   * **The `d_reproj` weighting bug above.** We scored only the confidently
     matched subset where Eq. 16 scores all of Ω. This alone can account for
     anywhere from 1× to >100×, and our observed gap is 14–30× — squarely inside
     that range, and far beyond what the depth-convention bug (~1 px) explains.
     The first rerun measures it directly: compare `d_reproj` against
     `d_reproj_conf` in the same `results.json`. **Note this affects only
     `d_reproj`, not `R°`/`t°`**, so it does not explain the inverted rotation
     ordering.
   * **Pair sampling** — the paper evaluates consecutive pairs of the full
     sequence, we use `--stride 10` on a windowed subset. This one *does* reach
     `R°`, so it remains the candidate for the rotation story.
2. **300 iterations may be far too few** (gap 1 above). `experiments/iters_sweep.py`.
3. **FOV** — §4 above. `experiments/fov_sweep.py`.

The paper's own limitation (v) is the mechanism behind our degenerate stride-1
runs: with small inter-frame displacement the self-supervised constraints go weak,
because large depth or translation-direction errors induce only small reprojection
errors.

## 6c. Ticket 9 — the VGGT vanilla protocol sweep, and why its hit does not count (2026-08-07)

GPU-Claude ran `experiments/vanilla_repro.py` with VGGT-1B on `3f15a9266d`,
100 pairs per configuration, 20 configurations. Full table in
`results/vanilla-repro-3f15a9266d/` on the `results` branch.

The headline looked like a hit: **stride 40, `is_bad` honoured, square 504×504,
`seq_len` 2 → `R°` = 7.189 against the paper's 7.21**, off by 0.021°. GPU-Claude
flagged the objection in the same message, and the objection is right.

### The stage-1 sweep is an affine function of the identity score

Fitting the six stage-1 rows (square off, `seq_len` 2, `is_bad` dropped):

| span | identity `I` | `R°` measured | `R°` fitted | resid |
|---|---|---|---|---|
| 1 | 0.939 | 0.460 | 0.581 | −0.121 |
| 2 | 1.805 | 0.811 | 0.729 | +0.082 |
| 5 | 4.104 | 1.200 | 1.120 | +0.080 |
| 10 | 8.312 | 1.828 | 1.836 | −0.008 |
| 20 | 16.096 | 3.110 | 3.160 | −0.050 |
| 40 | 30.777 | 5.673 | 5.657 | +0.016 |

**`R° = 0.42 + 0.170·I`, R² = 0.9984.**

`I` — the identity predictor's score, i.e. the median GT rotation — is fixed by
the frame span alone. So `R°(span)` is a smooth monotone curve, and *every*
target between 0.46° and 5.67° is attained at exactly one span. **Fitting one
free parameter to one target number cannot fail, so it is not evidence.** The
stride-40 agreement is a curve crossing. Three of the twenty configurations
land within 1° of 7.21 for the same reason.

The two constants are the useful output, because both are span-invariant:

* **floor 0.42°** — the part of the error that survives as rotations shrink to
  zero. This is the fisheye damage, and it is what the adapter has to remove.
* **slope 0.170** — the fraction of the rotation magnitude left unrecovered.

For the paper's 7.21° to be a consecutive-pair number (`I` = 0.939° here), its
floor would have to be ≥ 7.05° — **17× ours**. That is the quantity actually in
dispute, and no choice of stride changes it.

### Squaring is a degradation, not a protocol

`square=True` costs a flat ~27% of `R°` at fixed span (5.673 → 7.189 at span 40;
6.114 → 7.772 at span 40 reached as `seq_len` 3 × stride 20). It stretches the
fisheye anamorphically to 504×504 and pushes VGGT off-distribution. Reaching the
paper's number by degrading the input is not a protocol match.

### `seq_len` and stride are one axis, not two

Identity depends only on the span `(seq_len − 1) × stride`, confirmed exactly in
the data: `seq_len` 3 at stride 20 and `seq_len` 2 at stride 40 both give
`I` = 30.777. At fixed span, the extra middle frame slightly *hurts*
(5.673 → 6.114, and 7.189 → 7.772 squared) — worth noting for a multi-view model,
where more context should help.

### What replaces it

`experiments/protocol_identify.py`. One target cannot falsify a one-parameter
fit; two can. `vanilla` and `Center-PH` are both training-free and matcher-free,
Tab. 2 gives both on this scene (7.21 and 2.45), and they share one unknown — the
span. Each method's curve crosses its own target at some `I*`; the test is
whether the two `I*` agree. Agreement identifies the protocol and validates the
harness at two independent operating points; disagreement rules the protocol out
as the explanation and points at the backbone or its preprocessing.

Guarded by `MIN_R2 = 0.90`, since every `I*` is read off a fitted line.

### Also settled by this round

* **Frames are sorted before use.** `transforms.json` does *not* store `frames`
  in filename order (`names == sorted(names)` is False), but `data.py:167` sorts
  by `file_path`, so "stride 1" really is temporally consecutive. Confirmed
  numerically: consecutive-pair median rotation over the 752 good frames is
  0.943°, matching the 0.939° the sweep measured.
* **Dropping `is_bad` creates jumps.** Consecutive-pair rotation over good frames
  only has median 0.943° but max 64.4° (vs 4.46° with bad frames kept), because
  removing a contiguous run of up to 132 frames splices its two ends together.
  The median is robust to this; a mean would not be.
* **`test_frames` is not the eval sequence** — 10 frames, consecutive-pair
  rotations of 165.9°/45.1°/119.0°/…, median 45.1°. Too few and too wild.
* **The input really is raw fisheye**, visually confirmed on the staged sample:
  strong barrel distortion, `OPENCV_FISHEYE` with `k1..k4`, 146.3° horizontal.
  `has_mask: true` in `transforms.json`, so masks exist in the dataset even
  though the staged sample carries none.

## 6d. Correction: how many training-free targets ScanNet++ actually gives (2026-08-07)

Tickets 004 and 005 said "DA3-Small has no per-scene vanilla number, so only VGGT
can be checked". That is true as literally stated and **wrong in effect** — it was
used to justify running one backbone against one number, which is exactly the
degeneracy §6c is about. The full inventory:

| source | backbone | scene | training-free rows available |
|---|---|---|---|
| Tab. 2 | VGGT | `3f15a9266d` (**named**) | vanilla 7.21, Center-PH 2.45 |
| Tab. 2 | π³ | `3f15a9266d` (**named**) | vanilla 6.17, Center-PH 2.28 |
| Tab. 1 | DA3-Small | ScanNet++ (mean, unnamed scenes) | vanilla 10.21, Center-PH 3.27, **Multi-PH 1.66** |
| Tab. 5 | DA3-Small | `3f15a9266d` (**named**) | none — RayTun3R/LoRA/CalTok only, all fitted |
| Tab. 3 left | DA3-Small | ScanNet++ (mean) | vanilla 0.282/0.601, Center-PH 0.066/0.961 — **depth** |

So there are **seven** training-free ScanNet++ pose numbers, not one. Tab. 2's four
are the tight ones (named sequence, one protocol). Tab. 1's three cannot be matched
in absolute terms from a single scene, but they are the only row carrying Multi-PH,
and **ratios between methods survive both the unknown span and the unknown scene
set** — which is the property §6c showed absolute `R°` lacks.

`experiments/protocol_identify.py` now carries all of them keyed by backbone, runs
`multi_ph`, prints every pairwise ratio, and refuses to give an `I*` verdict on an
aggregate row.

### Tab. 3 left is the one target that does not slide with span

`AbsRel` and `δ₁.₂₅` are per-pixel depth metrics: unlike `R°` they have no
dependence on how much rotation is in the pair, so **the whole curve-crossing
degeneracy does not apply to them**. That makes them the strongest available check
on backbone health, independent of the protocol question.

They are blocked on ground-truth depth — see below.

## 6e. What the phone transfer settled about masks and depth (2026-08-07)

GPU-Claude re-staged the sample with the mask directories and left `MASKS_ADDED.json`.

**The mask bug was ours, twice over.** `transforms.json` gives `mask_path` as a
**bare filename** (`DSC07484.png`, no directory), and ScanNet++ stores masks in a
sibling directory named after the image set — `dslr/resized_anon_masks/` beside
`dslr/resized_images/`. `dslr/masks/` does not exist. Every earlier candidate path
(`masks/<stem>.png`, `dslr/<mask_path>`, `masks/<basename>`) missed, and a miss
reads exactly like "this dataset ships no masks". `data.py` now has `anon_mask()`
with the right resolution and a test.

**Both mask sets are anonymisation masks, and neither defines Ω.** Measured on all
24 staged frames of `3f15a9266d`:

| set | usable fraction | identical across frames? | radial structure |
|---|---|---|---|
| `resized_anon_masks` | 0.9978 (0.9940–0.9993) | no | none — off-fraction 0.0003/0.0039/0.0010/0.0017 by radius band |
| `resized_undistorted_masks` | 0.9977 (0.9937–1.0000) | no | none — 0.0010/0.0045/0.0016/0.0004 |

A lens/valid-region mask would be constant across frames, radially symmetric, and
would remove a large fraction at high radius. Neither does any of the three; the
off-fraction actually *peaks* in the mid-radius band, where objects are. These are
blacked-out faces and screens (visible as a magenta blob in `DSC06719`).

**So ScanNet++ ships no lens mask, and Ω stays `camera.valid_mask` / `theta_max`.**
The hypothesis that the paper's 115° came from a shipped mask is dead — §4's FOV
disagreement is not explained by masking, and `has_mask: true` in `transforms.json`
means only that anonymisation masks exist.

**`render_depth` is genuinely absent, not a path bug.** No `dslr/render_depth`, no
`depth_file_path` on any frame, on this scene or any other GPU-Claude checked. The
scene ships `scans/mesh_aligned_0.05.ply` (35 MB) and a 3-frame panoramic
`panocam/depth`. Dense DSLR depth therefore has to be **rendered from the mesh with
the ScanNet++ toolkit** — a real job, not a copy. Until it exists, Tab. 3 left and
every `AbsRel`/`δ₁.₂₅` number stay untestable, which is the one target immune to
the span degeneracy. That makes rendering it worth doing.

## 6f. π³ backbone implemented (2026-08-07)

`--backbone pi3` (`backbones.Pi3Backbone`), so Tab. 2's second named-sequence row
is reachable: vanilla 6.17 / 19.7 / 38.6 and Center-PH 2.28 / 25.7 / 5.2 on the
same scene, same protocol as the VGGT rows.

Four things about upstream `yyfz/Pi3` that the wrapper has to handle:

* **`camera_poses` is camera-to-world**, unlike VGGT's `extrinsics` and unlike
  everything else here. Verified against π³'s own definition — it computes
  `points = camera_poses @ homogenize(local_points)`, so a correct cam-from-world
  `(R, t)` must satisfy `local = R @ points + t`. Measured on a random-weight
  forward: max residual **4e-6**, `R Rᵀ = I` to 0.0, `det R = 1`. A test pins the
  algebra without needing the checkpoint. Inverting this backwards is invisible at
  small rotations and grows with baseline — the worst way for it to be wrong here.
* **Depth is `local_points[..., 2]`** — planar z, so `native_depth = "z"` and
  `_finalize` converts as usual.
* **Only `decoder_size='large'` works upstream.** `small`/`base` build a
  `dec_embed_dim`-wide register token and concatenate it with the 1024-wide encoder
  output with no projection, so `decode()` raises. `large` is the released config.
* **No DPT positional grid to correct** — π³'s heads are `LinearPts3d`, so the
  `dpt_grid` parameter-free correction has no attachment point and warns rather
  than silently no-op'ing, since a `pi3` run is then not comparable to a VGGT/DA3
  run with `dpt_grid` on.

Install: not on PyPI, but `pyproject.toml` declares the package, so
`pip install git+https://github.com/yyfz/Pi3.git` works on Python ≥ 3.10; otherwise
clone it beside this repo or set `$PI3_ROOT`. Needs **torch ≥ 2.3** —
`pi3/models/layers/attention.py` imports `torch.nn.attention`, which fails at
*import* time on older torch, not at run time. Encoder is `dinov2_vitl14_reg`, so
`has_abs_pe` is true and Eq. 5 applies; decoder is RoPE-only (`rope100`).

`BACKBONE_NAMES` is now the single source for every `--backbone` choice list —
adding a backbone used to mean editing seven argparsers, and five of them would
have kept silently rejecting `pi3`.

## 6g. Ticket 10 — the protocol is identified, and `R°` decomposes (2026-08-07)

GPU-Claude ran `protocol_identify` on all three backbones, twice (100 and 300
pairs). Full tables in `results/protocol-identify-3f15a9266d/` on `results`.

### Vanilla agrees at stride 60, on two backbones at once

| backbone | our `R°` @ stride 60 (n=300) | paper | off by |
|---|---|---|---|
| VGGT | **7.242** | 7.21 | 0.032 |
| π³ | **6.392** | 6.17 | 0.222 |

And it is not a curve crossing: stride 40 gives 5.312 / 4.627 and stride 80 gives
8.867 / 7.594, missing both targets. The fits concur — `I*` = 44.30 (VGGT,
R²=0.9957) and 43.19 (π³, R²=0.9851), bracketing stride 60's `I` = 43.706.

**This also kills the ticket-9 configuration.** These runs are `square=False`, so
both `square=True, stride 40` and `square=False, stride 60` reach 7.21 for VGGT —
exactly the degeneracy §6c predicted. π³ is the tiebreak: at stride 40 it gives
4.627 against its own target of 6.17, a clear miss. Only stride 60 satisfies both.
**The protocol is the span, not the resolution.**

Honest weight: this is *one* degree of freedom checked, not two. The two `I*`
values are not fully independent — their agreement is close to the statement that
our vanilla `R°` ratio between backbones matches the paper's. That is a real test
that could have failed, and it passed; it is not the same as two independent
confirmations.

**The vanilla path of the harness is validated** — loader, camera model, pose
convention, metric, on two architectures.

### `R° = a + b·I` is not a curve fit, it is a measurement of rotation gain

If a model recovers a fraction `α` of every rotation about roughly the right axis,
its error is exactly `(1−α)·I`. So **`b = 1 − α`**, and `α` is a span-invariant
property of the model. Confirmed on the Mac with a different statistic — regressing
*predicted* rotation angle on *GT* angle through the origin, rather than error on
identity:

| | measured α | implied `1−α` | GPU-Claude's fitted `b` |
|---|---|---|---|
| DA3 vanilla | 0.816 | 0.184 | 0.1668 |
| DA3 Center-PH | 0.867 | 0.133 | 0.1375 |

Two independent code paths, two statistics, agreement to ~0.02. `protocol_identify`
now reports `gain` per span directly.

On raw fisheye all three backbones sit at α ≈ 0.82–0.88: **they under-read every
rotation by 12–18%**, which is what fisheye angular compression does to a
pinhole-trained model. That is the paper's thesis, measured directly.

### Our Center-PH is not leaking — that reading inverts

GPU-Claude flagged π³ Center-PH as non-physical: error grows 4× (0.048° → 0.206°)
while the rotation to estimate grows 68× (0.84° → 57.2°). Checked and rejected:

* **No ground truth can reach that path.** `ProjectionBaseline.__call__` receives
  only images; the hooks return immediately when `adapter is None`.
* **The rectification is correct.** Rendered and inspected: ceiling and shelf
  edges that curve in the fisheye are straight in the virtual view.
* **The virtual view is not degenerate** — at 110° it is 95.3% live (4.7% black
  caps) and covers 66% of the fisheye pixels. The dead-pixel confound that bit the
  earlier FoV sweep does not apply here.
* **A flat error curve is what a *correct* estimator looks like.** Error
  proportional to rotation is a *bias* (α < 1); error independent of rotation is a
  noise floor. Rectifying removes the bias, so flatness is the expected signature,
  not a suspicious one.

Gains after rectification: VGGT 0.849 → **0.992**, π³ 0.878 → **0.998**. Center-PH
does exactly what it is supposed to do.

### The "4–14× Center-PH disagreement" is 4–5% of rotation gain

Ratios of two near-zero errors explode. Converting the paper's numbers to gain at
the span its own vanilla implies:

| backbone | method | our gain | paper's implied gain | our `R°` | paper `R°` | `R°` ratio |
|---|---|---|---|---|---|---|
| VGGT | vanilla | 0.849 | 0.849 | 7.21 | 7.21 | 1.0× |
| VGGT | Center-PH | 0.992 | **0.949** | 0.56 | 2.45 | 4.4× |
| π³ | vanilla | 0.878 | 0.878 | 6.17 | 6.17 | 1.0× |
| π³ | Center-PH | 0.998 | **0.949** | 0.20 | 2.28 | 11.4× |

The vanilla rows match by construction (`I*` is defined from them). The Center-PH
rows are free, and both land on **0.949** — the paper's Center-PH recovers ~95% of
rotation, ours ~99%. A 4–5% gain difference, not a 4–14× failure. *(Caveat: that
0.949/0.949 agreement is mutual consistency between the paper's two backbones; it
holds under any common rescaling of the span, so it does **not** independently
confirm stride 60.)*

`protocol_identify` prints this block whenever vanilla yields an `I*`.

### The real anomaly is DA3, in the opposite direction

Rectification restores VGGT and π³ to gain ≈ 0.99. **DA3-Small only reaches 0.867**
— it still under-reads rotation by 13% on a clean pinhole image, so its Center-PH
barely beats its vanilla (paper: 3.12×, ours: 1.2–1.4×). This is a property of
DA3's pose head, not of the fisheye or of our baseline, and it is consistent with
the paper's own Tab. 6, where DA3-Small is the weakest pose backbone by a wide
margin (ETH3D vanilla 8.59 vs π³ 2.66, VGGT 3.19).

It matters because DA3-Small is the paper's *primary* backbone, so Tab. 1, 3, 4
and 7b all rest on it.

### The tension this leaves

Stride 60 means the paper evaluates pairs **~44° of rotation apart**, on a
sequence whose consecutive frames are 0.94° apart. The paper says "consecutive
image pairs". Those cannot both be literally true of the 896-frame DSLR set.

The gain framing makes the alternative testable, and it fails badly: for the
paper's vanilla 7.21 to be a consecutive-pair number (`I` = 0.94), its gain would
have to be `1 − (7.21−0.54)/0.94 = −6.1`. A negative gain means predicted rotation
*anti-correlated* with truth. Their Center-PH would need −1.39 and even RayTun3R
only 0.22. An entire published table of methods with near-zero or negative
rotation gain is not credible.

**So the most likely reading is that "consecutive" refers to a subsampled frame
set, not to adjacent DSLR frames** — 896/60 ≈ 15 frames, which is a plausible
keyframe count. Whether ScanNet++ ships such a list is a cheap thing to check and
is the first item of the next ticket.
