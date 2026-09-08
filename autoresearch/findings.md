# Research Findings

## Research Question

How can a frozen (or lightly finetuned) depth foundation backbone be adapted so the
high-incidence peripheral region of Aria ~110° fisheye **improves without sacrificing
central performance** — and what is the periphery actually good for (depth vs
cross-frame alignment vs nothing)?

The four novelty axes (from the human, 2026-08-18):

- **N1 — Pareto**: improve high-FOV region without hurting low-FOV region.
- **N2 — Efficiency**: distortion-adaptive processing (deformable/radial tokenization keyed to local distortion level).
- **N3 — Periphery-for-alignment**: the rim is hard for depth but rich in parallax; exploit it for cross-frame alignment and overall spatial understanding.
- **N4 — Dynamics**: egocentric video has hands; handle dynamic regions.

## Current Understanding

(Rewritten 2026-08-20 after 12 runs + 3 GPU tickets. The bootstrap-era survey
context lives in `docs/research/fisheye-wide-fov-adaptation.md`.)

The story, in five measured acts:

1. **The periphery's alignment value is span-under-noise, not per-point
   quality.** Rim correspondences are not individually better for rotation
   (H1 refuted, ideal-noise control flat), but at fixed count, widening the
   admitted field wins on essentially every real pair while the ideal-noise
   effect is ~20× smaller (H1.1) — the value is robustness to real feature
   noise. Replicates on real Aria; saturates by ~45° there (H1.3).
2. **Frozen FMs already run their pose on the rim.** Center deletion barely
   moves DA3's rotation; rim deletion costs far more than area-matched random
   deletion, on both a 170° DSLR and Aria (H1.2, runs 004–007). So adapters
   must not perturb rim features — pose stability is a mandatory third eval
   axis.
3. **The fisheye depth failure is a precise, radially-modulated range
   compression** — dispersion 2–10% everywhere, bias up to 3.3× at the
   near rim (runs 008b–009); the rim penalty survives the GT-depth control on
   raw fisheye across 5 models (ticket 024A), and multi-frame context buys
   the center, not the field (024B). **The rim gives alignment and does not
   receive fusion.**
4. **Output-indexed recalibration cannot invert the compression** (48-param
   table: near-rim transfers −18…−25% but near-center collateral, run 010);
   **frozen features can** — a 25k-param readout head trained in minutes on
   CPU cuts held-out near-rim AbsRel 51–67% with every other zone improving
   and pose untouched by construction (run 011). Transfer beyond one scene:
   ticket #29 pending.
5. **The four novelty axes collide at one object: the near-field rim.**
   Hand/body pixels are 0.8–4% of the cone, 80%+ beyond θ=41°, at median
   0.26–0.94 m (ticket #28) — inside both the worst-calibrated cells and the
   pose-critical band. Whether they actively corrupt pose is ticket #31.

Standing facts that shape any next step: VGGT-Ω has the largest controlled
rim penalty (1.81×) and no camera-input channel — the highest-headroom, and
hardest, adapter target; the eval of record is `finetune/eval/metrics.py`
scale_shift, range domain, full joint tables (zone pools hide collateral).

## THE INPUT WAS SIDEWAYS (2026-09-03/04) — read before citing any number below

Every run of the **H1 / H5 / H12 / H14 / H15 / H9** line fed the backbone frames
in ADT's native sensor orientation, a quarter turn off upright. Measured on
seq136, 30 frames, frozen DA3-Small, with the prediction rotated back so every
row scores identical pixels, GT and masks:

| input rotation | whole | near_rim | near_ctr | center | far |
|---|---|---|---|---|---|
| k=0 (what ran) | 0.5503 | 1.3925 | 0.5420 | 0.3241 | 0.2558 |
| k=1 | 0.6132 | 1.4843 | 0.9461 | 0.3562 | 0.2774 |
| k=2 | 0.5505 | 1.3669 | 0.5529 | 0.3199 | 0.2614 |
| **k=3 (upright)** | **0.1975** | **0.4101** | **0.3230** | **0.2051** | **0.1148** |

k=0 and k=2 agree to the fourth decimal (the two sideways orientations), k=1 is
upside-down and worst, k=3 upright and best — the shape of a model with an
up-prior. Pose too: median rotation error 12.07° → **5.77°**, RRA@15 0.550 →
**0.925** (40 pairs, once the predicted pose is un-rolled; both wrong unroll
choices are worse than not turning at all, which is why the sign is pinned by
measurement in `common/upright.py`).

**How it happened.** The repo has two ADT loaders. `raytun3r.data.ADTSequence`
— behind depthfisheye, fovbench and the bench rows — documents the turn and
applies it (`rotation=270`, `aria_intrinsics(rotated=True)`). `AriaLocalPairs`
was written for **H1.3**, a *classical* pose experiment where a quarter turn is
irrelevant provided the camera matches the pixels; it did, and the hand-eye gate
passed at 0.77–0.96° on it. It was then inherited unexamined when the line
crossed into a **pretrained depth network**, which is not rotation-invariant.
H5's `Seq` wraps it and H12/H14/H15/H9 all import H5's `Seq`. Every internal
check stayed self-consistent, so nothing ever failed. It took the human looking
at an example image.

**What survives:** every *within-experiment* arm comparison — all arms ran at
the same wrong orientation. `rect` beat `roundtrip`; `shuffled` beat `jac`
16/16; `raycal` beat both controls 6/6.

**What does not:** every absolute AbsRel in that line (inflated ~2.8×), and —
the load-bearing one — **the size of the rim penalty itself**:
near_rim/center is **4.30× sideways and 2.00× upright**. A large part of the
radial error field this line has been studying is an orientation artefact. The
remaining 2.00× is still large and still agrees in direction with ticket 024A's
controlled ratios (1.25–1.81, measured through a different pipeline), but the
magnitude has to be re-established.

**Already re-measured upright:** H14's teacher pre-check on seq131 — the rect
teacher's near-rim advantage falls from **−35.3% to −14.7%**. So more than half
of what the rect teacher was buying was compensating for the orientation, not
for the lens. The premise (024A) survives at less than half its apparent size.

**A SECOND defect, found while diagnosing the first (2026-09-04).** Three ADT
loaders were in play, each internally self-consistent:

| loader | quarter turns | orientation | used by |
|---|---|---|---|
| `AriaLocalPairs` | k=0 | sideways | H1/H5/H12/H14/H15/H9 |
| `raytun3r.data.ADTSequence` | k=1 | **upside down** | `depthfisheye`'s ADT arm, `raytun3r` train/eval on ADT |
| `fovbench.run` | k=3 | **upright ✓** | fovbench, and therefore **ticket 024A** |

`ADTSequence`'s map is `{0:0, 90:3, 180:2, 270:1}` — degrees the stored frame is
from upright, mapped to the turns that undo it — and its default said `270`,
i.e. k=1. Rendered and looked at: the floor comes out at the top. The stored
frame is 90° off, not 270°; the default is now `90`. k=1 is not a near miss, it
is the **worst of the four** orientations, worse than not turning at all.

**fovbench was right all along, so ticket 024A — the premise H14 rests on — is
unaffected.** `depthfisheye`'s ADT arms (`results/depthfisheye/{depthfisheye,
full,lora}`) ran upside down and would need re-running before citation; its
SynWoodScape arms use a different loader and are unaffected.
`tests/test_orientation_convention.py` now pins all three declared defaults to
one quarter turn, with no data and no GPU.

**HOW MUCH OF THE RIM PENALTY WAS ORIENTATION — per backbone (H16, 2026-09-04).**
seq136, 20 frames, prediction rotated back so every cell scores identical pixels:

| backbone | sideways whole | upright whole | Δ | rim/ctr sideways | rim/ctr upright |
|---|---|---|---|---|---|
| da3:small *(this line's backbone)* | 0.5674 | 0.1954 | −65.6% | **5.28** | **2.13** |
| da3:large | 0.2620 | 0.1113 | −57.5% | 2.58 | 2.04 |
| vggt (1B) | 0.1730 | 0.1277 | −26.2% | 2.49 | **2.57** |
| vggt_omega *(512 px, 16 fr)* | 0.0993 | 0.0804 | −19.0% | 2.15 | 2.33 |

Two readings, and the second is load-bearing:

1. **Orientation robustness varies 3.5× across backbones** and does not track
   model size: VGGT-Ω loses 19% to a sideways input where DA3-Small loses 66%,
   and VGGT-Ω is also the most accurate of the four.
2. **"The rim penalty is an orientation artefact" is FALSE as a general claim.**
   The ratio collapses only for DA3-Small (5.28 → 2.13); the other three barely
   move. Measured upright, **all four** sit at **2.0–2.6×**, which agrees in direction with 024A's controlled ratios
   (1.25–1.81, different pipeline, upright by construction).

So **the project's central claim survives** — the rim really is worse, by about
2× — and what has to change is the magnitude this line quoted and the fact that
**the backbone is now an experimental variable that must be reported**: this
line happened to run on the least orientation-robust of the three.

**Fix:** `autoresearch/experiments/common/upright.py`, applied at the backbone
boundary only — the loader keeps returning stored-frame images because the rig
grids, lens warps, θ binning, GT and pose conjugation all live there and none of
them care about orientation. Depth crosses the boundary as planar **z**, because
`_finalize` would otherwise divide a rotated prediction by the camera's cos map
and Aria's principal point is 4.5 px off centre (≈1° of θ at the rim, a 2.5%
radial error). `forward_z` refuses a `range` install rather than trusting the
caller.

## Upright re-runs, 2026-09-04 — what survived and what did not

**H14 (rect-teacher distillation): REFUTED.** near_rim % change against the
frozen model, held out, sideways → upright:

| arm | seq136 | seq132 |
|---|---|---|
| `rect` | −26.0% → **−13.2%** | −12.9% → **+4.7%** |
| `rect_ring` | −19.1% → +3.7% | +5.1% → +7.9% |
| `roundtrip` (control) | +0.1% → −0.1% | 0.0% → 0.0% |
| `gt` (labelled) | −82.0% → −57.9% | −33.0% → −27.2% |

P1 required `rect` to beat `roundtrip` on **both** held-out sequences; upright it
wins seq136 and loses decoration_seq132, where it is worse than doing nothing.
P2 (22.8% / −17.5% of the labelled arm) and P3 (near_centre +43.3% where `gt`
improves −39.0%) also fail. **More than half of what the rect teacher appeared
to buy was compensating for the input orientation.** The pre-check said so
before the students trained: the teacher's own near-rim advantage fell −35.3%
→ −14.7%.

**H9 (RayCal-TTA): the model-class fix flipped it.** near_rim AbsRel, upright:

| seq | none | raycal | **raycal_inv** | raycal_quad | global | shuffled |
|---|---|---|---|---|---|---|
| clean_seq136 | 0.4269 | 0.4930 | **0.3613** | 0.4331 | 0.6222 | 0.6256 |
| decoration_132 | 0.2320 | 0.3607 | **0.2602** | 0.2951 | 0.3878 | 0.3910 |
| clean_seq131 | 0.3805 | 0.4122 | **0.3050** | 0.3171 | 0.5316 | 0.5340 |
| clean_seq133 | 0.4021 | 0.5033 | **0.3365** | 0.4438 | 0.6112 | 0.6177 |
| clean_seq134 | 0.3719 | 0.4330 | **0.3201** | 0.3621 | 0.5519 | 0.5527 |
| clean_seq135 | 0.5039 | 0.5602 | **0.4107** | 0.4537 | 0.7552 | 0.7549 |

The ORIGINAL `raycal` now loses to doing nothing **6/6** — upright the field is
half the size and a mis-specified correction costs more than it buys.
`raycal_inv`, the same per-θ model fitted in the direction the correction is
*applied* in, is best **6/6** and beats the frozen model **5/6**. `raycal_bal`
(the sampling reading) is worst 6/6, exactly as its own synthetic predicted.
`global` and `shuffled` stay far behind everything (0.55–0.76), so **P2 passes
6/6 by a wide margin** — the radial structure is doing real work.

Still failing: the locked bar (0/6, though the gap now *shrinks* — decoration
+0.5247 → +0.2935, a 44% cut against a 50% bar — where sideways it **grew**),
and P3 (near_centre 21–52% worse). `raycal_shrunk` — evidence-weighted
shrinkage toward identity, driven by each bin's anchor count and depth spread
rather than by θ — is the current attempt at P3 and is running.

**H15 (lens conditioning): still refuted, but the failure changed shape.**
near_rim head-to-head over all 16 (lens × sequence) cells, sideways → upright:

| | sideways | upright |
|---|---|---|
| `shuffled` better than `jac` | **16/16** | 10/16 |
| `mismatched` better than `jac` | 8/16 | **0/16** |
| `none` better than `jac` | 4/16 | 10/16 |

Upright, `jac` beats the **mismatched** control — a real lens's real field,
equally smooth, describing the wrong lens — on 16/16, while `shuffled` and plain
LoRA each beat it on 10/16. Spread across arms is ~2% on values of 0.18–0.31.
So the network is **sensitive** to the field (a plausible-but-wrong one is
consistently worse than anything) and does not **benefit** from it being right.
P1 still fails on the held-out lenses (jac beats shuffled on 2 of 4). The
sideways headline — "a scrambled field beats the real one 16/16" — was itself
partly an orientation artefact; what survives is the weaker, more precise claim
that the field is inert.

**Orientation tolerance (H16).** ±20° of roll costs +28% and leaves the
near_rim/centre ratio flat at 2.10–2.23; ±40° costs +197% and inflates it to
4.9. Ordinary head roll is inside the flat part, so a per-frame gravity
alignment is **not worth building** — recorded as a decision not to build.

## Method-phase status (post-pivot, 2026-08-27)

- **H5 (rim-targeted LoRA): REFUTED BY ITS OWN CONTROL (2026-08-22, #35 evals).**
  Plain LoRA matches or beats the full rim-loss arm on both held-out sequences
  (near-rim −83.5% vs −80.6% on seq136; −33.5% vs −33.3% on dec_seq132) and wins
  pose (13.51→11.52° vs 13.51→13.73°). Mac-side paired bootstrap on the delivered
  per-frame values: full-minus-plain +0.038 [+0.017,+0.059] on seq136 — the rim
  losses are significantly WORSE there, a tie on the other. The gain is bought by
  LoRA finetuning per se. Plain-LoRA is now the standing adaptation baseline.
- **H6 (rim-restricted cross-frame attention): REFUTED ON HELD-OUT (2026-08-22,
  #36 evals).** H6.1's train-scene "rim-KV==full-KV" does not survive: all-token
  −75.9% vs rim −52.2% on seq136, and the rim arm makes dec_seq132 worse (+3.2%
  vs −6.2%). Bootstrap: rim worse on BOTH seqs outside error bars (+0.177 and
  +0.027). The module works; the restriction is what fails. 0.48× FLOPs is not free.
- **#38 (RayTun3R rows) v2:** the double-conversion fix FLIPPED the conclusion —
  adaptation helps seq136 (−7.2% whole, −15.6% near-rim) and hurts dec_seq132
  (+47.3%/+37.6%). Mixed, not uniformly negative; no v1 number citable.
- **Oracle null refinement (results 1158e27):** the null is not one number — it
  grows with the model's own error size (σ 0.05→0.30 m: aea 1.068→1.169) and
  aea's residual plateaus at ~1.10 regardless of strata. A per-model null is
  required; a single shared null under-credits noisy models.
- **BENCH:** frozen rows done; #40 = cross-room bedroom probe; #41 unblocked.

## The reframe (2026-08-22): the rim deficit is a global lens-prior mismatch

Three rim-targeted interventions have now lost to their own controls — H5's rim
losses to plain LoRA, H6's rim-KV to all-token, H7's θ-gated LoRA to uniform
(and the center/rim MoE was killed unanimously before any GPU spend). Meanwhile
every intervention that helps is GLOBAL: plain LoRA, rect_derect on slambench,
the whole-image feature head. Read together: **the rim deficit is not a
region-shaped capacity problem; it is the loudest symptom of a global mismatch
between the backbone's near-pinhole image-formation prior and the fisheye's.**
"Add capacity where the symptom is loudest" has now failed four ways — that
refutation chain is itself paper material.

GPU-Claude independently registered **H12 (lens-Jacobian FiLM conditioning)**
as the constructive form of this reframe, with a mechanism that explains H5's
failure rather than restating it: on Aria's calibration, log_area peaks at
48.9° then FALLS, and log_aniso crosses ZERO near 50° (−0.191 at 54.83°) — the
rim band differs in KIND, and a monotone-in-θ scalar weight cannot represent a
field that turns over and changes sign inside the band it weights. The pilot's
kill bar is pre-registered (real field must beat a position-shuffled field at
equal capacity), and a silent checkpoint bug (zero LoRA tensors saved) was
caught by tensor-norm audit before any wrong number shipped.

### H12 RAN AND LOST (2026-08-22, results da38331) — the fifth controlled negative

near-rim AbsRel, held-out: seq136 jac 0.2477 / shuffled **0.2354** / theta 0.2400;
dec_seq132 jac 0.2394 / shuffled 0.2337 / theta **0.2334**. **`jac` is the worst
of the three on both sequences** — the real geometry field loses to a scrambled
one carrying identical values at identical capacity. GPU stopped on the
pre-registered criterion and did not proceed to ScanNet++. This was the
strongest available form of the idea (hand the network the geometry rather than
tell it where to try harder), and the reframe that motivated it survives an
experiment built to exploit it and failing.

**Mac post-mortem (2026-08-24, `data/h12_gradient_and_field_sensitivity_2026-08-24.md`),
neither half a rescue:**

1. **The real field's advantage is monotone in eccentricity and reverses at the
   rim.** Count-weighted corr(jac−control, θ) = **+0.66** on seq132 and +0.24 on
   seq136, against *both* controls; seq132's nearest-depth column is perfectly
   monotone across all eight θ rings (−0.092 on axis → +0.019 at 51.4°). Real
   geometry helps on axis — beating θ-only too, so it is the Jacobian *content*,
   not merely a smooth radial field — and hurts at the rim.
2. **The field is 10–40× less determined at the rim.** A ±1% perturbation of a
   single KB4 coefficient swings log_aniso by ~2% of its value at 30° but ~12%
   at 54.83° (≈5% vs ≈31% summed over the four coefficients). Caveat: 1% is a
   conditioning probe, not Aria's published coefficient uncertainty, so this
   does not settle whether the rim turnover is physical or a fit artefact.

Together: **conditioning pays where the field is well-determined and costs where
it is not**, with the crossover near the same 45–50° band where log_aniso turns
over. A confidently-wrong input is worse than a scrambled one — which is exactly
what the `shuffled` control measured. Consequence for the line: **geometric
conditioning is a centre tool on this lens** and must not be sold as a rim fix.

Process gap worth fixing: `eval_cond.py` emits no `per_frame`, so H12 alone
among the kills has no error bars, and its largest margin (the centre effect,
+0.31% vs +16.73% on seq132) is a single unbarred number. Cheap re-emit asked for.

Consequences for the brainstorm survivors: **H11 is blocked** (its precondition
— rim-KV==full-KV — failed held-out; only the temporal-scale claim survives,
corroborated by #22 stride-10). **H9 gains weight** (GT-free, per-lens, same
global-field diagnosis; unaffected by the kills). **H10 unaffected** (pose leg).

**Standing after H12 (2026-08-24).** Every hypothesis of the form "treat the rim
as a separate sub-problem" is now closed: H5, H6, H7, the MoE, and H12. The
constructive space that remains has exactly two shapes, and neither is
region-targeted:

- **H9 (RayCal-TTA)** — the last untested *depth* method. Its locked bar ("one
  affine must serve the whole image") is global by construction, which is the
  one property every surviving intervention shares. It is now priority 1.
- **H10 (FrozenMatch)** — the pose leg, untouched by the depth kills; classical
  geometry is ~10× more accurate than the learned pose path on solvable pairs.
- **H13 (distillation ladder)** — GPU's new efficiency leg (teacher caching
  built 2026-08-24). Orthogonal to the rim question rather than another attempt
  at it, with two guardrails from evidence already in this repo: the VGGT
  teacher has the **steepest** rim error field of the five models we
  benchmarked, so a student that matches it everywhere inherits the worst rim
  behaviour available; and VGGT confidence gating already measured *worse* than
  ungated on this repo's DAv2 work, because conf is high on easy central pixels
  and low exactly on the band we care about. Distil the centre; supervise the
  rim from GT or geometry; use conf as a probe target, never a loss weight.

## Multi-agent architecture brainstorm (2026-08-19, human-directed)

Three agents (Fable/Opus/Sonnet), ~18 independent proposals, then adversarial
cross-critique. Full record: `literature/2026-08-19-arch-brainstorm/`
(3 brainstorms + 3 critiques + synthesis.md). Survivors registered as H9/H10/H11;
kills and pre-checks in `research-state.yaml: brainstorm_2026_08_19`.

- **Organizing thesis (consensus):** pose→depth, inverting DUSt3R/MASt3R/VGGT's
  arrow — "spend the rim's pose surplus on the rim's depth deficit."
- **H9 RayCal-TTA** (top pick): classical pose → triangulated anchors → label-free
  removal of the radial compression field; sold as GT-free adaptation to an unseen
  camera (answers F8 structurally). Locked bar: the scale_shift↔frozen-affine gap
  must collapse. Application-step fork (analytic inverse vs pseudo-labeled feature
  head) is deliberately unresolved — both arms run on the same anchors.
- **H10 FrozenMatch pose stream**: three agents converged independently. Stratify
  the RANSAC sample, never θ-weight (H1); full-field, not rim-only (run_007);
  never headline RRA@15 (CAM3R saturated it).
- **H11 epipolar rim attention**: invented independently by two agents, then
  re-scoped by baseline arithmetic — metric evidence needs t−8/t−16, appearance
  evidence saturates at t−2; two temporal scales, a testable claim.
- **Data-sufficiency verdict (human's Q1):** 4 sequences/one apartment is enough
  to learn the lens (fixed radial function, 0.49M params — pilot works), NOT
  enough to support a generalization claim: the review's aux-only control shows
  features do the work, and features may be apartment appearance; training loss
  still falling at epoch 20 → widen data, don't lengthen. Order: more ADT scenes
  (#40), ScanNet++ 170°, ego-synth. H9's label-free adaptation is the structural
  answer (needs video, not labels).
- **Blind spots caught only by cross-critique** (none of the three saw their own):
  the two-annuli problem (pose value ~35–45°, depth liability 45–55° — threatens
  the thesis, run_006 outer arm was n=11, cheap to settle); Aria's SLAM cameras
  give a static stereo baseline at every timestamp (bypasses rotation-dominant
  motion, the #1 named killer); moving hands poison static-scene triangulation
  precisely in the worst cells (motion gate mandatory); rolling shutter at the rim.
- **Killed unanimously with our own measurements**: center/rim dual-expert MoE
  (H7 flat gate + F2 no-interference + F4 wrong-axis) — the human's direction-B
  first bullet, and the refutation chain is itself paper material.
- **The premise survives its own confound on real footage (2026-08-21, slamfov #23)**:
  an oracle answering from GT with a fixed 0.15 m error — zero field dependence —
  reads a 1.63x (aea) / 2.84x (nymeria) pooled "rim is worse" effect on real MPS
  SLAM points: the depth-vs-eccentricity confound caught in the act, at full
  coverage. After distance standardisation (null residual 1.10/1.26, strata
  coarseness), ALL five models still exceed the null (vggt_omega 1.83/1.96,
  vggt_1b 1.80/2.07, da3_large 1.59/1.68, da3_small 1.56/1.57, dav2_large
  1.41/1.57) — the rim degradation is real, not a depth artifact. Two rules to
  carry: (1) the confound's SIGN varies per cell (dav2/nymeria went UP under
  standardisation) so pooled numbers can't be discounted blind — only the
  two-way table + an oracle-null row is citable; adopt both in our bench
  protocol. (2) **Accuracy and evenness are different axes**: vggt_omega is the
  most accurate AND has one of the steepest fields; dav2_large is the flattest
  and far from the best. Our project's claim is exactly "buy evenness without
  selling accuracy" — this is the paper's Figure-2-grade motivation.
- **Independent corroboration of H11's temporal re-scope (2026-08-20)**: the
  slambench #22 delivery (real MPS SLAM points, aea+nymeria, models incl.
  VGGT-Omega/DA3) shows multi-frame context at stride 1 buys ~nothing (ratios
  0.98–1.07) while stride 10 buys real gains (DA3-L nymeria −10…−13%, VGGT-Ω
  rect −6…−7% at 10fr) — adjacent frames lack baseline; metric evidence needs
  temporal distance. Exactly Opus's t−8/t−16 arithmetic, on independent data.
  Caveat: whole-image averages, no zone split in that harness.

## Corrections from the external H2 review (2026-08-19, FEEDBACK-2026-08-19.md)

An independent session re-audited the H2 line against our JSONs + 3 new
matched-capacity controls (imported to h2 code/feedback-controls/). Verdict
and consequences, all accepted:

1. **"Cross-scene" is renamed "cross-sequence (one apartment, one device)"**
   everywhere. ADT metadata: all six sequences are `Apartment_release_*`,
   same `M1292` device, ~25 m² room; decoration is a redecoration of the
   same room. A second real scene (ADT Office if available, else ScanNet++)
   is now load-bearing, not nice-to-have.
2. **Report absolute `after` first.** The head converges to ~0.29-0.36
   near-rim from any baseline (before spread 4.4×, after spread 1.26×);
   percentage headlines flatter easy baselines. decoration's "weak −18.7%"
   was its baseline already sitting at the floor — not a transfer failure.
3. **~82% of the seq131 near-rim "penalty" is the eval affine's placement**
   (refit on near pixels: 1.47→0.26 uncorrected). The compression *field*
   survives (run_009 is alignment-free) but every scale_shift AbsRel zone
   number is a rim-under-this-alignment property. Protocol of record gains
   alignment-robustness rows (frozen-affine / scale_only / zone-restricted).
4. **The feature-vs-geometry control now exists and lands our way**: a
   same-capacity MLP on (θ, d̂) recovers <half the rim gain AND reproduces
   the H2.1 center damage; only frozen features double the gain and fix the
   center. The aux-only arm becomes a permanent baseline row in every
   adapter experiment.
5. **Splits are not independent replications**: seq131 halves (0.639) sits
   below the even/odd bootstrap 90% range (0.879-1.633) — temporally
   correlated halves. Error bars (bootstrap over frames) required on
   per-sequence numbers.
6. **REOPENED: the "held-out scenes have no near-rim mass" claim.** run_012's
   harness (uniform frames) measures seq136 before-near-rim at 1.359 while
   #37's harness (FIRST-60 frames) reports 0.149 — the mass difference may be
   a frame-window artifact, not scene content. Until resolved, neither claim
   is citable. (Resolution: rerun one #37 row with uniform/full frames.)

## External positioning (2026-08-19 survey + comparison study)

- **Closest lanes** (literature/2026-08-19-distortion-crossframe-survey.md):
  Wid3R (camera-model token, trains a new wide-FOV model), UniDAC (trained
  spatially-varying scale map + latitude-aware RoPE — their premise is our
  run_009 measurement, solved with dataset-scale training where we use 48
  params on a frozen model), Spark3R (saliency-driven token reduction on the
  same backbones, incl. VGGT-Ω — ours is geometry-driven and adds rim
  capacity instead of pruning; stackable). **Unclaimed and still ours:**
  diagnosis-driven adaptation (probe the frozen FM's error field, aim the
  parameters at it) and asymmetric periphery treatment (rim = pose asset +
  depth liability, exploited rather than averaged over).
- **RayTun3R's own tables state our thesis** (paper/comparison-protocol.md):
  Center-PH wins ScanNet++ depth 2.5× *by discarding the rim* and loses
  rotation 3× for it; RayTun3R wins pose but concedes depth on 4/5 datasets.
  Nobody holds both — that is the N1 lane.
- **#38 first delivery INVALIDATED (2026-08-19)**: my raytun3r_row.py
  double-converted GT (ADTSequence already returns range; the script divided
  by cos again), inflating rim GT up to 1.73×. Fixed (8b5c13d), reopened
  #38 for a 4-row re-run (adapters unaffected). The "RayTun3R adaptation
  hurts on ADT" reading is UNVERIFIED until v2 lands; do not cite the v1
  numbers anywhere. Lesson logged: any new eval script must assert its GT
  domain against the loader's declared convention, not the variable name.
- **Center-PH measured on ADT (2026-08-19, local anchor, exploratory)**: its
  ScanNet++ depth win does NOT transfer — on identical pixels the rectified
  input leaves the center flat and makes the near-field center (egocentric
  workspace, ≤2m) **62% worse**, while covering only 49.6% of the near-rim
  zone. "Crop the problem away" fails exactly where egocentric products
  live. (bench/analysis.md; held-out-scene reproduction queued behind #35-38.)
- **To compare the same way as RayTun3R** we still need: a Center-PH
  baseline (CPU-buildable), a ScanNet++ 3f15 row (their named sequence =
  external published anchor; raytun3r/ already loads the dataset), and an
  explicit adaptation-data column (their protocol is unsupervised test-scene
  TTA; ours is supervised cross-scene — different rows, never the same row).
- **H7 and H8 both REFUTED same-day by CPU pilots (2026-08-19)** — and the
  refutations draw one line: θ-gating LoRA is redundant (gate stays flat;
  PE already conditions the adapter spatially) and equal-solid-angle
  resampling degrades every zone (+16..+31% at equal tokens; the radial
  failure is NOT sampling density). Together with H3 (patch undistortion
  no-op) and Center-PH (near-field +62%), the pattern is: **input-space
  geometric surgery hurts a frozen FM, geometry-conditioning the adapter is
  redundant — the interventions that work live behind the encoder (rung-1
  readout), in the objective (rung-2 weighted losses), or add evidence
  (rung-3 cross-frame attention).** This "where to intervene" gradient,
  each step measured, is itself a paper-level organizing result.

## Key Results

- **H1 REFUTED (runs 001–002, 2026-08-18):** on ScanNet++ 3f15 (~170° DSLR
  fisheye, the pre-verified classical harness), incidence-angle quartile bins of
  SIFT matches at equal count show **no per-correspondence rim advantage for
  rotation**: the synthetic control (same pixels, GT-consistent targets, 1 px
  noise) is flat across bins (0.30–0.39°), the real arm is non-monotone with the
  rim bin worst, and paired rim-minus-center is a coin flip in both arms.
  Rules out: methods premised on "rim correspondences are individually better for
  rotation". Reproducible band signature: rim-annulus estimates *overshoot*
  rotation (gain 1.06–1.14) even in the noise-only control.
  Details: `experiments/h1-rim-pose-value/analysis.md`.

- **H1.1 SUPPORTED (run 003):** at fixed correspondence count, widening the
  admitted disk θ≤T from 35° to 85° improves rotation on **17/17 pairs** (median
  −2.15°) and cuts translation-direction error 43°→16°. The synthetic control
  shows the ideal-geometry conditioning effect is ~20× smaller (−0.11°): **span
  pays through robustness to real feature noise, not conditioning.** The 65→85°
  band still contributes.
- **H1.2 REFUTED, informatively (run 004):** DA3-Small's pose does NOT ignore the
  periphery — it depends on it. Deleting all central content (θ≤45°, 39% of
  pixels) leaves rotation error unchanged (4.93° vs 5.00°); deleting the rim
  (61%) more than doubles it on every pair (12.32°). Run 005 closed the area
  confound: at equal 61% masked area, rim deletion costs 3× random deletion.
- **H1.3 SUPPORTED (runs 006–007): both findings transfer to real Aria** (local
  ADT seq131, GT via a gate-verified hand-eye extrinsics bootstrap: 0.77–0.96°
  residual, angle(C)=40.6° vs the box's ~38°). Span value replicates
  (10.1°→1.6° from θ≤25° to θ≤45° at fixed count) but **saturates by ~45°** —
  the 45–54.8° ring's pose value is unproven at n=11. Rim-dependence replicates
  softer: rim-masked 38.5° ≫ area-matched random 25.3° > center-masked 20.0° >
  vanilla 14.8° — on Aria's narrower cone the center is not disposable, but the
  rim is still the most load-bearing region per unit area.

- **H2.0/H2.0b (runs 008b–009): the fisheye depth failure is a precise,
  radially-modulated RANGE COMPRESSION, not noise.** Alignment-free maps show
  dispersion of only 2–10% everywhere (the model *sees* the near rim fine),
  while bias is huge and structured: 0–1 m content placed 1.7–3.3× too far
  (worst at the rim), 5–10 m content 1.4–1.8× too near. Matches the RayTun3R
  repro's depth-gain-0.406 signature and UniK3D's wide-FOV contraction.
- **H2.1 (run 010): a 48-param (θ × predicted-depth) table transfers the
  near-rim fix (−18…−25% on held-out frames) but damages the near center — in
  every variant, including with the eval affine frozen.** Mechanism: the
  compression makes predicted depth many-to-one in true depth, so an
  output-indexed correction pushes the majority's fix onto minorities.
  **Post-hoc recalibration cannot invert a compression; input evidence is
  necessary** — measured, not argued. The table is now the mandatory baseline
  for any adapter.
- **Official calibration landed (ticket #27, closed): device→RGB rotation
  38.44°, vs the hand-eye bootstrap's 40.55° — 2.33° apart**, validating the
  bootstrap; H1.3 conclusions unaffected (effects were ≥10°).
- **Cross-lane result (GPU ticket 024, `results` branch digest 601fcb22767e —
  not produced by this workspace, provenance: 6 ADT sequences, 300 frames):
  the fisheye rim DEPTH penalty is real, not furniture.** After the GT-depth
  control, 57–85% of the log-penalty survives on raw fisheye (controlled
  rim/center AbsRel ratio: VGGT-Ω 1.81, DA3-Large 1.66, DA3-Small 1.25,
  VGGT-1B 1.33), while on rectified input it collapses to ≈1.0 (−0.5…0.2
  survives — "the rect rim penalty is mostly the furniture"). This is the
  depth half of the N3 tension, measured independently of this workspace, and
  it sizes the N1 prize per backbone: **VGGT-Ω has the largest controlled rim
  penalty and is therefore the highest-headroom adapter target.**

- **H4 measured (ticket #28, GPU): hand/body pixels are 0.8–4% of cone
  pixels, 80%+ of them beyond θ=41°, at median depth 0.26–0.94 m** (the
  wearer's own body enters from the frame edge, extremely near). **The hand
  zone IS the near-field rim** — the same (θ, depth) cells where the
  compression is worst and inside the band the pose path relies on. All four
  novelty axes now point at one measured object: the near-field rim of
  egocentric fisheye. H4.1 (ticket #31): do these pixels actively disrupt
  pose, or are they mere occlusion?
- **H2.2 flagged cell resolved:** the even/odd 0–1 m/3.4° regression sits on
  392 of 2.45 M pixels (0.016%) — noise, de-flagged.
- **H2.2 SIX-SEQUENCE CONFIRMATION (ticket #29):** near-rim AbsRel drops on
  every sequence and every split, −21%…−75%; 5/6 clear −30% on both splits;
  near-center worst case +5.8% (noise-order). One scene wasn't a fluke.
- **H4.1 (ticket #31): hands ≈ plain occlusion** — GT-dynamic masking vs
  area-matched random differs by only 0.35–1.0° on 10–25° baselines; depth
  arm inconsistent. (Depth-GT provenance under verification, #34.)
- **H3 REFUTED with mechanism (runs 014/014b):** correct per-patch local
  undistortion changes nothing (3rd-decimal identical) because **within-patch
  distortion on Aria KB4 at patch 14 is ≤0.21 px even at the rim** — the
  distortion lives between patches, not inside them. Closes N2's
  patch-content branch and quantitatively explains RayTun3R's "patch
  undistortion minimal" ablation row. (run_014 discarded: rotation bug caught
  by the protocol's mandated visual check.)

## Patterns and Insights

- Using a quarter of the matches costs ~2–3× rotation error regardless of which
  quarter — count/coverage dominates band identity. The wide-FOV pose story is
  **span**, not per-point rim quality — and specifically span-under-real-noise
  (the effect nearly vanishes with ideal features).
- **The emerging narrative (revised after run 004):** (i) the periphery carries
  large alignment value, delivered by *span under real noise*, not per-point rim
  superiority (H1 refuted, H1.1 supported); (ii) the frozen FM already extracts
  it — its pose survives center deletion untouched but collapses without the rim
  (H1.2 refuted in the informative direction); (iii) meanwhile fisheye *depth* is
  worst exactly there. So the user's N3 intuition ("rim: bad for depth, good for
  alignment") is now **measured on both the classical and the learned side**, and
  the design constraint it imposes is new: an adapter that improves rim depth
  must NOT perturb the rim features the pose path depends on. That argues for
  zero-init, late/readout-side corrections (RayTun3R-style PE residual or
  decoder-grid fixes) over anything that re-writes early rim features, and it
  adds a required eval metric: pose stability alongside rim depth for every
  adapter variant (the Pareto front gains a third axis).

## Lessons and Constraints

Learned inside this workspace:

- **Zone aggregates hide collateral damage** — run_010's near-center harm was
  invisible in pooled zones; always read the full (θ × depth) joint table.
- **Check the eval-of-record before locking protocol details** — run_008's
  scale_only vs fovbench's scale_shift cost a rerun.
- **Under per-frame re-alignment, local corrections move remote cells** (the
  affine couples them); when isolating a local effect, also evaluate with the
  affine frozen from the uncorrected prediction.
- **The hand-eye bootstrap works** (2.33° from factory calibration) — usable
  whenever a GT frame conjugation is missing locally.

Inherited from prior repo work — treat as hard constraints:

- **Verify, don't fit**: never tune unstated settings until a published number appears.
  Compare against independent references (classical pose harness, triangulated floors).
- **Denominator bugs are the dominant bug class** (5 instances in raytun3r): always
  check which pixels a mean is over; score competing methods on the intersection.
- **Depth conventions**: planar z vs euclidean range differ by up to 2.15x at the Aria
  rim; conversion happens once at a declared boundary (`native_depth` / `install`).
- **One frozen affine per frame before binning**; report spread (max/min), not
  rim-over-center (U-shape artifact of least-squares scale placement).
- **Layout confound**: widening per-view FOV without re-spacing the ring just adds
  dead pixels (110° arm was 38% dead). Design rule: `tilt + fov/2 ≈ 54.8°`.
- **alpha (rotation gain) is span-invariant; bare R_deg is not** — compare gains.
- **CPU/GPU split**: this Mac writes and verifies code (25s test suites, no weights);
  GPU evidence comes from lambda_63 via gpu-labelled GitHub issues. Don't claim GPU
  numbers from CPU work.

## Open Questions

- Q1: Does the rim actually help pose per-correspondence, or is it only "more pixels"? (→ H1)
- Q2: Can a radial adapter be made *provably* center-neutral (zero-init center bins), and does band-restricted loss buy rim accuracy? (→ H2)
- Q3: Does parameter-free KB4 patch resampling already recover most of the rim loss (RayTun3R ablation says patch undistortion alone is minimal — but that was whole-image; per-band unknown)? (→ H3)
- Q4: How big is the hand/dynamic-pixel problem in ADT, measured? (→ H4)
- Q5: What fraction of apparent radial depth degradation survives the distance control? (feeds every axis)

## Optimization Trajectory

No runs yet.

## The three measurements the report was missing, 2026-09-04

**Teacher-FOV sweep, re-measured, on two sequences.** The shape reproduces and
the magnitudes roughly halve:

| fov | cone cov | frame fill | rim-zone seen | near_rim seq131 | near_rim seq136 |
|---|---|---|---|---|---|
| 85° | 0.666 | 100% | 40.1% | **−23.49%** | −18.61% |
| 89° | 0.735 | 100% | 52.2% | −22.95% | −19.23% |
| 95° (adopted) | 0.837 | 98.7% | 70.4% | −14.68% | −9.55% |
| 110° | 1.000 | 77.5% | 100% | **+33.28%** | +39.31% |

Two things follow. (a) The black-corner inversion is real and reproduces on a
second sequence — an unseen image statistic costs more than the rim coverage it
buys. (b) **Accuracy and coverage trade against each other**: 85° is the best
teacher per pixel and sees 40% of the rim zone; 95° sees 70% and is worse where
they overlap. H14.2's ring is that trade at its limit — 99.3% coverage and
**+9.49%** at the rim, i.e. worse than the raw model, which is the whole reason
its student is worse in both held-out sequences.

The adopted 95° pre-check over the four training sequences: near_rim −14.68 /
−11.48 / −12.76 / −14.01%, near_centre +32.44 / +20.38 / +32.58 / +40.02%,
`roundtrip` control −0.04%. The premise holds 4/4 but is **thin**, and that is
the sufficient explanation for H14 losing P1.

**Cross-room is two sequences now.** `BlackCeramicBowl_seq030`'s depth finally
downloaded; extracted at stride 40 and scored through its own `camera.json`:

| sequence | frozen near_rim | `rect` (label-free) | `gt` (labelled) |
|---|---|---|---|
| DinoToy_seq030 | 0.1920 | **−28.5%** | −11.8% |
| BlackCeramicBowl_seq030 | 0.4970 | **−10.7%** | −5.0% |
| (apartment held-out seq136) | 0.4269 | −13.2% | **−57.9%** |

The ordering **inverts** when the room changes, on both sequences. `gt` fits one
room's depth and does not carry it; `rect` fits a function of the image and
does. This is the strongest evidence for the H14 idea in the record, and it is
orthogonal to H14's own pre-registered bar, which it still fails.

**Data scaling: more frames from one room buy memory, not generalisation.**
Labelled arm, 20 epochs, per-sequence frame cap raised 60 → 240 → 600:

| frames/seq | steps | seq136 rim | seq136 near_ctr | dec_132 rim | dec_132 near_ctr |
|---|---|---|---|---|---|
| 60 | 4,800 | −58.3% | −38.2% | **−27.3%** | +27.8% |
| 240 | 19,200 | −64.1% | −52.9% | −16.9% | +89.0% |
| 600 | 48,000 | **−69.3%** | −45.4% | −18.9% | **+100.7%** |

The same-room held-out sequence improves monotonically; the **rearranged** one
stalls at the rim and its near centre degrades to +100.7%. So "use the 97.9% of
the data we already have" is **not** the cheap answer to the overfitting
question — I had recorded it as the likely one. What the line needs is a second
room, and the LiteOffice rows above are the only measurement that supplies one.

**H9 closing state.** `raycal_shrunk` (evidence-weighted shrinkage toward
identity, driven by anchor count and depth spread, *not* by θ) holds near-centre
damage to −3.6…+8.4% on 5/6 sequences while keeping the rim win 5/6. The locked
bar stays **0/6**: the affine gap shrinks 6.1–25.5% under `raycal_shrunk`
(seq135 *grows* 4.3%), and at best 32.6% under `raycal_inv` on
decoration_seq132 — never halved. Correct summary: **an effective rim
recalibration, not a solution to the radial distortion.**

## Report re-check and the two follow-ups it produced, 2026-09-07

Re-verified that every number in `to_human/fisheye-rim-report.html` comes from
upright input: all training caches (`/netapp/datasets/f.zhang2/h14_teacher_cache_upright`,
git cfb8b09 / 8ff39d7 / f9e7cd4), checkpoints and eval JSONs under
`results/autoresearch-{h14,h15}-upright`, `h9-upright-v2`, `h16-orientation`,
`followups` were produced after 9d53963 (the second upright fix), and every
train/eval script forwards through `common/upright.py`. Nothing needed
re-running.

The report was reorganised so each experiment (A roll, B H14, C H14.2, D H15,
E H9, F generalisation) has idea / input-and-label examples / controls /
results / bar / "one layer deeper". New example figures on the same seq136
frame (`frame_001918`, index 1410; generator at `to_human/assets/report_figs.py`):

- **Roll** on that frame: near_rim 0.440 / 0.402 / 0.405 / 1.034 / 1.757 at
  0/10/20/30/40°; center 0.161 / 0.232 / 0.238 / 0.399 / 0.322. The error maps
  show what blows up past 30°: the near table top, hand and floor — near,
  upward-facing planes — while far walls barely move. Reading: the rolled
  frame breaks the model's gravity prior, and near ground planes are where
  that prior is load-bearing; the "rim penalty" at 30°+ is mostly that.
- **H14** on that frame, on the teacher's covered pixels: teacher near_rim
  0.273 vs raw 0.441 (a friendly frame; the 60-frame mean is −14.7%),
  near_rim coverage 75.5%.
- **H9** on that frame: 3,000 pixels matched in both partners → 1,775 anchors
  kept, 544 rejected by the agreement gate (textureless wall, floor, hand/tray
  edges; the palm never enters the candidate set — matcher weight is 0).
  **Anchor range error vs GT: median 4.1%**, against the model's 44% at the
  near rim on the same frame. The anchors are an order of magnitude better
  than the 32-coefficient curve they are spent on.

Two experiments designed from this, not run:

1. **Roll augmentation** (cheapest; reuses H14's trainer). Same random roll
   (±40°, about the principal point) applied to input and label. Control: the
   `gt` arm as is. Evaluate both at 0/20/30/40°. Bar: (i) the augmented arm's
   near_rim rise at 30° vs 0° at most half of the current +121%; (ii) its 0°
   near_rim no worse than the control by >2% — the gravity prior is useful at
   0° and augmentation may erase it; (iii) both held-out sequences. If (i)
   passes and (ii) fails, the finding is "equivariance vs gravity prior is a
   trade-off, pick by deployment pose distribution".
2. **Anchor distillation**: H9's metric anchors as sparse labels for H14's
   student. H14's bottleneck is a thin, holed teacher (−11…−15%); H9 has a 4%
   teacher that is sparse. Bars P1–P3 from H14, primary sequence
   decoration_seq132, plus: the student must also improve on the outermost
   ring where no anchors exist, otherwise it memorised the anchors. Known
   limit: anchors need motion (LiteOffice's static wearer starves them).

Also recorded: dec_seq132 is the only sequence that separates "learned the
lens" from "memorised the room"; every future bar should make it the primary
sequence and treat seq136 as a sanity check.

## Roll controls: the black wedge was two thirds of the fall-off, 2026-09-07

The roll curve (`fine_rotation.py`) rotates the fisheye frame with zero
padding, so the black region changes with angle — a model that dislikes hard
black edges traces the same curve as one that dislikes being rolled. Three arms
(`h16-orientation/code/roll_controls.py`, seq136, 20 frames, all scored on the
same theta<=44° disc; `results/autoresearch-h16-orientation/roll_controls_seq136.json`):

| roll | pinhole (no black) all / rim / rim÷ctr | fisheye_frame | fisheye_disc (hard mask, fixed) |
|---|---|---|---|
| 0° | 0.1545 / 0.3292 / 2.40 | 0.1623 / 0.3788 / 2.47 | 0.2037 / 0.5255 / 3.03 |
| ±20° | +13% / +10% / 2.4–2.6 | +28% / +34% / 2.6 | +60% / +88% / 3.5–3.8 |
| ±30° | +46% / +39% / 2.2–2.4 | +89% / +126% / 3.2 | +150% / +225% / 5.9–6.0 |
| ±40° | +131% / +138% / 3.0–3.6 | +193% / +303% / 4.9–5.1 | +200% / +260% / 5.8–6.5 |

- `pinhole` rolls the *virtual camera* of an 89° co-axial view (corner ray
  54.3° < 54.83° at every roll, fill 1.000) — no padding anywhere. This is the
  clean curve: not equivariant (+46% at 30°), but rim/ctr does not move until 40°.
  Centre and rim degrade together (30°: centre +44%, rim +39%).
- `fisheye_frame` is the old arm re-run with the rotation centre corrected to the
  upright principal point ((256.0, 256.5) → (246.0, 255.5); the old script used
  the stored-frame coordinates — ≤4 px translation at 40°, numbers change only in
  the third decimal). Its "rim doubles at 30°" is two thirds black wedge.
- `fisheye_disc` masks the frame to the inscribed disc about the principal point
  (r=245 px, 98.3% of the cone) at every angle. At **0°** that alone costs near_rim
  +38% (0.536 vs 0.389) and all +25%. A hard black edge is a bigger perturbation
  than a 20° roll. The arm then degrades faster than `fisheye_frame`, so it is a
  measurement of hard-edge sensitivity, not a clean roll curve.
- `resample0` (0° through the interpolator) = 0° to four decimals.

Consequences written into the report (§02, §00, §08): the gravity-alignment
decision stands (not worth it under 20°; if ever built, rotate the camera and
re-image, never the picture); the roll-augmentation design now specifies
camera-rolled rendering (`RolledView`) and the `pinhole` protocol for its bar
(30° near_rim +39% → ≤ +20%; 0° within 2% of control); and a standing rule:
**no hard black borders into the model** — no crops, masks, or zero padding —
which is the same mechanism that inverted H14's 110° teacher.

## Data plan, 2026-09-07

The 60-frames-per-sequence cap (2.1% of the frames) was a default inherited
from H5's trainer, never argued. Raising it within the Apartment (§07 ③) did
not buy generalisation because every frame is the same room. What is on disk:

- Apartment: 19 sequences downloaded (clean seq131–150 + decoration seq132),
  ~2,850 RGB frames each; only 6 have depth extracted. 13 more (clean
  seq137–150) have raw depth on disk and need `tools/extract_adt_sequence.py`.
  Also downloaded, unextracted: golden_skeleton_seq100, meal_skeleton_seq131,
  multiskeleton_party_seq101. ADT has 236 sequences but **two physical rooms**.
- LiteOffice: 2 sequences downloaded (~2,750 frames each), 60 extracted; 52
  sequences exist upstream, all the same room, device 61283.
- **ScanNet++ on /netapp: 1,018 scenes, DSLR fisheye (~115° FOV) with rendered
  depth, ~260 frames/scene, and a loader already in `raytun3r/data.py`
  (`ScanNetPPFisheye`).** This is the many-rooms fisheye+depth set the line
  has been missing.

Plan (in order):
1. Extract all frames of the 6 + 13 Apartment sequences and both LiteOffice
   sequences (CPU, hours); stride 5 for training (30 fps → 6 fps, ~570
   frames/seq), all frames for eval.
2. Held-out by *scene*, not by sequence: train on Apartment clean seq131–150
   (~11k frames at stride 5), hold out decoration_seq132 (rearranged),
   LiteOffice (second room, second device), and a ScanNet++ split.
3. Add ScanNet++ as the main training set for the labelled arm and as the
   cross-device held-out for the label-free arms: e.g. 200 scenes train /
   50 held-out, ~50k / 13k frames. Its lens is a different KB4 (115°), so it
   also tests H15's "real geometry is inert" on a real second lens.
4. Re-run H14 `rect` (label-free) on the enlarged Apartment + LiteOffice set —
   it needs no depth, so every RGB frame counts — with dec_seq132 and
   LiteOffice as the primary bars.
Frame counts at 240×20 epochs took 3 min; 11k×5 epochs is ~1 h, 50k×5 is ~4 h
on one RTX 6000 Ada — all feasible on lambda_63.

## H17.1 — what a real head actually does, 2026-09-08

Read from Aria's MPS closed-loop trajectory (gravity vector + device pose, 30 Hz),
not estimated: `h17-roll-prior/code/roll_distribution.py`, 60,105 frames over 19
Apartment + 2 LiteOffice sequences.

| | median &#124;roll&#124; | p90 | p99 | max | >20° | >30° | >40° |
|---|---|---|---|---|---|---|---|
| pooled | **3.7°** | 10.2° | 21.8° | 34.5° | 1.5% | 0.1% | **0%** |

Roll rate: median 5.4 °/s, p99 40.7 °/s (1.4 °/frame at 30 fps).

Two things fall out.

1. **The model's flat zone and the data's distribution coincide.** h16's clean
   (pinhole) curve is flat to ±20°; the data's p99 is 21.8°. 98.5% of real
   frames sit inside the tolerance. Measured independently, they match. On ADT
   the horizontal prior is not a bug — it is a prior matched to its deployment
   distribution. Per-sequence spread is real though (seq144/145: 26–30% of
   frames beyond 10°; decoration_seq132: 0.8%), so this is activity-dependent
   and does not transfer to a wearer lying down or working overhead.
2. **UPRIGHT_K=3 confirmed from geometry.** Of the four quarter-turn offsets
   only 270° puts the distribution on zero (median 3.7° vs 87–176° for the
   others). Independent of any loss.

Literature: [arXiv 2608.00678](literature/2608.00678-breaking-horizontal-prior.md)
reports the same "horizontal prior" on Marigold/GenPercept/DAv2/DistillAD, fixes
it with a training-time regulariser, uses **no** gravity/IMU input, and its best
*algorithmic* roll estimator is off by 25.9° — which is the argument for using
the IMU an egocentric device already has. No fisheye, no egocentric data, and no
measurement of the actual roll distribution anywhere in that paper.

## CORRECTION to the h16 roll write-up, 2026-09-08

The report claimed "two thirds of the fall-off is the black wedge". **That is
not supported and has been removed.** Relative rise of all-image AbsRel from
each arm's own 0°, at 30°:

| arm | 0° AbsRel | +30° | rise |
|---|---|---|---|
| `pinhole` (rectified, no black at any angle) | 0.1545 | 0.2292 | **+48%** |
| `fisheye_frame` (rotate the picture) | 0.1623 | 0.2949 | **+82%** |
| `fisheye_disc` (black region held FIXED) | 0.2037 | 0.5018 | **+146%** |

The arm that pins the black region down is the *steepest*, so steepness is not
"the wedge moves". `pinhole` is gentler because it feeds the model a rectified,
full-frame image — its native domain — and that change is confounded with the
absence of black. **Projection and boundary are not separated by these three
arms.**

Why the intended decomposition cannot work as designed: the imaged disc does not
fit inside the square frame (1.7% of the cone falls outside the inscribed
circle), so *any* roll performed in the fisheye frame disturbs the boundary, and
the model is hypersensitive to exactly that. To separate them, a fourth arm is
needed: a virtual fisheye camera whose cone is strictly inside its frame, rolled
about its optical axis. Its 0° point carries its own black annulus, but
comparisons *within* the arm across angles are clean.

What survives, and is clean:
- **Not roll-equivariant**: +48% all-image at 30° on a rectified full-frame
  input. Model property, nothing to do with fisheye.
- **`resample0`**: one bilinear pass is free (0.1955 vs 0.1954).
- **A hard black edge costs as much as a 20° roll**: masking 1.7% of the cone at
  0°, content otherwise untouched, costs +25% all-image and +38% near_rim. Same
  mechanism as H14's 110° teacher inverting at 22.5% black frame. Standing rule:
  **no hard black borders into the model** — no crops, masks, or zero padding.
- rim/ctr stays 2.2–2.6 to 30° on the rectified input and rises only at 40°;
  it does rise (2.47→3.24) on the fisheye input. "The rim penalty doubles under
  roll" is true of the deployed fisheye input, not of the model per se.

## Data ladder at CONSTANT gradient steps, 2026-09-08 (partial)

The earlier 60/240/600 ladder ran all three at 20 epochs, so data and compute
moved together (4.8k → 19.2k → 48k steps). Re-run with steps pinned near 48k:

| frames/seq × epochs | steps | distinct | 136 rim | 136 ctr | **132 rim** | **132 ctr** |
|---|---|---|---|---|---|---|
| 60 × 20 | 4.8k | 240 | −58.3% | −38.2% | **−27.3%** | +27.8% |
| 60 × 200 | 48k | 240 | −67.1% | −54.3% | −21.7% | +58.0% |
| 600 × 20 | 48k | 2,400 | −69.3% | −45.4% | −18.9% | +100.7% |
| 1200 × 10 | 48k | 4,800 | running | | | |
| 2880 × 4 | 46k | 11,418 | running | | | |

**At equal compute, 10x the distinct frames buys ~nothing** (seq136 −67.1 →
−69.3) and is *worse* on the rearranged room (−21.7 → −18.9), with near-centre
damage going +58% → +101%. Most of the apparent gain in the old ladder was the
10x in gradient steps, not the 10x in data. The best result on the honest
held-out sequence is still the smallest, shortest run (60 × 20, −27.3%).

One seed, no error bars; the dec_seq132 rim numbers all sit in a −17…−27% band,
while the near_centre trend is monotone in *steps*. Read the direction, not the
gaps.

## H17.3 — is roll legible in the frozen features? Between the bars, 2026-09-08

`h17-roll-prior/code/roll_probe.py`. Ridge from last-encoder-block tokens
(4x4 pooled, PCA 128) to the roll angle; 4 training sequences (1,600 samples),
2 held-out (800). Null = the identical pipeline with permuted training labels.

| arm | MAE | median | R2 | permuted null | seq136 | dec_seq132 |
|---|---|---|---|---|---|---|
| `camera` (no black at any angle) | **12.75°** | 7.72° | 0.600 | 25.8° | 6.8° | **18.7°** |
| `picture` (rotate the picture) | 12.76° | 8.25° | 0.600 | 27.3° | 6.7° | 18.8° |

Against the locked bars (<10 deg => conditioning inert; >20 deg => IMU carries
information the model lacks): **12.75 fires neither**. The median (7.7) is
inside, the mean is dragged out by dec_seq132.

Two things the protocol did not expect:

1. **The black-wedge control is falsified.** The two arms are identical
   (12.75 vs 12.76) where the prediction was 2x. Reason: Aria's imaged disc is
   near rotationally symmetric about the principal point, so rolling the
   picture barely changes the black region. This independently supports the h16
   correction above — the wedge was never what made the roll curve steep.
2. **The probe mostly learned the room**: 6.8 deg on the same-room held-out
   sequence, 18.7 deg on the rearranged one (null 25.8). So the features encode
   roll *within a known room* and only weakly in a new one. H15's "conditioning
   on what it already knows is inert" therefore does NOT transfer here — it does
   not already know, not in a new room. Conditioning keeps some headroom, but
   not enough to promote it.

**Methodological lesson, worth keeping.** The first version regressed
(sin, cos) and took atan2: MAE 70 deg, far worse than predicting a constant,
which reads like "roll is unreadable". It was a readout bug. Within +-45 deg
cos only spans [0.707, 1], its variance is tiny, R2_cos came out -72, and it
destroyed an already-decent sin fit (R2 0.61). Direct angle regression (no
wraparound in this range) took MAE from 70 to 12.75. **A linear probe cannot
legitimately do worse than the constant predictor** — the label-permutation
null is what made that visible, and it is now printed by default.

### Where this leaves the roll line

- Scientific value, holds up: the horizontal prior is real (§02, matches
  arXiv 2608.00678 on four other backbones); it is *matched to ADT's actual
  pose distribution* (§02b); the features encode roll only within-room (§02c).
- Engineering value on ADT: **small**. Only the 1.5% of frames beyond 20 deg
  can be helped by anything gravity-related.
- The bottleneck for this line is **a head-mounted dataset with a wider pose
  distribution and dense depth**, not a method. We do not have one.

## H17.2 — the horizontal prior is a PRETRAINING-DATA property, 2026-09-08

CONFIRMATORY, against the bar locked in `h17-roll-prior/protocol.md`.
`code/roll_backbones.py`, seq136, 20 frames, clean arm only (89 deg co-axial
view, virtual camera rolled, fill 1.000 at every angle), every backbone scored
on the same theta <= 44 deg disc, view size rounded to each backbone's own patch
size (VGGT-Omega is patch 16 -> 624; the rest patch 14 -> 630).

Rise in all-image AbsRel relative to **each backbone's own 0 deg**:

| backbone | pretraining | 0° AbsRel | ±10° | ±20° | **±30°** | ±40° |
|---|---|---|---|---|---|---|
| da3:small | single image | 0.1545 | +2% | +13% | **+46%** | +131% |
| da3:large | single image | 0.0690 | −4% | +10% | **+52%** | +114% |
| vggt | multi-view geometry | 0.0828 | +2% | +7% | **+11%** | +23% |
| vggt_omega | multi-view geometry | 0.0507 | +1% | +3% | **+11%** | **+11%** |

**Locked prediction: both VGGT variants rise less than DA3-Small at 30 deg, and
under +30%. Result: +11% and +11%. PASSES.** Neither falsification condition
fired (no VGGT variant rose more than DA3-Small; the four are not within 5 pp).

Three readings:

1. **It is the pretraining task, not the depth task.** Backbones trained on
   multi-view geometry — which see cameras at arbitrary orientations by
   construction — are 4-5x less roll-sensitive at 30 deg and 5-10x at 40 deg
   than single-image depth models. This extends arXiv 2608.00678, whose four
   models (Marigold, GenPercept, DAv2, DistillAD) are all single-image; they
   attribute the prior to the orientation long tail in photo collections but
   never test a backbone whose pretraining lacks that tail.
2. **Capacity does not fix it.** DA3-Large is 2.2x more accurate than DA3-Small
   at 0 deg and *more* roll-sensitive at 30 deg (+52% vs +46%). Scaling the
   single-image recipe does not buy roll robustness.
3. **The cheapest fix for roll robustness is the backbone, not the method.**
   Cheaper than augmentation, cheaper than IMU conditioning, and it comes with
   better absolute accuracy. It also deflates H17.4 for the VGGT family: at
   +11% at 30 deg with ADT's p99 at 21.8 deg, there is essentially nothing left
   for a gravity input to recover.

Caveats: one sequence, 20 frames, one seed; rim/ctr under this protocol is not
comparable to the section-01 table (different mask and projection). VGGT and
VGGT-Omega are run single-frame here, so their multi-frame machinery is idle —
the robustness is coming from the pretrained representation, not from fusion.

## H17.5 + dose curve — the border rule was the wrong shape, 2026-09-08

### H17.5: locked bar says "both contribute", but the design had a flaw

`code/roll_boundary.py`. Masking the 89 deg rectified view to its inscribed
disc and rolling as before:

| arm | 0° all | ±20° | ±30° | ±40° |
|---|---|---|---|---|
| `pinhole` | 0.1545 | +13% | **+46%** | +131% |
| `pinhole_masked` | 0.3178 | +33% | **+84%** | +97% |

+84% is between the locked thresholds (>=100 boundary, <=70 projection), so the
recorded verdict is **both contribute, neither dominates**.

**But the protocol's premise was wrong and this must be said.** It claimed the
inscribed-disc mask was "a near-exact analogue" of `fisheye_disc`. It is not:
the disc removes **21.9% of the rectified frame**, while `fisheye_disc` removes
**1.7% of the cone** (the fisheye frame's corners were already dark). Those
differ by an order of magnitude in severity, so the 30 deg comparison is not
the clean decomposition it was designed to be. What is clean is the 0 deg
number: masking 21.9% of a full rectified frame costs **+106% all-image /
+141% near_rim** — the largest boundary measurement in the project, and in the
model's own native domain.

### The dose curve: there isn't one

`code/border_dose.py`, same view, black border of width w, scored on
theta <= 30 deg (an 89 deg view's half-edge ray is 44.5 deg, so at h16's 44 deg
cap only 5 px of frame sit outside the scored region and no meaningful border
fits — the script refuses doses that would eat scored content).

| border | % of frame black | 0° all AbsRel | cost vs no border |
|---|---|---|---|
| 0 px | 0.0% | 0.1261 | — |
| 5 px | 3.1% | 0.1538 | +22.0% |
| 12 px | 7.5% | 0.1427 | +13.2% |
| 25 px | 15.2% | 0.1476 | +17.1% |
| 45 px | 26.5% | 0.1472 | +16.8% |
| 70 px | 39.5% | 0.1344 | **+6.6%** |
| 100 px | 53.4% | 0.1667 | +32.2% |

**There is no dose-response.** 3.1% black costs more than 39.5% black. The seven
non-zero doses scatter non-monotonically over +6.6…+32.2%, which is the noise
scale at 20 frames. The only supported statement is the *step*: **border present
vs absent, ~+18%.**

### So the standing rule had the wrong shape — corrected

The rule was written as if damage scaled with how much of the frame is black,
which is how H14's "22.5% black -> the teacher inverts" was read. The dose curve
says area is not the variable. **Distance is.** Compare like for like:

| where the score is taken | border | cost |
|---|---|---|
| theta <= 30 deg (14 deg inside the border) | 26.5% black | +17% |
| theta <= 44 deg (right against the border) | 21.9% black | +106% |

Same model, same projection, comparable black fraction, 6x different damage.
Within the dose experiment the same gradient is visible at every dose (`outer`
20–30 deg is consistently worse than `center` <= 11 deg). Corrected rule:

> **A hard border wrecks the region adjacent to it and costs the rest of the
> image a roughly constant ~18%. What matters is whether your zone of interest
> is near the border, not how much of the frame the border occupies.**

This is a better explanation of H14's 110 deg teacher than the one recorded:
its black corners sat directly against `near_rim`, the very zone it was supposed
to improve. It also means the 95 deg teacher's 98.7% fill was not "nearly
harmless because only 1.3% is black" — it was nearly harmless because that 1.3%
sits at the corners, away from most of the rim band.

### Roll interacts with the border

30 deg costs +47.6% on top of 0 deg with no border, and +64…+138% with one.
Borders make the model roughly 1.4–2.9x more roll-fragile — direction consistent
across all six doses, magnitude noisy and non-monotone like the rest of the
curve.

Caveats: 20 frames, one seed, one sequence, DA3-Small only. H17.2 showed roll
sensitivity is strongly backbone-dependent, so border sensitivity probably is
too and none of this is established for VGGT.

## H17.6 — border sensitivity is NOT a family property. Prediction falsified, 2026-09-08

`code/roll_boundary.py --models da3:small,da3:large,vggt,vggt_omega`, seq136,
20 frames. Mask the 89 deg rectified view to its inscribed disc and score on
theta <= 44 deg, i.e. **right against the border**.

| backbone | no border | + border | **cost all** | cost rim | roll@30 no border | roll@30 with border |
|---|---|---|---|---|---|---|
| da3:small | 0.1545 | 0.3178 | **+106%** | +141% | +46% | +84% |
| da3:large | 0.0690 | 0.5077 | **+636%** | +814% | +52% | +24% |
| vggt | 0.0828 | 0.1710 | **+107%** | +127% | +11% | −29% |
| vggt_omega | 0.0507 | 0.0655 | **+29%** | +53% | +11% | +10% |

**The locked prediction — both VGGT variants under +40% — is FALSIFIED.**
`vggt` pays +107%, a dead heat with DA3-Small. Only `vggt_omega` qualifies.

**Roll robustness and border robustness are different axes.** H17.2 split the
four cleanly by pretraining (single-image 46-52%, multi-view 11%). Border
sensitivity does not split that way at all:

    vggt_omega (+29%)  <<  da3:small (+106%) ~ vggt (+107%)  <<  da3:large (+636%)

So "multi-view pretraining buys robustness" is true for roll and **not** true
for borders. Whatever makes VGGT-Omega border-tolerant is not the thing it
shares with VGGT.

### The finding with the widest reach: DA3-Large's accuracy is brittle

At 0 deg on a clean frame DA3-Large is the best of the DA3 pair by 2.2x
(0.0690 vs 0.1545). Put one hard border on it and it becomes **the worst of all
four models** (0.5077, against DA3-Small's 0.3178 and VGGT-Omega's 0.0655) —
a 7.4x degradation from a change that touches no scored pixel. Anyone choosing
a backbone on clean-benchmark accuracy would pick DA3-Large and then lose an
order of magnitude to any real framing artefact: a letterbox, a crop pad, a
lens shade, a masked-out region.

### The door this opens for H14

The protocol pre-committed the decision: if both VGGT variants came in under
+40%, run a 110 deg teacher. Only VGGT-Omega did — so the door opens for it
alone, and it happens to be the strongest candidate anyway:

* border-tolerant (+29% where DA3-Small pays +106%);
* the most accurate backbone here at 0 deg (0.0507);
* roll-robust (+11% at 30 deg, H17.2).

**Next H14 arm: a 110 deg VGGT-Omega teacher.** 100% cone coverage, so H14's
accuracy-vs-coverage tension is dissolved rather than traded along, and it is
still label-free. Tested against `rect` (95 deg DA3) and `roundtrip` on the two
held-out sequences with decoration_seq132 primary.

### One anomaly, flagged not explained

`vggt` with a border gets *better* when rolled: masked 0 deg 0.1710 vs masked
+/-30 deg 0.1202/0.1238, i.e. −29%. Every other cell in the table has roll
costing something. Either the masked-0 deg case is anomalous for VGGT
specifically, or this is 20-frame noise. Not interpreted.

Caveats: one sequence, 20 frames, one seed. The DA3-Large effect (7.4x) is far
outside any plausible noise; the VGGT roll-with-border sign is not.

## H18 — the 110 deg VGGT-Omega teacher. H14's idea was right; its teacher was the problem. 2026-09-08

Bars locked in `h17-roll-prior/protocol.md` before training. Everything except
the teacher held at H14's shipped settings (4 sequences x 60 frames, 20 epochs,
seed 0, LoRA r=8). **No depth labels touch the teacher, the targets or the
student in any arm except `gt`.**

### Pre-check: the same view, only the backbone changes

| teacher | cone cov | rim-band cov | frame fill | near_rim vs raw DA3-Small |
|---|---|---|---|---|
| DA3-Small 110 deg | 100% | 100% | 0.775 | **+33.3% / +39.3%** (inverts) |
| **VGGT-Omega 110 deg** | **100%** | **100%** | 0.775 | **-65.0%** |
| VGGT-Omega 95 deg | 83.7% | 71.9% | 0.987 | -70.4% |
| DA3-Small 95 deg (H14 shipped) | 83.7% | 70.4% | 0.987 | -14.7% |

Identical geometry, identical 22.5% black frame; one teacher inverts and the
other is 4.4x stronger than H14's best. This is H17.6's border finding cashed
out: VGGT-Omega is the backbone that tolerates a border adjacent to the zone of
interest, so it can run the 110 deg view that covers the whole cone.

### Students

| arm | teacher | seq136 rim (near_ctr) | dec_seq132 rim (near_ctr) |
|---|---|---|---|
| `omega110` | VGGT-Omega, 110 deg **rectified** | **-52.7% (-34.5%)** | **-22.8% (+12.4%)** |
| `omega_rt` | VGGT-Omega, **on the fisheye** | -47.1% (-38.7%) | -17.9% (+2.0%) |
| `rect` | DA3-Small, 95 deg rectified | -13.2% (+43.3%) | +4.7% (-4.1%) |
| `roundtrip` | DA3-Small, on the fisheye | -0.1% | +0.0% |
| `gt` | dense depth labels | -57.9% (-39.0%) | -27.2% (+24.4%) |

### Bars

1. **Beat the control on both held-out sequences - PASS.** Against the DA3
   `roundtrip` (-0.1 / +0.0) trivially, and against the *matched* control
   `omega_rt` (-47.1 / -17.9) on both. This is H14's original P1, which `rect`
   failed on decoration_seq132.
2. **Recover at least half the labelled gain - PASS, by a lot.** 91% on seq136,
   84% on dec_seq132. `rect` managed 23% and a negative fraction.
3. **near_center not worse than +10% - PASS on seq136 (-34.5%), FAIL on
   dec_seq132 (+12.4%), by 2.4 points.** The labelled `gt` arm degrades the
   same zone by +24.4% on that sequence, so the label-free arm is *gentler*
   than the ceiling on this axis; the bar is still failed as written.
4. dec_seq132 primary - reported first throughout.

### The control is what makes this readable

`omega_rt` - the same teacher, the same two resamplings, no change of
projection - was added because without it the result is uninterpretable.

* **Most of the gain is "a stronger model was distilled"**: the fisheye teacher
  alone recovers **81% / 66%** of the labelled gain.
* **The rectified projection adds a further 5.7 / 4.9 points** at the rim
  (10% / 18% of the labelled gain), consistent in direction on both sequences.
* It also **causes** the bar-3 failure: `omega_rt` holds near_center at
  -38.7% / +2.0% where `omega110` is -34.5% / +12.4%. The rectification buys
  rim and spends near-centre.

So H14's original mechanism (rectify to unlock what the model already knows) is
**real but second-order**; teacher strength is first-order. That is a different
claim from the one H14 set out to make, and it is the one the data supports.

### What the claim now is, precisely

Not self-distillation any more. "**A frozen stronger model can be distilled
into a small fisheye-domain model with no depth labels at all, recovering
66-84% of what dense ground truth buys at the near rim; rectifying the view
first adds another 10-18%.**" It needs a stronger model to exist, which
self-distillation did not - that cost has to be stated whenever this is quoted.

Caveats: one seed; the student is DA3-Small throughout; the teacher runs
single-frame so VGGT-Omega's multi-frame machinery is idle; `omega_rt` resizes
504 -> 512 -> 504 because a patch-16 teacher cannot take a 504 px frame and
padding was not an option (h17's dose curve), a 1.6% rescale against `resample0`
measuring one bilinear pass as free.
