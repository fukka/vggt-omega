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

## H18 follow-up — three seeds, and a cross-room result that INVERTS the reading, 2026-09-08

### Three seeds

| arm | seq136 rim | dec_seq132 rim | dec_seq132 near_ctr |
|---|---|---|---|
| `omega110` | −51.5 ± 2.0 | −22.4 ± 1.8 | **+16.4 ± 3.9** |
| `omega_rt` | −47.6 ± 1.1 | −17.8 ± 1.0 | +6.6 ± 4.3 |

Bar 2 (>= half the labelled gain) passes on all three seeds; **bar 3
(near_center <= +10%) fails on all three** (12.4 / 20.1 / 16.9), so that failure
is robust, not a one-seed artefact. The rectification advantage at the rim is
+3.9 / +4.6 points in the mean and favours `omega110` on 6/6 seed-sequence
cells, but one seed nearly ties (49.2 vs 48.9 on seq136), so it is directionally
consistent and small rather than decisive.

### Cross-room, and this is the part that changes the story

Same seed-0 checkpoints, evaluated on LiteOffice (other room, other device,
scored with its own `camera.json`). near_rim:

| arm | DinoToy_seq030 | BlackCeramicBowl_seq030 |
|---|---|---|
| `omega110` (VGGT-Ω, **rectified**) | **−20.0%** | **−9.0%** |
| `omega_rt` (VGGT-Ω, on the fisheye) | **+0.8%** | **−0.1%** |
| `rect` (DA3-Small, rectified) | **−28.5%** | **−10.7%** |
| `gt` (dense labels) | −11.8% | −5.0% |

**The arm that carried most of the in-room gain transfers nothing.**
`omega_rt` goes from −47.6% in-room to +0.8% / −0.1% across rooms. Every
rectified arm transfers; the un-rectified one does not.

So the decomposition **inverts** between the two tests:

| | in-room (held-out sequence) | across rooms |
|---|---|---|
| teacher strength alone (`omega_rt`) | −47.6% — first-order | **≈ 0** |
| adding the rectified projection | +3.9 points — second-order | **the entire effect** |

**This corrects what I published an hour ago.** The earlier write-up said
"teacher strength is first-order, H14's mechanism second-order" without the
qualifier. That is true **in-room only**. Across rooms it is the other way
round, and the cross-room test is the one this project has repeatedly found to
be the honest one.

### What it suggests, stated as a hypothesis and not a result

Both targets are deterministic functions of the image, so "determinism" is not
the distinction. The plausible one: the rectified path teaches a *systematic
radial relation* — the same geometric correction everywhere, which is a
low-dimensional thing a 122.9k-parameter LoRA can only represent globally. The
fisheye path teaches "be like VGGT-Ω on these images", a much
higher-dimensional target the student can fit room-specifically. Untested.

### A trade-off falls out

The **weaker** DA3-Small rectified teacher transfers *best* (−28.5% / −10.7%),
better than the far stronger VGGT-Ω rectified teacher (−20.0% / −9.0%), while
being far worse in-room (−13.2% / +4.7% vs −52.7% / −22.8%). Stronger teacher →
more in-room gain, slightly less transfer. Both rectified arms beat the labelled
ceiling across rooms, which extends the earlier cross-room reversal to a much
stronger teacher.

### Where that leaves the claim

* **In-room**: a frozen stronger model distilled label-free recovers 82-89% of
  the dense-label gain at the rim; the rectified projection is a small part of
  that.
* **Across rooms**: only the rectified projection survives, and it beats dense
  labels. "Distil a stronger model" does not transfer at all.
* Anyone quoting the in-room number without the cross-room one gets the
  mechanism backwards.

## H18.2 — everything that transfers is 16 numbers. Locked prediction passes hard. 2026-09-08

Replace the 122.9k-parameter LoRA with something that can only express a radial
relation, fit it on the four training sequences **only**, apply it unchanged
everywhere. `code/radial_probe.py`, supervision = the same `omega110` teacher
targets the LoRA saw, scale-aligned the same way. No depth labels anywhere.

near_rim AbsRel, relative to the frozen model:

| sequence | frozen | `global` (2 params) | **`radial` (16 params)** | `omega110` LoRA (122,900) |
|---|---|---|---|---|
| seq136 — held out, same room | 0.4269 | +8.7% | **−16.6%** | **−51.5%** |
| decoration_seq132 — rearranged | 0.2320 | +6.5% | **−6.5%** | −22.4% |
| **DinoToy_seq030 — other room** | 0.1920 | +2.7% | **−26.0%** | −20.0% |
| **BlackCeramicBowl_seq030 — other room** | 0.4971 | −1.6% | **−8.9%** | −9.0% |

**Locked prediction: `radial` reproduces at least half of `omega110`'s
cross-room gain and beats `global` on both LiteOffice sequences. It reproduces
130% and 99%, and beats `global` everywhere. PASSES, emphatically.**

`global` — the same functional form with the theta dependence removed — is
**worse than doing nothing** on three of the four sequences. So the transferable
content is radial *structure*, not a rescale. That was the falsification
condition and it did not fire.

### The finding

**Sixteen numbers carry everything that crosses a room. The other 122,884
parameters buy a large in-room gain that transfers not at all.**

* in-room the LoRA is 3.1x / 3.4x better than the radial curve;
* across rooms the radial curve **matches or beats it** (−26.0 vs −20.0,
  −8.9 vs −9.0).

That is the mechanism the cross-room inversion was pointing at, now measured
rather than hypothesised. The rectified teacher's transferable content is a
per-theta recalibration of the frozen model's depth — a property of the **lens**,
which is why it carries to another room and another device. The LoRA's extra
capacity fits the **room**.

Fitted curve: `log(target) = a(theta) * log(pred) + b(theta)`, with a running
1.337 → 1.468 → 1.341 from centre to rim and b falling monotonically
+0.012 → −0.252. A power law near 1.4 with a rim-dependent offset.

### Convergent validity with H9 — two unrelated label-free sources, one object

H9 fit the same functional form from **parallax anchors** and got seq136
0.4269 → 0.3613 (−15.4%, `raycal_inv`) / 0.3724 (−12.8%, `raycal_shrunk`).
This fits it from a **rectified teacher** and gets 0.4269 → 0.3560 (−16.6%).

Two supervision sources with nothing in common — triangulated metric anchors
from camera motion, and a stronger model's predictions under a change of
projection — converge on the same 16-parameter object and the same magnitude.
H9 additionally needed per-sequence test-time adaptation; this one curve is fit
once and applied to a different room unchanged.

### What this changes practically

A 16-number radial calibration can be shipped instead of a LoRA, and it
transfers **better**. The LoRA is the right choice only when the deployment room
is the training room. Neither needs a depth label.

Caveats: one fit, no seeds (the fit is least squares on 240 frames, so seed
variance is not the relevant uncertainty; frame sampling is). Cross-room is
still 120 frames over two near-static sequences and is now carrying two claims
rather than one. `radial` beating `omega110` on DinoToy suggests the LoRA's
room-specific fitting actively costs transfer, which is consistent but untested.

## H18.3 / H18.4 — the 16 numbers are NOT a lens constant. Withdrawing yesterday's practical claim. 2026-09-08

### H18.3 (locked): the transfer works, the coefficients do not

Fit on LiteOffice (2 sequences, 120 frames), apply to the Apartment:

| bar | result | verdict |
|---|---|---|
| recover >= 60% of the Apartment-fitted near_rim gain | seq136 78%, dec_seq132 95% | **PASS** |
| mean abs difference in `a(theta)` < 0.10 (falsify > 0.20) | **0.493** | **FAIL by 2.5x the falsification line** |

`a(theta)` fitted on the Apartment is smooth and unimodal
(1.337 → 1.468 → 1.341); fitted on LiteOffice it is jagged
(0.596, 0.951, 1.263, 1.228, 0.896, 0.737, 0.793, 0.915). The global exponents
differ too: 1.368 vs 0.891.

### Exploratory follow-up: is that just the fitting set?

A log-log slope is not identified independently of the range it is fitted over,
and LiteOffice spans 0.41–4.66 m against the Apartment's 0.44–10 m. So both fits
were redone restricted to predicted depth in 0.5–4.5 m (a filter on the
prediction, so still label-free):

| fitted on | seq136 | dec_seq132 | DinoToy | Bowl |
|---|---|---|---|---|
| Apartment, unrestricted | −16.6% | −6.5% | −25.9% | −9.0% |
| Apartment, range-matched | −16.6% | −6.5% | −25.9% | −9.0% |
| LiteOffice, unrestricted | −12.9% | −6.2% | −18.2% | −3.9% |
| **LiteOffice, range-matched** | **−7.3%** | **+1.2%** | −13.1% | −2.7% |

**The Apartment fit does not move at all; the LiteOffice fit gets worse and turns
harmful on decoration_seq132.** So range matching does not rescue it — the
LiteOffice-fitted curve is simply unstable to how its fitting pixels are chosen,
and 120 frames of near-static capture is not enough to estimate the object. The
Apartment fit is not fragile in the same way.

### H18.4 (locked): backbone-specific, as predicted

Same Apartment targets, DA3-Large as the frozen student instead of DA3-Small.
Predicted mean abs `a(theta)` difference > 0.20 (i.e. it IS backbone-specific);
**measured 0.297. PASSES.** DA3-Large's own curve is smooth
(1.063 → 1.144 → 1.109), shape-correlated with DA3-Small's at 0.763 but sitting
about 0.30 lower. It works on its own baseline: −15.8% / −12.8% / −18.9% / −5.0%.

So the *shape* of the curve is largely shared across backbones and the *level*
is not.

### CORRECTION to what I published one tick ago

The last report said: **"ship a 16-number radial calibration instead of a LoRA;
it transfers better."** That is not supported and is withdrawn.

What survives:

* **A per-theta curve fitted on a sufficient set transfers.** 240 Apartment
  frames with real motion and a wide depth range give a curve that carries to
  another room and another device, matching a 122,900-parameter LoRA there.
  That result (H18.2) stands and is unaffected by range matching.
* **The numbers are not a constant.** They are specific to the backbone
  (H18.4, mean |da| 0.297) and they depend on having a good fitting set
  (H18.3's LiteOffice curve is unstable).
* So it is a **calibration procedure**, not a lens constant: fit 16 numbers per
  (lens, backbone) on a few hundred frames with motion, and they then transfer
  to new rooms. Useful, and much weaker than what I wrote.

This is the second claim I have had to walk back in this session — the first was
the black-wedge attribution in section 02. Both were published before the
control that would have caught them existed. In both cases the control was
cheap and I ran it one tick later.

## H18.6 (CPU, during the outage) — the 16 numbers are really 6, and smoothness is a free fit-quality check

Purely descriptive: fit a low-order polynomial in theta/theta_max through the
eight per-bin coefficients already computed, and ask how much of each curve it
captures. No GPU, no new data.

| fit | a: R^2 quadratic | b: R^2 quadratic | max residual as % of that curve's own spread |
|---|---|---|---|
| Apartment, DA3-Small | 0.936 | 0.993 | a 13.9%, b 3.3% |
| Apartment, DA3-Large | 0.989 | 0.979 | a 5.9%, b 7.1% |
| LiteOffice, raw | 0.301 | 0.329 | a 38.3%, b 33.1% |
| LiteOffice, range-matched | 0.590 | 0.071 | a 33.0%, b 43.7% |

Two things follow.

**1. The working curves are smooth, so 16 numbers are really 6.** A quadratic in
each of `a` and `b` captures 94–99% of the Apartment fits' variation. If that
holds up, the calibration procedure ships **three coefficients for the exponent
and three for the offset**, not sixteen bin values — and a smooth form
extrapolates to bin edges instead of stepping across them.

**2. Smoothness is a fit-quality check that costs nothing and needs no held-out
data.** The LiteOffice curves — which H18.3 showed transfer badly and which I
described as "jagged" from eyeballing the numbers — are quantitatively not
smooth: a quadratic misses a third to a half of their variation. So "is a
quadratic enough?" flags a bad calibration **at calibration time**, before any
evaluation. That is exactly what a deployment procedure needs, since in
deployment there is no held-out room to check against.

**Caveat, and it is not small.** These are eight points fitted with three
parameters, so the absolute R^2 values are optimistic. The signal here is the
**contrast** — 0.94–0.99 against 0.07–0.59, consistent across four independent
fits — not any single number.

**Prediction this generates, for the box (added to H18.5's protocol).** If the
LiteOffice curve's jaggedness is noise rather than structure, then *smoothing it
with the quadratic form should improve its transfer*. If smoothing does not help,
the jaggedness is real and the smooth-form story is wrong. Either answer is
worth having and neither has been tested.

## H18.3 follow-up (CPU) — the depth-range confound I flagged is NOT what separates the two fits

H18.3 recorded a worry: a log-log slope is not identified independently of the
depth range it is fitted over, LiteOffice spans 0.41–4.66 m against the
Apartment's 0.44–10 m, so comparing their coefficients might compare the fitting
sets rather than the lenses. I ran the range-matched refit but never actually
checked whether that worry was operative. It is not.

Effect of restricting the fitting pixels to predicted depth 0.5–4.5 m:

| fit | max change in `a` | as % of that curve's own spread |
|---|---|---|
| Apartment | **0.0057** | 4.4% |
| LiteOffice | **0.5653** | 84.8% |

**The Apartment fit's mass was already inside 0.5–4.5 m** — restricting it
changes nothing, which is why the range-matched Apartment gains were
bit-identical. So the two fitting sets do **not** differ in effective depth
span, and the confound is not the explanation. The finding above stands but its
stated reason was wrong.

**What actually differs**: a large part of LiteOffice's fitting mass sits where
the frozen model predicts **outside the scene's own 0.41–4.66 m range** — that
is, on its own gross errors. Restricting the range removes exactly that mass.

### And a puzzle worth recording rather than explaining away

Removing it makes LiteOffice's coefficients look **more** Apartment-like in the
inner bins (bin 0: 0.596 → 1.161, bin 2: 1.263 → 1.522, against the Apartment's
1.337 and 1.468) while making its transfer **worse** (−12.9% → −7.3% on seq136's
rim). The outer bins, which are the ones near_rim actually scores (bin centres
44.6 and 51.4 deg), barely move: 0.793 → 0.835 and 0.915 → 0.939, both far below
the Apartment's 1.400 and 1.341.

So the rim correction is weak in both versions, and the transfer difference is
coming from the **inner** bins. The plausible route is the evaluation's per-frame
scale+shift alignment: changing the centre changes where the affine sits, which
changes the rim residual. This project has measured that channel before — 82% of
one sequence's rim penalty came from affine placement — so it is not a new
mechanism, but it has not been checked here.

**Testable when the box returns**, and cheap: score the same corrected
predictions under `scale_only` and under a frozen affine. If the −12.9 / −7.3
gap collapses, the difference was alignment, not correction.

## H17.1 follow-up (CPU) — the −2.7° bias is not a calibration constant, and it comes in a block

§02c recorded an uncertainty and left it: the roll distribution is not symmetric,
signed median −2.7°, and *"a constant offset cannot be distinguished from a real
head tilt by this measurement"*. That was wrong — it can, and the data to do it
was already in the same JSON.

A constant extrinsics offset lives in `T_device_camera`, which is **per device**.
It would give one fixed bias on every sequence recorded with M1292. A physical
tilt — the wearer's habit, or how the glasses happened to sit that session — is
per session and should vary.

Signed median roll per sequence, all on **the same device** (M1292):

| block | sequences | signed median |
|---|---|---|
| seq131–141 | 9 sequences | −2.74 … +0.44 |
| **seq142–148** | **7 sequences** | **−4.65 … −7.50** |
| seq149, 150, dec132 | 3 | −2.16 … −3.83 |

Mean −3.27°, **sd 2.38°, range +0.44 to −7.50 — an 8-degree spread within one
device.** A calibration constant cannot do that. **The hypothesis is dead.**

And it is not per-sequence noise either: **seq142–148 form a contiguous block**
that sits 4–5° further tilted than everything before it. That shape points at a
per-session cause — a different wearer, or the device re-seated between recording
sessions — not at per-frame behaviour.

This also sharpens §02c's other statement. The sequences flagged there for having
26–30% of frames beyond 10° (seq144, seq145) are **inside that block**. So the
"activity-dependent" reading was close but not right: what varies is a
**per-session mounting/posture offset**, and the frames beyond 10° are that
offset plus ordinary head motion on top.

**Practical consequence: none, on this data — and that is worth saying.** A
per-session constant de-roll is much cheaper than per-frame gravity alignment and
would remove a systematic 5–7° on a third of the sequences. But §02's clean curve
prices 10° at +2%, so 6° is worth well under 1%. The finding closes an
uncertainty and explains a pattern; it does not buy accuracy.

The two LiteOffice sequences (device 61283) sit at +0.43 and −2.81, mean −1.19 —
consistent with the same picture, but two sequences say nothing on their own.

## H18.6 follow-up (CPU) — the caveat asked for an honest criterion; it widens the gap

H18.6 recorded its own limit: "eight points fitted with three parameters, so the
absolute R^2 values are optimistic; the signal is the contrast". Leave-one-out
answers that directly — the held-out bin never touches the fit, so extra
parameters buy nothing.

| fit | coef | in-sample R² | **LOO R²** |
|---|---|---|---|
| Apartment, DA3-Small | a | 0.936 | **0.708** |
| Apartment, DA3-Small | b | 0.993 | **0.961** |
| Apartment, DA3-Large | a | 0.989 | **0.948** |
| Apartment, DA3-Large | b | 0.979 | **0.884** |
| LiteOffice, raw | a | 0.301 | **−1.984** |
| LiteOffice, raw | b | 0.329 | **−2.078** |
| LiteOffice, range-matched | a | 0.590 | **−0.808** |
| LiteOffice, range-matched | b | 0.071 | **−3.381** |

The reference is the LOO R² of a **constant** predictor, which is −0.306 at n=8.

**The honest criterion widens the gap rather than closing it.** In-sample the
contrast was 0.94–0.99 against 0.07–0.59; under LOO it is **0.71–0.96 against
−0.81 to −3.38**. Every Apartment curve genuinely interpolates a bin it never
saw. Every LiteOffice curve predicts a held-out bin **worse than the mean of the
other seven** — there is no smooth curve there to interpolate, only scatter.

**This also turns the fit-quality check into a calibrated one.** H18.6 proposed
"is a quadratic enough?" as a red flag that needs no held-out room. It now has a
threshold that needs no tuning either: **LOO R² above the constant-predictor
baseline (−0.306 for 8 bins) means there is a curve; below it means there is
not.** Both quantities come from the fit alone.

Caveat that remains: eight bins is few, and LOO on eight points is itself noisy.
What carries weight is that all four Apartment measurements land above the
baseline and all four LiteOffice ones land below it, with no overlap.

## H18.5 pre-study (CPU) — the simulation failed, and the failure is the useful part

With the box down indefinitely I tried to answer H18.5's mechanistic question —
"is the binding constraint pixels or viewpoints?" — by characterising the
estimator synthetically (`code/radial_estimator_sim.py`). It does not work, and
recording why is worth more than the numbers.

Generative model: each frame is a scene view whose pixels occupy a narrow band
of log-depth, plus a per-frame bias (scene-level) and i.i.d. per-pixel noise.
`spread` draws band centres across the whole depth range, `single` from a narrow
slice.

| frames | `spread` mean pairwise \|da\| | `single` | ratio |
|---|---|---|---|
| 2 | 0.113 | 0.048 | 0.43 |
| 60 | 0.0067 | 0.0060 | 0.90 |
| 240 | 0.0061 | 0.0045 | 0.75 |

**Both cross \|da\| < 0.10 at 2–4 frames.** The real data needs somewhere
between 120 (insufficient) and 240 (sufficient). The simulation is off by two
orders of magnitude, so **it does not reproduce the phenomenon and its ordering
prediction cannot be trusted.**

It also inverts the ordering I expected: `single` comes out *more* stable than
`spread` at every count. That is explicable inside the model — a per-frame bias
is a pure intercept shift, and once band centres are spread out those biases
become correlated with `log(pred)` within a bin and alias into the slope. Real
or not, it is a property of my model, not evidence about ADT.

**What the failure says, and this is the part to keep.** A per-frame intercept
plus i.i.d. pixel noise **cannot** produce "120 frames insufficient, 240
sufficient". Real frames must carry far more correlated structure than that —
most plausibly the model's *errors* are spatially structured within a frame (a
whole wall predicted wrong together), which is neither an intercept nor
independent noise. So:

* H18.5's real stability curve will be governed by something this model lacks;
* **analytic or simulated sample-size estimates should not be trusted for this
  problem** — the effective sample size per frame is far below any naive count;
* the pixel-count sweep (500 → 60,000 px/frame moved \|da\| from 0.0147 to
  0.0049, roughly root-n) is equally model-dependent and says nothing real.

Kept out of the human report: it is a negative result about my own instrument,
not about the problem.

## H18.5 — the sweep ran, and it falsified its own pre-registered bar

The runner finished on GPU0 (`autoresearch/experiments/h14-rect-distill/results/omega/radial_sweep.json`,
5 draws per cell, seed 0). The locked bars were: *stability reaches mean pairwise
|da| < 0.10 by 60 spread frames; spread beats single at every matched count;
falsified if single matches spread.*

```
  mode  n_fr   stab      seq136        dec132          Dino          Bowl
spread     2   0.485   -11.0+-12.9    -1.4+-13.3   -20.4+-14.8    -8.0+- 2.5
spread     4   0.237   -13.0+- 5.5    -3.0+- 5.2   -21.9+- 3.2    -8.4+- 0.4
spread     8   0.085   -15.1+- 2.6    -5.0+- 0.9   -23.1+- 3.6    -8.5+- 0.9
spread    15   0.190   -15.5+- 3.9    -5.3+- 4.3   -24.7+- 4.3    -8.8+- 0.6
spread    30   0.069   -13.9+- 2.1    -4.3+- 1.5   -23.0+- 2.3    -8.6+- 0.5
spread    60   0.081   -15.3+- 1.9    -5.3+- 1.7   -24.7+- 1.9    -8.8+- 0.1
spread   120   0.025   -16.4+- 0.5    -6.2+- 0.5   -25.9+- 0.6    -9.0+- 0.1
spread   240   0.000   -16.6+- 0.0    -6.5+- 0.0   -26.0+- 0.0    -8.9+- 0.0
single     2   0.530   -17.1+- 8.7     0.3+-18.2   -23.0+-12.9    -8.6+- 1.8
single     4   0.225   -15.8+- 5.0    -5.3+- 3.4   -25.4+- 5.1    -8.8+- 2.0
single     8   0.152   -12.7+- 3.1    -1.6+- 2.8   -21.9+- 3.1    -8.4+- 0.5
single    15   0.141   -16.7+- 2.9    -6.4+- 3.5   -26.1+- 3.3    -9.0+- 0.2
single    30   0.060   -16.1+- 1.1    -6.6+- 0.6   -25.4+- 1.6    -8.7+- 0.5
single    60   0.000   -17.9+- 0.0    -7.1+- 0.0   -27.9+- 0.0    -9.3+- 0.0
```

**Bar 1 passed, and by a wider margin than asked.** |da| < 0.10 arrives at 8
spread frames, not 60. By 30 frames every sequence sits within ~1–3 points of
the full-set fit.

**Bar 2 failed, which was the pre-registered falsification condition.** Across
the five non-degenerate matched counts, single wins three (4, 15, 30) and spread
wins two (2, 8). Single *matches* spread. **H18.5 is refuted.**

Read the degenerate rows carefully before quoting them: `spread_240` and
`single_60` exhaust their own pools, so every draw is the same draw and their
0.000 is arithmetic, not convergence. The `single_60` vs `spread_240` transfer
comparison (-17.9 vs -16.6 on seq136) is therefore one point estimate against
another with no error bar on either. It is suggestive, not measured.

**What this rules out.** The mechanism the CPU pre-study modelled — that within-
bin `log(pred)` spread is what identifies the slope, so frames from different
viewpoints buy precision that same-viewpoint frames cannot — is not what governs
the real estimator. The simulation (see the section above) predicted a large
spread/single gap that grows with viewpoint separation. There is no gap. The
pre-study was internally consistent and empirically wrong, which is the outcome
it was explicitly built to be checkable against.

**What replaces it.** Frame count alone, and it saturates by ~30. The most likely
reason single does as well: a single ADT sequence is a person walking around a
room, so consecutive frames already sweep a wide depth range — the "narrow slice
of depth" the simulation assumed a single viewpoint sees does not describe a
head-worn recording. That is a hypothesis, not a measurement; it predicts the
gap should reappear on genuinely static footage, which LiteOffice nearly is.

**Practical consequence, and it is the useful one.** The radial calibration costs
about 30 frames — a second of video — from one ordinary recording. That is a very
different deployment story from "collect varied footage per device".

**Still open, and now the blocking question:** every fit here was scored against
ground truth afterwards. Whether the curve can be fitted from the teacher alone
on an unseen device is what decides deployability.

## H19 — native calibration loses to a foreign curve. What the fit needs is motion.

Ran on GPU0, both configs (`h19-native-calib/results/native_{unmatched,matched}.json`).
Full reading in `h19-native-calib/analysis.md`.

The deployment question was: fit the radial curve on the device it will run on,
teacher-supervised, no ground truth anywhere. The answer is that this is worse
than using a curve fitted on a *different* device — by about 10x.

| fitted on | device | footage | → DinoToy | → Bowl |
|---|---|---|---|---|
| `apartment`, 240 fr | wrong | walking | **-26.0%** | **-8.9%** |
| `bowl`, 60 fr | right | near-static | -2.6% | — |
| `dino`, 60 fr | right | near-static | — | -3.8% |

**The evaluation frames are identical across these rows.** The only thing that
differs is the fitting set. So this is a direct comparison with no confound in
the scoring: wrong-device-with-motion beats right-device-without.

**B2 failed**, which was the pre-registered falsification condition, on both
sequences and in both configs.

**B3 passed**, and it supplies the mechanism. A 30-frame fit's mean pairwise
|da|: 0.060 from one walking Apartment sequence (H18.5), against 0.100-0.111 for
Bowl and 0.127-0.132 for DinoToy. Near-static footage is **half as informative
per frame**. H18.5 explained its own refutation by pointing at exactly this and
predicted the spread advantage would reappear on static footage. It did, on a
test it did not have to survive.

**B1 is not established.** Radial beats its `global` control on both
cross-sequence cells unmatched (3.0 and 2.9 points against a 2-point bar), but
`bowl`->DinoToy inverts under range matching and radial goes 3.5 points *worse*.
A margin that flips with a fitting-range choice is not a result. Do not quote it.

### This corrects something already published

Both reports said the calibration costs "about 30 frames, and they can all come
from a single recording". Wrong as stated. It needs **30 frames with motion in
them**, and having the right device does not substitute for that. Both reports
amended; the English one carried the claim in its recommendations, which is the
worst place to be wrong.

### Lesson for the constraints list

"N frames is enough" is never a property of N alone. H18.5 measured 30 on
walking footage and the number does not survive a change in what the frames
contain. Any future "how much data" answer here has to name the content of the
data, not just its size.

### The confound, stated plainly

LiteOffice is the only second device available AND the only near-static footage
available. "Target-device fitting is bad" and "this footage is static" cannot be
separated by this experiment. B3 measuring staticness directly, and the failure
being that lens's *own* curve, is why the staticness reading is preferred.
Moving footage from a third device would settle it.

### The reading it opens

`apartment` transfers to a different lens better than that lens's own static
footage does. That points at the curve being substantially **device-independent**
— closer to a one-off fit on any well-moving footage than to a per-device
calibration. That is a much better deployment story than the one H19 set out to
test, and it is now the thing worth testing.

## H20 — the mechanism survives a within-device control, and becomes a dose-response

`h20-motion-split/results/motion_split.json`, full reading in that experiment's
`analysis.md`. All three pre-registered bars pass.

### H20a — "near-static" was never measured. Now it is.

Camera centre per frame from GT pose (`C = -R^T t`):

| sequence | spread from centroid | path length |
|---|---|---|
| Apartment seq131 / 133 / 134 / 135 | 1.52 / 1.94 / 1.97 / 1.61 m | 38 / 46 / 39 / 36 m |
| **DinoToy** | **0.105 m** | **2.34 m** |
| **BlackCeramicBowl** | **0.210 m** | **2.43 m** |

9-19x in spread, 15-20x in path, no overlap. The Apartment wearer walks ~40 m
and the LiteOffice wearer moves 2.4 m in total.

**This closes a hole rather than opening one.** H19's published mechanism rested
on calling LiteOffice near-static, and that label had been inherited from H9's
remark about a static wearer starving the parallax anchors — never measured in
this line. It was right, but it was an assumption until now.

### H20b — same device, same room, only the motion differs

| arm | camera spread | \|da\| | seq136 | dec132 | DinoToy | Bowl |
|---|---|---|---|---|---|---|
| `high` | 2.021 m | 0.069 | -20.0 +- 1.1 | -9.4 +- 1.4 | -30.5 +- 1.0 | -9.3 +- 0.2 |
| `low` | 0.499 m | 0.377 | -11.5 +- 9.0 | -1.4 +- 9.4 | -21.1 +- 7.6 | -8.9 +- 0.7 |

5.5x worse stability, worse transfer on all four sequences, and the draw-to-draw
spread explodes (+-9.0 against +-1.1). **The mechanism published on 2026-09-08
survives a control that holds the device fixed**, so H19's confound is closed
without a third device.

### The upgrade: a dose-response, not a binary

Rim gain on DinoToy against camera travel. First three rows are the same camera
in the same room:

| fitting set | spread | -> DinoToy |
|---|---|---|
| Apartment, 30 most spread out | 2.021 m | -30.5% |
| Apartment, all 240 | 1.768 m | -26.0% |
| Apartment, 30 least spread out | 0.499 m | -21.1% |
| LiteOffice Bowl, its own 60 | 0.210 m | -2.6% |

Monotone across a 10x range of camera motion, three points device-matched.

**Practical form: the fitting frames should span roughly 2 m of walking.** That
is a specification someone can act on; "30 frames with motion in them" was not.

### Two traps this leaves behind

**A single low-motion fit looks fine.** The `low` arm's mean a(theta) over five
draws is a smooth arch (1.454 -> 1.630 -> 1.412). Averaging hides what
|da| = 0.377 reports. Eyeballing one fit would not catch this — H18.6's LOO R^2
check is the thing that does, and this is a second reason to keep it.

**|da| is not comparable across arms with different draw structure.** `low`
re-draws a random anchor, so its fits sample different corners of the apartment;
LiteOffice has one place to be, so its draws agree with each other while
agreeing on something wrong. Compare transfer, not |da|, across those two.

### Lesson for the constraints list

Two "how much data" answers in this line now needed a qualifier that was not in
the number: H18.5's 30 frames needed "with motion", and H19's "motion" needed
"about 2 m of it". Each time the number alone was published first and the
qualifier arrived from the next experiment. The general form is in the H19
entry; H20 is the second instance, which makes it a pattern rather than a slip.

## H21 — the curve is not lens-specific. "A property of the lens" was wrong.

`h21-lens-identity/results/lens_identity.json`, full reading in that
experiment's `analysis.md`. Four lenses over the Aria cone via H15's lens family;
every arm warped once from the real Aria camera so resampling blur is common.
Costs no teacher inference — the cache stores range and `grid_between` resamples
by ray, so range is invariant under a fixed-cone lens change.

**B3 passes**: `global` is worse than nothing on every diagonal (+2.3 to +5.2%)
against `radial`'s -21 to -36.6%. The setting is sound.

**B2 refuted** — the pre-registered falsification. Cross-lens minus same-lens is
**+0.84 points** (seq136) and **+0.47** (dec132), both inside H20's +-1.0
draw-to-draw sd. **Matching the lens buys nothing.**

**B1 marginal, do not lean on it**: mean pairwise |da| 0.0783 against a 0.069
floor, with two of six pairs BELOW the floor (aria-stereographic 0.0224,
equidistant-equisolid 0.0459).

### What dominates instead: which geometry you fit on

Row means span **7.5 points** (aria_kb4 -24.2, stereographic -27.3, equidistant
-30.6, equisolid -31.7 on seq136) against a 0.84-point matching effect —
**about 9x**. And `equisolid`'s curve is the best curve for *every* target lens,
beating the real Aria lens's own curve by **5.8 points** (-26.8% vs -21.0%) and
4.4 on dec132.

Likely mechanical rather than deep: equisolid allocates more image radius to
high theta than KB4, so its fitting set holds more rim pixels — and the rim is
where a(theta) is least determined and where the score is taken. Same object,
better estimated. That is the same shape of explanation as H20's motion result:
what limits the fit is information about the rim.

### The restatement both reports needed

Wrong: "a recalibration of depth against viewing angle — a property of the lens".
Right: **a property of how the backbone's error grows with incidence angle,
largely independent of how the lens distributes pixels across that angle.**

Calibration is therefore a genuine one-off, not a per-camera step.

### A new actionable recommendation

**Resample the footage to a rim-stretching geometry before fitting**, then apply
the curve to the real images. Worth 5.8 points on the real Aria lens here, for
free. This is the first recommendation in this line that improves a result rather
than only constraining it.

### Two limits, recorded rather than discovered later

Synthetic re-renderings, not four real cameras: same footage, cone, optics, MTF,
vignetting and native resolution throughout. Lens-independence is established for
**radial mapping** — which is what the curve is indexed by — not for everything
"lens" covers.

Coefficients here are NOT comparable to H18.2's: every arm went through a
resample, so a(theta) runs 1.6-2.1 where the unwarped fit ran 1.34-1.47.
Within-experiment comparisons only.

### Third time for the same pattern

H18.5's "30 frames" needed "with motion"; H19's "motion" needed "about 2 m";
now H18.2's "property of the lens" needed "of the angle, not the lens". Each
time the published phrasing was tighter than the evidence, and the next
experiment supplied the qualifier. This is now a standing rule, not an
observation: **before publishing a characterisation, name the axis it was never
varied along.**

## H22 — Spearman +1.000, and H21's mechanism was backwards in the direction I corrected

`h22-rim-density/results/rim_density.json`, full reading in that experiment's
`analysis.md`. Six lens shapes, same machinery as H21.

### B1, B2, B3 all pass

| fitted on | rim area share | row mean (same room) | (rearranged) |
|---|---|---|---|
| orthographic | 68.3% | **-30.5%** | **-15.6%** |
| equisolid | 73.5% | -29.0% | -14.1% |
| equidistant | 75.0% | -27.7% | -13.3% |
| stereographic | 77.9% | -24.7% | -11.1% |
| the real lens | 79.4% | -21.8% | -9.3% |
| rectilinear | 86.6% | -11.3% | -2.4% |

**Spearman rho = +1.000 on both recordings.** The geometry giving the rim the
FEWEST pixels makes the best fitting set — the opposite of what H21's analysis
claimed, and the direction I had already corrected analytically before running.

### The recommendation, stated at the right strength

On the real Aria lens: equisolid -26.8%, orthographic -26.2%, equidistant
-26.0%, its own curve -21.0%. Any rim-compressing geometry is worth ~5-6 points;
**the 0.6 points between equisolid and orthographic is inside noise, so B2
passing does NOT license naming orthographic.** The finding is "fit on a
rim-compressing geometry", not a specific one.

### Where "matching buys nothing" holds — and where it stops

The raw diagonal beats the off-diagonal (-27.6 vs -23.5), which looks like it
contradicts H21. It does not: good fitting geometries are also easy targets, so
the diagonal is confounded by correlated row and column effects. The clean test
is an additive model `M[i,j] ~ mu + r_i + c_j` and the diagonal's residual:

  orthographic 8.5 | rectilinear 8.3 | equisolid 1.3 | real lens 1.1 |
  stereographic 0.7 | equidistant 0.5      (residual sd 2.98)

Among the four **realistic fisheye shapes, including the real lens**, matching is
worth 0.5-1.3 points — inside noise. **5 of 6 target lenses do better with
somebody else's curve.** H21's conclusion holds and is better supported. It
stops holding at the two extremes, where a lens's own curve is worth 8 points:
those renderings sit far enough outside the fisheye family that only their own
curve describes them.

Variance: fit geometry 49.8%/36.7%, target 39.4%/54.0%, rest ~10%.

### Mechanism — a hypothesis, deliberately not a finding

I got this wrong once, so it is labelled. Rim-EXPANDING geometries upsample the
rim: more pixels, interpolated from the same samples, so no new information plus
blur. Rim-COMPRESSING ones downsample: fewer pixels, each a real average. On that
reading the fit is limited by rim pixel QUALITY, not count — the same shape of
explanation as H20's motion result.

**A competing explanation this cannot rule out**: row and column effects are
ordered the same way, so the ranking may just be "how badly this rendering was
damaged by resampling", hitting fitting and scoring alike. The recommendation
survives either way because the row effect is measured against fixed targets;
the mechanism does not.

Falsifiable: if it is upsampling loss, rendering the source at higher resolution
before warping should shrink the spread. If it is the angular distribution
itself, it should not.

### On the standing rule

The rule added after H21 — *before publishing a characterisation, name the axis
it was never varied along* — worked as intended this time. H21's mechanism was
caught by computing r(theta)/r(theta_max) with no data at all, before H22 ran,
and the protocol opened by correcting it rather than testing around it.

## H23 — not upsampling loss either. Two mechanisms down, effect intact.

`h23-upsample/results/{src504,src1008}.json`, reading in that experiment's
`analysis.md`. ADT frames are natively 1408x1408 and this line had always read
them at 504, so a rim-expanding warp at 504 was inventing detail sitting in the
file. Two arms, byte-identical supervision, only the source resolution differs.

| fitted on | rim area | @504 | @1008 | change |
|---|---|---|---|---|
| orthographic | 68.3% | -28.4% | -27.0% | +1.4 |
| equidistant | 75.0% | -25.8% | -24.2% | +1.6 |
| the real lens | 79.4% | -20.1% | -18.1% | +2.0 |
| rectilinear | 86.6% | -10.5% | -8.6% | +1.9 |
| **spread** | | **18.0** | **18.5** | **+0.5** |

**B1 refuted** (bar: spread shrinks >=30%, falsified <10%) — it grew, on both
recordings. **B2 refuted** — rectilinear did not gain more than orthographic;
the drift is a near-uniform +1.4 to +2.0 across all four. **B3 passes**, so the
arms really did differ only in rim detail.

### The count of dead mechanisms is now two

1. "More rim pixels make a better fitting set" — H22, refuted, it is the exact
   opposite.
2. "Rim-expanding geometries lose because their extra pixels are invented" —
   H23, refuted.

**The hypothesis recorded in H22's analysis is withdrawn.**

### What the residual now has to be

Not pixel-level detail: doubling the available detail changes nothing. It has to
be about **where the angular content sits across the image radius** — same rays,
same sharpness, different arrangement. That rhymes with the border result, where
position in the frame mattered and area did not.

**No third guess is offered.** Two have been refuted on this one effect, and the
standing rule about naming the unvaried axis applies to mechanisms as much as to
characterisations.

### The recommendation survives and gains a robustness check

The monotone ordering reproduces at both source resolutions with the same slope,
so "fit on a rim-compressing geometry" is not an artefact of reading at 504.

Incidental worth keeping: at 1008 every geometry is ~1.5-2 points worse on the
same room. A uniform offset, not a differential — the sharper input sits further
from what the 504-derived teacher cache represents. Matters only if the cache is
ever re-derived at higher resolution.

### An experiment NOT to run as designed

Separating "where content sits" from "how much angle each pixel covers" by
scaling the same lens into a smaller disc introduces a border, and the border
result says that is catastrophic. Confounded by construction. Recorded as an
open problem rather than queued.

## H24 — the ordering generalises to another model. Most of the prize does not.

`h24-backbone-generality/results/da3_large.json`, reading in that experiment's
`analysis.md`. Identical to H22 except the depth model.

**B1 passes**: Spearman(rim area share, row-mean gain) = **+0.943** (same room)
and **+1.000** (rearranged), bar +0.8. The one inversion is orthographic and
equisolid swapping by 0.2 points. The ordering is a property of the task, not of
DA3-Small — and it holds on a model whose framing sensitivity is wildly
different (border alongside the scored region: +636% vs +106%).

**B2 passes**: on the real Aria lens the best foreign curve beats the lens's own
by **+9.1** points (equisolid -17.2 vs own -8.1) and **+4.3** (orthographic -2.2
vs own +2.1). The same-room margin is larger than DA3-Small's 5.8.

**B3 fails on rectilinear**, both recordings: global (+1.9 / +5.6) beats radial
(+4.6 / +9.9). Both positive — the sanity check fails in the one cell where the
method has collapsed for this model. Recorded rather than explained away.

### What the bars nearly hide

| fitted on | rim area | Small | Large | Small (rearr.) | Large (rearr.) |
|---|---|---|---|---|---|
| orthographic | 68.3% | -30.5% | -13.5% | -15.6% | **-0.7%** |
| equisolid | 73.5% | -29.0% | -13.7% | -14.1% | +0.8% |
| equidistant | 75.0% | -27.7% | -11.3% | -13.3% | +1.0% |
| stereographic | 77.9% | -24.7% | -8.1% | -11.1% | +1.0% |
| the real lens | 79.4% | -21.8% | -5.4% | -9.3% | +2.6% |
| rectilinear | 86.6% | -11.3% | +3.6% | -2.4% | +5.0% |

**On the harder recording DA3-Large's method sits at zero.** Best geometry
-0.7%, everything else positive. The ordering is perfectly monotone around
nothing. So B2 "passing" there is -2.2% against +2.1% — *slightly helps* versus
*actively hurts*. A relative bar cleared while the absolute numbers hover at
zero, which is exactly the kind of pass that reads better than it is.

### Three claims at three strengths

1. **Which geometry to fit on** generalises. Strong.
2. **Preferring a foreign rim-compressing curve over the lens's own** generalises.
   Solid.
3. **The size of the prize** does not. Less than half on the easy recording,
   essentially none on the hard one.

Headroom arithmetic, offered as such and NOT as a mechanism: DA3-Large starts
with half the rim error of DA3-Small (0.191 vs 0.389). A correction that removes
a systematic component should do less when there is less of it to remove.

### Consequence for both reports

The recommendation generalises **as a rule about which geometry to pick** and not
**as a promise about how much you gain**. Anyone applying it should fit and check
against ground truth on their own model first — "it helped a lot on DA3-Small"
does not carry, and on the most rim-expanding geometry it turns actively harmful.

### A pattern worth naming, again

This is the third time a result in this line was reported by a relative measure
that passed while the absolute quantity was near zero or reversed (H9's anchors,
H19's native fits, now H24's rearranged room). **Always print the absolute
numbers next to the relative ones**; every time this has bitten, the relative
number was the one that looked publishable.

## H25 — the headline in-room number was a 5.7-sigma outlier

`h25-sequence-variance/results/seq_variance.json`, reading in that experiment's
`analysis.md`. The Apartment-fitted 16-number curve, **unchanged**, scored on
thirteen never-used held-out recordings (seq137, 138, 140-150) plus the two
originals.

| | near-rim gain | centre damage |
|---|---|---|
| **13 never-used sequences** | **-7.02% +- 1.67** | +26.8% +- 12.3 |
| seq136 (the published number) | **-16.6%** -> **-5.74 sd** | +41.7% (+1.21 sd) |
| dec_seq132 | -6.5% -> +0.31 sd | +23.5% (-0.27 sd) |

**B1 passes** (sd/|mean| = 0.238, bar 1/3): sequence noise is small relative to
the effect, and all thirteen are negative. **B3 passes 13/13**: the 2-parameter
`global` control is worse than nothing on every one. So the effect is real.

**B2 is refuted, and not marginally.** seq136 sits **5.74 sd** above the mean of
thirteen recordings from the same room and the same device. The representative
value is **-7.0%, not -16.6%** — the published figure is **2.4x the typical one**.

It is not that seq136 is unusually hard or easy: its frozen rim error is +0.51 sd
from the thirteen-sequence mean, i.e. ordinary. It is specifically unusually
**responsive to this correction**.

### The uncomfortable pattern

seq136 has been THE held-out sequence for this whole line, and every in-room
number in both reports is quoted on it. Meanwhile **dec_seq132 — the recording
repeatedly described as "the harder test" — is the representative one** (+0.31
sd). The number that looked like a pessimistic check was the honest one; the
number that looked like the result was the outlier.

### Scope limit, stated up front

**Only the radial curve was re-measured.** The 122,900-parameter LoRA's -51.5%,
H18's student arms and every other in-room figure were also measured on seq136
and have NOT been re-measured. Nothing here shows they are inflated; nothing
here shows they are not. So any ratio between two arms both measured on seq136 —
including "the adapter is three times better in-room" — is now **untested**,
because both terms may scale together. That is the immediate next run.

### Centre damage was understated, in the other direction

The curve damages the near-centre zone by **+26.8% +- 12.3** across the thirteen
against seq136's +41.7%, so here seq136 was pessimistic by 1.2 sd. The damage is
large on every sequence and its spread (7.2% to 47.9%) is much wider than the
rim gain's.

### The methodological lesson, which is the real output

**A single designated held-out sequence, chosen once and quoted everywhere, is
not a held-out set.** Thirteen recordings that had been sitting extracted and
unused for one day were enough to show the number this line rests on is an
outlier in the flattering direction. Every future in-room claim here should be a
mean over these thirteen with an sd, never a single sequence.

This is the third standing rule, and it subsumes the second: printing absolute
numbers beside relative ones does not help if the absolute number itself comes
from one lucky recording.

## H26 — the adapter was inflated too, and the conclusions come out stronger

`h26-lora-variance/results/`, 52 evaluations (3 `omega110` seeds + `omega_rt`
control × 13 recordings). Checkpoints unchanged. Reading in that experiment's
`analysis.md`.

| | published (seq136) | thirteen recordings |
|---|---|---|
| adapter `omega110` | -51.5% | **-38.59% +- 4.97** |
| control `omega_rt` | -47.6% | -34.92% +- 5.67 |
| 16-number curve (H25) | -16.6% | -7.02% +- 1.67 |
| adapter / curve | 3.10x | **5.50x** |
| adapter - control | -3.9 pts | -3.66 pts |

**B1 fails at 2.60 sd** — the adapter figure was inflated by the same recording,
but less than half as badly as the curve's 5.74 sd.

**B2 fails in the opposite direction to the worry.** The ratio went UP, from
3.10x to 5.50x, because the curve was flattered about twice as much as the
adapter. **The published number understated the adapter's in-room advantage.**

**B3 passes 13/13 — the one that mattered.** `omega110` beats the un-rectified
control on every single recording, gap -3.66 points against seq136's -3.9. H18's
central in-room reading went from one recording to thirteen without moving.

### Which recording beats which seed

Three `omega110` seeds average -39.44, -39.62, -36.72 — a range of 2.9 points
against a within-seed sequence sd of ~5. **Which recording you score on matters
more than which seed you train.** Every future run here should budget for that.

### Incidental: the student improves the centre

`omega110` improves near-centre by **-31.4%** on average rather than damaging it,
the opposite of the 16-number curve's +26.8% (H25). Consistent with the student
being trained on the teacher's depth everywhere while the curve is a per-angle
correction fitted to rim-dominated statistics. **Not** the same measurement as
the data-ladder table's +58% to +145%, which comes from rim-weighted arms; the
two must not be quoted against each other.

### Still untested

The **"89% and 82% of what real labels buy"** ratio divides by a GT-trained arm
also measured on seq136 and not re-run. Given that everything on seq136 is
inflated and by different factors, that ratio should not be quoted until the
denominator is re-measured.

### Net

Two published headline numbers were wrong and both are corrected downward. All
three claims resting on them survive or strengthen: the adapter beats the curve
by more than published, the rectified teacher beats its control 13/13 instead of
1/1, and the effect is consistently signed everywhere tried. The correction cost
two numbers and bought a much firmer floor.

## H27 — the correction arc closes: every number down, every ratio up

`h27-gt-arm/results/`, 45 evaluations. Reading in that experiment's
`analysis.md`.

**B1 fails at -3.12 sd**: the GT arm's -57.9% is inflated too. That is **all
four** arms — nothing measured on seq136 was representative.

**B2 passes in the favourable direction**: the label-free fraction is **94.2%**
over thirteen recordings against the published 88.9%. The headline sentence was
**understated**, not wrong.

**B3 passes 13/13 with the full three-seed control**: gap -3.44 points against
H26's single-seed -3.66. H18's central comparison is not a one-seed accident.

### Every in-room number, corrected

| arm | published (seq136) | thirteen recordings | sd out | inflation |
|---|---|---|---|---|
| trained on real labels | -57.9% | **-40.99% +- 5.42** | 3.12 | x1.41 |
| adapter omega110 | -51.5% | **-38.59% +- 4.86** | 2.60 | x1.33 |
| control omega_rt (3 seeds) | -47.6% | **-35.14% +- 5.86** | — | x1.35 |
| 16-number curve | -16.6% | **-7.02% +- 1.67** | 5.74 | x2.36 |

| ratio | published | thirteen recordings |
|---|---|---|
| label-free / label-trained | 88.9% | **94.2%** |
| adapter / curve | 3.10x | **5.50x** |
| adapter - control | -3.9 pts | -3.44 pts, **13/13** |

### The pattern in the inflation column, which is a finding in itself

The three **learned** arms were inflated by nearly the same factor (1.33, 1.35,
1.41); the **16-number curve** by **2.36**, almost twice as much. That is why
every ratio improved instead of degrading — seq136 flattered the weakest method
most.

It also says something usable: **a 16-parameter fit on rim-dominated statistics
is far more sensitive to which recording you score on than a trained adapter
is.** sd/|mean| = 0.238 for the curve against 0.126 for the adapter. Any future
claim about the curve needs more recordings than a claim about the adapter does.

### What this arc did and did not close

**Closed.** Every in-room figure is now a mean over thirteen recordings with an
sd, in both reports. All three conclusions that rested on seq136 survived; two
improved.

**Not closed.** The rearranged-room ratio (82%) still rests on dec_seq132 alone
— much less worrying, since H25 showed it is the representative recording
(+0.31 sd), but it is one recording and should be described as one.

**Untouched.** The cross-room numbers were never measured on seq136, so this arc
does not reach them. They remain 120 frames over two near-static recordings and
are now, unambiguously, **the largest open exposure in this line.**

## H28 — the foundation holds; two published numbers do not

`h28-foundation-variance/`, both runners unchanged on the same thirteen
recordings. Reading in that experiment's `analysis.md`.

### The asymmetry, which is the main result

H25-H27 found seq136 inflated **every method** by x1.33 to x2.36. H28 finds it
was **ordinary for the phenomenon**:

| | thirteen recordings | seq136 | |
|---|---|---|---|
| rim / centre, DA3-Small | 2.171 +- 0.382 | 2.13 | **-0.11 sd** |
| rim / centre, DA3-Large | 2.127 +- 0.476 | 2.04 | -0.19 sd |
| raw rim error, DA3-Small | 0.396 +- 0.056 | 0.389 | ordinary |
| roll penalty +-20 deg | +12.4% +- 6.7 | +12.8% | +0.07 sd |
| roll penalty +-30 deg | +40.4% +- 11.1 | +46.3% | +0.52 sd |
| roll penalty +-40 deg | +91.7% +- 16.8 | **+130.6%** | **+2.32 sd** |

**That recording flattered what we built, not what we measured.** The opening
sections of both reports were resting on sounder ground than the distillation
sections were.

**B2 passes** (-0.11 sd). **B3 passes both halves** at +-30 deg (sd/mean 0.275
against a 1/3 bar; seq136 +0.52 sd). **B1 is marginal** — 0.176 against a pass
bar of 0.15 and a falsification bar of 0.25 — and is recorded as undecided
rather than rounded either way.

### Two numbers to restate

**"The rim is 2.0-2.6x worse than the centre, in every model."** That interval
was never a spread across models. It was four numbers, each from ONE recording,
that happened to land close together. Across thirteen, DA3-Small runs
**1.39-2.65** and DA3-Large **1.14-2.98**. The mean survives (2.17, 2.13 against
the published 2.13, 2.04); the tight interval does not. Honest form: **about
2.2x on average, spread roughly 1.4 to 2.7 between recordings.**

**"+131% at +-40 deg."** Representative value is **+92% +- 17**. This is the only
place in the orientation section where seq136 is not representative.

### What is unaffected

* the rim is about twice as bad as the centre on every recording tried;
* +12% -> +40% -> +92% is still a steep curve, and H17's DA3-vs-VGGT gap was a
  ratio between models on the same frames, so a shift in the common baseline
  does not touch it;
* **+-20 deg stays cheap** (+12.4%), which is what H17's "98.5% of real frames
  sit inside the comfortable range" depends on.

Caveat the spread adds: at +-20 deg the sd is 6.7 on a mean of 12.4 and the
range reaches 21.3%. "+-20 deg is nearly free" is true on average, not on every
recording.

### Gotcha

The roll arm failed on all fourteen sequences with `argument --angles: expected
one argument`. Values beginning with `-` are read by argparse as flags. The
runner's DEFAULT `-40,-30,...` works because it never passes through argparse;
passing the same string explicitly does not. Use `--angles=-40,...`, never a
space.

## H29 — the border warning survives 13/13; its numbers were the worst outliers yet

`h29-border-variance/`, both runners unchanged on the same thirteen recordings.
Reading in that experiment's `analysis.md`.

**B2 passes 13/13.** DA3-Large's border cost exceeds DA3-Small's on every single
recording. **B3 passes 13/13.** A 70 px border blacking out 39.5% of the frame
costs LESS than a 5 px border blacking out 3.1%, everywhere. The non-monotonic
dip that is the entire evidence for "proximity, not area" is completely robust.

| border | % of frame | thirteen recordings | seq136 |
|---|---|---|---|
| 5 px | 3.1% | +19.5% +- 13.0 | +22.0% |
| 12 px | 7.5% | +14.7% +- 10.2 | +13.2% |
| 25 px | 15.2% | +18.1% +- 11.8 | +17.1% |
| 45 px | 26.5% | +21.8% +- 15.6 | +16.8% |
| **70 px** | **39.5%** | **+7.3% +- 4.9** | +6.6% |
| 100 px | 53.4% | +35.6% +- 20.9 | +32.2% |

**B1 fails by the largest margin in the whole campaign.**

| | thirteen | seq136 | |
|---|---|---|---|
| DA3-Small border cost | +59.0% +- 14.1 | +105.7% | **+3.32 sd** |
| DA3-Large border cost | +268.9% +- 61.5 | +635.5% | **+5.96 sd** |
| ratio Large / Small | **4.56x** | 7.4x | |

**+5.96 sd beats H25's 5.74 sd.** Both published border figures are extreme
outliers and the headline "7.4x collapse" is really 4.56x.

### This refines H28's asymmetry rather than confirming it

H28 concluded seq136 "flattered what we built, not what we measured". But the
border test IS a measurement and here seq136 is +3.3 and +6.0 sd. The sharper
statement, offered as an observation and NOT a mechanism (two mechanism guesses
have already been refuted in this line):

**seq136 responds unusually strongly to INTERVENTIONS.** Corrections applied to
it help unusually much (H25-H27); perturbations applied to it hurt unusually
much (H29). Its UNPERTURBED structure is ordinary (H28). That covers all five
experiments without special pleading, and predicts that any future intervention
measured on seq136 alone will read high.

### What changes

**Not the advice.** "Never put a hard black border next to the region you care
about" and "cost is proximity, not area" are 13/13 — better supported now than
when published.

**Three numbers.** +106% -> +59% +- 14; +636% -> +269% +- 62; 7.4x -> 4.56x. A
4.6x collapse from a change touching no scored pixel is still the strongest
practical warning in either report.

**One phrase.** "DA3-Large is the best of the four on a clean frame and the worst
with a border" was four backbones on one recording; H29 re-measured two. The
"worst of the four" half is now supported for the two tested and untested for
VGGT and VGGT-Omega.

## H30 — the "squeeze before you fit" gain holds 13/13; the prediction landed weakly

`h30-lens-variance/results/lens_13.json`. Curves fitted on the same four
training sequences from the same cache; only the evaluation set changed.

**B1 passes 13/13.** On the real Aria lens a foreign rim-compressing curve beats
the lens's own on every recording.

| | thirteen recordings | seq136 |
|---|---|---|
| the real lens's own curve | -14.9% +- 3.2 | -21.1% |
| best rim-compressing curve | -18.9% +- 4.4 | -26.8% |
| **advantage** | **+4.03 +- 1.33 pts** | +5.75 pts |

The advice stands and is worth **about 4 points, not 5-6**, and is never
negative on any recording. **B3 passes**: orthographic -26.5% <= equisolid
-25.2% <= aria_kb4 -18.4%.

### B2 — the pre-registered prediction was right, and one test is weak

H29's observation (seq136 responds unusually strongly to interventions) was
stated, not measured. H30 pre-registered that seq136's advantage would sit more
than 1 sd above the thirteen-recording mean. **It does: +1.29 sd.**

**That single result is weak** — P(z>1) is about 16%, so p ~ 0.16 alone. What
makes the observation credible is the campaign:

| intervention on seq136 | z | | unperturbed | z |
|---|---|---|---|---|
| H25 radial curve | **+5.74** | | H28 rim/centre | -0.11 |
| H26 adapter | +2.60 | | H28 roll +-20 | +0.07 |
| H27 GT arm | +3.12 | | H28 roll +-30 | +0.52 |
| H29 border small | +3.32 | | | |
| H29 border large | **+5.96** | | | |
| H30 lens advantage | +1.29 | | | |

**Six interventions, all positive** — sign test p ~ 0.016 before counting that
two exceed 5 sd. **Three unperturbed measurements, all at zero.** That split is
what the observation rests on; H30 contributes one more point on the correct
side, pre-registered, which is the most a single run can be.

Still an observation, **not a mechanism**: it says what seq136 does, not why.

### The standing consequence

**Any intervention measured only on seq136 should be assumed inflated.** Every
published in-room figure in this line has now been re-measured; this is the
instruction for anything measured in future.

The converse — that a *descriptive* number on seq136 is representative — held
every time it was checked, but rests on three data points and should not be
leaned on.

## H31 — the method choice was sound; the verification campaign closes

`h31-teacher-precheck/results/`, 42 pre-checks. Reading in that experiment's
`analysis.md`.

**B1 passes 13/13.** `omega_wide` beats `da3_wide` at the rim on every
recording. The bar was deliberately strict — a single failure falsifies, because
a method choice should not rest on a majority vote — and it never failed.

**B2 passes 13/13.** `da3_wide` is worse than doing nothing everywhere, +6.9% to
+31.1%.

| config | thirteen recordings | seq136 | published |
|---|---|---|---|
| DA3-Small, 95 deg narrow | -6.9% +- 6.2 | -9.9% | -15% |
| DA3-Small, 110 deg wide | **+19.8% +- 7.9** | +35.2% | +33% |
| VGGT-Omega, 110 deg wide | **-48.2% +- 7.2** | -69.1% | -65% |

Section 4.3's teacher was chosen correctly, and the reason — the wide view is
unusable for DA3 and fine for VGGT-Omega — now rests on thirteen recordings.

**Reproduction gap, flagged not smoothed:** the two decision-critical arms
reproduce closely on seq136 (+33 -> +35.2, -65 -> -69.1) but the narrow arm does
not (-15% published, -9.9% reproduced). Likely the frame count, 20 here against
60 originally, on the arm with the smallest effect. Nothing depends on that
number, but a 5-point gap should not go unmentioned.

### The pattern gets its proper form

H31 supplies two interventions with OPPOSITE signs, which forces a better
statement. `da3_wide` hurts and seq136 reads +1.95 sd MORE harmful; `omega_wide`
helps and seq136 reads 2.91 sd MORE helpful. So the rule is not "reads high":

**seq136 amplifies the magnitude of whatever you do to it, in whichever
direction the intervention points.**

All nine interventions in this campaign are amplified — |z| = 5.74, 2.60, 3.12,
3.32, 5.96, 1.29, 1.95, 2.91, 0.48 — **nine of nine, sign test p ~ 0.002, median
|z| = 2.91** — against three unperturbed measurements at -0.11, +0.07, +0.52.

Still an observation, **not a mechanism**.

### The campaign is closed

Nothing published in this line rests on a single recording now, except the
**rearranged-room** column (one recording, but H25 showed it is the
representative one) and the **cross-room** numbers, which were never measured on
seq136 and are blocked on a data decision put to the user and not acted on.

## H32 — roll costs 1.46% in practice, but 1.5% of frames carry a quarter of it

`h32-expected-cost/`, 11 angles x 13 recordings combined with H17's histogram
over 60,105 frames. No new data. Reading in that experiment's `analysis.md`.

**B1 passes.** Expected whole-image penalty under the real roll distribution:
**1.46% +- 1.09** (range 0.29-3.94; seq136 1.15%), against a 5% bar.

This replaces a coincidence with a measurement. The line had been stating three
numbers that were never multiplied together — flat to +-20 deg, p99 = 21.8 deg,
"98.5% comfortable". Multiplied, the answer is **about one and a half percent**.

**B2 fails, and it is the useful part.** Frames beyond +-20 deg are **1.48%** of
the data and carry **26.5% +- 16.8** of the expected cost — an **18x
concentration**. So *roll in general* is not worth handling, but **the rare
large roll is**: those are different products, a gravity prior applied
everywhere versus a detector for the rare tilted frame.

**B3 fails informatively.** The penalty is not monotone: -1.1% +- 2.8 at -5 deg
(negative on 9/13) and +0.1% at -10. The curve is flat to within noise between
-10 and +5 rather than rising from zero. That makes the integral smaller, not
larger, so B1 is unaffected — but B3 as written was the wrong sanity check.

**An alignment, not a finding.** The real distribution's signed median is
**-2.70 deg** (H17 checked and rejected calibration); the penalty minimum sits
near **-5 deg**. Wearers' typical tilt lands near the model's cheapest angle.
9/13, sign test p ~ 0.13 — a hint. No mechanism offered.

### What the +- averaging hid

Every published roll figure averaged the +- pair. Measured separately at 30 deg:
**-30 costs +24.4%, +30 costs +56.5%** — positive roll is **2.3x** costlier,
+30 costlier on **11/13**, gap +32.0 pts. At 15-20 deg the asymmetry is not
established (8-9 of 13), so it is a large-angle effect. The published
"+-30 deg = +46%" is the midpoint of 24 and 56.

**Second time in this line that averaging two arms hid something** — the first
was the rim/centre band in H28. Worth treating as a pattern: if two arms are
averaged before publication, check whether they differ before quoting the mean.

### The corrected practical statement

Ignoring head roll costs about **1.5%** of depth error on ordinary indoor
footage — cheap enough to ignore. If you spend effort on it, spend it on
**positive roll beyond 20 deg**: 1.5% of frames, a quarter of the cost, and more
than twice as expensive as the same angle the other way.

## H33 — the roll asymmetry test came out undecided; H32's claim narrows to DA3-Small

`h33-roll-asymmetry/`, four backbones on identical pixels, eight recordings.
Primary statistic per the recorded amendment: `(+30) - (-30)` in points.

| | mean diff | sd | +30 costlier on |
|---|---|---|---|
| `da3:small` | **+22.84** | 25.58 | 7/8 |
| `da3:large` | +4.60 | 20.13 | **4/8** |
| `vggt` | +8.38 | 5.39 | **8/8** |
| `vggt_omega` | +3.67 | 5.16 | 7/8 |

**B1 fails.** Single-image pair +13.72 +- 24.15, multi-view +6.03 +- 5.65 — a
7.69-point gap against a 24-point within-family spread. The family split is not
there. **B2 passes**: da3:small reproduces the asymmetry 7/8, consistent with
H32's 11/13. **The falsification did not trigger either**: the four do not agree
within noise, spanning 6x, with da3:large at chance.

### The experiment did not separate the two explanations

Designed as a clean either/or; the data gave neither.

**Toward the setup:** all four have a positive mean. A purely model-specific
effect should vanish or invert somewhere and does not.
**Against a pure artefact:** magnitudes differ 6x across models seeing literally
identical pixels, and da3:large is at chance (4/8) while vggt is 8/8.

**Defensible reading:** both are present — a small common asymmetry of roughly
+4 to +8 points that every model shows and the setup could account for, plus a
larger and much noisier **DA3-Small-specific** component (+22.8 +- 25.6). This
run cannot separate them. Separating them would need the upright convention
itself varied, which is invasive and was not attempted.

### H32's claim is narrowed, not withdrawn

H32 reported "+30 costs 2.3x what -30 costs" as a property of the roll penalty.
**That was DA3-Small only**, and DA3-Small turns out to have the largest and
noisiest asymmetry of the four.

* holds for **DA3-Small** (7/8 here, 11/13 in H32);
* **DA3-Large shows no consistent asymmetry** (4/8) — so not a DA3-family
  property;
* all four have a small positive mean, part of which may be our own rendering.

Quote the 2.3x as "on DA3-Small", never as a fact about depth models.

### Caution for the next person

The protocol originally asked for the +30/-30 **ratio**. The multi-view models'
-30 baseline is 0.6-3.7% and one cell is -0.2%, so ratios explode and one was
already `nan`. Uncritically quoted that reads "VGGT is 5x asymmetric" — a
headline produced entirely by dividing by noise. The switch to the difference is
recorded in that experiment's `protocol.md` as a change made AFTER seeing partial
data, with well-definedness as the reason.

## H34 — VOID. The mirror test had a design error, and B3 caught it.

`h34-mirror/`. **B3 failed and the test is void.** Level-0 whole-image error is
**0.1545 un-mirrored against 0.6145 mirrored** — a 4x degradation from mirroring
alone, before any roll. On seq136 near-rim goes 0.329 -> 1.537.

**The error:** I mirrored the source image and did NOT mirror the ground truth,
so the model predicted a mirrored scene and was scored against the unmirrored
one. Every mirrored-arm number is that mismatch.

**The trap it would have been:** B1 "passed" at 7/8 with a negative mean, which
reads exactly like "the asymmetry flips under mirroring, so it travels with the
scene". It is an artefact — with a baseline 4x too large, rolling reduces the
measured error, so every difference goes negative. **Without B3 this would have
been published as a clean positive result.**

**Why the fix is not cheap:** the mirror is applied in the fisheye source frame,
the scoring happens in the pinhole view frame, with the rig warp and the roll in
between, and the principal point is off-centre — so flipping the output back is
not the inverse of flipping the input. It needs source and GT mirrored together
before the rig, which changes how `zones` obtains ground truth.

**Decision: stop here.** The asymmetry is second-order and has now consumed H32,
H33 and H34, while the cross-room exposure — the largest open weakness in the
line — sits blocked on a data decision with the user. The asymmetry rests where
H33 left it (DA3-Small only, DA3-Large shows none, some part possibly our
rendering), which both reports already state. Nothing needs changing.

**Lesson worth keeping: any test that transforms the input needs a bar on the
untransformed baseline.** B3 was routine and it is the only reason this did not
go out as a finding.

## H35 — for the model you would actually ship, roll costs 0.46%

`h35-cost-per-backbone/`, dense grid, four backbones on the SAME six recordings,
integrated against H17's 60,105 frames. Reading in that experiment's
`analysis.md`.

| backbone | expected cost | +-20 deg | +-30 deg | tail share |
|---|---|---|---|---|
| DA3-Small | **1.80% +- 1.31** | +14.0% | +42.0% | **23.0%** |
| DA3-Large | 1.85% +- 1.33 | +9.3% | +27.8% | 14.9% |
| VGGT | **0.34% +- 0.23** | +2.9% | +7.0% | 17.1% |
| VGGT-Omega | **0.46% +- 0.25** | +2.6% | +4.8% | **9.8%** |

**B1 passes.** VGGT-Omega 0.46% against DA3-Small 1.80% — ratio 0.25, **4x
cheaper**. I expected the integral to compress H17's 4-5x gap, since it weights
the flat low-angle region far more heavily than H17's 30-degree comparison. **It
barely compresses.** The pretraining-family gap is not an artefact of looking at
large angles.

**B2 passes.** DA3-Small reads 1.80% here against H32's 1.46% +- 1.09 on
thirteen recordings — inside the spread, so subset and runner agree with H32.

**B3 — the tail advice is DA3-specific.** H32's "frames beyond +-20 deg carry
26% of the cost" holds for DA3-Small (23.0%) and NOT for VGGT-Omega (9.8%), a
13-point gap. The multi-view curves are flat enough that the rare tilt stops
being disproportionate. **"Handle the rare large roll" narrows to DA3-like
models**; on a multi-view model there is barely any cost to concentrate.

### Bigger is not the axis

DA3-Large costs 1.85%, statistically identical to DA3-Small's 1.80%, despite
being far better at large angles (+27.8% vs +42.0% at 30 deg). The integral is
dominated by the region under 15 deg where they are alike. **Scaling the
single-image model does not reduce the roll cost you would actually pay;
changing the pretraining does, by 4x.** Same shape as the rim-penalty finding —
a bigger model does not fix it, a differently-pretrained one does.

### The statement the roll line can close on

On ordinary indoor footage, ignoring head roll costs **0.3-0.5%** for a
multi-view-pretrained model and **~1.8%** for a single-image one. Both cheap. On
a single-image model the rare tilt beyond +-20 deg is where a quarter of the cost
sits; on a multi-view model there is nothing there worth chasing.

Measured, not inferred from three separately-quoted quantities.

---

# Outer-loop synthesis, 2026-09-08 — where the line stands after H17–H35

Written as a synthesis rather than another append, because eleven experiments
have run since the last one and `findings.md` had become a log rather than a
narrative.

## The four things this line actually established

1. **The rim penalty is real and scale does not fix it.** The rim is about
   **2.2×** worse than the centre (13 recordings, spread 1.4–2.7), in every model
   tried, and DA3-Large — twice as accurate overall — has the same ratio.

2. **A label-free student recovers ~94% of what depth labels buy, in-room.**
   Distil a stronger frozen model through a rectified view. The teacher choice
   that makes it work (VGGT-Omega on a wide view where DA3 fails) holds 13/13.

3. **Almost none of it crosses a room.** What crosses is a **six-number
   angle-dependent recalibration**, and that beats labelled training in a new
   room. It needs ~30 frames spanning ~2 m of walking, and it is **not** a
   property of the lens (matching the lens buys 0.8 pts, inside noise).

4. **Framing beats tilt.** A hard border next to the scored region costs
   **+59% to +269%** depending on model; ignoring head roll costs **0.3–0.5%**
   (multi-view) to **1.8%** (single-photo). **The thing to engineer around is
   what sits at the edge of the frame, not how the camera is held.**

## The methodological output, which may outlast the findings

* **One designated held-out recording is not a test set.** seq136 was used for
  every in-room number in this line and turned out to **amplify the magnitude of
  any intervention** — nine of nine, |z| up to 5.96 — while being ordinary for
  unperturbed measurements. Every published in-room figure had to be re-measured.
* **Print absolute numbers beside relative ones.** Three times a relative measure
  passed while the absolute quantity sat at zero.
* **If two arms were averaged before publication, check they differ.** Twice this
  hid something: the 2.0–2.6× band and the ±30° roll asymmetry.
* **Any test that transforms the input needs a bar on the untransformed
  baseline.** H34's routine sanity check is the only reason a 4×-broken input did
  not produce a clean-looking finding.
* **Name the axis a characterisation was never varied along.** "A property of the
  lens" was published twice before anyone varied the lens.

## What is left, honestly

**Blocked, and it is the largest exposure:** the cross-room evidence is still
**120 frames over two near-static recordings**. Every same-room figure is now a
mean over thirteen; every cross-room figure is two. More apartment footage cannot
fix it. The data that would — depth and ground truth for the untouched ADT
sequences — needs a download onto a shared filesystem at 98% capacity, which has
been put to the user and not acted on.

**Closed deliberately:** the roll asymmetry (three experiments, could not
attribute it, mirror test void) and the rim-density mechanism (two explanations
refuted, no third offered).

**Not worth more runs:** in-room verification. It is done.

## H36 — VOID. The external-domain question is still open, and I tried.

`h36-external-domain/`. Three runs, three B3 failures (bar: whole-image AbsRel
< 0.6):

| run | convention | cone | AbsRel | rim/centre |
|---|---|---|---|---|
| 1 | `z` | full ~190 deg | **2.640** | 1.23 |
| 2 | `range` | full ~190 deg | **5.993** | 4.28 |
| 3 | `range` | capped 80 deg | **3.150** | 7.17 |

**The numbers I am not quoting:** runs 2 and 3 give rim/centre of 4.28 and 7.17,
far above Aria's 2.2. Written carelessly that is "the rim penalty is 2-3x worse
in automotive fisheye" — striking, quotable, and built on a measurement B3 says
is broken. **Third time this session a broken measurement produced an attractive
number** (H34's mirrored "flip", H36's 1.5e6 predictions, this). The sanity bar
is the only thing between the artefact and the report each time.

### What it did establish, solidly

**The depth-convention machinery assumes a sub-90-degree cone.** `range` divides
planar z by cos(theta), cos crosses zero at 90, and the prediction reaches
1.5e6. `z` compares planar z against range ground truth instead. Neither is
correct past 90 degrees, and Aria's 54.83-degree cone meant twenty experiments
never surfaced it. Reusing this evaluation code on a wide automotive fisheye
hits it in the first run.

### Why the truncated run did not rescue it

Capping at 80 degrees removes the explosion (5.99 -> 3.15) but the CENTRE zone,
where distortion is mild and the conversion unambiguous, still reads **0.875**.
An 87% error in the easy part of the frame is not a rim problem. Either DA3 is
genuinely far out of domain here, or a residual plumbing difference remains.
Three runs could not separate those and I stopped rather than spend a fourth.

### The gap, stated plainly

**Nothing in this line has been validated outside Aria.** H25-H35 re-measured
extensively but all within one camera family, one apartment, one small office.
"Verified across thirteen recordings and four backbones" reads like external
validation and is not. Both reports now say so.

## H36b — the rim penalty replicates outside Aria, and at the same magnitude

After three void runs I stopped building the measurement and looked for one that
existed. The `depthfisheye` work stream in this repo has a **validated
by-theta evaluation on SynWoodScape**, 424M scored pixels, four cameras.

**It settles the void first**: that evaluation reports overall AbsRel **0.043**.
SynWoodScape is tractable, so my 2.64 / 5.99 / 3.15 were **my plumbing, not the
domain**.

| theta | AbsRel | ratio to centre |
|---|---|---|
| ~6.9 deg (centre) | 1.07% | 1.00 |
| ~34.4 deg | 2.10% | 1.96 |
| **~48.2 deg** | **2.55%** | **2.38** |
| ~75.7 deg | 5.06% | 4.73 |
| ~103.2 deg | 8.50% | **7.90** |

**Error grows monotonically with incidence angle on a completely different
fisheye domain** — synthetic, automotive, outdoor, ~190 degree lens.

### The comparison that matters

The headline 7.9x is partly reach: 103 degrees is far outside anything Aria can
see. **At a matched absolute angle the agreement is striking**: SynWoodScape at
~48 deg gives **2.38**, Aria's rim/centre is **2.17 +- 0.38** (H28, thirteen
recordings). Two datasets sharing nothing — indoor head-worn vs outdoor
automotive, real vs synthetic, 55 vs 190 degrees — land within a fifth of a
standard deviation over the same angular range.

### What it is not

Not a frozen model (LoRA fine-tuned ON SynWoodScape — it had every chance to fix
its own rim and did not, which makes the agreement more surprising, not less).
Not my measurement (another work stream's code and conventions; I am reading
their artifact). Not a controlled comparison (nothing held fixed but the
plotted quantity).

### What it changes

The caveat added one tick ago — "nothing in this line has been validated outside
Aria" — is **too strong**. Honest version: **the rim penalty itself replicates on
an unrelated fisheye domain at comparable magnitude; the METHODS built on it
remain Aria-only.**

### The lesson, which cost three runs

**Before building a measurement in a new domain, look for one that already
exists.** The answer had been sitting in `results/depthfisheye-sws-v3/` the whole
time. Same lesson as H21 (check what is invariant before re-running the expensive
model) and H25 (thirteen sequences sat extracted and unused for a day).

## H37 — the border ordering holds 13/13 at both ends, and seq136 predicted out of sample

H29 named the one thing it could not support: *"DA3-Large is the best of the
four on a clean frame and the worst with a border" was four backbones on ONE
recording; H29 re-measured two.* This ran the missing two.

| backbone | 13 recordings | seq136 (published) | z | range |
|---|---|---|---|---|
| `vggt_omega` | **+6.0% ± 4.6** | +29.1% | +5.01 | +1.2 … +19.0 |
| `vggt` | **+48.4% ± 18.0** | +106.5% | +3.22 | +28.3 … +86.7 |
| `da3:small` | **+59.0% ± 14.1** | +105.7% | +3.32 | +29.3 … +74.0 |
| `da3:large` | **+268.9% ± 61.5** | +635.5% | +5.96 | +148.6 … +345.7 |

**DA3-Large is worst on 13 of 13. VGGT-Omega is best on 13 of 13.** No
exception at either end. The ordering is a property of the backbones.

That matters beyond §4.2's warning. §4.3 chose the VGGT-Omega teacher *because*
H17.6 measured it as the border-tolerant one. H31 had verified the choice was
right; this verifies the reason was right. The two are not the same thing, and
until now only one of them was checked on more than one recording.

**The published gap was half the real one.** DA3-Large costs 21.8× VGGT-Omega
on seq136 and **45.1×** on the thirteen-recording means. seq136 amplified
DA3-Large by ×2.36 and VGGT-Omega by ×4.88, so it flattered the weaker side
more and compressed the ratio — the same mechanism that made every ratio in
H26/H27 improve when the absolute numbers came down. Fourth time now.

*Do not quote the mean of per-recording ratios* (70.4 ± 61.8, range 16.7–255.2).
VGGT-Omega's cost reaches +1.2%, so dividing by it explodes. H33's trap, live
again; the ratio of means is the citable form.

### The bar that was actually interesting

H31 restated the seq136 pattern as *it amplifies the magnitude of whatever you
do to it, in whichever direction the intervention points* — fitted on nine
interventions that all already existed. An observation fitted on its own
evidence explains everything and predicts nothing.

So B3 pre-registered a prediction: a border hurts, so seq136 must read **high**
for both new backbones. It read **z = +3.22** and **z = +5.01**.

Eleven of eleven now, two of them genuinely out of sample, against three
unperturbed measurements at −0.11, +0.07, +0.52. Still an observation. Still no
mechanism offered. But it has now earned the standing instruction that came
with it: *assume any intervention measured only on seq136 is inflated.*

### What this does not say

* **`vggt` vs `da3:small` is not separated.** Published as a near-tie (+106.5
  vs +105.7); on thirteen recordings `vggt` is lower on 8 of 13, means +48.4 vs
  +59.0, ranges overlapping almost completely. This experiment cannot order
  them and should not be read as doing so.
* **Border tolerance is still not a family property, and that is now firmer.**
  H17.6 concluded it from one recording; here the two VGGT variants differ by
  **8×** on thirteen-recording means with `vggt_omega` lower 13/13. Roll
  robustness *is* a pretraining-family property (H17.2, +11% for both VGGT
  variants); border tolerance is not. Two axes, each now on thirteen
  recordings, and they disagree about the same pair of models.
* **No mechanism.** Two mechanism guesses have already died on the neighbouring
  effect (H22, H23). None is offered.

### The pattern this is the third instance of

The actionable variable keeps turning out to be *which model*, not *how big*:

* rim penalty — VGGT-Omega best absolute **and** steepest field, but scaling DA3
  does not fix the field;
* roll — changing the pretraining family buys 4× (H35), scaling DA3-Small to
  DA3-Large buys nothing (1.80% → 1.85%);
* border — 45× between backbones (here), while DA3-Large, the *more accurate*
  DA3 on a clean frame, is the worst of the four.

Clean-benchmark accuracy does not predict any of the three robustness axes.

## H38 — the fourth part of the opening question, answered: read the roll, don't rotate the picture

The roll line was opened by a human question in four parts. Three were answered
(H17.2 they assume level; H17.1 it is violated mildly; H35 it costs 0.46–1.80%).
The fourth — *use the IMU* — was deprioritised as H17.4 with the argument that
there is almost nothing to recover. **That argument sized the prize and never
sized the price.**

The obvious way to use a known roll is to rotate the image level, infer, and
rotate the planar-z prediction back (which `upright.forward_z`'s docstring
establishes is exact). But rotating a square view leaves four black corner
triangles — and H37 had just measured what a border beside the scored zone
costs, 13/13 with no exception.

Three arms per angle, six recordings, four backbones. `null` rotates by −a then
+a: identical resampling, identical black corners, **no roll removed**.

### It is a net loss almost everywhere

| backbone | prize (+30° roll) | price (`null`) | net |
|---|---|---|---|
| `vggt_omega` | +6.0% | +6.8% | **−0.1%** |
| `vggt` | +12.3% | +42.5% | +26.8% |
| `da3:small` | +45.4% | +58.1% | **−12.0%** (6/6) |
| `da3:large` | +25.2% | +268.6% | +31.9% |

The one real win in eight cells is DA3-Small at +30°, which is **the most
roll-damaged (backbone, sign) pair in the experiment**. VGGT-Omega's prize and
price are the same size, so the most border-tolerant model available still
gains nothing. DA3-Large de-rotated sits +64 to +70% above its own
level-camera reference — worse than leaving the roll alone.

### The border cost reproduces on a different border shape

| backbone | H38 corners | H37 annulus |
|---|---|---|
| `vggt_omega` | +7.4% | +6.0% |
| `vggt` | +40.9% | +48.4% |
| `da3:small` | +64.1% | +59.0% |
| `da3:large` | +234.8% | +268.9% |

Four small corner triangles cost what a full annulus costs. Different shape,
different area, different script, different recordings, same ordering and
nearly the same magnitudes.

**Third independent route to "cost is proximity, not area."** H17.5 got it from
one border, H29's dose curve from sweeping the width (non-monotone, the weakest
of the three), and this from changing the shape at fixed proximity. This is the
cleanest of the three.

### B1 failed, and the failure is the asymmetry a third time

De-rotation helps DA3-Small on 6/6 recordings at +30° and 2/6 at −30°. Not
inconsistent: +30° costs it +45.4% and −30° only +27.8%, so there is less to
recover on the negative side while the price is higher (+70.1% vs +58.1%).
H32 found the asymmetry, H33 could not attribute it, H34 was void trying — and
here it decides whether a deployment recommendation holds at one sign or two.
Recorded as a fail because the pre-registered claim was that the operation
works, and it works in one cell out of eight.

### What to do instead — and this is the useful half

`raw` at 0° **is** the deployable operation: render the view level directly out
of the fisheye by folding the known roll into the resampling grid. No border,
no lost content, and it is exactly the level-camera reference every de-rotated
arm fails to reach.

> **Use the gravity vector in the resampling, not after it.**

The caveat is real. This only works if the pipeline resamples the fisheye at
all. Feed the raw fisheye frame straight in and there is no warp to fold the
roll into — and then rotating the image is the only move available, and it is a
net loss for three of four backbones.

That closes the fourth part of the opening question with a concrete answer
rather than a deprioritisation, and it does it without needing the wider-pose
dataset H17.4 was waiting for.

## H39 — the IMU pays, on the model that needs it, and H35's integral checks out

H38 said *don't rotate the picture, fold the roll into the warp* — but that was
an inference from H38's structure, not a measurement. H38's `raw@0` is aligned
to the **device**, and device and gravity coincide only when the head is level.

This measures the deployment operation on real frames with the real head roll,
read from MPS the way H17.1 read it. Three arms: `device` (roll_deg 0, today's
pipeline), `grav_p` (+ψ), `grav_m` (−ψ), scored on the per-frame intersection
of the three arms' coverage.

### DA3-Small: all three bars pass

| | value |
|---|---|
| gain, pooled over 6 recordings | **−2.37% ± 2.26** (better on 5/6) |
| H35's independently computed prediction | **1.80%** |
| wrong-sign arm | +9.94% |
| per-frame, &#124;ψ&#124; ≥ 8° | **+10.14%** |
| per-frame, &#124;ψ&#124; < 3° | −1.26% |

**B1 is the first external check this line has ever run.** H35 got 1.80% by
integrating a *synthetic* roll curve against a *measured* roll distribution.
H39 gets 2.37% ± 2.26 by a direct A/B on real frames. Two routes sharing almost
nothing agree to within a third. Every previous bar in this line was against a
null, a control or a permutation — never against an independent estimate of the
same number.

**B2 did its job twice.** It exists so a convention ambiguity could not be
resolved by whichever number came out smaller. The wrong sign costs +9.94%
against the right sign's −2.37%, a 4.2× asymmetry that identifies the sign by
shape. It had already killed the discarded first launch, where both gravity
arms came out +125% to +146% — equally bad, which is exactly its falsification
condition.

**B3's dose relation is clean, and its low cell is the honest null.** Frames
under 3° of roll get −1.26%, i.e. slightly *worse*: with no roll to remove,
aligning the render only costs resampling.

The pre-registered sub-prediction holds too. H17.1 read seq142–148 as a
contiguous block sitting 4–5° further tilted — a per-session mounting offset,
exactly what a gravity-aligned render removes. Gain on seq142/144/145 is −3.88%
against −0.86% on seq136/138/149.

### VGGT-Omega: nothing to collect, and the failure points the same way

Pooled +0.10% ± 3.26 against a wrong-sign arm of +0.72% — indistinguishable.
B3 fails in the informative direction: the per-frame gain is *more negative* at
high roll (−5.59%) than at low roll (−2.46%).

Reading, offered as a reading: for an already roll-invariant model there is no
roll to recover, so only the resampling difference between the two renders is
left, and that grows with the angle rotated through. DA3-Small pays the same
cost and simply wins by ten times more.

**This does not contradict H35.** H35 said 0.46% is *available*; H39 says it
cannot be collected this way because collecting it costs about that much.
Second time the price and the prize came out equal for this backbone — H38 had
+6.0% against +6.8%.

### The roll line, end to end

> **Read the roll from the IMU and fold it into the rendering warp — if you are
> running a single-image backbone.** Worth ~2.4% on ordinary indoor footage and
> ~10% on the frames that need it. One rotation of the sampling grid. No labels,
> no training, no change to the model. On a multi-view-pretrained backbone
> there is nothing to collect.

Four parts asked, four answered: they do assume level (H17.2); real footage
violates it mildly (H17.1); it costs 0.46–1.80% (H35); and you can get some of
it back, on one model family, by putting the gravity vector in the resampling
rather than after it (H38, H39).
