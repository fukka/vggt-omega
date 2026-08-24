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
