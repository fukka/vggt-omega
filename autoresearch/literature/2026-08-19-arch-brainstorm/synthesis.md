# Multi-agent architecture brainstorm — synthesis (2026-08-19)

Human-directed: "decouple pose estimation and depth estimation; try classical /
fisheye-specific / feature-matching pose; then center+rim experts, latent
fusion, etc.; launch Fable/Opus/Sonnet agents for brainstorm and discussion."

Process: three independent agents (Fable = full sweep, Opus = pose-decoupling
emphasis, Sonnet = efficiency emphasis) each produced 6–7 ranked proposals
against the measured F1–F8 fact sheet (raw files in this directory), then each
read the other two and wrote an adversarial critique (critique_*.md, archived
here too). ~18 proposals in, 3 survive with strong consensus.

## Convergence signals (independent, therefore meaningful)

- Fable and Opus **independently invented epipolar-curve KV attention**
  (constraining the H6 cross-frame module's KV to each rim query's KB4
  epipolar curve).
- Fable P2, Opus P4/P5, Sonnet #4 **independently converged on the same pose
  stream**: frozen-backbone features + matching + exact KB4 ray lifting +
  classical solver, decoupled from depth by construction.
- All three **independently killed the center-expert/rim-expert MoE** (the
  human's direction B, first bullet) on the same evidence: H7's flat gate
  (PE already spatially conditions a uniform adapter), F2's absence of
  center/rim interference (a single 25k head helps the rim *without* hurting
  the center — there is no conflict for a mixture to resolve), and F4 (the
  failure is a smooth low-dim field; capacity partitioning is the wrong axis).
  Consensus: **drop it entirely, don't even ablate** (Opus conceded its
  "one ablation row" hedge). The refutation chain itself is paper material:
  *spatial specialization is already latent in a frozen ViT's PE; you don't
  architect it, you supervise it.*
- Organizing sentence adopted by all three (Opus's formulation):
  **"The rim is where pose information is richest and depth is most biased;
  spend the rim's pose surplus on the rim's depth deficit"** — pose→depth,
  inverting the depth→pose arrow of DUSt3R/MASt3R/VGGT.

## Merged top-3 (the post-discussion consensus ranking)

### M1 — Self-supervised radial-field removal ("RayCal-TTA")
Pose (classical, bearing-space, harness exists at `raytun3r/matching.py:290`)
→ triangulate matches into sparse full-FOV range anchors → fit the backbone's
radial compression field from (log d̂, log d_tri) → remove it, so ONE affine
serves the whole image (F4's own criterion). Framed as **GT-free adaptation to
an unseen camera** (Fable's framing, adopted by Opus): needs video, not depth
labels — converts F8 (one apartment) from liability into headline; makes the
ScanNet++ 170° cross-camera row cheap instead of load-bearing-and-missing.

**Unresolved design fork, to be settled empirically, not by argument** — the
application step:
- Opus: analytic log-linear inverse `log d̂ = g(θ)·log d + c(θ)`, SNR-shrunk,
  one scale nuisance per sequence (can express the measured slope≠1; reports
  frozen-affine rows honestly).
- Fable's objection: any deterministic (θ, d̂)-indexed map is inside the class
  H2.1/F3 measured a hard ceiling for (d̂ is many-to-one in d; <half the gain;
  center damage) — and Fable's location test (run_010's collateral appeared at
  the near CENTER where 1/g≈0.9, not at the rim where 1/g≈3) argues the
  many-to-one mechanism, not noise amplification, is what bites.
- Fable's alternative: use the anchors as **pseudo-labels for the
  feature-conditioned, center-zero-basis K-dim field head** (features double
  the gain and fix the center, F3), geometry-bottlenecked against apartment
  memorization (invariant I5).
→ Falsifier runs BOTH application arms on the same anchors. Locked bar
(Sonnet's, adopted by Opus as harder to game): **the gap between scale_shift
AbsRel and frozen-affine AbsRel must collapse** — merely lowering the aligned
number could be re-fitting; collapsing the gap means the field is gone.
Day-1 arm: oracle (GT) pose; the g(θ) it identifies must match run_009's
GT-derived field. ~2 GPU-days.

### M2 — Decoupled pose stream ("FrozenMatch/RimPose")
Detector-free coarse-to-fine matching on the FROZEN backbone's own features
(zero extra encoder pass), exact KB4 ray lifting, classical/differentiable
spherical solver. Corrections locked in by the discussion:
- **Coverage-stratified RANSAC sampling, never θ-weighting** (H1 runs 001–002:
  no per-correspondence rim advantage; rim-annulus overshoot bias 1.06–1.14
  exists even in the noise-only control; the rim's pose value is span-of-the-SET).
- **Full-field, not rim-only** (run_007: deleting the center costs +5.3° on
  Aria; Fable withdrew its rim-only P2 on this evidence).
- Optional structural guarantee: tap the pose head at block L−4, below the
  H5 LoRA — Δpose under depth adaptation = 0 by construction (pre-check: run
  the H5 checkpoint through pose eval; if drift <0.2° the guarantee is solving
  a non-problem).
- **Report at RRA@2/@5 + median degrees + hard-pair splits. Never RRA@15**
  (CAM3R already reports 99.0/95.0 on ADT — saturated).
Motivating number: classical SIFT solved only 15/59 Aria pairs at 3.3s spacing
— the matcher, not the solver, is the bottleneck. Falsifier mostly CPU
(existing H1 harness). ~1 GPU-day.

### M3 — Parallax-gated epipolar rim attention (re-scoped EpiRim ∘ DeSweep)
The content-dependent second stage M1 needs (F3: geometry alone recovers <half).
Derotate previous-frame tokens on the KB4 sphere (token re-addressing, not
pixel resampling), then rim queries attend to D≈16–32 inverse-range hypothesis
KV along their epipolar curves. **Opus's arithmetic re-scope is binding**:
token pitch ≈3.0°; separating range hypotheses needs ≥5.5cm baseline, so KV
must come from t−8/t−16 (metric horizon), NOT F5's t−1…t−4 (appearance
horizon) — two different temporal scales for two different kinds of evidence,
itself a testable claim. Gate the module on measured post-derotation parallax.
Headline experiment: gain must rise monotonically with baseline-length bin and
vanish in the lowest bin (mechanism claim, not capacity claim). Training-free
epipolar-masking probe on the #36 checkpoint first. Efficiency claims in
wall-clock, not FLOPs (Sonnet: per-query gathers are memory-bandwidth bound).
~2 GPU-days.

Alternative reading to keep honest (Opus): rim-KV==full-KV may mean the H6
module aggregates band statistics rather than doing correspondence — in which
case epipolar restriction STARVES it and the probe fails. That outcome would
still be informative (kills the whole epipolar-KV family cheaply).

## Pre-checks scheduled before committing GPU weeks (from the "missed by all" round)

1. **Two-annuli problem (Opus's find, threatens the organizing thesis).**
   Pose value on Aria concentrates at ~35–45° (run_006: adding 45–54.8° moved
   the span curve 1.555°→1.572°, i.e. nothing, at n=11); depth liability peaks
   at 45–55°. If confirmed at adequate n with bootstrap bars, every
   geometry-fed fix is best-conditioned where the depth problem is smallest,
   and M1's g(θ) is extrapolated exactly where bias is 3.3×. The honest
   architecture then supervises 38–45° with geometry and extrapolates 45–55°
   with a smoothness prior — a sharper claim than any single agent proposed.
   Cheap: re-run run_006's outer arm at adequate n.
2. **Aria is not monocular (Sonnet's find).** The device carries two SLAM
   cameras with factory extrinsics to the RGB sensor — a static stereo
   baseline exists at EVERY timestamp, independent of head motion, which would
   sidestep M1/M3's #1 named killer (rotation-dominant motion) entirely.
   Check whether ADT exposes synchronized SLAM-camera streams + extrinsics
   (local check on seq131 possible).
3. **Dynamic hands poison triangulation (Fable's find).** H4.1 cleared hands
   for the FORWARD model only; two-view triangulation under a static-scene
   assumption turns a moving hand into a confidently-wrong anchor injected
   precisely into the worst (θ,depth) cells. Every M1 falsifier needs an
   anchor-level motion-consistency gate (≥3-view reprojection residual) and a
   poisoned-vs-gated ablation row from day 1.
4. **Rolling shutter (Opus, runner-up).** Aria RGB is rolling shutter; the rim
   has the largest image-plane velocity under rotation; every solver assumed
   global shutter. ADT has row timestamps — measure the bearing error before
   claiming sub-degree pose.

## Killed in discussion (do not revisit without new evidence)

| Idea | Killed by |
|---|---|
| Center/rim dual-expert MoE (any variant, incl. discrete θ-routing) | H7 flat gate + F2 no-interference + F4 wrong-axis; unanimous |
| Latent-space equal-area resampling | H8's mechanism (broken PE/token statistics) survives the move behind the encoder; ≤0.21px within-patch distortion removes the motivation |
| Rim-weighted RANSAC / θ-weighted pose loss | H1 runs 001–002: flat synthetic control, rim quartile worst, overshoot bias imported; correct form is stratified sampling |
| Ring-buffer KV cache as a contribution | "identical to recompute" proves zero capability; saves <0.5% of a frame vs sane streaming; no validity horizon under 150–200°/s head turns (needs pose conditioning → depends on M3) |
| Annular token merging (center 4:1) | DPT head mixes tokens spatially → center merge can move rim outputs; F1's lesson one level up |
| Pose headline at ADT RRA@15 | CAM3R 99.0/95.0 — saturated operating point |
| Distillation student (now) | F3's apartment-overfit question must close first, else the student inherits the overfit |

## Mapping back to the human's suggestions

- "pose via traditional methods / feature matching" → became the consensus
  core (M1+M2). Classical geometry is ~10× more accurate than the learned
  pose path on solvable pairs (1.5–1.6° vs 14.8° median, runs 006/007); the
  right role for learning is supplying the matches, not replacing the solver.
- "one model for center + one for rim" → killed with our own evidence (above);
  the refutation is itself a finding worth a paper paragraph.
- "fisheye → latent features that models merge" → survives only in
  F1-compliant forms: rim-KV attention (M3) and, low-ranked/speculative,
  polar re-indexing of rim tokens behind the frozen encoder (Sonnet #3;
  H7-redundancy smell, third priority at best).

## Shared preconditions

The whole rim-KV family (M3 + every efficiency variant) rests on H6.1, which
is train-scene + exploratory; the #36 held-out evals owed by the GPU box are
the gate. Nothing in M1/M2 waits on them; M3's training-free probe can run the
day they land.
