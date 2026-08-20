# Critique (Fable agent) — discussion round

## 1a. Strongest objection to Opus's RayCal

RayCal's *identification* half (pose → triangulated anchors → label-free g(θ),c(θ)) is the
best single idea on any of the three lists. Its *application* half is where it dies: the
SNR-shrunk inverse is a deterministic map indexed by (θ, d̂) — the exact hypothesis class
this repo has already measured a hard ceiling for, twice. H2.1/run_010: any output-indexed
correction fails because the compression makes d̂ **many-to-one in true d** — at fixed
(θ, d̂), pixels at different true depths coexist, so any deterministic inverse, shrunk or
not, pushes the majority's fix onto minorities. Shrinkage scales the *amplitude* of both
gain and collateral; it cannot resolve a many-to-one ambiguity. F3 then measured the
ceiling of the whole class: a matched-capacity *learned* function of (θ, d̂) recovers
<half the rim gain AND damages the center. A 40-param analytic member of that class is
strictly inside it. Opus concedes the ceiling ("roughly half") — i.e. the top pick's own
stated ceiling is below the existing 25k head (F2).

The reframing of run_010 as "unshrunk noise amplification" also fails a location test:
1/g amplification predicts collateral where 1/g is large (the rim, g≈0.3), but the
measured collateral was at the **near center**, where g≈1.1 and 1/g amplifies nothing.
The many-to-one mechanism predicts the collateral where it was observed. So the
paper-grade sentence Opus wants ("shrinkage explains run_010") is likely false as stated.
Salvage: keep the identification, discard the analytic inverse — feed the triangulated
anchors as *pseudo-labels* to a feature-conditioned head (F3: features double the gain
and fix the center). That merged object is my #1 below.

## 1b. Strongest objection to Sonnet's rim-KV cache

Its own falsifier's success criterion is "numerically identical to recompute" — a test
that, if passed, proves the contribution adds zero capability. What the cache saves is
K/V *projection* recompute for ~600 rim tokens/frame — a rounding error next to the
frozen backbone forward that must run every frame regardless (and next to a module
already measured at 0.23× of one cross-attention). Amdahl kills the headline: end-to-end
latency moves by well under 1%. Sonnet's stated virtue — "consumes zero new hypothesis
risk" — is precisely the indictment: no hypothesis, no method contribution; CVPR reads a
ring buffer as an implementation detail of whichever module it wraps. The one real claim
inside it ("center KV is the *harmful* half, deleted by geometry with no router") is
F5's finding, not the cache's, and Sonnet's own risk section admits it was measured on a
training-scene probe. Also, the t−2…t−4 gains were training-free probes on t−1-trained
gates; making the window a "standing architecture" reintroduces exactly the training risk
the proposal claims to avoid.

## 2. Where they beat mine — honest updates

- **Opus's H1 kill hits my P2 RimPose squarely.** My rim-ONLY matcher assumed the rim is
  where matches are best; H1 (runs 001–002) shows no per-correspondence rim advantage,
  H1.1 shows the payer is *span at fixed count*, and run_007 shows center-masked pose
  (20.0°) is *worse* than vanilla (14.8°) on Aria — a rim-only matcher amputates span and
  inherits the measured annulus overshoot bias (gain 1.06–1.14). Update: **P2 is
  withdrawn in favor of Opus's FrozenMatch** (full-field frozen-feature matching +
  coverage-stratified RANSAC — span enforced in the sampler, not by weights). The part of
  my P2 that survives: exact KB4 ray lifting + classical solver on frozen features, which
  FrozenMatch shares.
- **Obs-A strengthens, not weakens, my P1 EpiRim** — pose should feed the network, and
  EpiRim is that arrow. But Opus's independently-derived EpiRim variant argues my −30%
  gain guess down: F5's rim-KV already matches full-KV, so headroom in KV *selection* may
  be ~nil; the defensible wins are robustness across baseline length (H6's diagnosed 3.3s
  failure) and cost. I adopt Opus's number (−5…−12%) and their baseline-length-binned
  falsifier. Opus's DeSweep derotation also fixes the rotation-dominant-motion risk I
  listed as P1's main killer; derotate-then-sweep and epipolar-KV are one mechanism (Opus
  says so) and should ship as one module.
- **Sonnet #5 ≈ my P3** (constrained compression-field head). Mine keeps the stronger
  construction (basis identically zero below θ=15° → provable center neutrality; K-dim
  feature bottleneck vs their looser g(θ,d̂;features)); theirs contributes the sharpest
  falsifiable prediction (constrained head shrinks the scale_shift-vs-frozen-affine gap
  more than F2's free head). Merge both into one proposal.
- My P5 pyramid absorbs Sonnet #1 as its deployment engineering; Sonnet #2/#3 carry the
  H7-redundancy smell (Sonnet admits) and rank below everything above.
- My Periscope-TTA survives but its application step must change per §1a: it was already
  feature-head-based, so it merges cleanly with RayCal identification.

## 3. Merged top-3 (one coherent paper; <1 GPU-week; F8-robust)

1. **RayCal-TTA** (Opus P1 identification ∘ my P4/P3 ∘ Sonnet #5 application): classical
   pose → triangulated full-FOV anchors → *pseudo-label* a provably-center-neutral,
   feature-conditioned K-dim radial field head — label-free adaptation per (camera,
   backbone), dodging both the F3/H2.1 output-indexed ceiling and the F8 one-apartment
   trap (adaptation needs video only, so ScanNet++ 170° transfer is testable directly).
   Falsifier: oracle-pose arm + g(θ)-vs-run_009 agreement, ~1 GPU-day.
2. **DeSweep-EpiRim** (Opus P2+P3 ∘ my P1): derotate previous-frame tokens on the KB4
   sphere, inverse-range sweep along each rim query's epipolar curve as gated zero-init
   KV — the content-dependent metric evidence F3 says is required and 024B says global
   attention doesn't deliver; the pose→depth video mechanism. Falsifier: training-free
   epipolar masking of the #36 checkpoint + parallax-binned gain signature, ~1.5 GPU-days.
3. **FrozenMatch + coverage-stratified RANSAC** (Opus P5, absorbing my P2 and Sonnet #4):
   the shared pose stream feeding 1 and 2 — frozen-feature coarse-to-fine matching,
   span-stratified sampling implementing H1.1's actual finding; attacks the measured
   bottleneck (SIFT solved 15/59 Aria pairs). Falsifier mostly CPU, <0.5 GPU-day.

Total ≈ 4 GPU-days including reruns; one organizing sentence (Opus's, adopted): the rim's
pose surplus pays the rim's depth deficit — pose→depth, inverting DUSt3R/MASt3R/VGGT.

## 4. One thing all three of us missed

**Triangulation self-supervision reintroduces the dynamics problem that H4 dismissed.**
H4/F8: hand/body pixels sit at θ>41°, median 0.26–0.94 m — exactly the near-rim cells
every top pick targets. H4.1 cleared hands as "plain occlusion" for the *forward* model,
and all three of us cited that to skip a dynamics module. But RayCal/Periscope-TTA/
DeSweep pseudo-labels come from **two-view triangulation under a static-scene
assumption**: a moving hand triangulates to a confidently wrong range, injected precisely
into the worst (θ, depth) cells — self-supervision poisoned where it matters most. None
of the 18 proposals includes an anchor-level motion-consistency gate (e.g. reprojection
residual across ≥3 views, or epipolar-inconsistency rejection). It is a ~20-line
addition to the RayCal falsifier and should be in the day-1 oracle arm, with an
anchors-poisoned-vs-gated ablation row — otherwise reviewers with egocentric experience
will find it for us.
