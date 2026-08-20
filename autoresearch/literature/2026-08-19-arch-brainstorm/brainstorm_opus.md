# Brainstorm — Opus agent: pose–depth decoupling and geometry-grounded hybrids

Emphasis assigned: decouple pose from depth; classical/hybrid geometry on the fisheye
sphere; then use known pose to fix depth. Written against F1–F8 plus repo-measured
numbers I re-read today (h1 analysis runs 001–007, h6 analysis probes, bench analysis,
findings.md, research-state.yaml). Every quantitative claim below is either an F-number,
a repo number I cite by run/ticket, or a clearly-labelled **guess**.

---

## 0. Two framing observations that drive the whole ranking

**Obs-A — the frozen FM behaves like a span-limited classical estimator.**
Classical bearing-space pose restricted to θ≤35° has rotation gain α = 0.890 (run_003);
frozen depth FMs on raw fisheye sit at α = 0.82–0.88 (repo standing number). At full
span classical reads α ≈ 1.02 and median 1.5–1.6° on Aria (run_006), while DA3-Small
vanilla reads 14.8° median on the same pairs (run_007). So on the pairs classical can
solve, **classical is ~10× more accurate than the learned pose path**, and the learned
path's error signature is *exactly* that of a narrow-FOV estimator. Pose is not a place
where we should be trying to out-learn geometry on this camera. It is a place where
geometry should be feeding the network, not the reverse.

**Obs-B — the exploitable asymmetry is not "rim good / center bad", it is
"bias at the rim, variance nowhere".**
run_009: dispersion is 2–10% *everywhere*, bias is up to 3.3× at the near rim. F4: 82%
of the apparent penalty is the global affine's placement. Put together: the model's
depth output is, per pixel, an almost *deterministic* monotone function of true depth
whose parameters vary smoothly with θ. In log space

    log d̂(u) ≈ g(θ_u) · log d(u) + c(θ_u) + ε,   |ε| ≈ 0.02–0.10,  g < 1, g decreasing in θ

A single global scale-shift can absorb one (g, c) pair, not a *field* of them — which is
precisely F4. This is the single most actionable equation in the project, and nothing in
the current plan estimates (g, c) from anything other than depth GT.

**Why run_010 (the 48-param table) failed, restated as a design rule.** The stated
mechanism was "d̂ is many-to-one in d". With dispersion only 2–10%, the sharper statement
is: **inverting a compression amplifies residual noise by 1/g.** At the near rim bias is
3.3× and 1/g amplification of a 5% dispersion is harmless — invert hard. At the near
center bias is ~1.1× and the same inversion turns 5% dispersion into 10–25% error — that
is the measured near-center collateral. The bias/variance-optimal correction is a
**shrunk** inverse whose shrinkage is set per (θ, depth) cell by the locally measured
SNR. Run_010 applied an unshrunk inverse uniformly. This reframes a refuted result as a
mechanism, and it is a paper-grade sentence.

---

## Ranked proposals

### P1 — **RayCal**: pose-triangulated self-calibration of the radial depth-gain field (top pick)

**(1) Mechanism.** No training, no depth GT, no new network. For a short window of video
from one camera: (i) get dense correspondences between frames, (ii) lift matched pixels
to unit bearings through KB4 and solve relative pose on the sphere with the essential
matrix on bearings + MAGSAC++ (this harness already exists — `raytun3r/matching.py:290`
`relative_pose_magsac`, bearing-based by construction), (iii) triangulate each match to a
metric-up-to-one-unknown-scalar range on the sphere, giving a few thousand sparse depth
anchors *spread over the whole field of view*, (iv) regress the two-parameter-per-bin
field (g(θ), c(θ)) over ~20 θ bins by least squares on (log d̂, log d_triangulated) with
**one global scale nuisance per sequence** (absorbing the essential matrix's scale
ambiguity), (v) at inference apply the **SNR-shrunk analytic inverse**
`log d ← log d̂ − (1−λ(θ,d̂))·0 − λ(θ,d̂)·[(1−1/g)·log d̂ + c/g]`, with λ set from the
locally measured dispersion so the rim is inverted fully and the center is barely
touched. The field is a property of the (backbone, camera) pair, not the scene: fit it
once, on ~100 unlabelled frames, ship the 40 numbers with the camera.

**(2) Why it should work.** F4 is the whole argument: an oracle refit of the affine on
near pixels takes near-rim 1.47 → 0.26, i.e. 82% of the penalty is a misplaced global
affine, and the residual cause is a smooth radial field. RayCal estimates that field
*legitimately* — from geometry rather than an oracle — so that afterwards ONE affine
serves the whole image, which is F4's own stated criterion for a real fix. Obs-B says the
field is nearly deterministic (2–10% dispersion), so a 40-parameter parametric fit is not
underspecified. F1 is respected: nothing touches the input. F2's lesson (fixes live
behind the encoder) is respected in the strongest possible way — this lives *after* the
decoder and costs zero inference FLOPs. F3's warning is respected and used: a (θ, d̂) MLP
recovered <half the rim gain, so RayCal's ceiling is roughly half — which is exactly why
P1 is designed to be *composed with* the F2 feature head (below), not to replace it.
F6/Obs-A supply the estimator: the rim's pose value is what makes pose accurate enough to
triangulate at all.

**(3) Expected gain (guess).** Depth, under the F7 protocol: near-rim AbsRel −25…−40%
under scale_shift (alignment absorbs some of the win), and −60…−80% under the
**frozen-affine** and **one-affine-per-sequence** rows, which is where the real headline
is. Center within ±3% by construction (λ→0 there). Pose: unchanged, path untouched.
Cost: **zero added inference FLOPs and zero added latency** — one-time fit is minutes of
CPU per camera. Efficiency headline is unusually clean: 40 parameters, 0 FLOPs.
Secondary, and possibly the strongest claim in the paper: the fitted g(θ) curve should
reproduce the GT-derived compression curve of run_009 to within a few percent —
i.e. **the model's error field is identifiable without any depth ground truth.**

**(4) Cheapest falsifier (<1 day, 1×A100).** Two held-out ADT sequences, DA3-Small and
VGGT-Ω, dense video windows (not the 3.3 s sampling that broke the H6 pilot). Arm 1:
**oracle pose** (ADT GT trajectory + known extrinsics) → triangulate → fit g(θ), c(θ) →
compare against run_009's GT-derived field. This arm alone decides the paper: if
geometry-identified g(θ) ≠ GT-derived g(θ), P1 is dead in an afternoon and we learned
that content dependence dominates. Arm 2: **classical pose** from the existing bearing
harness → same pipeline → measures how much pose error costs. Arm 3: apply the shrunk
inverse and report all four F7 alignment rows plus the full (θ × depth) joint table (the
run_010 lesson: pooled zones hide near-center collateral). Success bar, locked in
advance: near-rim −25% under scale_shift AND near-center within +5% AND g(θ) agreement
within 10%.

**(5) Novelty positioning.** MoGe explicitly gives up scale and solves an optimal global
affine — F4 shows that affine is the thing that breaks at the rim, so RayCal is the
direct answer to MoGe's central design choice, and the paper can say so in one sentence.
UniK3D/UniDAC learn the spatially-varying scale field with dataset-scale supervised
training; RayCal identifies the same object per-camera at test time with 40 parameters
and no labels. DAC converts to ERP and retrains — input surgery, which F1 measures as
harmful on a frozen backbone. RayTun3R does unsupervised *per-scene* test-time adaptation
of network parameters; RayCal is per-*camera* and touches no network parameter, so the
two are stackable and comparable in one table with an explicit adaptation-data column
(the column `paper/comparison-protocol.md` already says we owe). DUSt3R/MASt3R/VGGT run
the arrow depth→pose (pointmap regression then alignment); RayCal runs pose→depth.
Fisheye ORB-SLAM-class pipelines have done bearing-space pose for a decade — **that is
not our claim and we must say so out loud**; the claim is using it as an unsupervised
identifier of a foundation model's systematic radial error field.

**(6) Main risk / how it dies.** (a) *Rotation-dominant motion.* Egocentric head turns
give tiny translation → triangulation is ill-conditioned → anchors are garbage. This is
the number-one killer and the mitigation is P2's derotation + a baseline gate (select
frame pairs by residual parallax after derotation, and reach to t−8/t−16 rather than the
t−2 that F5 found sufficient for *appearance* evidence — see P2). (b) *Content
dependence:* F3 says (θ, d̂) alone recovers <half; if the true field also depends on
surface, texture, or albedo, RayCal saturates at half the F2 head's gain. Response: ship
it as the geometric prior *under* the F2 feature head, and report the decomposition —
"geometry explains X%, features the rest" is itself a result. (c) *Noise amplification*
if λ is mis-set — the shrinkage must be fit, not guessed. (d) A reviewer says "this is
just per-camera calibration." Answer: it is calibration of a *learned model's* error
field, not of the lens, it is identified without GT, and it removes the alignment
protocol that F4 shows is fabricating 82% of the reported number.

---

### P2 — **DeSweep**: derotate-then-sweep — rotation-first decoupling with a spherical parallax cost volume

**(1) Mechanism.** Split pose into the well-conditioned and ill-conditioned halves and
treat them differently. Rotation R is estimated first, on bearings, from wide-span
correspondences (run_003/006: full span gives α ≈ 1.02 and 1.5° median; rotation needs no
depth and no translation). Then **derotate the previous frame in bearing space**: rotate
each previous-frame token's bearing by R and re-index. On the sphere this is exact — no
image resampling, so F1's resampling penalty does not apply (we move token *addresses*,
not pixels). After derotation, all residual token displacement is **pure parallax**,
radially outward from the epipole. Two payoffs: (i) correspondence search collapses from
2-D to a 1-D radial search along the derotated epipolar direction; (ii) build a
**spherical inverse-range cost volume** — sweep D ≈ 32 shells in inverse range, warp the
derotated previous-frame rim tokens onto each shell, correlate with the current rim
tokens, and feed the (rim-token × shell) cost tensor to a ~50k-parameter refiner that
outputs a residual to log-depth. Also ships a free **degeneracy detector**: if the
post-derotation residual flow is below the matching noise floor, translation is
unobservable this frame — emit a flag, skip triangulation, fall back to P1's cached
field.

**(2) Why it should work.** Ticket 024B measured "multi-frame context buys the center,
not the field", and the H6 CPU pilot failed at the rim. DeSweep is a mechanistic
explanation plus a fix: **implicit temporal attention interpolates appearance; the rim
needs explicit metric triangulation.** A cost volume supplies exactly the
content-dependent metric evidence that F3 showed (θ, d̂) lacks and that run_010's
output-indexed table could not manufacture. F6 says the rim has the largest apparent
motion under head rotation — i.e. the largest post-derotation parallax signal, so the
rim is where a cost volume is *best* conditioned, the mirror image of it being where
depth is worst. And the derotation is what makes the whole family survive
rotation-dominant egocentric motion, which is P1's main risk.

**(3) Expected gain (guess).** Near-rim AbsRel 0.68 → 0.40–0.48 (−30…−40%) on top of
whatever P1 gives, because this is the content-dependent half. Center untouched (rim
tokens only). Cost: correlation is 627 rim tokens × 32 shells × 384 dims ≈ 8 MFLOP, plus
a 3-layer refiner — **well under 0.5% of the backbone**, and cheaper than F5's already-cheap
0.23× attention module. Pose: the derotation step also yields a translation-direction
estimate; run_003 showed t-dir 43° → 16° from span alone, and a derotated 1-D search
should land under 10° on dense video pairs.

**(4) Cheapest falsifier.** Oracle-pose arm first, again: use ADT GT pose, build the
volume, train the 50k refiner for 2–3 hours on 4 sequences, evaluate on 2 held-out ones
with the full F7 rows. Locked predictions: (P1) near-rim improves ≥20% *beyond* the F2
readout-head baseline; (P2) the improvement scales with measured baseline length — bin
pairs by post-derotation parallax magnitude and show the gain vanishes in the
lowest-parallax bin (this is the falsifiable signature that the win is triangulation and
not extra capacity); (P3) the aux-only (θ, d̂) arm — a permanent baseline per the
2026-08-19 review — does not reproduce it. Half a day. A second half-day swaps oracle
pose for classical pose to price pose error.

**(5) Novelty positioning.** Plane-sweep cost volumes (MVSNet) and epipolar transformers
are perspective-native and assume a rectifiable geometry; the shell sweep here is on the
KB4 sphere over inverse *range*, on frozen FM tokens, with derotation. VGGT/VGGT-Ω fuse
frames with global attention and no explicit geometry — DeSweep is the explicit-geometry
counterpart and directly tests whether VGGT's implicit fusion is leaving triangulation on
the table (024B says it is). DUSt3R/MASt3R never build a cost volume. Derotation before
matching is classical SLAM practice — again, say so — but derotation *in token space of a
frozen depth backbone*, with the residual routed into a depth refiner, is not published.

**(6) Main risk / how it dies.** Token resolution: 14 px patches at 504 px may be too
coarse to resolve near-field parallax shells; if the shells are indistinguishable in the
cost tensor the refiner learns a constant. Second risk: Aria RGB is rolling-shutter and
the rim has the largest image-plane velocity under head rotation, so the rim is
simultaneously the parallax-richest and the most motion-blurred / skewed band (the repo
already has blur-benchmark work — this is the tension to cite). Third: derotation
re-indexing needs interpolation between token centers; even in token space that is a
resampling, and F1 is a warning that resampling artifacts can exceed the effect being
chased. Mitigation: compare gather-nearest vs bilinear-token derotation as an ablation.

---

### P3 — **EpiRim**: pose-conditioned epipolar-curve KV for the F5 cross-frame module

**(1) Mechanism.** F5's module has rim queries attend to previous-frame rim KV, with
rim-only KV matching full KV (0.684 vs 0.685) at 0.23× all-token cost. Given pose from
P2, each rim query's true correspondence lies on a **great circle** through the two
epipoles — which projects through KB4 to a *curve*, not a line, in the image. Replace the
"all 627 rim tokens" KV set with K = 32 tokens sampled along that query's epipolar curve
(plus a small fixed off-curve set as a hedge against pose error). Cross-attention becomes
a gather over a per-query index tensor. Additionally: use the *arc-length position along
the curve* as a relative positional encoding, which makes the attention distribution
directly interpretable as a depth posterior (the position along the epipolar curve *is* a
depth hypothesis) — so this module and P2's cost volume are two readings of the same
geometry and can share the sweep.

**(2) Why it should work.** F5 already measured that the useful KV is rim-band and that
center KV actively hurts (+18%) — evidence that the module wants *geometrically
plausible* correspondences and is damaged by irrelevant ones. Epipolar restriction is the
sharpest possible version of that filter. More importantly it repairs H6's diagnosed
failure: the CPU pilot failed because "adjacent" frames were 3.3 s apart and there was
little findable correspondence — pose conditioning removes the need to *find* it. That
turns H6 from a small-motion-only module into one that works across large baselines,
which is what egocentric head motion actually produces.

**(3) Expected gain (guess).** Near-rim 0.684 → 0.60–0.65 (−5…−12%) — modest; the win is
robustness across baseline length plus efficiency. Attention cost 627 queries × 32 KV vs
1296 × 1296 all-token = **0.012×**, ~19× cheaper than F5's already-reduced 0.23×. Honest
caveat: on an A100 a gather-attention of this shape is memory-bandwidth bound, so measure
wall-clock, not FLOPs — the realized speedup may be 3–5×, not 19×.

**(4) Cheapest falsifier.** Training-free first, exactly as H6.1/H6.2 were run: take the
delivered #36 rim checkpoint, restrict KV at eval time to the epipolar-curve set using GT
pose, and read near-rim/center/far. If a training-free epipolar restriction is within 2%
of full rim-KV (the bar H6.1 used) then the module tolerates it and training will exploit
it; if it *improves* near-rim training-free, that is the whole result for one GPU-hour.
Then the real test: bin evaluation pairs by baseline length and show curve-restricted KV
holds at 1 s / 3.3 s spacing where plain rim-KV collapses.

**(5) Novelty positioning.** Epipolar Transformer / epipolar-attention exist for
perspective multi-view; on fisheye the epipolar geometry is curved and nobody has put
curved-epipolar attention inside a frozen depth foundation model. Spark3R prunes tokens
by saliency on these same backbones — EpiRim prunes *KV* by geometry, which is the
orthogonal and stackable axis (findings already flag Spark3R as stackable). LoFTR uses
attention for matching, not for depth-feature routing.

**(6) Main risk / how it dies.** Pose error widens the curve: at 2° rotation error the
epipolar band at the rim is wide enough that K = 32 samples miss the true match, and the
module degrades below plain rim-KV. Mitigation: predict a band width from the pose
covariance and sample within it. Second risk: the accuracy gain is genuinely nil (F5's
rim-KV is already at full-KV parity, so there may be no headroom in *selection*, only in
cost) — in which case this is an efficiency-section paragraph, not a contribution.

---

### P4 — **RimPose**: an annulus-dominant pose head with a differentiable spherical solver, tapped below the adapter

**(1) Mechanism.** Two coupled ideas. *(i) Structural pose–depth decoupling by tap
point.* F2's working adapter is LoRA on the **last four** ViT MLP blocks; therefore tokens
at block L−4 are bit-identical to the frozen model's. Route the pose head off block L−4
and the depth head off the adapted top: pose invariance under depth adaptation becomes a
**structural guarantee, not an empirical hope**, at zero extra encoder cost (single
forward). *(ii) The head itself.* It reads the rim annulus (θ ≥ 38°, ~627 of 1296 tokens)
plus a sparse 64-token center anchor set, cross-attends to the previous frame's
corresponding set, emits per-token bearing correspondences with confidences, and feeds
them to a **differentiable weighted solver on the sphere** (weighted Procrustes for R, a
Gauss–Newton step on the epipolar residual for t-direction) — learned matches, classical
solver. Supervise with the classical harness's poses as targets plus a direct α-gain loss
so the head is trained to be *unbiased*, not merely low-error.

**(2) Why it should work.** F6 (rim carries the pose value, 17/17 pairs, −2.15° median)
and F5 (rim KV suffices, 0.23× cost) jointly argue that the pose-relevant information is
in the annulus. Obs-A says the current learned path is a span-limited, gain-0.82–0.88
estimator — a differentiable solver on bearings cannot have that bias because the
geometry is imposed, not learned. The tap-point idea converts the standing design
constraint from findings ("an adapter that improves rim depth must NOT perturb the rim
features the pose path depends on") from a regularization problem into an architectural
one.

**(3) Expected gain (guess).** Pose: median rotation from 14.8° (DA3 vanilla, run_007) to
1.5–3° on the same pairs, i.e. classical-competitive but with **~100% solve rate and
deterministic latency** — classical SIFT solved only 15 of 59 of those pairs. Cost: two
cross-attention layers over 691 tokens ≈ 1.3 GFLOP, ~3% of a ViT-S/14 forward at 504 px,
no RANSAC. Depth: unchanged by construction. Δpose under depth adaptation: **exactly
0.00°**, which is a table row no competitor can write.

**(4) Cheapest falsifier.** Before building anything: run the delivered H5 LoRA
checkpoint through the pose eval and measure how much pose actually drifts. If drift is
<0.2°, the tap-point guarantee is solving a non-problem and P4 loses half its motivation
— that check is one GPU-hour and should be run first. Then a 1-day arm: train the head on
4 ADT sequences with classical poses as targets, evaluate on 2 held-out with RRA/RTA and
median, ablating rim-only vs rim+center-anchors vs all-token.

**(5) Novelty positioning.** VGGT's camera head reads all tokens with global attention;
RimPose reads 48% of them, imposes the solver, and is provably decoupled from the depth
adapter. MASt3R-SfM and DUSt3R derive pose from regressed pointmaps (depth→pose);
RimPose never touches depth. CAM3R already reports ADT RRA@15 = 99.0 / RTA@15 = 95.0
(`cam3r/adt.py`) — **the @15 operating point on ADT is saturated and must not be our
headline**; report RRA@2 / RRA@5 and median degrees, and add hard-pair splits (low
overlap, large rotation, high blur) where the interesting differences live.

**(6) Main risk / how it dies.** Three measured hazards. (a) run_007 says on Aria's
narrower cone **the center is not disposable** (center-masked 20.0° vs vanilla 14.8°) —
a strictly rim-only head is contradicted by our own data, hence the center anchors; if
they turn out to be load-bearing, the efficiency claim shrinks. (b) runs 001–002 found a
reproducible **annulus overshoot**: estimating rotation from a high-θ ring gives gain
1.06–1.14 *even in the ideal-noise synthetic control* — a rim-dominant head may inherit
that bias, which is exactly why the α-gain loss is in the design. (c) ADT pose is
near-saturated for good methods, so the result may be unpublishable-as-accuracy and
survive only as efficiency + the decoupling guarantee.

---

### P5 — **FrozenMatch**: detector-free coarse-to-fine matching on the frozen backbone's own features, with spherical epipolar priors and coverage-stratified RANSAC

**(1) Mechanism.** LoFTR-style: coarse matching between the frozen ViT's 36×36 patch
tokens of two frames (mutual-nearest + dual-softmax), then a fine refinement head on the
intermediate-resolution feature map to sub-patch accuracy. **Zero extra encoder cost** —
we already ran the backbone for depth. Fisheye-specific parts: matches are scored against
a *curved* epipolar prior rather than a line; and the RANSAC minimal-sample draw is
**coverage-stratified** — force each minimal sample to span θ bins and azimuth sectors
instead of drawing uniformly.

**(2) Why it should work.** The stratified sampler is the direct, correct implementation
of F6/run_003's actual finding: span at fixed count improves rotation on 17/17 pairs and
t-direction 43° → 16°, and the ideal-noise control shows the effect is ~20× smaller —
i.e. **span pays through robustness to real feature noise**, which is precisely what a
RANSAC sampler controls. Standard uniform sampling on a fisheye draws most samples from
the dense center, producing narrow-span hypotheses, which is a mechanical explanation for
why the frozen models read like span-limited estimators (Obs-A). And the motivating
number is stark: classical SIFT solved **only 15 of 59** Aria pairs at 3.3 s spacing
(run_007) — the matcher, not the solver, is the bottleneck on this camera.

**(3) Expected gain (guess).** Solve rate on hard pairs 25% → 75–90%; median rotation on
solved pairs comparable to SIFT's 1.5–2.6° with better tails. Stratified sampling alone:
−0.3…−1.0° median and a visible tail improvement, essentially free. Cost: coarse matching
is one 1296×1296 similarity ≈ 0.6 GFLOP + a small fine head, ~2% of the backbone. This is
the enabling module for P1, P2, P3 — none of them work if matching is the bottleneck.

**(4) Cheapest falsifier.** Entirely CPU-feasible and therefore the cheapest item on this
list: on the local ADT sample, run the existing `adt_pose_value.py` harness with the
matcher swapped from SIFT to frozen-DA3-feature matching, and report solve rate and
median rotation against the hand-eye-verified GT. Separately, ablate uniform vs
coverage-stratified RANSAC sampling with everything else fixed — a 20-line change with a
locked prediction (median improves, solve rate unchanged).

**(5) Novelty positioning.** LoFTR/ASpanFormer/RoMa are perspective-trained dedicated
matchers; MASt3R adds a matching head to a 3D backbone. The differentiator here is
*reusing a depth foundation model's frozen features as the matcher on a fisheye*, so a
deployed AR system runs one encoder for depth, pose, and matching. Fisheye ORB-SLAM-class
pipelines use hand-crafted features on bearings — mature prior art we cite, and whose
25% solve rate on our data is the motivation. The stratified-sampling result is small but
it is a clean, cite-able geometry finding for wide-FOV RANSAC.

**(6) Main risk / how it dies.** Depth-trained features may simply be bad at
correspondence — depth training rewards smooth, locally-invariant representations, the
opposite of what matching wants; and the rim features are the most distortion-affected.
If frozen matching underperforms SIFT, P5 becomes "use an off-the-shelf matcher" and the
zero-extra-encoder efficiency claim dies (P1–P3 survive, just with a second network).

---

### P6 — **ScaleBridge**: replace per-frame affine fitting with a geometry-locked, temporally-persistent scale

**(1) Mechanism.** F7's protocol re-fits a scale-shift *per frame*. That is an evaluation
convenience which, per F4, is also fabricating 82% of the reported rim penalty — and in a
product it is worse than an artifact: it means the depth map's scale flickers frame to
frame. ScaleBridge fits a single affine per *sequence*, anchored by the triangulated
metric points of P1/P2 (and optionally by Aria's VIO metric translation, which makes it
truly metric rather than up-to-scale), propagated with a one-state Kalman filter. Ships
with a new reported metric: **temporal scale drift** — the frame-to-frame variation of
the optimal per-frame affine, in percent.

**(2) Why it should work.** It is the direct consequence of F4's diagnosis. And it turns
P1's win into a benchmark statement no baseline can match: report AbsRel with *one affine
per sequence*, or with no alignment at all, and beat scale-shift-aligned baselines. That
comparison is far more convincing to a reviewer than another aligned-AbsRel delta,
because the alignment is exactly what the field's numbers hide behind.

**(3) Expected gain (guess).** Under a single sequence-level affine, near-rim for a
RayCal-corrected model should land 0.35–0.55 where an uncorrected model reads 1.2–1.5 —
a much larger separation than the aligned protocol shows. Temporal scale drift: a guess
of 8–20% frame-to-frame for a vanilla single-image FM, → <3% with anchoring. Cost:
negligible.

**(4) Cheapest falsifier.** Purely an eval-side experiment on existing checkpoints:
compute the per-frame optimal affine on held-out sequences and plot its trajectory. If
drift is already <3%, the metric is uninteresting and ScaleBridge reduces to a row in the
alignment-robustness table (which F7 already requires anyway). Under an hour.

**(5) Novelty positioning.** MoGe's contribution is literally an optimal global affine
solve; UniK3D/Metric3D chase metric scale by training with camera embeddings. ScaleBridge
gets scale from *this sequence's own geometry* at test time. As a protocol contribution it
also fixes the credibility problem F4 exposed in our own numbers, which is worth a
paragraph on its own.

**(6) Main risk / how it dies.** Heavy overlap with P1 — if presented as a separate
contribution a reviewer will call it a subsection. Correct framing: it is P1's evaluation
protocol and its AR-facing claim, not a fourth method. Also, two-view scale is only up to
a scalar unless VIO is used, and leaning on Aria VIO weakens the "any camera" story.

---

### P7 — Direction C, briefly: center-expert / rim-expert with latent fusion (**ranked last, with a bad prior**)

**Mechanism.** Two lightweight decoders over the shared frozen tokens — one specialized to
θ ≤ 20°, one to θ ≥ 38° — fused by a learned per-token gate, so each expert can adopt a
different depth prior without the compromise a single head makes.

**Why it is ranked last.** Three measurements point against it. H7 (θ-gated LoRA) was
refuted with a mechanism: the gate stayed flat (|g−1| ≈ 0.06) because **positional
encoding already conditions the adapter spatially** — a θ-routed expert is the same idea
one level up and should be expected to collapse the same way. Center-PH's rectified crop
made the near-field center **62% worse** (F1), which kills the most natural "center
expert = perspective expert" instantiation. And the H2.4 result showed VGGT-Ω's final-block
tokens already produce center collateral up to +50%, so more heads reading those tokens
is more risk, not less. **Worth exactly one ablation row** — a 2-expert head vs a
matched-capacity 1-expert head — as evidence the paper considered and rejected it. If the
gate again comes out flat, that is a second data point for a genuinely interesting claim:
*spatial specialization is already latent in a frozen ViT's PE; you do not need to
architect it, you need to supervise it.*

---

## Explicit kill: **rim-weighted robust pose** (and its cousin, a θ-weighted pose loss)

The seductive version: "F6 says the rim carries the pose value → upweight rim
correspondences in RANSAC/IRLS, or add a θ-weighted term to the pose loss." It is the
first thing anyone proposes after reading F6, it is trivially implementable, and it is
**refuted by our own data**. H1 (runs 001–002) tested exactly the per-correspondence rim
advantage: the ideal-noise synthetic control is **flat across θ quartiles** (0.30–0.39°),
the real arm is non-monotone with the rim quartile *worst*, and the paired rim-minus-center
difference is a coin flip (rim better on 9/20 real, 14/23 synth). Rim correspondences are
not individually better. Worse, runs 001–002 found a reproducible **rim-annulus rotation
overshoot** (gain 1.06–1.14) present in the noise-only control — so upweighting the rim
would import a systematic bias, not just fail to help. The rim's pose value is **span and
count under real feature noise** (F6/run_003), a property of the correspondence *set*, not
of any correspondence. The correct salvage, which is P5's second half: do not weight by θ
— **stratify the RANSAC minimal sample by θ and azimuth** so every hypothesis is
wide-span. Same intuition, opposite implementation, and the one our measurements support.

A second, softer kill: **training a learned two-view pose head whose headline is beating
baselines on ADT RRA@15.** CAM3R already reports 99.0 / 95.0 there. That row is saturated;
any pose contribution must be argued at RRA@2/@5, on median error, on hard-pair splits,
or on cost and determinism — never at @15.

---

## How these compose into one system (for the paper's Figure 1)

One frozen encoder forward per frame. Off block L−4, the **RimPose** head (P4) reads the
annulus plus sparse center anchors, emits bearing correspondences via **FrozenMatch**
(P5), and a differentiable spherical solver returns R and t-direction — pose is complete
and structurally independent of everything downstream. R **derotates** the previous
frame's tokens (P2); residual parallax feeds both an **epipolar-curve KV** attention (P3)
and a spherical inverse-range **cost volume**, which drives a tiny rim depth refiner. In
parallel, the sequence's triangulated anchors identify the backbone's **radial depth-gain
field** (P1) with no labels, and the SNR-shrunk inverse removes it — after which
**ScaleBridge** (P6) needs only one affine for the whole sequence. Total added inference
cost: roughly 5% of the backbone. Total added parameters: ~0.1 M plus 40 calibration
numbers.

The single sentence that organizes the paper: **the rim is where pose information is
richest and depth is most biased, so we spend the rim's pose surplus on the rim's depth
deficit.**

---

## Three-line summary

**Top pick: P1 RayCal (with P2 DeSweep as its learned, content-dependent second stage).**
F4 says 82% of the reported rim penalty is a global affine misplaced by a smooth radial
compression field; RayCal identifies that field with zero labels — classical bearing-space
pose on the sphere, triangulated sparse anchors spanning the full FOV, a 40-parameter
log-linear fit, and an SNR-shrunk analytic inverse that explains why run_010's unshrunk
version damaged the center — so afterwards one affine serves the whole image, at zero
inference FLOPs and on any backbone or camera.
It is CVPR-grade rather than an adapter tweak because it is not another set of trained
parameters bolted behind an encoder: it shows a depth foundation model's systematic error
field is **geometrically identifiable without ground truth**, inverts the field-standard
depth→pose arrow that DUSt3R/MASt3R/VGGT all follow, and lets us report fisheye depth
under one-affine-per-sequence (or no) alignment — retiring the per-frame alignment that
our own F4 proves is manufacturing most of the numbers everyone in this area reports.
