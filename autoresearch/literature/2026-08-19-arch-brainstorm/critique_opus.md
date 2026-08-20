# Critique (Opus agent) — round 2

## 1. Strongest objections

### 1a. Fable's EpiRim / PERISCOPE

**The primary objection is arithmetic, and Fable files it as a "secondary risk".** EpiRim
samples D≈16 inverse-range hypotheses and gathers the tokens they project to. The
*separation* between adjacent hypotheses is disparity, which scales with the translation
baseline — not with rotation, and egocentric motion is rotation-dominant. Token angular
pitch at 504 px / patch 14 is ≈ 2·54.8°/36 ≈ 3.0° per token. The angular separation
between the 1 m and 10 m hypotheses at baseline t is t·(1/1 − 1/10) rad; to exceed **one
token** you need t ≥ 5.5 cm, and to spread 16 hypotheses over even 3 tokens you need
t ≥ 10–15 cm. F5's operating point is t−1…t−4 on *dense* windows chosen because
appearance evidence saturates there. At 30 fps that is 3–15 cm only if the wearer is
walking; for the standing/manipulating segments that dominate ADT it is ~1 cm, and all 16
hypotheses collapse into one or two tokens. The cost volume becomes a delta function and
the "matching distribution over range" is uniform noise. EpiRim is not wrong, it is
**mis-scoped in time**: metric evidence needs t−8/t−16, appearance evidence saturates at
t−2, and Fable inherited the appearance horizon for a metric mechanism.

**Second objection: F5 is weak support and arguably counter-evidence.** Fable argues
"rim-only KV == full KV (0.684 vs 0.685) ⇒ the attention is finding correspondences on its
own; giving it the epipolar answer should only sharpen it." Read it the other way: a
module whose output is *invariant to halving its KV set* is not performing correspondence
— it is consuming band-level context statistics. Center-KV costing +18% is equally
explained by center tokens carrying different depth statistics, not by failed matching. If
the module is a statistics aggregator, an epipolar prior has nothing to sharpen and the
D=16 restriction will strictly hurt by starving it. Fable's own kill condition
("epipolar-restricted ≤ unrestricted ⇒ geometry adds nothing") is therefore the *likely*
outcome, which is fine — but it means EpiRim cannot be the flagship until the probe runs.

**Third: RimPose's falsifier is transported from the wrong lens.** Fable predicts
"full-image ≈ rim-only (F6)". On ScanNet++ 170° that held (run_004: center-masked 4.93 vs
vanilla 5.00). On **Aria it was measured and it softened**: run_007 gives center_masked
20.0° vs vanilla 14.8° — deleting the center costs +5.3°. A strictly rim-only pose stream
is contradicted on the target lens by our own data. Also n=17 pairs (H1.1's set) is below
the bootstrap error bars the 2026-08-19 review made mandatory.

**Fourth: Field-Token's "provable" center neutrality is not provable.** A basis identically
zero for θ<15° does not make the *evaluated* center neutral, because findings state
plainly that under per-frame re-alignment local corrections move remote cells — the affine
couples them. Changing rim log-depth changes the fitted affine, which moves center AbsRel.
Zero-basis ≠ zero-effect under the eval of record. (And run_009's bias is (θ×depth)-
structured, so a θ-only *additive* spline cannot express a slope; Fable concedes this in
the risk section but still ranks it as the single-frame leg.)

### 1b. Sonnet's sliding-window rim-KV cache

**Its own falsifier disqualifies it as a contribution.** Sonnet specifies that the cached
result should be "numerically identical" to recompute. Something identical by construction
is an implementation detail, not a method. The FLOPs claim ("~75% of K/V-projection
matmuls avoided") is measured against a baseline nobody would build — recomputing the
whole window every frame. Against the sane streaming baseline (project each frame's KV
once, ever) the cache saves **exactly zero**. And the absolute size is small either way:
KV projection for ~627 rim tokens at d=384 is ≈ 0.18 GFLOP against ≈ 40 GFLOP for a
ViT-S/14 forward at 1296 tokens — **under 0.5% of a frame**. Sonnet also states the
accuracy is "roughly flat vs F2 alone". A sub-1% FLOPs win over a strawman, with no
accuracy win, cannot be rank 1.

**The substantive version of the objection: cached KV has no validity horizon.** A ring
buffer assumes the cached tokens still depict content the current rim sees. F6 says the
rim has the *largest* apparent motion under head rotation; at a routine 150–200°/s head
turn the 110° cone's rim band is fully replaced in ~0.2 s ≈ 6 frames, so at T=4 the buffer
can hold tokens with zero world overlap and the module has no way to notice. Sonnet
half-flags this and keeps the idea at rank 1 anyway. The only fix is to condition the
buffer on pose/parallax — i.e. Sonnet's #1 is only correct if Fable's #1 (or my P2) is
built first, which inverts the ranking. Compounding this: the whole thing rests on H6.1,
which the h6 analysis explicitly labels *training scene, small inter-frame motion,
exploratory*, and the #36 held-out evals are still owed by the box.

Sonnet's #2 (annular token merging) has a non-sequitur at its core: F2 shows the center
needs little added *correction*, not that center *features* are redundant — and the h6
analysis already corrected the record that the DPT head mixes tokens spatially, so merging
center tokens can move rim outputs. Merge 4:1 in the center and you are testing F1's
resampling lesson one level later, which is exactly Sonnet's own stated risk.

## 2. Where they beat or invalidate parts of mine

**Periscope-TTA vs RayCal — same loop, three real differences.**
- *Parameterization (mine is stronger).* RayCal fits a forward model
  `log d̂ = g(θ)·log d + c(θ)` and inverts it analytically. Periscope-TTA fits P3's K≈8
  **additive** radial spline. An additive offset cannot represent a gain, and run_009's
  signature — near content pushed 1.7–3.3× *far* while 5–10 m content is pulled 1.4–1.8×
  *near*, at the same θ — is a slope ≠ 1. A θ-only additive basis must pick one depth
  regime per angle. Same parameter order, strictly less expressive, wrong functional form.
- *Scale (mine is stronger).* Periscope-TTA declares the monocular scale ambiguity
  "harmless by protocol, because eval is per-frame scale-shift aligned." That concedes F4
  entirely — it keeps the per-frame affine that is manufacturing 82% of the number. RayCal
  carries one global scale nuisance per sequence inside the fit, so it can report the
  frozen-affine and one-affine-per-sequence rows where the real claim lives.
- *Framing (Fable is stronger, and I should adopt it).* I framed this as per-camera
  calibration; Fable frames the identical machinery as **GT-free adaptation to an unseen
  scene/camera**, which converts F8 (one apartment, generalization untested) from the
  project's biggest liability into the headline capability. "Calibration procedure" invites
  a reviewer shrug; "adapts to a new 170° camera with video and no depth sensor" does not.
  Fable's I5 (prefer geometry-bottlenecked or self-supervised paths over feature maps that
  can memorize one apartment) is a better-stated invariant than anything in my document.

**Sonnet #5 gives a sharper success criterion than mine.** I proposed "near-rim −25% under
scale_shift". Sonnet's is better and harder to game: the falsifiable prediction of F4's
mechanism is that the **gap between scale_shift AbsRel and frozen-affine AbsRel shrinks**.
A method that merely lowers the aligned number could be re-fitting; a method that collapses
the gap has actually removed the field. Adopt this as the locked bar for M1 below.

**Both of them beat my hedge on the dual-expert idea.** I ranked center/rim experts last
"worth one ablation row". Fable's kill point 2 is the argument I missed: F2 shows a single
25k head already improves the rim *without* center collateral, so there is **no measured
interference between regimes** for a mixture to resolve — it solves a problem absent from
the data. Concede; drop it, don't ablate it.

**Sonnet is right that my P4 overclaimed.** A rim-only pose head's ~3% backbone cost is
marginal to the end-to-end budget; it is a component, not a result, exactly as Sonnet's
own risk section says.

## 3. Merged top-3 (from all ~18), for one coherent paper, <1 GPU-week, F8-robust

**M1 — Self-supervised radial-field removal: pose → triangulation → field inversion.**
Fable's Periscope-TTA *loop* with my RayCal *parameterization* (log-linear g(θ), c(θ) with
an SNR-shrunk inverse, one explicit scale nuisance per sequence), sold as GT-free adaptation
to an unseen camera. Locked bar = Sonnet's: the scale_shift↔frozen-affine gap collapses.
Oracle-pose arm day 1, estimated-pose arm day 2. ~2 GPU-days.
*Reason:* the only item that is simultaneously the depth contribution, the payoff of the
pose contribution, and a structural answer to F8 — it needs video, not labels, so the
ScanNet++ 170° cross-camera row becomes cheap instead of load-bearing-and-missing.

**M2 — RimPose: frozen-feature ray-space pose (Fable P2 ≡ my P4/P5 ≡ Sonnet #4).**
Dense mutual-NN matching on frozen tokens, exact KB4 bearing lift, classical solver, plus
my H1-derived correction: **coverage-stratified RANSAC sampling, never θ-weighting**.
Reported at RRA@2/@5 and median degrees — @15 is saturated (CAM3R 99.0/95.0 on ADT).
Must include the Aria center-anchor arm, since run_007 shows center deletion costs +5.3°.
~1–2 GPU-days.
*Reason:* a pose result at ~0 trained parameters and no extra encoder pass, and M1 has no
engine without it.

**M3 — Parallax-gated epipolar rim attention (Fable P1, re-scoped by §1a).**
Keep the D-hypothesis KV, but gate the module on measured post-derotation parallax and
draw KV from t−8/t−16 for metric evidence rather than F5's appearance horizon. The headline
experiment is the **binned** one: the gain must rise monotonically with baseline length and
vanish in the lowest-parallax bin. Training-free probe first on the #36 checkpoint.
~2 GPU-days.
*Reason:* F3 says (θ, d̂) alone recovers under half the gain, so M1 needs a content-
dependent second stage; the parallax-binned prediction is what makes it a mechanism claim
rather than a capacity claim.

Dropped for this paper: the ring buffer (engineering, <1% of a frame), annular token
merging, polar re-tokenization (H7 prior), distillation (premature while F3's overfit
question is open), the angular KV pyramid (an ablation inside M3), every MoE variant.
Total ≈ 5–6 GPU-days.

## 4. The thing all three of us missed

**"The rim" is two different annuli, and we all conflated them.** Depth's near_rim zone is
θ ≥ 38° and runs to θ_max 54.8°; the depth penalty grows with θ, so the *worst* depth cells
are the outermost ring. But run_006 measured the pose value of that same outer ring and
found **nothing**: the count-matched span curve is 1.555° at θ≤45° and 1.572° at θ≤54.8° —
the 45–54.8° band contributes zero, and H1.3's write-up flags its pose value as unproven.
Pose value on Aria is concentrated in roughly 35–45°; depth liability is worst in 45–55°.

This threatens the shared thesis of all three documents. "The periphery is a pose asset and
a depth liability, so route the asset to pay the liability" (Fable's P6 headline, my P1's
motivation, Sonnet's cross-task justification for #1) silently assumes one band. If the
asset lives at 35–45° and the liability peaks at 45–55°, then every geometry-fed method —
triangulated anchors, epipolar KV, cost-volume shells — is best conditioned precisely where
the depth problem is *smallest*, and degrades into the band we most need to fix. Concretely:
M1's triangulated anchors will be dense at 38–45° and sparse and noisy at 50–55°, so the
fitted g(θ) is extrapolated exactly where the bias is 3.3×.

It is cheap to settle and nobody scheduled it: re-run run_006's span curve with the
outer-ring arm at adequate n (it was n=11) and with per-frame bootstrap error bars, and
separately report M1's anchor density and residual as a function of θ. If the outer ring
really carries no pose signal, the honest architecture is asymmetric in a second way —
geometry supervises 38–45°, and a *smoothness prior extrapolating outward from there*
carries 45–55° — which is a more interesting and more defensible design than any of us
proposed, and it is a claim only this project is positioned to make.

*(Runner-up, one line: Aria RGB is rolling shutter, the rim has the largest image-plane
velocity under head rotation, and every solver in all three documents assumes global
shutter — a systematic, radially-structured bearing error sits unmodelled underneath our
sub-degree pose requirements, and ADT has the row timestamps to measure it.)*
