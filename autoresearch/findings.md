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

(Bootstrap state, 2026-08-18. No autoresearch experiments run yet; everything below
is inherited from this repo's prior work and the verified literature survey at
`docs/research/fisheye-wide-fov-adaptation.md`.)

1. **The field's converged recipe** is geometry-aware tokenization on a frozen
   pretrained backbone (survey §2): leave weights alone, fix the interface. The three
   anchors are all reproduced in this repo: VGGT-360 (training-free tangent views,
   `VGGT-360-fisheye/`), RayTun3R (10.7k-param polar PE residual, `raytun3r/`),
   CAM3R (SH ray module + ray-aware alignment, reports ADT RRA@15 99.0, `cam3r/`).
2. **Nobody reports the center/periphery Pareto front.** Adapter papers quote
   whole-image means; the question "did the center get worse" is unasked. N1 is open.
3. **Radial error claims are confounded** — error-vs-eccentricity is partly
   depth-vs-eccentricity (an oracle with no field effect reads 1.86x on ego-synth).
   The distance control in `slambench/fov.py` is mandatory for any N1/N3 claim.
4. **Periphery-for-pose is SLAM folklore with no FM-era measurement**: features at
   outer FOV carry large parallax and stabilize rotation (LF-VISLAM etc.), and this
   repo already measured that raw-fisheye backbones under-read rotation by 12–18%
   (alpha 0.82–0.88) while rectified-input restores alpha≈1. Whether the *rim
   specifically* carries the pose value is testable on CPU with the existing
   SIFT+MAGSAC++ harness (`raytun3r/experiments/harness_verify.py`).
5. **VGGT-Ω cannot be told the camera** (forward takes images only; RoPE normalized
   to [-1,1] carries no angular scale), so on unfamiliar content its FoV estimate
   regresses to the training prior and depth bends through it. Any adapter for it
   must inject geometry (rays/PE/tokens), not just override a number.

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
