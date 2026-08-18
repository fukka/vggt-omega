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
  (61%) more than doubles it on every pair (12.32°). Not area-matched yet
  (random-mask control pending), but center-masked ≈ vanilla is already decisive
  in one direction: **the frozen model's alignment signal lives in the rim.**

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
