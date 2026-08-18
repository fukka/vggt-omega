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

None yet (bootstrap).

## Patterns and Insights

None yet.

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
