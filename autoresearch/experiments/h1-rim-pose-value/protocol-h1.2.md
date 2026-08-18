# H1.2 — Do frozen depth FMs actually use the periphery for alignment?

Locked before running. Parent: H1 (refuted per-point form) / H1.1 (supported span
form: at fixed count, θ≤85° beats θ≤35° by 2.15° rotation on 17/17 pairs, and the
value is robustness to real feature noise).

**Hypothesis.** Frozen feed-forward 3D FMs (DA3-Small as the CPU-runnable
representative) effectively ignore the fisheye periphery for cross-frame
alignment: their rotation estimate behaves like a span-limited classical
estimator (repo prior: DA3 vanilla gain ≈ 0.72–0.85 on fisheye; classical at
θ≤35° reads 0.890).

**Predictions.**
1. Rim-masked input (pixels θ > T masked to the image mean color, T ≈ 45°):
   DA3's rotation gain changes by **less** than the classical span curve's drop
   over the same restriction (classical t85→t45: 0.925→0.973 gain, err
   1.67→3.23°). If the model ignores the rim, masked ≈ vanilla within noise.
2. Center-masked input (θ ≤ T masked, complementary region): gain collapses far
   more than rim-masking — the model's alignment lives in the center.
3. Consequence if both hold: the classically-recoverable span value (H1.1) is
   NOT being extracted by the frozen model — quantifying the headroom an
   adapter should target, with the classical curve as the attainable reference.

**Method.** Same scene/pairs machinery as runs 001–003 (ScanNet++ 3f15 local
sample, pairs with GT rotation 0.5–30°, ~20–40 pairs). Conditions: vanilla /
rim-masked(T=45°) / center-masked(T=45°). Backbone: `raytun3r` `da3` backbone,
small, pretrained, CPU, `depth_convention="range"`, install without patch
undistortion (pure vanilla behavior); pose read out via `.relative(0,1)` exactly
as `harness_verify.py` does. Report per condition: median rotation error, gain,
masked-pixel fraction. Denominator: only pairs where all three conditions
produce a pose. Runtime note: DA3-Small CPU forward is slow; cap pairs to keep
the run under ~1.5 h and record the cap.

**Refutation.** Rim-masking degrades DA3's gain by at least the classical drop
⇒ the model already uses the rim, and the adapter story shifts from "make it use
the rim" to "make its rim use *robust*" (H1.1's noise-robustness mechanism).
