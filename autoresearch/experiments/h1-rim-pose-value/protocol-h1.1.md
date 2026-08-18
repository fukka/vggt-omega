# H1.1 — Is the periphery's pose value angular SPAN rather than per-point quality?

Follow-up to H1 (refuted: no per-correspondence rim advantage at equal count;
runs 001–002). Locked before running.

**Hypothesis.** At fixed correspondence count, rotation (and translation-direction)
accuracy improves as the admitted field of view widens, because a wider bearing
spread conditions the essential-matrix problem better — i.e. the periphery's value
is *coverage*, not per-point quality.

**Prediction.** Restricting matches to cumulative disks θ ≤ T for
T ∈ {35°, 45°, 55°, 65°, 85°(=all)} with count subsampled to the smallest
condition's count (seeded, 5 resamples, median), median rotation error decreases
monotonically with T in both real and synthetic arms; the marginal gain from the
last band (65→85, the "Aria-and-beyond" band) is nonzero. Translation direction
error, reported for pairs with GT translation ≥ 0.10 m, shows the same or stronger
trend (parallax argument).

**Method.** Same data, machinery, seeds, and denominator discipline as H1
(`rim_pose_value.py --mode span`). Count matching: N* = min over conditions of
that pair's in-disk match count; every condition estimates from exactly N*
matches (5 seeded resamples, per-pair median error). Pairs enter the summary only
if every condition answered. Synthetic arm identical but with GT-consistent
targets + 1 px noise.

**Refutation.** Flat (or non-monotone beyond noise) error vs T in the synthetic
arm ⇒ span itself carries no conditioning value at fixed count, and the wide-FOV
pose folklore reduces to feature *count* — which would redirect N3 to "the
periphery matters because it adds correspondences, so don't crop it, but don't
specially weight it either".
