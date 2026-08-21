# Verify hand GT-depth provenance; use the synthetic stream as the hands-free control

**Owner:** gpu
**Status:** **done** — `results/autoresearch-h4-provenance` (meta `ticket: 32`).
**Files I may touch:** scripts under `autoresearch/experiments/h4-dynamics/code/`
(create/extend), results to `results/autoresearch-h4-provenance/`.
**Blocked by:** none — the three #28 skeleton sequences are on lambda_63.

## Part 1 — which depth did we score against, and what does each variant contain

For the three #28 sequences, ~20 high-dyn_frac frames each:

1. State explicitly (from the #28/#31 scripts) which `AriaDigitalTwinDataPathsProvider`
   variant (`skeleton_flag`) supplied (a) the segmentation and (b) the depth
   in each ticket. If they differ between seg and depth, that alone is the bug.
2. Load BOTH depth variants for the same frames. At dynamic-seg pixels report:
   median |depth_with_skeleton − depth_without| (m), fraction of dynamic
   pixels where the two differ by > 5 cm (= "the person is actually rendered
   into the with-skeleton depth"), and the median of each variant there.
   This settles whether GT "sees" the hand or pretends it doesn't exist.
3. Re-state #28's dyn-depth medians under the *correct* variant (whichever
   that turns out to be), so the "hands live at 0.26–0.94 m" claim is either
   confirmed, corrected, or withdrawn.

## Part 2 — the synthetic stream as the hands-free counterfactual (human's idea)

4. For the same frames, check whether the person is rendered in the synthetic
   RGB stream: simplest read = per-frame |synthetic − real| restricted to
   dynamic-seg pixels vs static pixels (a person present only in real shows a
   large gap concentrated on dynamic pixels); include a 2–3 frame visual grid
   (real | synthetic | seg | both depth variants) as a PNG on `results`.
5. If synthetic is indeed hands-free: DA3-Small depth on real vs synthetic
   input for high-dyn_frac frames, scored against the SAME GT variant
   (protocol of record), static and dynamic pixels separately. This replaces
   #31's mean-color masking with a photometrically natural "remove the hands"
   — report whether #31's "hands ≈ plain occlusion" conclusion survives.
   Also worth one line: does the meal_seq131 anomaly (random mask beating
   vanilla by ~4°) reproduce under the synthetic stream?

## Acceptance

- JSONs + the visual grid on `results`, and a comment answering, in order:
  (i) which variants #28/#31 actually used; (ii) does with-skeleton depth
  contain the person; (iii) are #28's dyn-depth numbers confirmed/corrected;
  (iv) is synthetic hands-free; (v) does the #31 conclusion survive the
  synthetic-stream re-test.
