# H2.0b — Is the (θ × depth) structure real, or the affine's seesaw?

Locked before running. Pre-announced in run_008b's analysis and on the
dashboard. The risk: one frozen scale-shift per frame can redistribute a
structured bias across the (θ, depth) plane, so both the near-rim blowup and
the far-rim inversion could partly be fit artifacts (the repo's U-shape
warning, in 2D).

**Method (alignment-free, per cell).** For each frame: residual map
`r = log(GT_range) − log(pred_range)` — no alignment applied at all. Per
(θ-bin × depth-band) cell, pooled over frames:
- **bias** = median(r in cell) − median(r in frame) (per frame, then pooled
  median): where the model is systematically off, relative to its own frame
  level — reproduces the miscalibration geometry without any affine;
- **dispersion** = MAD of (r − cell median) within cell, pooled: how *noisy*
  the prediction is there, independent of every scale/shift question.

**Predictions.**
1. The near-field-rim cells (0–2 m, θ>38°) show BOTH large |bias| and large
   dispersion — a genuinely worse prediction, not a misallocated affine.
2. The far-rim inversion: if it is real, far-rim cells show smaller dispersion
   than far-center cells; if it was the affine's seesaw, their dispersions are
   comparable and only the bias map inverts.

**Interpretation rules.** Dispersion is the artifact-proof read (invariant to
any per-frame monotone affine in log space up to its slope; scale_shift in
depth space is not exactly affine in log, noted). Bias inherits "relative to
the frame's own level", which is the honest analogue of what any single-frame
consumer experiences.

**Refutation.** If near-rim dispersion ≈ center dispersion at fixed depth, the
run_008b blowup was mostly calibration geometry — the adapter rung (H2.1
recalibration table) becomes MORE attractive, not less; but claims of "the
model can't see the near rim" would be withdrawn in favor of "the model
mis-scales the near rim".
