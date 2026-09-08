# H41 — the residual-roll curve, and a quantitative test of H40's Jensen claim

## What H40 left as an argument

H40 corrected H39b's labelling: its "price" S is not pure resampling, because
in H39 `grav_p` has residual roll 0, `grav_m` has 2ψ and `device` has ψ, so the
symmetric part contains

    S = [e(0) + e(2ψ)] / (2·e(ψ)) − 1

which is a **Jensen gap** whenever the roll penalty is convex, plus whatever the
operation itself costs. H40 measured the interpolation part at ~1% and argued
the rest is curvature. **That is an argument, not a measurement**, and it rests
on a roll curve that has never been sampled where the data actually lives: H16
measured ±10/20/30/40 on one recording, H28 re-measured ±20/30/40 on thirteen,
and H32 found the curve *not monotone* near zero (−1.1% at −5°, +0.1% at −10°).
Ninety percent of real frames sit inside ±10°.

## The measurement

Not a synthetic curve on nominally level frames — a **residual**-roll curve with
an exact per-frame zero. For each frame, ψ is read from MPS (H17.1's method) and
the view is rendered at

    roll_deg = ψ + d      for d ∈ {0, ±2, ±4, ±6, ±8, ±11, ±15, ±22}

so the residual roll the backbone sees is exactly **−d**, and `d = 0` is that
frame's own gravity-aligned reference. Every render is normalised by that
frame's `d = 0` error, so the curve g(residual) = e(residual)/e(0) needs no
binning and carries no drift between recordings.

`da3:small` only — the backbone where the roll penalty is large enough to have a
shape. Thirteen recordings, 20 frames, scored exactly as H39 scored, on the
intersection of every arm's coverage.

## Bars, locked before running

* **B1 — is the curve convex where the Jensen argument needs it?**
  g(22°)/2 + 1/2 > g(11°), i.e. the Jensen gap at ψ = 11° is positive.
  **Falsified if it is ≤ 0**, which would kill H40's correction and hand H39b's
  label back.
* **B2 — does the curve predict H39b's *prize*?**
  A_pred(ψ) = [g(2ψ) − 1] / (2·g(ψ)) at ψ = 11° lands within a factor of 2 of
  H39b's measured **+21.52%** on frames with |ψ| ≥ 8°. **Falsified outside
  [10.8%, 43.0%]**.
* **B3 — does it predict H39b's *price*?**
  S_pred(ψ) = [1 + g(2ψ)] / (2·g(ψ)) − 1 at ψ = 11° against H39b's measured
  **+11.51%**. **Falsified outside [5.8%, 23.0%]**.
  If S_pred lands well *below* the measurement, the shortfall is a genuine
  operation price and H39b's label was partly right after all — that outcome is
  as informative as agreement and is not a failure of the experiment.

## What the low-roll cell says either way

H39b measured S = **+0.95%** and A = **+0.26%** on frames under 3°. Both are
near zero, and H40 puts interpolation at ~0.81%. If the curve is flat near zero
(H32's finding), Jensen predicts ≈0 there and the measured 0.95% is almost
entirely the operation. That is a consistency check across three experiments and
it is reported whatever B1–B3 do.

## Cost

Evaluation only, one backbone: 15 renders × 20 frames × 13 recordings. Split
across the two GPUs by the sign of d, with d = 0 rendered on both so each half
normalises itself.
