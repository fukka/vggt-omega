# H41 — the roll curve, measured properly, and H40's argument becomes arithmetic

**Status: B1, B2 and B3 all PASS. And H32's near-zero anomaly is explained
away.**

Thirteen recordings, 20 frames each, `da3:small`. Each frame rendered at
`roll_deg = ψ + d`, so the residual roll is exactly −d and `d = 0` is that
frame's own gravity-aligned reference. Every render normalised by that frame's
own `d = 0` error.

| residual | g | sd | | residual | g | sd |
|---|---|---|---|---|---|---|
| −22° | 1.3068 | 0.099 | | +2° | 1.0111 | 0.035 |
| −15° | 1.1577 | 0.064 | | +4° | 1.0341 | 0.027 |
| −11° | 1.0859 | 0.046 | | +6° | 1.0379 | 0.040 |
| −8° | 1.0499 | 0.038 | | +8° | 1.0642 | 0.045 |
| −6° | 1.0305 | 0.037 | | +11° | 1.0987 | 0.048 |
| −4° | 1.0147 | 0.028 | | +15° | 1.1883 | 0.068 |
| −2° | 1.0136 | 0.037 | | +22° | 1.3538 | 0.080 |
| **0°** | **1.0000** | — | | | | |

## H32's "the curve is not monotone near zero" was an artefact of its zero

H32 reported the penalty **falling** below baseline at −5° (−1.1% ± 2.8,
negative on 9/13) and flat from −10° to +5°, and called B3 "the wrong sanity
check". With an exact per-frame gravity-aligned zero the curve is **monotone
from zero on both sides**, with a clean minimum at 0 and no dip anywhere.

The explanation is the zero itself. H32's 0° was the **device**-aligned render,
and those frames carry a median 4° of real head roll, so its "0°" was already
several degrees off level. Sweeping around that point puts the true minimum at
roughly −4°, which is exactly where H32 saw its dip. **A curve measured against
the wrong origin looks non-monotone; nothing about the model was odd.**

This is the second correction in two experiments that comes from fixing a
reference rather than from measuring something new.

## H40's Jensen argument, now arithmetic

H40 argued that H39b's "price" S is mostly the curvature of this curve, because
`grav_p` has residual 0, `grav_m` has 2ψ and `device` has ψ. The curve predicts
both of H39b's numbers with no free parameters:

| at ψ = 11° | predicted from the curve | H39b measured |
|---|---|---|
| prize A = [g(2ψ) − 1] / 2g(ψ) | **+16.10%** | +21.52% |
| price S = [1 + g(2ψ)] / 2g(ψ) − 1 | **+7.11%** | +11.51% |

Both inside the locked factor-of-2 windows, and **both under-predict in the
direction the construction requires**: the |ψ| ≥ 8° cell has median 11° but
reaches 28°, so 2ψ reaches 56°, far outside the ±22° grid. The interpolation
clamps at the endpoint and therefore *must* understate g(2ψ). The shortfall is
not a discrepancy; it is the clamp.

**The low-roll cell reconciles three experiments.** At ψ = 1.5°:

| | value |
|---|---|
| Jensen prediction for S | **+0.29%** |
| H39b measured S | +0.95% |
| H40's measured interpolation cost | +0.81% |

Jensen contributes almost nothing there, and what H39b measured is essentially
what H40 charged for two grid samples. Three experiments, three methods, one
number.

## What this settles

* **H40's correction to H39b stands, quantitatively.** The symmetric part of
  H39's two arms is mostly the convexity of the roll curve, not a price. The
  label "the price: resampling" was wrong; "the symmetric part" is right.
* **H39's headline is still untouched** — −2.65% ± 2.43 is a direct A/B with no
  decomposition in it. That has now survived three experiments trying to
  reinterpret its parts.
* **The curve itself is now the line's best-measured object**: 13 recordings,
  15 angles, an exact per-frame zero, recording-to-recording sd under 0.05
  inside ±11°.

## The asymmetry, at this angle and against this origin

+22° costs 1.354 and −22° costs 1.307 — positive residual costlier, matching the
sign H32 and H33 found, but by **3.6%**, not H32's 24.4% against 56.5% at ±30°.
The two are not comparable: H32 measured against a device zero, at a larger
angle, on frames whose own roll biased one side. **This does not resolve the
asymmetry question and is not offered as doing so** — H33 left it undecided and
H34 was void trying, and nothing here changes that.

## Limits

* One backbone. `da3:small` was chosen because it is the one with a roll penalty
  large enough to have a shape; the curve is not claimed for the others.
* ±22° is the outer edge. The frames themselves reach 28°, and every quantity
  that needs g beyond 22° is an extrapolation, which is why B2 and B3
  under-predict.
* 20 frames per recording, 16–19 surviving the timestamp join.
