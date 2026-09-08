# H22 — all three bars pass. Spearman +1.000, and matching only matters at the extremes.

`results/rim_density.json`. Six lens shapes, same machinery as H21, every arm
warped once from the real Aria camera.

## B1 — monotone, as strongly as it can be

Row mean = the average gain a curve fitted on that geometry gives across all six
target lenses.

| fitted on | rim area share | row mean, same room | row mean, rearranged |
|---|---|---|---|
| orthographic | 68.3% | **−30.5%** | **−15.6%** |
| equisolid | 73.5% | −29.0% | −14.1% |
| equidistant | 75.0% | −27.7% | −13.3% |
| stereographic | 77.9% | −24.7% | −11.1% |
| the real lens (aria_kb4) | 79.4% | −21.8% | −9.3% |
| rectilinear | 86.6% | −11.3% | −2.4% |

**Spearman ρ = +1.000 on both recordings.** Perfectly monotone, and in the
direction opposite to H21's stated mechanism: **the geometry that gives the rim
the fewest pixels makes the best fitting set.**

**B2 passes**: orthographic beats equisolid by row mean (−30.5 vs −29.0;
−15.6 vs −14.1). **B3 passes**: rectilinear is the worst by a wide margin.

## The practical number, on a fixed target

Applied to the **real Aria lens**, which is the only target that matters in
deployment:

| curve fitted on | → real lens, same room |
|---|---|
| equisolid | **−26.8%** |
| orthographic | −26.2% |
| equidistant | −26.0% |
| the real lens's own | −21.0% |

Any of the three rim-compressing geometries beats the real lens's own curve by
about 5–6 points. Between equisolid and orthographic the difference is 0.6
points, inside noise — **B2 passing does not mean the recommendation should
switch to orthographic.** "Fit on a rim-compressing geometry" is the finding;
naming one is not supported.

## Where H21's "matching buys nothing" holds, and where it stops

The raw diagonal looks better than the off-diagonal (−27.6 vs −23.5), which
would seem to contradict H21. It does not: the good fitting geometries are also
the easy targets, so the diagonal is confounded by the row and column effects
being correlated. The clean test is an additive model, `M[i,j] ≈ μ + r_i + c_j`,
and then how much the diagonal beats it:

| lens | extra benefit from its own curve |
|---|---|
| orthographic | **8.5 pts** |
| rectilinear | **8.3 pts** |
| equisolid | 1.3 |
| the real lens | 1.1 |
| stereographic | 0.7 |
| equidistant | 0.5 |

Residual sd is 2.98. So among the four **realistic fisheye shapes** — including
the real lens — matching is worth 0.5 to 1.3 points, comfortably inside noise.
H21's conclusion holds and is now much better supported: **5 of 6 target lenses
do better with somebody else's curve.**

It stops holding at the two extremes, where a lens's own curve is worth 8 points.
Those renderings are far enough outside the fisheye family that only a curve
fitted on that same geometry describes them.

Variance decomposition: fit geometry 49.8% / 36.7%, target 39.4% / 54.0%,
everything else ~10%. The additive picture dominates.

## Mechanism: a hypothesis, explicitly not a finding

H21's guess was wrong in direction, so this one is labelled clearly and left
untested. Rim-**expanding** geometries upsample the rim: more pixels, but
interpolated from the same underlying samples, so no new information plus
interpolation blur. Rim-**compressing** geometries downsample: fewer pixels, each
a genuine average of real samples. On that reading the fit is limited by the
**quality** of rim pixels, not their count — the same shape of explanation as
H20's motion result.

**A competing explanation that this experiment cannot rule out:** row and column
effects are ordered the same way, so the whole ranking may be "how much this
rendering was damaged by resampling", affecting fitting and scoring alike. The
practical recommendation survives either way, because the row effect is measured
against *fixed* targets — but the mechanism does not.

Falsifiable next step: if it is upsampling loss, rendering the source at higher
resolution before warping should shrink the spread; if it is something about the
angular distribution itself, it should not.
