# H40 — at a matched 30°, the resampling is free and the border is everything

**Status: B1 PASSES on both backbones. B3 PASSES cleanly — the two costs
compose. B2 FAILS on DA3-Small, and that failure corrects H39b.**

Fourteen recordings, 20 frames each. Construction check clean: **no black in any
crop, in any recording, at either rotation.**

| | `da3:small` | `vggt_omega` |
|---|---|---|
| two resamplings, no border | **+0.81% ± 0.67** (positive 13/14) | **+1.19% ± 0.84** (14/14) |
| a border, no extra resampling | **+105.73% ± 29.61** | **+6.63% ± 8.26** |
| both | +114.50% ± 31.97 | +7.21% ± 7.81 |
| multiplicative prediction | +107.40% | +7.95% |

## B1 — the caveat §03z had to carry is gone

§03z's headline was "the border was ~95% of de-rotation's price", ratio 0.047,
and its own callout named the axis never varied: H38 measured at ±30° and H39b
at the real roll. **At a matched 30°: the border costs 105.73% and the whole
rest of the rotation costs 0.81%.** That is a ratio of **130×** for DA3-Small
and 5.6× for VGGT-Omega, against a bar of 4×.

> **The border is not most of the price. It is essentially all of it.**

The angle-matched number is *stronger* than the upper bound §03z published, not
weaker — which is not what an unvaried axis usually does.

## B3 — the two costs multiply, and that is worth knowing

`both` lands within 7% (DA3-Small) and 9% (VGGT-Omega) of the product of the two
single-factor costs. Border and resampling are **independent multiplicative
factors** on the error, not competing or interacting ones. Nothing in this line
had established that; it makes "price = border × resampling" the right sentence
and licenses reasoning about them separately.

## B2 fails, and it corrects H39b's labelling

The pre-registered expectation was that resampling at 30° would exceed H39b's
real-roll price of 3.03%, since H39b's P2 showed that price growing with the
angle. It does not: **0.81% at 30° against 3.03% at 4–7°.**

The two are not the same operation, and that is the finding:

* **H40's `rt`** re-interpolates an already-rendered view. Same source pixels,
  twice through a grid sample. That costs **~1%**.
* **H39's gravity render** warps from a *differently oriented view*, so it reads
  a different region of the lens, at a different local sampling density, with
  different distortion. H39b's S is the cost of *that*, and only about a point
  of it is interpolation.

**And a second term H39b did not account for.** In H39, `grav_p` has residual
roll 0 and `grav_m` has residual roll 2ψ, while `device` has ψ. So

    S = [f(0) + f(2ψ)]/2 − f(ψ)   (plus the true operation cost)

and the roll penalty f is convex (flat to ±20°, steep after — §02, §03u). By
Jensen's inequality that bracket is **positive and grows with ψ**, which is
exactly the growth H39b's P2 reported and attributed to resampling. So a
meaningful part of H39b's S — most of it at high roll — is **the curvature of
the roll penalty, not a price at all**.

**What this does and does not touch.** H39's headline is unaffected: −2.65% ±
2.43 is a direct A/B between `device` and `grav_p`, with no decomposition in it.
What needs restating is H39b's *interpretation* of S as "the price: resampling".
Correct version: **S is the symmetric part of the two arms, which contains the
operation's real cost (~1 point of interpolation, plus reading a different part
of the lens) and a Jensen term from the convexity of the roll curve.** The
prize/price crossing and the gating advice survive as descriptions of the
symmetric and antisymmetric parts; they are not clean cost accounting.

## The construction, and the design that was void first

**The first construction was wrong and its own sanity check said so**, before a
single number was interpreted. I reasoned that a 60° view's corner ray of 42.4°
sits inside Aria's 54.83° cone, so rotating it makes no border. That confuses
the view's **rays** staying inside the cone with the square **raster** keeping
its corners: rotating any square raster by 30° throws 15.3% of it outside
itself. The arms it produced — DA3-Small "resampling" **+80.4%** — were
measuring a border again, the one quantity the experiment exists to remove.

The repair rotates on a canvas of side ≥ vs·√2 and crops afterwards. The
canvas's inscribed disc contains every corner of the rotated crop, so the crop
stays full; at 630 px and 60° the canvas is 896 px spanning 78.8° with its own
corners at 49.1°, still inside the cone, so nothing is black anywhere. Printed
per recording and per backbone: **0.0000 black, direct and rotated.**

## Limits

* One angle (30°). The multiplicative composition is established there and
  nowhere else.
* A 60° crop, not the 89° view the rest of this line uses, so absolute errors
  are not comparable to other sections — every number here is a ratio within
  this experiment.
* The border here is an inscribed-disc annulus. H38's was four corner triangles.
  H37 and H38 both found shape matters little, but they did not test it at this
  FOV.
