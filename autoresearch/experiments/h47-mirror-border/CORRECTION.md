# Correction — the 89° view never had black corners, so H47 tested the wrong thing

**Found 2026-09-09**, tenth tick of the outage, while checking the geometry of a
proposed reciprocal arm. The repo already contained the right number.

## The arithmetic error

A square view's corner ray is `atan(√2 · tan(fov/2))`, **not** `(fov/2) · √2`.
I used the second in H40's and H47's protocols:

| fov | I wrote | correct | Aria cone |
|---|---|---|---|
| 60° | 42.4° | **39.2°** | 54.83° |
| 89° | 62.9° | **54.2°** | 54.83° |

`run_h14_sweep.sh` has carried the correct table since long before these ticks —
*"89 → corner ray 54.2, 100% real frame content"*, with black appearing only from
95° — and `findings.md:754` says the same: *"corner ray 54.3° < 54.83° at every
roll, fill 1.000 — no padding anywhere."*

## What this voids

**H47's premise.** It was written as *"the 89° view has black corners, the 60°
one does not; remove the border and see if the effect survives."* **Neither view
has black corners.** What H47 actually varied is the **field of view**.

**H46's candidate explanation for the non-replication.** *"Aria's 89° view has
black corners and ScanNet++'s does not, and the off-centre principal point makes
that pattern asymmetric under mirroring"* — the whole suspicion rests on the same
miscalculation and is **void**. The four candidate differences H46 listed
(content, capture, ground truth, scored region) therefore stand **unreduced**;
nothing has been eliminated.

**H44/H45's black-corner mechanism**, same origin, same status.

## What survives, and is strengthened

**H47's conclusion.** *The mirror effect is not caused by black corners* — now
trivially true, because there were never any black corners in either arm. The
conclusion needed less evidence than it got, and the measurement that supports it
is still valid.

**H40's factorial is untouched.** Its border arm applies a **deliberate inscribed-disc
mask**; that border is real and measured. Only its incidental remark about the
89° view "already running off the cone" is wrong.

## What the H47 measurement now says

Restated correctly, H47 compared **89° against 60° field of view**, both fully
filled, and found the effect at 60° is **60–90%** of the 89° value:

| backbone | 89° | 60° | ratio |
|---|---|---|---|
| `da3:small` | +214.5% | +154.8% | 0.72 |
| `da3:large` | +295.5% | +261.4% | 0.90 |
| `vggt` | +59.0% | +45.7% | 0.79 |
| `vggt_omega` | +50.1% | +29.9% | 0.60 |

So **the effect grows with field of view**. That is a real observation and it was
sitting under a wrong label. It is *not* offered as a mechanism — a wider view
changes the resampling, the angular content and the scored region together.

## The lesson

**The number was already in the repo, in two places, correct.** The error came
from re-deriving a geometric quantity instead of looking it up — the same failure
mode as H39's convention bug, where re-deriving `R_cd` instead of copying it cost
a discarded launch. *Copy a constant from the code that established it.*

And it survived three protocols because it was **conservative in the safe
direction**: the true corner ray is *smaller* than I wrote, so every "this view
fits inside the cone" claim held anyway, and nothing failed loudly.
