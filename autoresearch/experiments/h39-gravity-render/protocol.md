# H39 — fold the real gravity vector into the warp. Does it actually pay?

## What H38 left as an inference

H38 measured that rotating the picture costs more than the roll it removes, and
recommended the alternative: **render the view gravity-aligned in the first
place**, so no border is ever created. But that recommendation rests on an
inference, not a measurement — H38's `raw@0` is a view aligned to the *device*,
and device and gravity coincide only when the head is level.

This measures the deployment operation directly, on real frames, with the real
head roll read from the MPS closed-loop trajectory the same way H17.1 read it.

It also closes a loop. H35 computed the expected cost of ignoring roll by
integrating a *synthetic* roll curve against the *measured* roll distribution:
**1.80% ± 1.31 for DA3-Small, 0.46% ± 0.25 for VGGT-Omega**. Nothing has ever
checked that integral against a direct A/B on real frames. This is that check.

## Design

Three arms per frame, same frames, same backbone, scored on the intersection of
the three arms' coverage (rolling a square view moves its corners, so the
covered part of the cone differs and must be intersected per frame):

| arm | view rendered at | residual roll |
|---|---|---|
| `device` | `roll_deg = 0` | ψ — what the pipeline does today |
| `grav_p` | `roll_deg = +ψ` | 0 or 2ψ |
| `grav_m` | `roll_deg = −ψ` | 2ψ or 0 |

**Both signs are rendered on purpose.** The convention chain — `rolls_deg`'s
`atan2`, `RolledView.rotation`, and what `RT.Rig` does with it — is ambiguous
enough that asserting a sign would be guessing, and picking the sign that scores
better afterwards would be choosing the answer. One arm leaves a residual roll
of zero and the other leaves 2ψ, so the identification is made by the **shape**
of the result, which B2 states as a falsifiable prediction.

Six recordings spanning the measured roll range (H17.1 per-sequence median
|roll|): seq145 7.50°, seq144 6.61°, seq142 6.46° at the high end; seq136 2.69°,
seq138 2.94°, seq149 2.92° at the low end. 60 frames each, two backbones — the
two extremes of H35's integral. Evaluation only.

## Bars, locked before running

* **B1 — does the integral survive a direct test?** The better gravity arm's
  improvement over `device`, pooled over the six recordings, is within a factor
  of **3** of H35's independently computed expected roll cost (1.80% for
  DA3-Small, 0.46% for VGGT-Omega). **Falsified** if the improvement has the
  wrong sign, or lands outside [⅓×, 3×] of the prediction. A factor of 3 is
  wide on purpose: H35 integrated a curve measured at ±10/20/30/40° over a
  distribution whose mass is under 5°, and the two arms here differ in
  resampling as well as in roll.
* **B2 — is the sign identified by shape rather than by score?** The worse
  gravity arm must be worse than `device` by roughly what a 2ψ roll costs on
  the H16 curve, i.e. it must be **clearly** worse, not marginally.
  Operationally: `worse_arm − device` is at least **3×** the magnitude of
  `device − better_arm`. **Falsified** if the two gravity arms land within
  noise of each other, in which case nothing here is identified and B1's
  number is not interpretable.
* **B3 — does the gain scale with the roll?** Per-frame, the improvement of the
  better gravity arm over `device` is larger on frames with |ψ| ≥ 8° than on
  frames with |ψ| < 3°. **Falsified** if it is not, which would mean whatever
  the arms differ by is not the roll.

## The sub-prediction worth recording

H17.1 found that seq142–148 sit 4–5° further tilted as a **contiguous block**,
which it read as a per-session mounting offset rather than head motion. A
constant per-session offset is exactly what a gravity-aligned render removes.
So the gain should be **larger on seq142/144/145 than on seq136/138/149**, and
by more than the |ψ| difference alone would suggest. Not a bar — B3 already
tests the dose relation — but recorded before running.

## Cost

Evaluation only: 2 backbones × 3 arms × 60 frames × 6 recordings. Split across
the two GPUs by backbone.
