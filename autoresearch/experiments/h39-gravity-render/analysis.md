# H39 — the IMU pays, on the model that needs it, and the integral checks out

**Status: DA3-Small passes all three bars. VGGT-Omega fails B1 and B3, and that
failure is the same finding from the other side.**

Six recordings, 60 frames each (48–51 survive the timestamp join), real head
roll read from the MPS closed-loop trajectory. Three arms per frame: the view
rendered at `roll_deg` 0 (`device` — what the pipeline does today), `+ψ` and
`−ψ`. Scored on the per-frame intersection of the three arms' coverage.

## DA3-Small — all three bars pass

| recording | median &#124;ψ&#124; | `grav_p` | `grav_m` |
|---|---|---|---|
| seq136 | 2.52° | **−2.69%** | +3.57% |
| seq138 | 2.84° | +1.33% | +3.16% |
| seq142 | 6.19° | **−3.33%** | +12.61% |
| seq144 | 6.86° | **−5.40%** | +9.00% |
| seq145 | 6.70° | **−2.90%** | +18.91% |
| seq149 | 2.90° | **−1.21%** | +12.42% |
| **mean** | | **−2.37% ± 2.26** (5/6) | **+9.94%** |

**B1 — the integral survives a direct test.** H35 computed the expected cost of
ignoring roll at **1.80%** by integrating a *synthetic* roll curve against the
*measured* roll distribution. This recovers **2.37% ± 2.26** by a direct A/B on
real frames with their real roll. Two routes with almost nothing in common
agree to within a third.

That is the first time anything in this line has been checked against an
independent estimate of the same quantity rather than against a null.

**B2 — the sign is identified by shape, not by score.** The wrong sign costs
**+9.94%**, 4.2× the magnitude of the right sign's gain. Both signs were
rendered precisely so that the convention could not be chosen by looking at
which number was smaller; the asymmetry is what identifies it.

**B3 — it is the roll, and the dose relation is clean.** Per-frame, the gain is
**+10.14%** on frames with &#124;ψ&#124; ≥ 8° and **−1.26%** on frames with
&#124;ψ&#124; < 3°. The near-level frames going slightly *negative* is the
honest null: with no roll to remove, aligning the render only costs resampling.

**The sub-prediction recorded before running also holds.** H17.1 read seq142–148
as a contiguous block sitting 4–5° further tilted — a per-session mounting
offset rather than head motion, which is exactly what a gravity-aligned render
removes. Mean gain on seq142/144/145 is **−3.88%** against **−0.86%** on
seq136/138/149.

## VGGT-Omega — nothing to collect

| statistic | value |
|---|---|
| better arm, pooled | **+0.10% ± 3.26** (better on 3/6) |
| wrong sign | +0.72% |
| per-frame, &#124;ψ&#124; < 3° | −2.46% |
| per-frame, &#124;ψ&#124; ≥ 8° | −5.59% |

B1 fails: H35 said 0.46% was available and the measurement finds nothing —
the two arms are indistinguishable (+0.10% vs +0.72%, against a within-arm sd
of 3.26).

**B3 fails in the informative direction.** The per-frame gain is *more negative*
at high roll (−5.59%) than at low roll (−2.46%). Reading offered, and it is a
reading and not a measurement: for a model that is already roll-invariant there
is no roll to recover, so all that is left is the resampling difference between
the two renders — and that difference grows with the angle rotated through. The
same cost is present in DA3-Small and simply loses to a recovery ten times
larger.

**This does not contradict H35.** H35 said 0.46% is *available*; H39 says it
cannot be collected this way, because collecting it costs at least that much.
Second time the price and the prize came out the same size for this backbone —
H38 measured +6.0% against +6.8%.

## What the roll line now says, end to end

> **Read the roll from the IMU and fold it into the rendering warp — if you are
> running a single-image backbone.** It is worth about 2.4% on ordinary indoor
> footage and about 10% on the frames that actually need it, it costs one
> rotation of the sampling grid, and it needs no labels, no training and no
> change to the model. On a multi-view-pretrained backbone there is nothing to
> collect; the operation costs roughly what it would save.

That answers the fourth part of the question this line opened with, in the
affirmative, for exactly one of the two model families — and H38 already
established the other half of the how: **in the resampling, never after it**.

## Limits

* Six recordings, one apartment, one device. The B1 agreement is between two
  estimates that share the same footage, though not the same method.
* 9–12 frames per recording were dropped because the depth frames start about a
  second before the trajectory and no roll can be read for them.
* `grav_p` was better on 5 of 6 for DA3-Small; seq138 (+1.33%) is the exception
  and it is one of the low-roll recordings, consistent with B3.
* The VGGT-Omega reading — resampling cost growing with the angle — explains
  the sign of B3's failure but was not measured separately. No arm isolates it.
