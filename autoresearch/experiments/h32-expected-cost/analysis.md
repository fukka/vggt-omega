# H32 — roll costs 1.46%. But 1.5% of frames carry a quarter of it.

`results/dense_seq*.json`, 11 angles × 13 recordings, combined with H17's 2.5°
histogram over 60,105 frames. No new data.

## B1 passes — and the line's headline is now a number

**Expected whole-image penalty under the real roll distribution: 1.46% ± 1.09**
across thirteen recordings (range 0.29–3.94; seq136 1.15%). The bar was 5%.

This replaces a coincidence with a measurement. The line has been saying "the
model is flat to ±20°, real roll's p99 is 21.8°, therefore 98.5% of frames are
comfortable" — three numbers that were never multiplied together. **They now
have been, and the answer is small.** Ignoring roll costs about one and a half
percent of depth error.

## B2 fails, and this is the useful part

Frames beyond ±20° are **1.48%** of the data and carry **26.5% ± 16.8** of the
expected cost — an **18× concentration**. The bar was 25%, so this fails, if
narrowly and with a wide spread across recordings.

The engineering reading changes accordingly. *Roll in general* is not worth
handling: 98.5% of frames sit in a region costing essentially nothing. **The
rare large roll is worth handling**, because superlinear growth means a
1.5% tail buys a quarter of the total. Those are different products: a gravity
prior applied everywhere versus a detector that flags the rare tilted frame.

## B3 fails, and the failure is informative rather than fatal

The penalty is not monotone in |θ|. At −5° it is **−1.1% ± 2.8**, negative on
9 of 13 recordings; at −10° it is +0.1%. The curve is **flat to within noise
between −10° and +5°** rather than rising from zero.

That does not undermine B1 — a flatter bottom makes the integral smaller, not
larger — but it does mean the curve should not be described as monotone, and
B3 as written was the wrong sanity check.

**An alignment worth noting and not over-reading.** The measured roll
distribution has a signed median of **−2.70°** (H17, which checked and rejected
calibration as the cause). The penalty curve's minimum sits near **−5°**. The
wearers' typical tilt lands close to the model's cheapest angle. On 9/13 that
is a hint, not a finding — sign test p ≈ 0.13 — and no mechanism is offered.

## What the ± averaging hid

Every published roll figure averaged the ± pair. Measured separately:

| | −θ | +θ | +θ costlier on |
|---|---|---|---|
| 15° | −? | — | 9/13 |
| 20° | +6.9% | +17.8% | 8/13 |
| 30° | **+24.4%** | **+56.5%** | **11/13**, gap +32.0 pts |

**Positive roll costs 2.3× what negative roll costs at 30°.** At 15–20° the
asymmetry is not established (8–9 of 13). So it is a large-angle effect, and the
published "±30° = +46%" is the midpoint of 24% and 56%.

This was invisible until the grid was dense enough to be worth reading
per-sign — and it is the second time in this line that averaging two arms hid
something (the first was H28's rim/centre band).

## The corrected practical statement

> Ignoring head roll costs about **1.5%** of depth error on ordinary indoor
> footage. That is cheap enough to ignore. If you do decide to spend effort on
> it, spend it on **positive roll beyond 20°** — 1.5% of frames, a quarter of
> the cost, and more than twice as expensive as the same angle the other way.
