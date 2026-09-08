# H39 on thirteen recordings, and H39b — what the operation costs vs what it buys

## Part 1 — H39 rerun on all thirteen never-used recordings

Standing rule 3 asks for thirteen before a deployment number goes out, and
H39's DA3-Small gain had an sd nearly the size of its mean. Extended.

| backbone | 6 recordings | **13 recordings** | bars |
|---|---|---|---|
| `da3:small` | −2.37% ± 2.26 (5/6) | **−2.65% ± 2.43 (11/13)** | B1 B2 B3 **pass** |
| `vggt_omega` | +0.10% ± 3.26 (3/6) | −1.04% ± 4.11 (7/13) | B1 pass, **B2 fail**, B3 fail |

**DA3-Small holds and tightens.** The gain is 2.65% against H35's independently
computed 1.80%, still inside B1's window. Per-frame: **+10.02%** at |ψ| ≥ 8°,
**−0.68%** at |ψ| < 3°. Eleven of thirteen recordings improve.

**B2 earned its keep a second time.** On thirteen recordings VGGT-Omega's better
arm reads **−1.04%**, which passes B1 and would make a quotable headline — "even
the multi-view model gains 1%". B2 says it is not interpretable: the wrong-sign
arm is **+0.08%**, not the ≥3× worse that a real roll removal requires, and the
better arm wins on 7 of 13, a coin flip. Nothing is identified, so nothing is
claimed.

That is the second time this bar has caught something it was not written for.
It was written to stop a *convention* being chosen by score; it has now stopped
a discarded run (both arms equally bad) and a spurious positive.

## Part 2 — H39b, splitting the two arms

    grav_p = S − A      S = (grav_p + grav_m)/2   the price: same for both signs
    grav_m = S + A      A = (grav_m − grav_p)/2   the prize: the roll removed

| backbone | price S | prize A | net | pays? |
|---|---|---|---|---|
| `da3:small` | **+3.03% ± 2.78** | **+5.68% ± 3.40** | −2.65% | **yes** |
| `vggt_omega` | −0.48% ± 3.07 | +0.56% ± 1.81 | −1.04% | not established |

### P1 passes on DA3-Small, and this is the number the recommendation rests on

H38 measured the price of the same rotation **with a border**: 64.1%. Folding it
into the warp instead costs **3.03%** — **0.047 of it**.

> **The border was essentially the whole price. The resampling is almost free.**

That is what makes "use the gravity vector in the resampling, not after it"
worth saying: not that the sign is easier to get right, but that ~95% of the
cost of the obvious approach comes from an artefact the warp never creates.

**Name the axis this was never varied along: the rotation angle.** H38's 64.1%
is at ±30°; H39's S is at the real roll, median 4–7°, and P2 shows S grows with
the angle. So 0.047 is an upper bound on the advantage, not an angle-matched
ratio. The nearest matched-ish comparison inside this data is the |ψ| ≥ 8° cell
(median ~11°), where S is **11.51%**, still under a fifth of H38's ±30° price.
Isolating S at 30° needs frames with a real 30° roll, and ADT's maximum is ~28°
with p99 at 21.8°, so it cannot be done on this data.

### P2 passes, P3 fails on VGGT-Omega — and that failure is a measurement

| backbone | S at &#124;ψ&#124;<3° → ≥8° | A at &#124;ψ&#124;<3° → ≥8° |
|---|---|---|
| `da3:small` | +0.95% → **+11.51%** | +0.26% → **+21.52%** |
| `vggt_omega` | +1.86% → **+5.62%** | −0.38% → +0.72% |

P2 holds on both: the price grows with the angle rotated through, which is what
a resampling cost must do.

P3 asked for A's slope to beat S's on both backbones. It does on DA3-Small
(21.3 against 10.6) and **fails on VGGT-Omega** (1.1 against 3.8). H39 had
offered exactly this as a *reading* — "for an already roll-invariant model there
is no roll to recover, so only the resampling is left, and it grows with the
angle". It is now measured: **the price rises with the angle and the prize does
not**, which is why the operation cannot pay on that backbone at any roll.

### The bug this part produced, recorded rather than deleted

VGGT-Omega's S is −0.48% ± 3.07 — indistinguishable from zero. The first run of
the script divided H38's 7.4% by it and printed **"7.4e9× cheaper"**. That is
H33's divide-by-noise trap in a new costume, three experiments later, and it
took a glance at the output to catch. The script now refuses to print a ratio
unless S is distinguishable from zero, and P1 is reported as undecided for that
backbone rather than passed.

## What changes in the reports

* DA3-Small's deployment number moves from 2.37% ± 2.26 (6) to **2.65% ± 2.43
  (13, 11/13)**, and the high-roll cell from +10.14% to **+10.02%**.
* VGGT-Omega's line changes from "nothing to collect" to "**nothing
  identified**" — the arms do not separate, so neither the 6-recording +0.10%
  nor the 13-recording −1.04% is a result.
* The §03x recommendation gains its quantitative justification: the border was
  ~95% of de-rotation's price, measured, with the angle caveat stated.
