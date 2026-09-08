# H32 — what does real head roll actually cost?

## The question this line has never answered

The roll result is stated as two numbers that were measured separately and then
put side by side:

* the model is roughly flat to **±20°** (H16/H28);
* real head roll has **p99 = 21.8°** over 60,105 frames (H17);
* therefore "**98.5% of real frames sit inside the model's comfortable
  range**".

That is a coincidence of two independent measurements, and it is *not* the
quantity anyone actually cares about. The useful number is the integral:

> **expected penalty = ∫ penalty(θ) · P(roll = θ) dθ**

Nobody has computed it, because the penalty curve was only ever measured at
0, ±20, ±30, ±40 — and **90% of the roll mass sits below 15°**, where there are
no measurements at all. The published claim rests on an interpolation nobody
did.

## Design

`roll_controls.py` on the thirteen recordings with a **dense low-angle grid**:
`-30,-20,-15,-10,-5,0,5,10,15,20,30`. The `pinhole` arm and the `common` cap, so
it is directly comparable to H28.

Combined with H17's measured distribution, which is on file as a 2.5° histogram
over 60,105 frames (`h17-roll-prior/results/roll_distribution.json`). No new
data and no new model runs beyond the denser angle grid.

Reported per recording, so the integral gets a spread rather than a point.

## Bars, locked before running

* **B1 — is roll actually cheap in practice?** The expected whole-image penalty
  under the measured distribution is **below 5%**. **Falsified above 10%**, in
  which case "roll is nearly free" is wrong and the line's headline conclusion
  changes.
* **B2 — does the rare tail dominate?** The share of the expected cost coming
  from frames beyond **±20°** — which are only **1.5%** of the data — is under
  **25%**. **If it exceeds 25%**, the advice flips from "ignore roll" to
  "the tail is what matters", which is a different engineering decision
  entirely: you would handle rare large rolls rather than ignore roll wholesale.
* **B3 — sanity.** The penalty is monotone in |θ| on a majority of the
  thirteen. If the curve is not monotone in the region carrying the mass, the
  integral is not meaningful and B1/B2 should not be quoted.

## Why this is worth a tick

It converts the line's most-quoted qualitative claim into a number with an
uncertainty, using only data already on disk. And B2 can change the
recommendation without changing any measurement — superlinear penalty growth
means a 1.5% tail can carry most of the cost.

## Cost

11 angles × 3 arms × 20 frames × 13 recordings. No training.
