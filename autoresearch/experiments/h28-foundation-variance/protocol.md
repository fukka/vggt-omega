# H28 — does the foundation of this line hold across recordings?

## The question the correction arc forces

H25–H27 established that seq136 is unrepresentative: 5.74 σ for the radial
curve, 2.60–3.12 σ for the three learned arms. Every in-room *distillation*
number has been re-measured across thirteen recordings.

**The orientation results were never re-measured, and they are the foundation
both reports open with.** Checking the artifacts, they are all seq136:

* the four-backbone table — "**the rim is 2.0–2.6× worse than the centre, in
  every model**", §4.1 / §01;
* the roll sensitivity table — "**+13% / +46% / +131% at ±20/30/40°**", §4.1 /
  §02.

If seq136 is an outlier for these too, it is not a decimal that changes. It is
the opening claim of the whole line.

## Design

Two existing runners, unchanged, on the same thirteen recordings H25–H27 used.

| arm | runner | what it produces |
|---|---|---|
| `ratio` | `rotation_sweep.py --models da3:small,da3:large` | near-rim, centre, and the **rim ÷ centre ratio** |
| `roll` | `roll_controls.py --angles -40,-30,-20,0,20,30,40` | whole-image error against the level baseline |

Nothing is refitted or retrained. seq136 is included so the published values land
in the same table.

## Bars, locked before running

* **B1 — is the rim/centre ratio stable?** Its standard deviation across the
  thirteen is **under 15% of the mean** for DA3-Small. **Falsified at 25% or
  above**, which would make "2.0–2.6× in every model" a statement about one
  recording rather than about fisheye depth.
* **B2 — was seq136 representative here?** Its published 2.13 lies within
  **2 sd** of the thirteen-recording mean.
* **B3 — is the roll penalty stable?** DA3-Small's ±30° penalty (published +46%)
  has sd under one third of its mean across the thirteen, and seq136 sits within
  2 sd.

## What each outcome means

* **All three pass** → the foundation is sound and, unlike the distillation
  numbers, was never distorted by the choice of recording. The reports keep §4.1
  as written, now with error bars.
* **B1 or B3 fails** → the most-quoted sentences in both reports describe one
  recording. Every figure in the orientation sections needs restating, and the
  "2.0–2.6×" range in particular would have to be replaced by a measured spread.
* **B2 fails while B1 passes** → the ratio is stable but seq136 sits at an edge
  of it; the numbers get restated on the mean without the conclusions moving.

## Why this and not more data

The download that would scale the *cross-room* evidence is blocked on a decision
about a shared filesystem at 98% capacity, and has been put to the user rather
than started. This needs no new data: the thirteen recordings are already
extracted, and the question is at least as important.

## Cost

`ratio`: 2 backbones × 14 recordings × 20 frames. `roll`: 7 angles × 3 arms ×
14 × 20. No training.
