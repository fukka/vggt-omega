# H19 — can the radial curve be calibrated on the device it will run on, with no ground truth?

## Where this comes from

H18.2 established that what crosses a room is a 16-number (really 6) radial
curve, and H18.5 established that ~30 frames are enough to fit it. Both fitted
on the **four Apartment sequences** and applied the result unchanged to
LiteOffice, which is a different room *and* a different pair of glasses.

Two things about that are worth being precise about, because the English report
briefly blurred them:

* The fit was **already label-free**. `radial_probe.py` regresses the cached
  omega110 teacher depth on the frozen prediction. Ground truth enters only at
  evaluation. So "fit it without labels" is not the open question.
* What is genuinely untested is fitting on **the device it will run on**. In
  deployment nobody has four Apartment sequences from a Meta device M1292; they
  have some footage from the camera in their hand. The curve is a property of
  the lens, and LiteOffice's lens is not M1292's.

## Hypothesis

A curve fitted on the target device's own footage — teacher-supervised, zero
ground truth anywhere in the fitting loop — transfers to other footage from that
device better than a curve fitted on a different device.

## Arms

All fits regress `log(teacher) = a(theta) * log(frozen_pred) + b(theta)`. Ground
truth is used for **scoring only**, never for fitting.

| fit set | frames | role |
|---|---|---|
| `apartment` | 4 Apartment sequences, 240 | the existing foreign-device curve, the reference |
| `dino` | DinoToy_seq030, 60 | native, LiteOffice device |
| `bowl` | BlackCeramicBowl_seq030, 60 | native, LiteOffice device |
| `lite_both` | both, 120 | native, more frames, but includes the eval sequence — optimistic |

Each fitted in two forms:

* `radial` — 16 parameters, one log-log line per theta bin.
* `global` — 2 parameters, no theta dependence. **The control that decides the
  reading**: if `global` matches `radial`, the native fit is a rescale and not a
  radial relation.

The cells that matter are the **cross-sequence, same-device** ones —
`dino` applied to BlackCeramicBowl, and `bowl` applied to DinoToy. Self-fit
cells (`dino` on DinoToy) are reported but are an upper bound, not a result.

A second configuration repeats everything with `--fit-pred-range` matched
across devices. A log-log slope is not identified independently of the range it
is fitted over, and LiteOffice spans 0.41-4.66 m against the Apartment's
0.44-10 m, so the unmatched comparison compares fitting sets as much as lenses.

## Bars, locked before running

* **B1 — is it radial?** `radial` beats `global` on both cross-sequence cells.
  Falsified if `global` is within 2 percentage points of `radial`, which would
  say the native fit carries no radial structure.
* **B2 — does native beat foreign?** The cross-sequence native curve beats the
  `apartment` curve on both LiteOffice sequences. The apartment curve's numbers
  are already on record: **-26.0%** on DinoToy, **-8.9%** on BlackCeramicBowl.
  Falsified if native fails to beat apartment on either sequence.
* **B3 — H18.5's own follow-up prediction.** H18.5 found single-sequence fitting
  as good as spread fitting, and explained it by noting that an ADT sequence is
  a person walking, so consecutive frames already sweep a wide depth range. It
  predicted that the advantage of spread should *reappear* on genuinely static
  footage. LiteOffice is nearly static. So: the stability of a 30-frame DinoToy
  fit should be **worse** (larger mean pairwise |da|) than the 0.060 measured
  for 30 frames from one Apartment sequence. Falsified if LiteOffice is as
  stable or better.

## What each outcome means

* **B2 passes** — per-device calibration is worth doing, costs ~30 frames of
  ordinary footage and no labels at all. That is a deployable procedure.
* **B2 fails** — the foreign curve is good enough, and near-static footage from
  the target device is worse than moving footage from a different one. The
  recommendation becomes "fit once on moving footage and ship it", which is a
  weaker but cleaner story.
* **B3 passes** — H18.5's explanation of its own refutation survives a test it
  did not have to survive, and "30 frames" acquires a condition: 30 frames *with
  motion in them*.

## Cost

One GPU pass over 6 sequences at 60 frames. Every fit and every score after
that is numpy. No training.
