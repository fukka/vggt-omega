# H25 — how much does the headline number move between held-out recordings?

## Why this, now

Nearly every number in this line rests on **one or two** evaluation recordings.
Both reports say the cross-room test is the weakest link, and H24 has just added
a standing rule about relative measures passing while absolute ones sit at zero.
There is a companion question that conditions all of it and has never been
asked: **is a single-sequence number representative, or is sequence-to-sequence
variance comparable to the effects being reported?**

Thirteen Apartment sequences (seq137, 138, 140–150) were extracted during the
2026-09-07 outage and have **never been used for anything**. They are the same
room and the same device, so they cannot strengthen the cross-room claim. They
can settle the variance question, which is cheaper and arguably more urgent.

## Design

The Apartment-fitted 16-number radial curve, **unchanged**: fitted on seq131,
133, 134, 135 from the omega110 teacher cache exactly as in H18.2. Nothing is
refitted, so this cannot flatter itself by tuning.

Evaluated on the thirteen never-used sequences plus the two originals (seq136,
dec_seq132) so the published numbers appear in the same table as the new ones.

Scoring needs only the frozen prediction and ground truth, so this costs **no
teacher inference**.

## Bars, locked before running

* **B1 — is the effect large against sequence noise?** The standard deviation of
  near-rim gain across the thirteen new sequences is **less than one third** of
  the mean gain. **Falsified if sd ≥ half the mean**, in which case single-
  sequence numbers throughout this line are unreliable and both reports need
  error bars on every figure that currently has none.
* **B2 — was seq136 a lucky draw?** Its published −16.6% lies within **1 sd** of
  the thirteen-sequence mean. **Falsified if it is beyond 2 sd.**
* **B3 — sanity.** The 2-parameter `global` control is worse than doing nothing
  on a majority of the thirteen, as it was on all four originals. If the radial
  structure were sequence-specific this would not hold.

## What each outcome means

* **B1 and B2 pass** → the single-sequence numbers published across this line are
  representative, and the reports can keep quoting them. It also gives every
  future claim here a known noise floor to be judged against, which nothing has
  had so far.
* **B1 fails** → the most consequential outcome. It would mean the effects
  reported in §4.3, §4.4 and §4.6 are of the same size as the variation between
  recordings, and that much of this line has been reading noise. Both reports
  would need error bars retrofitted throughout, and several conclusions would
  need weakening.
* **B2 fails while B1 passes** → seq136 specifically is unrepresentative, and
  every number quoted on it has to be restated on the multi-sequence mean.

## Cost

One GPU pass: 240 fitting frames (cached teacher) plus 15 × 60 evaluation
frames. No training, no teacher inference, no new data — the sequences have been
sitting extracted and unused for a day.
