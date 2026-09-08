# H26 — both numbers were inflated. The conclusions come out stronger.

`results/` holds 52 evaluations: three `omega110` seeds and the `omega_rt`
control, on the same thirteen recordings H25 used. Checkpoints unchanged,
nothing retrained.

## The three bars

**B1 fails, but far less badly than the curve did.** Pooled over three seeds and
thirteen recordings, `omega110` gives **−38.59% ± 4.97**. The published −51.5%
sits **2.60 sd** out, against a bar of 2. So the adapter figure was inflated by
the same recording — but at 2.6 σ, not the curve's 5.7 σ.

**B2 fails, in the opposite direction to the worry.** The adapter-to-curve ratio
was supposed to stay near the published 3.10×. On thirteen-recording means it is
**5.50×** (−38.59 ÷ −7.02). The published figure **understated** the adapter's
in-room advantage, because the curve was flattered roughly twice as much as the
adapter was.

**B3 passes 13 of 13 — the important one.** `omega110` beats the un-rectified
`omega_rt` control on **every single recording**, mean −38.59% against −34.92%,
a gap of **−3.66 points** against seq136's −3.9. H18's central in-room reading
went from one recording to thirteen without moving.

## Corrected numbers

| | published (seq136) | thirteen recordings |
|---|---|---|
| adapter, `omega110` | −51.5% | **−38.59% ± 4.97** |
| control, `omega_rt` | −47.6% | −34.92% ± 5.67 |
| 16-number curve | −16.6% | −7.02% ± 1.67 |
| adapter ÷ curve | 3.10× | **5.50×** |
| adapter − control | −3.9 pts | −3.66 pts |

Seed spread is small next to sequence spread: the three `omega110` seeds average
−39.44, −39.62 and −36.72, a range of 2.9 points against a within-seed sequence
sd of about 5. **Which recording you score on matters more than which seed you
train.** That is worth knowing for every future run here.

## An incidental worth keeping

The `omega110` students **improve** the near-centre zone by −31.4% on average
rather than damaging it. That is the opposite of the 16-number curve, which
damages the centre by +26.8% (H25), and it makes sense: the student is trained
on the teacher's depth everywhere, while the curve is a per-angle correction
fitted to rim-dominated statistics.

This is *not* the same measurement as the data-ladder table's "+58% to +145%
centre damage", which comes from different arms trained with rim weighting. The
two must not be quoted against each other.

## What is still untested

The **"89% and 82% of what real labels buy"** figure divides by a GT-trained arm
that was also measured on seq136 and has **not** been re-run. That ratio remains
untested. Given the pattern here — everything on seq136 is inflated, and by
different factors — it should not be quoted until the denominator is re-measured.

## The net effect on the line

Two published numbers were wrong and both have been corrected downward. But the
three claims that rest on them all survive or strengthen:

* the adapter beats the curve in-room — **by more than published**, 5.5× not 3.1×;
* the rectified teacher beats the un-rectified control in-room — now **13/13**
  instead of 1/1;
* the effect is real and consistently signed on every recording tried.

The correction cost two headline numbers and bought a much firmer floor under
the conclusions.
