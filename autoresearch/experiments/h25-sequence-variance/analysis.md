# H25 — B2 refuted. The headline in-room number was a 5.7-sigma outlier.

`results/seq_variance.json`. The Apartment-fitted 16-number curve, **unchanged**,
scored on thirteen never-used held-out recordings plus the two originals.

## The result

| | near-rim gain | centre damage |
|---|---|---|
| **13 never-used sequences** | **−7.02% ± 1.67** | +26.8% ± 12.3 |
| seq136 (the published number) | **−16.6%** → **−5.74 sd** | +41.7% (+1.21 sd) |
| dec_seq132 | −6.5% → +0.31 sd | +23.5% (−0.27 sd) |

**B1 passes.** sd/|mean| = 0.238, comfortably inside the one-third bar. So
sequence-to-sequence noise is *small* relative to the effect — the effect is
real and consistently signed. All thirteen are negative.

**B3 passes 13/13.** The 2-parameter `global` control is worse than doing nothing
on every one of the thirteen (+5.8% to +12.3%). The radial structure is not
sequence-specific.

**B2 is refuted, and not marginally.** seq136 sits **5.74 standard deviations**
above the mean of thirteen recordings from the same room and the same device.

## Why this matters more than a corrected decimal

seq136 has been **the** held-out sequence for this entire line. Every in-room
number in both reports is quoted on it. The representative value for the radial
curve is **−7.0%, not −16.6%** — the published figure is **2.4× the typical
one**.

And it is not that seq136 is unusually hard or easy: its frozen rim error
(0.4269) is +0.51 sd from the thirteen-sequence mean (0.3996 ± 0.0536), i.e.
ordinary. It is specifically unusually *responsive to this correction*.

The pattern to notice: **dec_seq132, the recording repeatedly described as "the
harder test", turns out to be the representative one** (+0.31 sd). The number
that looked like a pessimistic check was the honest one all along, and the
number that looked like the result was the outlier.

## The scope limit, stated rather than discovered later

**H25 re-measured only the radial curve.** The 122,900-parameter LoRA's −51.5%,
H18's student arms, the teacher pre-checks and every other in-room figure were
*also* measured on seq136 and have **not** been re-measured. Nothing here shows
those are inflated — but nothing here shows they are not, and they share the
suspect sequence. That is the immediate next run, and it is cheap.

Until then the correct statement is narrow: *the radial curve's in-room gain on
seq136 is 2.4× its typical value across thirteen held-out recordings from the
same room.* Any ratio computed against another arm measured on seq136 (for
instance "the adapter is three times better in-room") is **untested**, because
both terms may scale together.

## Centre damage was also understated

The curve damages the near-centre zone by **+26.8% ± 12.3** across the thirteen,
against +41.7% on seq136 — so here seq136 was *pessimistic*, by 1.2 sd. The
damage is real and large on every sequence, and the spread (7.2% to 47.9%) is
much wider than the rim gain's.

## What this line should adopt

A single designated held-out sequence, chosen once and quoted everywhere, is not
a held-out set. It took thirteen unused recordings sitting on disk for one day to
show that the number the whole line rests on is an outlier in the flattering
direction. **Every future in-room claim here should be a mean over these
thirteen with an sd, not a single sequence.**
