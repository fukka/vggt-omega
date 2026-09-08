# H37 — all three bars pass, and the seq136 observation predicted out of sample

**Status: B1, B2, B3 all PASS 13/13. Sanity check OK (no negative cell).**

## The table

Border cost = all-image AbsRel with a hard border adjacent to the scored zone,
against the same frames without one. Thirteen never-used Apartment recordings
plus seq136, the one every published number came from.

| backbone | 13 recordings | seq136 (published) | z | range |
|---|---|---|---|---|
| `vggt_omega` | **+6.0% ± 4.6** | +29.1% | +5.01 | +1.2 … +19.0 |
| `vggt` | **+48.4% ± 18.0** | +106.5% | +3.22 | +28.3 … +86.7 |
| `da3:small` | **+59.0% ± 14.1** | +105.7% | +3.32 | +29.3 … +74.0 |
| `da3:large` | **+268.9% ± 61.5** | +635.5% | +5.96 | +148.6 … +345.7 |

(`da3:small` and `da3:large` are H29's; this experiment added the two VGGT rows
with the same runner, recordings and level angle.)

## B1 and B2 — the ordering is a property of the backbones

DA3-Large is the worst of the four on **13 of 13**. VGGT-Omega is the best of
the four on **13 of 13**. Neither ordering has a single exception.

That closes the one claim H29 flagged it could not support, and it closes it in
the strongest available form: not "on average", but on every recording.

**§4.3's stated reason for the teacher was right.** H31 had already shown the
*choice* was right (omega_wide beats da3_wide at the rim 13/13). This shows the
*reason* was right too — VGGT-Omega really is the border-tolerant one, not a
one-recording accident that happened to point at a good teacher.

## The number that got better when it was re-measured

Published, from seq136: DA3-Large's border cost is **21.8×** VGGT-Omega's.
On thirteen recordings the ratio of means is **45.1×**. The published figure
understated the gap by half.

Same shape as H26/H27, and for the same reason: seq136 amplified the two
extremes unequally — ×2.36 for DA3-Large, ×4.88 for VGGT-Omega — so the ratio
between them shrank. The recording flattered the *weaker* side more, exactly as
it flattered the 16-number curve (×2.36) more than the adapter (×1.33).

**Do not quote the mean of the per-recording ratios**, which is 70.4 ± 61.8
with a range of 16.7–255.2. VGGT-Omega's border cost reaches down to +1.2%, so
dividing by it explodes. This is H33's metric trap and it is live here: the
ratio of means (45.1) is the citable form.

## B3 — the first out-of-sample test of the seq136 observation

H31 restated the pattern as *seq136 amplifies the magnitude of whatever you do
to it, in whichever direction the intervention points*, fitted on nine
interventions that already existed. It had never predicted anything.

Pre-registered here: a border hurts, so seq136 should read high for both new
backbones. It reads **z = +3.22** (`vggt`) and **z = +5.01** (`vggt_omega`).

The tally is now **eleven of eleven** interventions, two of them predicted
before the data existed, against three unperturbed measurements at −0.11, +0.07
and +0.52. It remains an observation with no mechanism offered — but it is no
longer only post-hoc pattern-matching.

## What this does NOT establish

* **`vggt` versus `da3:small` is not separated.** Their published seq136 values
  were a near-tie (+106.5 vs +105.7). On thirteen recordings `vggt` is lower on
  8 of 13 with means of +48.4 vs +59.0 and heavily overlapping ranges. The
  honest reading is that these two are comparable and this experiment cannot
  order them.
* **Border tolerance is still not a pretraining-family property.** This is
  H17.6's conclusion and it survives, strengthened: the two VGGT variants
  differ by **8×** on the thirteen-recording means and `vggt_omega` is lower on
  13 of 13. Whatever makes VGGT-Omega tolerant is not what it shares with VGGT.
  Roll robustness *is* a family property (H17.2); border tolerance is not. Two
  different axes, confirmed on thirteen recordings each.
* **No mechanism.** Four backbones ordered consistently across thirteen
  recordings says nothing about why. This effect has already refuted two
  mechanism guesses in a neighbouring experiment (H22, H23) and no third is
  offered here.

## Practical form

"Never let a hard border sit next to the zone you care about" now has a second
half worth quoting: **how much it costs depends on the backbone by a factor of
45**, and the ordering is stable enough that the cheapest defence against bad
framing is choosing the model, not fixing the framing. Same shape as the roll
result (H35: changing the pretraining buys 4×, scaling the model buys nothing)
and the rim result. Third time in this line that the actionable variable turned
out to be which model, not how big.
