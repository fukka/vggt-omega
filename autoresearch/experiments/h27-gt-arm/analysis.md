# H27 — the arc closes. Every number down, every ratio up.

`results/` holds 45 evaluations: the GT-trained arm and the two remaining
control seeds, on the same thirteen recordings plus seq136 and dec_seq132.

## The bars

**B1 fails at −3.12 sd.** The GT arm's published −57.9% is inflated by seq136
too. That makes it **all four** arms: nothing measured on that recording was
representative.

**B2 passes, and in the favourable direction.** The label-free fraction is
**94.2%** on thirteen recordings against the published 88.9%. The headline
sentence in §4.3 is not just safe — it was **understated**.

**B3 passes 13 of 13 with the full three-seed control.** `omega110` beats
`omega_rt` on every recording, gap **−3.44 points** against H26's single-seed
−3.66. So H18's central comparison is not a one-seed accident.

## Every in-room number, corrected

| arm | published (seq136) | thirteen recordings | σ out | inflation |
|---|---|---|---|---|
| trained on real labels | −57.9% | **−40.99% ± 5.42** | 3.12 | ×1.41 |
| adapter `omega110` | −51.5% | **−38.59% ± 4.86** | 2.60 | ×1.33 |
| control `omega_rt` (3 seeds) | −47.6% | **−35.14% ± 5.86** | — | ×1.35 |
| 16-number curve | −16.6% | **−7.02% ± 1.67** | 5.74 | ×2.36 |

| ratio | published | thirteen recordings |
|---|---|---|
| label-free ÷ label-trained | 88.9% | **94.2%** |
| adapter ÷ curve | 3.10× | **5.50×** |
| adapter − control | −3.9 pts | −3.44 pts, **13/13** |

## The pattern in the inflation column

The three **learned** arms were all inflated by about the same factor — 1.33,
1.35, 1.41 — while the **16-number curve** was inflated by **2.36**, nearly
twice as much.

That is why every ratio improved rather than degrading: seq136 flattered the
weakest method most. It also says something usable about the curve itself: a
16-parameter fit on rim-dominated statistics is **far more sensitive to which
recording you score on** than a trained adapter is. Its sd/|mean| is 0.238
against the adapter's 0.126.

## What is now settled, and what is not

**Settled.** Every in-room figure in this line is now a mean over thirteen
recordings with a standard deviation. Both reports carry the corrected numbers.
The three conclusions that rested on seq136 — the label-free fraction, the
adapter-over-curve gap, and the rectified teacher beating its control — all
survived, and the first two got better.

**Not settled.** The *rearranged-room* ratio (82%) still rests on dec_seq132
alone. That is much less worrying than seq136 was, because H25 showed
dec_seq132 is the **representative** recording (+0.31 sd), but it is one
recording and should be described as such.

**Also not settled.** The cross-room numbers were never measured on seq136, so
this arc does not touch them — but they remain 120 frames over two near-static
recordings, which both reports still call the weakest link. That is unchanged
and remains the largest open exposure in this line.
