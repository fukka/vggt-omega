# H27 — the last in-room figure still resting on one recording

## What is left

H25 corrected the 16-number curve (−16.6% → −7.02% ± 1.67, a 5.74 σ outlier).
H26 corrected the adapter and its control (−51.5% → −38.59% ± 4.97, 2.60 σ).

One headline survives on seq136 alone, and it is the one §4.3 leads with:

> **With no depth labels, the student recovers 89% and 82% of what training on
> real labelled depth buys.**

Both figures are ratios whose denominator is a **GT-trained arm measured only on
seq136** (−57.9% same room, −27.2% rearranged). Everything else measured on that
recording has turned out inflated — and by *different factors*, 5.7 σ for the
curve against 2.6 σ for the adapter — so the ratio cannot be assumed safe.

## Design

Three arms, checkpoints unchanged, on the same thirteen recordings, plus seq136
and dec_seq132 for direct comparison with the published values.

| arm | checkpoint | why |
|---|---|---|
| `gt` | `autoresearch-h14-upright/gt/lora_last.pt` | the denominator of the headline ratio |
| `omega_rt`, `omega_rt_s1` | `autoresearch-h14-omega/omega_rt{,_s1}` | completes the control to three seeds; H26 ran only `_s2` |

Statistics are computed over the **thirteen** so they stay comparable with H25
and H26; seq136 and dec_seq132 are reported separately.

## Bars, locked before running

* **B1 — is the denominator inflated too?** `gt`'s published −57.9% lies within
  **2 sd** of the thirteen-recording mean. Given the pattern, expect this to
  fail; recorded so the expectation is on the record either way.
* **B2 — does the headline ratio survive?** The label-free fraction recomputed
  on thirteen-recording means is within **±15 percentage points** of the
  published 89%. **Falsified outside that**, in which case §4.3's leading
  sentence must be restated with the new number.
* **B3 — does the control hold up with three seeds?** The three-seed `omega_rt`
  mean stays worse than the three-seed `omega110` mean on a majority of the
  thirteen. H26 showed this 13/13 for a single control seed; this checks it is
  not a one-seed accident.

## What each outcome means

* **B2 passes** → the headline is safe as stated, because numerator and
  denominator were inflated by comparable factors. Both reports keep it, with the
  corrected absolute numbers underneath.
* **B2 fails** → the most-quoted sentence in this line is wrong and gets
  rewritten with the thirteen-recording figure.
* **B3 fails** → far more serious than either. It would mean H18's central
  comparison depends on which control seed was picked, and §4.3 would need
  rebuilding rather than renumbering.

## Cost

Evaluation only: 3 arms × 15 recordings. No training, no teacher inference.
