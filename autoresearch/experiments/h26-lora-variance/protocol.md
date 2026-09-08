# H26 — is the LoRA's −51.5% inflated the same way?

## Why this is urgent rather than merely interesting

H25 found that seq136, **the** held-out sequence for this entire line, is a
**5.74-sigma outlier** for the radial curve: −16.6% published against −7.02% ±
1.67 across thirteen never-used recordings from the same room and device.

H25 re-measured **only the radial curve**. The 122,900-parameter LoRA's −51.5%,
the roundtrip control's −47.6% and every other in-room figure in §4.3 and §4.4
were measured on that same sequence and have not been re-measured. Until they
are, the biggest headline in the whole line rests on a recording now known to
flatter at least one method by 2.4×.

Two published claims depend directly on it:

* "**With no depth labels, the student recovers 89% and 82% of what real labels
  buy**" — a ratio of two numbers, both from seq136.
* "**At home the adapter is three times better**" than the 16-number curve —
  51.5 ÷ 16.6. The denominator is already known to be inflated 2.4×. If the
  numerator is inflated by the same factor the ratio survives; if not, it does
  not.

## Design

The existing checkpoints, unchanged, evaluated on the same thirteen recordings
H25 used (seq137, 138, 140–150). Nothing is retrained.

| arm | checkpoint |
|---|---|
| `omega110` × 3 seeds | `results/autoresearch-h14-omega/omega110{,_s1,_s2}/lora_last.pt` |
| `omega_rt` (control) | `results/autoresearch-h14-omega/omega_rt_s2/lora_last.pt` |

Absolute `--out` paths throughout: `eval_lora.py` resolves a relative `--out`
against its own directory, which has silently misplaced results before.

## Bars, locked before running

* **B1 — is the LoRA inflated too?** seq136's −51.5% lies within **2 sd** of the
  thirteen-sequence mean for `omega110`. **Falsified beyond 2 sd**, which would
  mean the headline of the distillation section is an outlier as well.
* **B2 — does the ratio survive?** The adapter-to-curve ratio computed on the
  thirteen-sequence means is within **±30%** of the published 3.1×. Falsified
  outside that, in which case "three times better" must be restated.
* **B3 — does the control keep its place?** `omega_rt` remains worse than
  `omega110` on a majority of the thirteen, as it was on seq136. This is the
  comparison H18's whole reading rests on, and it has never been checked on more
  than two recordings.

## What each outcome means

* **B1 passes** → the LoRA number is representative, the radial curve was
  specifically flattered by seq136, and the adapter-vs-curve gap is *larger*
  than published, not smaller.
* **B1 fails** → both numbers were inflated by the same recording. The ratio may
  survive (B2) while every absolute figure in §4.3 needs restating.
* **B3 fails** → the most serious outcome available here. H18's conclusion that
  the rectified teacher beats the un-rectified control in-room would rest on one
  unrepresentative recording, and the section would need rewriting rather than
  renumbering.

## Cost

Evaluation only: 4 arms × 13 recordings × 60 frames. No training, no teacher
inference, no new data.
