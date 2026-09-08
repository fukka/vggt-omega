# H31 — the last decision-critical number still on one recording

## What is left

H25–H30 re-measured every published in-room figure across thirteen recordings.
One remains, and it is not a result — it is **the measurement that chose the
method**:

| teacher | rim coverage | frame black | rim gain vs raw |
|---|---|---|---|
| DA3-Small, 110° wide | 100% | 22.5% | **+33% (worse)** |
| VGGT-Omega, 110° wide | 100% | 22.5% | **−65%** |
| DA3-Small, 95° narrow | 70% | 1.3% | −15% |

That table is why §4.3 uses VGGT-Omega as the teacher at all. It is one
recording, and by H30's standing instruction — *any intervention measured only
on seq136 should be assumed inflated* — its magnitudes are suspect.

The **ordering** is what the decision rests on, and H29 showed a between-model
ordering can hold 13/13 while the magnitudes are 6 σ off. That is an argument by
analogy, not a measurement.

## Design

`cache_teacher.py --precheck-only`, which runs the teacher and the raw fisheye
model and compares them at the rim without writing a cache. Three configurations,
unchanged, on the thirteen recordings at 20 frames each.

| config | flags |
|---|---|
| `da3_wide` | `--teacher-fov 110 --variant small` |
| `omega_wide` | `--teacher-fov 110 --teacher-model vggt_omega` |
| `da3_narrow` | `--teacher-fov 95 --variant small` |

## Bars, locked before running

* **B1 — does the decision survive?** `omega_wide` beats `da3_wide` at the rim on
  **every** recording. **Falsified by a single failure**, because the choice of
  teacher was made on this comparison and a method choice should not rest on a
  majority vote.
* **B2 — is the wide view really unusable for DA3?** `da3_wide` is *worse than
  the raw model* (positive gain) on a majority of the thirteen. This is the
  finding that forced the switch.
* **B3 — prediction, not a bar.** Per H30's standing instruction, `omega_wide`'s
  published −65% should read **more than 1 σ better** than the
  thirteen-recording mean. Recorded so the instruction stays falsifiable rather
  than becoming an unexamined assumption after one confirmation.

## What each outcome means

* **B1 passes** → the method choice was sound, the verification campaign closes,
  and nothing published in this line rests on one recording except the
  cross-room numbers, which are blocked on data.
* **B1 fails** → §4.3's teacher was chosen on an artefact. That would not just
  restate a number; it would mean the label-free result was built on the wrong
  teacher and the section needs redoing.

## Cost

Pre-check only, no cache written, 20 frames per recording. VGGT-Omega at patch
16 is the expensive arm; 13 × 20 frames is affordable.
