# H37 — the border ordering across all four backbones, on thirteen recordings

## The claim being closed

H29 re-measured the border result on thirteen recordings and, in doing so,
flagged what it could **not** support:

> "DA3-Large is the best of the four on a clean frame and the worst with a
> border" was four backbones on ONE recording; H29 re-measured two. The "worst
> of the four" half is untested for `vggt` and `vggt_omega`.

This is the last published multi-backbone claim in the line still resting on
seq136 alone, and it is load-bearing twice over:

1. §4.2's headline warning is *ordering* ("DA3-Large is worst"), not magnitude.
2. §4.3 picked the VGGT-Omega teacher **because** H17.6 measured it as the one
   backbone tolerant of a border adjacent to the scored zone (+29% against
   +106 / +107 / +636). H31 verified the *teacher* choice on thirteen
   recordings, but not the *border tolerance* that motivated it.

Published, all from seq136 (H17.6):

| backbone | border cost, all-image AbsRel |
|---|---|
| `vggt_omega` | +29% |
| `da3:small` | +106% → H29: **+59.0 ± 14.1** over 13 |
| `vggt` | +107% |
| `da3:large` | +636% → H29: **+268.9 ± 61.5** over 13 |

## Design

The H29 `models` arm, unchanged, with the two missing backbones:

```
roll_boundary.py --models vggt,vggt_omega --angles 0 --seq <each of 14>
```

Fourteen recordings: seq136 (the published one) plus the thirteen never-used
Apartment recordings under `adt_apartment_extra`. Single camera angle, level,
because every published comparison is at level. No training; evaluation only.
Border cost = `pinhole_masked` / `pinhole` − 1 on all-image AbsRel, the same
statistic H29 used, so the two experiments' numbers are directly comparable.

## Bars, locked before running

* **B1 — is DA3-Large still the worst of the four?** DA3-Large's border cost
  exceeds all three others on a **majority** of the thirteen.
  **Falsified** if any backbone beats it on a majority, in which case §4.2's
  ordering claim is withdrawn, not softened.
* **B2 — is VGGT-Omega still the most tolerant?** VGGT-Omega's border cost is
  the **smallest of the four** on a majority of the thirteen.
  **Falsified** if `vggt` or `da3:small` is smaller on a majority — which
  would mean §4.3's teacher was chosen on a one-recording accident even though
  H31 showed the choice itself was right.
* **B3 — does the seq136 amplification observation predict out of sample?**
  H31 restated it as *seq136 amplifies the magnitude of whatever you do to it,
  in whichever direction the intervention points*, on nine of nine
  interventions. A border is an intervention that hurts, so seq136 should read
  **high** for both new backbones: z > 0 against the thirteen-recording mean.
  This is the first genuinely out-of-sample test of that observation — it was
  formed on nine measurements that already existed. **Falsified** if either
  backbone reads z < 0.

## Sanity check (not a bar)

The border cost must be **positive on every recording for every backbone**. A
border adjacent to the scored zone helping anywhere means the arms are
mismatched, and the run is void. This is the `lesson_transform_tests_need_a_
baseline_bar` rule applied: H34 was saved by exactly this kind of check.

## What each outcome means

* **B1 and B2 pass** → the ordering is a property of the backbones, not of
  seq136, and both the warning and the teacher choice are on thirteen
  recordings.
* **B1 fails** → the strongest ordering claim in §4.2 goes.
* **B2 fails** → §4.3's *stated reason* for the teacher was wrong even though
  H31 showed the *choice* was right. Both reports would have to say so.
* **B3** is cheap and settles whether the seq136 observation has any predictive
  content or is post-hoc pattern-matching on nine points.

## Cost

Evaluation only, 2 backbones × 14 recordings × 2 arms at one angle. VGGT-Omega
needs the 512 checkpoint; both are inference-only.
