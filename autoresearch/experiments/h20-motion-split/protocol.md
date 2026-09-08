# H20 — is it the staticness, or is it the device?

## The hole this closes

H19 found that a curve fitted on Apartment footage beats a curve fitted on
LiteOffice's own footage, on LiteOffice, by ~10x. I published a mechanism for
that in both reports: **what the fit needs is motion, not a lens match.**

That mechanism rests on calling LiteOffice "near-static". Checking the record,
**that has never been measured in this line.** It is inherited from H9's remark
that "LiteOffice's static wearer starves the anchors", which is indirect, and
from H18.6's finding that LiteOffice's fitted curves are scatter rather than
curves — which is a property of the *fit*, not of the *camera*.

So the published claim rests on an unmeasured premise, and the confound H19's
analysis.md admits ("target-device fitting is bad" vs "this footage is static")
is still open. This closes both without needing a third device.

## Two parts

**H20a — measure the motion.** For all six fitting sequences, compute the camera
centre per frame from the GT pose (`C = -R^T t`) and report positional spread
about the centroid, path length, and bounding-box diagonal. Puts every sequence
on one scale and turns "near-static" into a number or kills it.

**H20b — the within-device control.** This is the part that breaks the confound.
From the **same Apartment cached frames** — same device, same lens, same room —
select two 30-frame subsets:

* `high` — maximum camera spread (farthest-point sampling from a random start)
* `low`  — minimum camera spread (the 30 frames nearest a random anchor frame)

Fit the same 16-number curve on each, five draws apiece, and read the same two
quantities as H18.5 and H19: mean pairwise |da| between draws, and near-rim
transfer on the four evaluation sequences.

**The only variable between `low` and `high` is how far the camera travelled.**
Device, lens, room, scene, frame count and teacher are all held fixed.

## Bars, locked before running

* **B1** — LiteOffice's positional spread is materially below every Apartment
  sequence's. **Falsified if it is not**, in which case "near-static" is simply
  wrong, and the mechanism published in both reports has to be withdrawn.
* **B2** — within the Apartment device, `low` fits are less stable than `high`
  fits (larger mean pairwise |da|), and `low` lands near the 0.100-0.132 the
  LiteOffice sequences gave in H19.
* **B3** — `low` transfers materially worse than `high` on the two cross-room
  sequences.

## The falsification that matters

**If `low` ≈ `high` on both stability and transfer, staticness is not the
mechanism.** The difference in H19 would then be the device after all, the
reading published in both reports on 2026-09-08 would be wrong, and both would
need correcting a second time. That outcome is the reason to run this.

## Cost

One GPU pass for the frozen predictions over six fitting sequences and four
evaluation sequences; every fit, selection and score after that is numpy. No
new teacher inference — the selection draws only from frames the omega110 cache
already holds, which is what makes the within-device contrast free.

## Known limitation, recorded up front

The selection draws from cached frames, which are already subsampled every ~10th
frame of the source video. So `low` is "30 frames from a short stretch of the
walk", not "30 consecutive video frames". That makes the contrast **weaker** than
a true static capture, so B2/B3 passing is conservative evidence and B2/B3
failing is not conclusive on its own.
