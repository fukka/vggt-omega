# H42 — the residual-roll curve for the other three backbones

## The last thing in the orientation section still resting on one recording

H41 measured the roll penalty curve properly for the first time — 13 recordings,
15 angles, an exact per-frame gravity-aligned zero — and named its own limit in
the same breath: **`da3:small` only**. §08's uncertainty row says the same: the
four-backbone comparison is still H17.2's single pass at ±10/20/30/40 on
seq136, the recording H25–H31 showed amplifies every intervention.

This runs H41's measurement on `da3:large`, `vggt` and `vggt_omega`.

## Why it is worth a run rather than an assumption

Three things it can settle that nothing else can:

1. **Is the *shape* a pretraining-family property, or only the ±30° number?**
   H17.2's headline — "roll sensitivity is a property of the pretraining data,
   not of depth estimation" — rests on one number per backbone on one recording.
   A whole curve on thirteen recordings either reproduces that grouping or does
   not.
2. **Does VGGT-Omega's null in H39 follow from its own curve?** H39 found
   nothing for that backbone and B2 refused to call the −1.04% a result. H39b
   measured S = −0.48% ± 3.07 and A = +0.56% ± 1.81. If its curve is flat, the
   same Jensen arithmetic that predicted DA3-Small's numbers must predict
   *these* — near zero. That is a second, independent closing of the loop.
3. **Is H41's origin correction a DA3-Small quirk?** §03u's dip and the
   "coincidence" it recorded were both explained by the device-aligned zero
   sitting a median 4° off level. If that explanation is right, **every**
   backbone's curve must bottom out at residual 0.

## Design

`residual_curve.py` unchanged, `--models da3:large,vggt,vggt_omega`, the same
13 recordings, 20 frames, deltas {0, ±2, ±4, ±6, ±8, ±11, ±15, ±22}. Split
across the two GPUs by the sign of d, with d = 0 on both so each half
normalises itself. Evaluation only.

## Bars, locked before running

* **B1 — does the family grouping survive a whole curve?**
  g(+22°) and g(−22°) are **lower for both VGGT variants than for both DA3
  variants**. **Falsified** if either VGGT variant exceeds either DA3 variant at
  either end, in which case H17.2's family claim is a one-recording,
  one-angle artefact and must be narrowed to "+30° on seq136".
* **B2 — does VGGT-Omega's own curve explain its null?**
  From its curve alone at ψ = 11°: |A_pred| < 5% and |S_pred| < 3%, against
  H39b's measured +0.56% and −0.48%. **Falsified** if either prediction is
  larger, which would mean the flat curve is not why nothing was recoverable
  there and H39's null needs another explanation.
* **B3 — is the minimum at zero for everyone?**
  Every backbone's curve takes its minimum at residual 0 (ties within ±2°
  allowed, since that is the grid spacing). **Falsified** if any backbone
  bottoms out elsewhere, which would say H41's origin correction is specific to
  DA3-Small rather than a property of the reference.

## Cost

Evaluation only: 3 backbones × 15 renders × 20 frames × 13 recordings, both
GPUs. VGGT-Omega needs the 512 checkpoint.
