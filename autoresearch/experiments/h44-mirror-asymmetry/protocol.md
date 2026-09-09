# H44 — the mirror test, done in the frame where it is exact

## The one thing left in this line, and the test that was owed it

After H43b the **residual** asymmetry — measured about each frame's own
gravity-aligned zero — is all that survives: `da3:small` +4.70%, `da3:large`
+7.77%, `vggt` **−6.99%**, `vggt_omega` −0.87%. Four models seeing identical
pixels lean **different ways**, which already rules out a purely geometric cause
(a principal-point offset or a rendering handedness would push all four the same
way). Nothing says what it is.

H33 named the missing test — *"separating them needs the upright convention
itself varied"* — and H34 attempted a mirror version and was **void**: it
mirrored the source image and not the ground truth, so the model predicted a
mirrored scene scored against an unmirrored one, a 4× broken baseline that
produced a clean-looking result.

## Why it is exact now, and was not then

H34 mirrored in the **fisheye source** frame, where the principal point sits
4.5 px off centre and the rig warp and the roll both intervene before scoring.
Mirror in the **rendered view** instead:

* the virtual pinhole has `cx = (width−1)/2` exactly (`Camera.to_pinhole`), so a
  horizontal flip about the centre column is an exact symmetry of that frame;
* planar z is a scalar field, so flipping the prediction back is exact;
* **the ground truth is never touched**, because the prediction returns to the
  original frame before it is scored.

    normal:  z = forward_z(view)
    mirror:  z = flip( forward_z( flip(view) ) )

Everything else — frames, mask, GT, roll, origin — is identical.

## What each outcome means

The asymmetry is `g(+r) − g(−r)`. A horizontal mirror turns "content rolled +r"
into "mirrored content rolled −r", and it **preserves up and down** — floors stay
at the bottom, ceilings at the top.

* **If the asymmetry travels with the scene as the model sees it** (up/down
  structure, hands low, floors below), mirroring should **flip its sign**.
* **If it is a handedness fixed in the model's own image space**, mirroring
  should **not** flip it.

Two backbones are run because their residuals have **opposite signs**:
`da3:small` (+4.70%) and `vggt` (−6.99%). A discriminator that behaves the same
way on both is worth far more than one that works on one.

## Bars, locked before running

* **B0 — the H34 bar, mandatory.** At r = 0 the mirror arm's cost against the
  normal arm must be **under 25%** for both backbones. A mirrored room is still
  a room. **If it is catastrophic the construction is wrong and the run is
  void** — this is exactly what H34 failed and did not check.
* **B1 — does it flip?** `Δ_mirror` has the **opposite sign** to `Δ_normal` on
  both backbones. **Falsified if either keeps its sign**, which would say the
  asymmetry is fixed in the model's image space rather than carried by the
  scene.
* **B2 — is it the same size?** `|Δ_mirror|` is within a factor of **2** of
  `|Δ_normal|` on both. A sign flip with a collapsed magnitude would mean the
  mirror destroyed the effect rather than reflecting it.

## What it cannot do

It cannot say *which* scene structure carries the asymmetry if B1 passes, and it
cannot say *what* the model-space handedness is if B1 fails. It separates two
families of explanation and stops there. **No mechanism will be offered on the
strength of this run**, whichever way it goes.

## Cost

Evaluation only: 2 backbones × 2 arms × 5 angles (0, ±11, ±22) × 20 frames ×
13 recordings, split across both GPUs by the sign of the angle.
