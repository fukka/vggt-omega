# H33 — is the roll asymmetry a property of the models, or of our setup?

## What H32 found, one tick ago

Every published roll figure averaged the ± pair. Measured separately at 30°,
**−30° costs +24.4% and +30° costs +56.5%** — positive roll 2.3× costlier,
consistent on 11 of 13 recordings. H32 also found the penalty *minimum* sits near
**−5°** rather than at zero.

I published that as a finding about depth models. **It might not be one.**

## The competing explanation I should have tested first

Three things in this pipeline are not left-right symmetric, and any of them
could produce an apparent asymmetry that has nothing to do with the models:

* every image is rotated by a **quarter turn** (`UPRIGHT_K = 3`) before the
  model sees it;
* the **principal point is 4.5 px off centre** in the upright frame —
  (246.0, 255.5) against (256.0, 256.5) — recorded early in this line and never
  revisited;
* the **scene content** is not symmetric either: the wearer's hands and the desk
  sit low and to one side in most frames.

If the asymmetry comes from any of those, it is a property of **our rendering**,
not of depth estimation, and H32's finding has to be restated.

## The test

Run the same roll sweep across **four backbones from two different pretraining
families**. The rendering, the upright convention and the scene content are
**identical** for all four — only the model changes.

* If the asymmetry is a model property, the **single-image** pair (`da3:small`,
  `da3:large`) and the **multi-view** pair (`vggt`, `vggt_omega`) should differ.
  H17 already showed those families differ 4–5× in overall roll sensitivity, so
  they are the natural split.
* If the asymmetry is a setup artefact, **all four should show the same ratio**,
  because they all see exactly the same pixels.

`roll_boundary.py --models da3:small,da3:large,vggt,vggt_omega`,
`--angles=-30,-20,0,20,30`, on eight recordings, `pinhole` arm.

## Bars, locked before running

* **B1 — is it the model?** The +30/−30 cost ratio differs between the
  single-image pair and the multi-view pair by more than the recording-to-
  recording spread within either.
* **B2 — sanity.** `da3:small` reproduces the asymmetry on this subset:
  +30° costlier than −30° on a majority of the eight.
* **FALSIFIED if all four backbones agree within noise.** Then the asymmetry is
  a property of the rendering or the upright convention, **H32's finding is
  withdrawn as a claim about depth models**, and the honest statement becomes
  "our pipeline is not left-right symmetric, and we do not know which part".

## Why this matters more than it looks

H32's asymmetry is one tick old and already in both reports. It is exactly the
kind of finding that reads well and could easily be an artefact of a convention
chosen months ago for unrelated reasons. Testing it now costs one run; leaving
it costs whatever anyone builds on it.

## Cost

5 angles × 4 backbones × 8 recordings × 20 frames. VGGT-Omega is the expensive
arm. No training.
