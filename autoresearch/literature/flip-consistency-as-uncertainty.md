# Flip consistency is an established uncertainty signal — and what that does to H45

**Searched 2026-09-09**, seventh tick of the hardware outage, following the
first literature pass. H45's limits section asked for an external anchor; this
is the closest thing available without the box.

## What the field does with the flip

*Revisiting Gradient-based Uncertainty for Monocular Depth Estimation*
(arXiv 2502.05964) uses exactly H45's construction — **flip the image, predict,
flip the prediction back, compare against the unflipped prediction** — as an
augmentation-consistency signal from which to derive **uncertainty**, with no
ground truth needed. In its ablation tables the variant is simply labelled
**"Flip"**, and the survey literature reports flipping as the *best-performing*
of the transformations tried for this purpose, on the reasoning that a flip
"observes the scene from a different context while preserving the geometry".

Flip-**averaging** — predict twice, average the two maps — is separately a
common post-processing trick, though not used in that paper.

**No paper found reports a magnitude** for the flip inconsistency itself, so
H45 still has no published number to sit beside. What the literature supplies
instead is the **assumption** the technique rests on.

## What that assumption is, and where it fails

Both uses — uncertainty and averaging — assume the flipped prediction is a
**comparably good** estimate of the same scene, differing in ways that carry
information about genuine ambiguity.

On Aria egocentric fisheye that assumption fails for the DA3 pair. H45/H47
measure the flipped-back prediction at **2.2–3.6× the unflipped error**
(AbsRel 0.131 → 0.412 for `da3:small`, 0.058 → 0.226 for `da3:large`), on
**96–100% of frames**, with the least-affected tenth still 31–69% worse (H49).
That is not a small perturbation carrying information about ambiguity; it is a
large systematic penalty with a handedness.

## Two consequences, stated as implications and not as measurements

1. **Flip-based uncertainty would be miscalibrated on this data.** The signal it
   reads would be dominated by a systematic handedness effect rather than by
   per-pixel ambiguity — and dominated *unequally*, since the two VGGT variants
   pay 1.3–1.5× where the DA3 pair pays 2.2–3.6×. **Not measured here**: this
   line has not run a flip-uncertainty estimator and scored its calibration.
   The inference is from the size of the inconsistency, not from a calibration
   experiment.
2. **Flip-averaging would be actively harmful for the DA3 pair here** —
   averaging a 0.131 prediction with a 0.412 one. This follows directly from
   the measured pair, with no further assumption.

## Why this matters for how H45 is described

It moves H45 from *"a curious property of some models"* to *"a caution about a
standard technique on this kind of data"*. And it settles the credit: the
construction is **the field's**, applied to a new setting. H45 should say so.
