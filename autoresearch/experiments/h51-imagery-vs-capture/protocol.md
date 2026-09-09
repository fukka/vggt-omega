# H51 — imagery or capture? The one separator ADT already contains

**Locked before any number was computed.** Unlike H50, this is a **genuine
blind pre-registration**: no arm of this comparison has ever been run in this
project, and the thresholds below were chosen without seeing a single value.

## The question H48 left

H48 put the mirror effect on the **content** rather than the lens or the warp,
and named what it could not separate — a resample carries each dataset's
*sharpness, motion blur and exposure* through the other's geometry, so
"content" there means *the imagery*, not *the scene*. Two candidates remain:

- **Imagery / scene**: what an egocentric camera looks at — everything close,
  the floor in a fixed place, the wearer's own body and reach volume, a
  characteristic layout of surfaces.
- **Capture**: how a head-worn 30 Hz sensor records it — motion blur, rolling
  shutter, auto-exposure, sensor noise, compression.

## Why this is answerable without any new data

ADT ships **Blender renders of the same recordings**. Five sequences carry both
streams — `seq131` (2,878 frames) and `seq133`–`seq136` (399 each), verified
as a **three-way overlap** of `videos_synthetic`, `videos_rgb` and `depth_npy`,
so every scored frame is:

| held identical | varied |
|---|---|
| scene, layout, objects | photometry: real sensor vs rendered |
| camera trajectory and pose (same frame index, same timestamp) | motion blur |
| lens and intrinsics | auto-exposure, sensor noise |
| ground-truth depth (the same `.npy`) | compression artefacts |

This is a **paired** design at the level of the individual frame. Nothing has
to be resampled, so the entire class of failure that voided H48's reciprocal
arm cannot occur here.

## Arms and construction

H47's construction, unchanged: 60° co-axial view (corner ray 39.2°, inside
Aria's 54.83° cone), scored at θ ≤ 28°, four backbones, three arms per image
(`normal` / `mirror` / `twice`), mirror cost as the **ratio of means**.

    real   = videos_rgb        (what H45/H47 measured)
    synth  = videos_synthetic  (same frames, rendered)

20 frames per sequence drawn from the three-way overlap, 5 sequences.

## The statistic and the rule (fixed here, before the run)

    retention_m = mirror_cost(synth) / mirror_cost(real)      per backbone

- **CAPTURE** — the effect is an artefact of how the sensor records:
  `retention_m < 0.30` for **at least 3 of 4** backbones.
- **IMAGERY** — the effect is a property of what is in front of the camera:
  `retention_m ≥ 0.60` for **at least 3 of 4** backbones.
- **NEITHER CLEANLY / BOTH CONTRIBUTE** — anything else, including a split
  across backbones. Reported as inconclusive with the numbers, not narrated
  into one of the other two.

3-of-4 rather than 4-of-4 because `vggt_omega`'s native effect is the smallest
of the four (+29.9% in H48's Aria cell) and a ratio on a small denominator is
the least stable of the four; the band between 0.30 and 0.60 is left explicitly
undecided rather than split at a midpoint.

## Gates, checked before the rule is applied

- **Plumbing**: flip twice = identity, worst deviation `< 0.001%` in both arms.
- **Resolving power**: know-nothing floor ÷ baseline AbsRel `≥ 2.0` for every
  backbone in **both** arms. The synthetic arm is the one at risk here — if
  the models are far worse on rendered imagery, the cell may be unable to show
  an effect, which is exactly the mistake H48's first read made.
- **Pairing**: every scored frame must exist in all three of
  `videos_synthetic`, `videos_rgb` and `depth_npy`, checked by stem, and the
  same frame list must be used by both arms. Recorded per run.

## What it cannot do — named before the run

**The renders differ in more than capture.** A Blender render of a scanned
apartment also has simplified materials, no specular subtlety, cleaner edges
and different noise statistics. So a *drop* on the synthetic arm supports
"the effect needs real-sensor photometry" but does **not** isolate motion blur
from renderer fidelity; and it would not distinguish "the sensor's artefacts
cause it" from "photorealism is required for it". A *survival* is the cleaner
outcome: it would put the effect on the scene and its geometry, which no
renderer difference can explain away.

Said plainly so a drop is not later reported as "motion blur causes it".

## Cost

Evaluation only. 4 backbones × 2 arms × 3 sub-arms × 20 frames × 5 sequences,
one GPU, no training, no resampling, no new data downloaded.

## Implementation note

`AriaLocalPairs` hardcodes `videos_rgb`; it gains an `rgb_subdir` keyword
defaulting to the current value, so no existing caller changes. H5's `Seq` is
deliberately **not** touched — issue #35 (`cpu`) names H5's training runs, and
although it is unassigned, this experiment does not need it: H51 builds its own
frame list from the three-way stem overlap, which it has to do anyway.
