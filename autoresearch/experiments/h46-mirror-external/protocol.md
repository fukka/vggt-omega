# H46 — does the mirror effect survive a different dataset?

## The check H45 asked for, in its own words

H45's limits section says it plainly: *"One rectified 89° view of one fisheye,
indoors, per-frame scale-shift alignment. The magnitude is large enough that an
independent check on a different dataset would be worth having."*

Reflecting the model's input and reflecting the answer back cost **+58%**
(`vggt_omega`), **+67%** (`vggt`), **+254%** (`da3:small`) and **+350%**
(`da3:large`) over thirteen Aria recordings, with the two pretraining families
not overlapping. That is a large claim resting on one camera, one apartment and
one rectification pipeline.

## Why ScanNet++ is the right check, and why it is cheap

It is already on the box (`/netapp/datasets/f.zhang2/scannetpp`, 1,018 scenes)
and this repo already has a loader for it — `raytun3r.data.ScanNetPPFisheye`,
with the frame-quality and depth-convention traps already documented in its own
docstring. **H36's lesson applies: look for the measurement that already exists
before building one.**

It is genuinely independent of everything H45 rested on:

| | H45 | H46 |
|---|---|---|
| camera | Aria 214-1, 110° | ScanNet++ DSLR fisheye, ~115° |
| what the model sees | a rectified 89° pinhole view warped out of the fisheye | the captured frame itself, no rig, no warp |
| scenes | one apartment, 13 recordings | ten different scenes |
| ground truth | ADT rendered depth | ScanNet++ mesh-rendered depth |

The pipeline overlap is the backbone and `forward_z` — which is exactly what the
claim is about.

## Design

Ten scenes, 20 frames each, four backbones, three arms per frame:

* `normal` — the frame as captured
* `mirror` — flip the frame, predict, flip the planar z back
* `twice` — flip and unflip before predicting (**the identity**)

`depth_convention="z"` on both the loader and the backbone, so ground truth and
prediction are both planar z and nothing is converted. Frames ScanNet++ itself
flags `is_bad` are dropped by the loader. Scored on whole-image AbsRel with
per-frame scale-shift alignment, on pixels with valid depth.

## Bars, locked before running

* **B1 — the plumbing bar, and it is not optional.** The `twice` arm differs
  from `normal` by under **0.01%** on every scene and backbone. **Falsified
  otherwise, and then nothing here is a measurement.** This is the bar that
  turned H44's void into H45's result and it travels with the method.
* **B2 — does it replicate at all?** Both DA3 variants cost more than **+25%**.
  **Falsified below**, which would mean H45's effect is a property of the
  rectified-Aria pipeline rather than of the models — a far more interesting
  and far more damaging outcome for H45's framing.
* **B3 — does the family split replicate?** Both DA3 variants cost more than
  both VGGT variants. **Falsified otherwise**, which would leave the effect real
  but not the pretraining-family story.

## What it cannot do

It cannot say *why*. Chirality cues in natural photographs remain the obvious
guess and remain untested. And a second dataset is a replication, not a proof of
generality: both are indoor, human-captured, and both models were pretrained on
web-scale imagery that includes rooms.

## Cost

Evaluation only: 4 backbones × 3 arms × 20 frames × 10 scenes.
