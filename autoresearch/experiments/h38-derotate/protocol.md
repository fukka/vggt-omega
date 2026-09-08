# H38 — you have the gravity vector. Should you de-rotate before inference?

## The question this line was actually asked

The roll line opened with a human question: *do depth foundation models assume a
level camera; is that violated by egocentric data; can the device's IMU supply
the roll; and can we feed it in to boost performance?* Three of the four parts
are answered:

* **H17.2** — they do assume it, by pretraining family: DA3 rises +46/+52% at
  30°, both VGGT variants +11%.
* **H17.1** — it is violated, mildly: ADT's real head roll is median 3.7°, p99
  21.8°, and the per-session mounting offset adds 4–5° on a third of sequences.
* **H35** — integrated against 60,105 real frames, ignoring roll costs
  **1.80%** (DA3-Small) to **0.46%** (VGGT-Omega).

The fourth part — *use it* — was deprioritised as H17.4 with the argument
"there is almost nothing to recover". That argument was about the size of the
prize. It never measured the **price**.

## Why the price is not obviously small

The cheapest way to use a known roll is to rotate the image level before
inference and rotate the planar-z prediction back. `upright.forward_z`'s own
docstring establishes that second step is exact: planar z is invariant under a
roll about the optical axis.

But rotating a square view leaves **four black corner triangles**. And H37 just
measured, on thirteen recordings with no exception at either end, what a hard
border beside the scored zone costs:

| backbone | border cost |
|---|---|
| `vggt_omega` | +6.0% ± 4.6 |
| `vggt` | +48.4% ± 18.0 |
| `da3:small` | +59.0% ± 14.1 |
| `da3:large` | +268.9% ± 61.5 |

So the intervention that removes a ~2% problem introduces a class of artefact
that costs between 6% and 269% depending on the backbone. **The prediction this
line's own results make is that de-rotation is a net loss for most backbones** —
which, if true, is the useful answer to the human question and the opposite of
the obvious one.

The scored region is the inscribed disc (θ ≤ 44°) of an 89° square view. A
rotation about the centre maps that disc to itself, so the black corners land
**entirely outside the scored pixels** — exactly the H17.5 / H29 / H37 geometry.

## Design

Three arms at each roll angle, same frames, same scored mask:

| arm | what the model sees | roll removed? | border paid? |
|---|---|---|---|
| `raw` | the view rendered at roll *a* | no | no |
| `derot` | that view rotated by −*a* | **yes** | yes |
| `null` | that view rotated by −*a* then +*a* | no | **yes** |

`null` is the untransformed-baseline bar the H34 rule demands: identical
resampling, identical black corners, identical lost content, **zero roll
removed**. Without it, any improvement in `derot` could be resampling blur
flattering a scale-shift-aligned metric, and any degradation could be blamed on
the roll.

**The rotation sign is fixed by geometry, not by the score.** Before any model
runs, both signs are applied to the roll-*a* render and compared against the
roll-0 render on the inscribed disc; the sign with the smaller L1 is used and
the two residuals are printed into the result JSON. This is deliberate: picking
the sign by which one scores better would be choosing the answer.

Angles 0, +30, −30 — reported **separately, never averaged**, because H32 found
the ±30 pair differs (+24% vs +56% on DA3-Small) and H33 could not attribute
it. Four backbones, six recordings, 20 frames each.

## Bars, locked before running

* **B1 — does de-rotation help at all, before counting its price?**
  `derot` is below `raw` at ±30° for DA3-Small on a majority of recordings.
  **Falsified** if it is not: then the operation does not even remove the roll
  it is supposed to remove, and the design is wrong rather than the idea.
* **B2 — the deployment question. Is it a net win?**
  Pre-registered prediction: **no, except possibly for VGGT-Omega**.
  Concretely, `derot` at ±30° is *above* `raw` at 0° (the level-camera
  reference it is trying to recover) for `da3:large`, and the gap between
  `derot` and `raw`-at-0 is ordered like H37's border costs.
  **Falsified** if `derot` reaches within 5% of `raw`-at-0 for `da3:large`.
* **B3 — out-of-sample test of H37's ordering on a different border shape.**
  The pure price of the operation, `null`/`raw` − 1 at ±30°, is ordered
  `vggt_omega` < `vggt` ≈ `da3:small` < `da3:large` — H37's ordering, measured
  on four corner triangles instead of an annulus. **Falsified** if
  `da3:large` is not the most damaged.

## What each outcome means

* **B2 confirmed** → the answer to the human question is *"you can read the
  roll, and you should not act on it this way"*, with a measured reason: the
  artefact costs more than the roll. That is a publishable negative and it
  points at the alternative (rotate the **sampling grid**, not the image, so no
  border is ever created — which the rig already does and which is what
  `raw` at roll 0 secretly is).
* **B2 falsified** → de-rotation is a real free win on a known-roll device, and
  H17.4's deprioritisation was wrong.
* **B3 confirmed** → H37's ordering is about borders in general, not about the
  particular annulus H29/H37 used.

## Cost

Evaluation only. 4 backbones × 3 arms × 3 angles × 6 recordings × 20 frames.
Split across both GPUs by backbone.
