# H45 — how far from mirror-equivariant are these models?

## Where this came from: a void run's construction check

H44 set out to use a mirror to separate two explanations for the residual roll
asymmetry. Its mandatory B0 — *mirroring a level frame must cost under 25%,
because a mirrored room is still a room* — **failed by an order of magnitude**,
so H44 is void for its purpose.

But the failure has two possible causes and the bar cannot tell them apart:
the construction is wrong, or **these models really are far from
mirror-equivariant**. A flip-twice arm settled it — flipping the view and back
is the identity, so that arm must equal `normal` exactly, and it does:
**0.0000% maximum relative difference**, both backbones.

**So the plumbing is exact and the number is a measurement.** On the pilot
recordings it is large and it splits by pretraining family:

| backbone | mirroring a level frame costs |
|---|---|
| `da3:small` | +183% … +258% |
| `vggt` | +51% … +77% |

`D(flip(I))` should equal `flip(D(I))` for any model that has no preferred
handedness. These are not close.

## Labelled honestly

**CONFIRMATORY, and the direction was already seen.** Four recordings of pilot
data came out of the void H44 run before it was stopped, and they show the split
above. This re-runs it on thirteen recordings and four backbones with the bars
written down first; it is not a blind test and is not presented as one.

## Design

`mirror_curve.py` at `--deltas=0` only — no roll, nothing else moving. Three
arms per frame: `normal`, `mirror` (flip the rendered view, predict, flip the
planar z back), and `twice` (flip and unflip before predicting — the identity).
Thirteen recordings, 20 frames, four backbones. The ground truth is never
touched; the prediction returns to the original frame before it is scored.

## Bars, locked before running

* **B1 — does it split by pretraining family?** Both DA3 variants cost more
  than both VGGT variants, on the thirteen-recording means. **Falsified if any
  VGGT variant exceeds any DA3 variant** — which would make this a different
  axis from roll robustness rather than the same one.
* **B2 — is it large?** DA3-Small's mean exceeds **+100%**. **Falsified below**,
  in which case the pilot was unrepresentative and the effect is ordinary.
* **B3 — is the plumbing exact everywhere?** The `twice` arm differs from
  `normal` by under **0.01%** on every recording and every backbone.
  **Falsified otherwise, and then nothing here is a measurement** — this is the
  bar that turns H44's void into H45's result and it is not optional.

## What it will not claim

That mirror sensitivity *causes* the roll asymmetry — H44 was void precisely
because the mirror changes the error regime too much to serve as that
discriminator. And no mechanism for why a depth model should prefer one
handedness; chirality cues in natural photographs are the obvious guess and
this run does not test it.
