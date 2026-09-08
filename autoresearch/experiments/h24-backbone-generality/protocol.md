# H24 — does the rim-density ordering hold for a different depth model?

## Why this and not another mechanism probe

Two mechanism guesses have now been refuted on this effect (H22's "more rim
pixels", H23's "invented pixels"), and H23's analysis says plainly that no third
guess is being offered. The next useful question is therefore not *why* but
*how far it generalises* — because that is what the recommendation in both
reports actually depends on.

**Everything from H21 to H23 used DA3-Small.** H18.2 already established that the
fitted curve is backbone-specific. What has never been checked is whether the
*ordering of fitting geometries* — the thing "fit on a rim-compressing geometry"
rests on — is a property of the task or of that one model.

`DA3-Large` is the right probe. It is the model whose framing sensitivity is
most unlike DA3-Small's: a hard border alongside the scored region costs it
**+636%** against DA3-Small's +106%, a 7.4× collapse from a change that touches
no scored pixel. Two models that differ that much in how they respond to what
sits at the edge of the frame are a real test of whether the ordering is about
the task.

## Arms

The same six lenses as H22, over the Aria cone, every arm warped once from the
real Aria camera. Only the depth model changes: `--variant large`.

Everything else is held: same fitting sequences, same 240 frames, same cached
omega110 teacher targets, same evaluation recordings, same scoring.

## Bars, locked before running

* **B1 — does the ordering survive?** Spearman(rim area share, row-mean gain)
  ≥ **+0.8** for DA3-Large, on both evaluation recordings. DA3-Small gave
  +1.000. **Falsified below +0.5.**
* **B2 — does the recommendation still pay?** On the real Aria lens, the best
  rim-compressing geometry beats the real lens's own curve by at least
  **3 points**. DA3-Small gave 5.8.
* **B3 — sanity, not a finding.** The 2-parameter `global` control is worse than
  `radial` on every diagonal, as in H21 and H22. If it is not, the setting is
  broken for this model and B1/B2 mean nothing.

## What each outcome means

* **B1 and B2 pass** → the ordering is a property of the task, not of one model,
  and "fit on a rim-compressing geometry" can be stated without a backbone
  caveat. That is a real strengthening of the only recommendation in this line
  that improves a result rather than fencing one.
* **B1 fails** → the recommendation is DA3-Small-specific. Both reports have to
  say so, and the §4.6 advice narrows to "this helped for the model we tested".
  Given that H18.2 already found the curve itself is backbone-specific, this
  outcome is entirely plausible and would not be a surprise.
* **B2 fails while B1 passes** → the ordering is real but the headroom is not,
  which would mean the trick is worth less on stronger models. Worth knowing
  before recommending it.

## Expectation, recorded but not leaned on

Given DA3-Large's extreme framing sensitivity I expect the ordering to be at
least as strong, not weaker. That is an expectation, not a prediction I have
earned — two mechanism guesses on this effect have already been wrong, and this
one has no more grounding than those did.

## Cost

Six lenses × 360 frames of DA3-Large at 504². No training, no teacher inference.
