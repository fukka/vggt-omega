# H30 — does the "squeeze before you fit" gain hold, and does H29's observation predict?

## Two questions, one run

**The recommendation.** §4.6's advice — re-render to a rim-compressing geometry
before fitting the curve, then apply it to the real images — is worth "5–6
points" on the real Aria lens. Every number behind it (H21–H24) was **evaluated
on seq136**.

**My own observation.** H29 ended with a statement I made rather than measured:

> seq136 responds unusually strongly to **interventions**. Corrections applied
> to it help unusually much (H25–H27); perturbations hurt unusually much (H29);
> its unperturbed structure is ordinary (H28).

Every H21–H24 result is an intervention scored on seq136. **If that observation
is right, the 5.8-point gain should read high**, and the thirteen-recording mean
should be smaller. Writing the prediction down before running is the only thing
that makes it worth anything — the same discipline that made H22's correction
clean rather than embarrassing.

## Design

`lens_identity.py` unchanged. Curves fitted on the same four Apartment training
sequences from the same cache — **nothing is refitted**. Only the evaluation set
changes: the thirteen recordings instead of seq136.

Lenses: `orthographic`, `equisolid`, `aria_kb4`. The column that matters is
`aria_kb4` — the real lens, the only target that exists in deployment.

## Bars, locked before running

* **B1 — does the recommendation hold?** On the real lens, the best
  rim-compressing curve beats the real lens's own curve on a **majority** of the
  thirteen. **Falsified below 7/13**, in which case §4.6's advice is a
  one-recording artefact and is withdrawn rather than restated.
* **B2 — does H29's observation predict correctly?** seq136's 5.8-point
  advantage is **more than 1 sd above** the thirteen-recording mean.
  **Recorded as a prediction, not a bar to pass**: if it lands inside 1 sd the
  observation is wrong and gets withdrawn from findings, which matters because
  it is currently offered there as covering all five experiments.
* **B3 — does the ordering hold?** Row means keep `orthographic` ≥ `equisolid` ≥
  `aria_kb4` as fitting geometries on the thirteen-recording evaluation.

## What each outcome means

* **B1 passes, B2 predicts correctly** → the advice stands with a smaller,
  honest number, and the intervention-amplification observation survives its
  first real test.
* **B1 passes, B2 wrong** → the advice stands and the observation is withdrawn.
  That is the more useful failure: it would mean seq136's inflation is
  experiment-specific and there is no general rule, so every future claim needs
  its own re-measurement rather than an appeal to the pattern.
* **B1 fails** → §4.6 loses its only actionable recommendation.

## Cost

Three lenses, one GPU pass: 240 fitting frames (cached teacher) and 13 × 60
evaluation frames per lens. No training, no teacher inference.
