# H30 — the recommendation holds 13/13, and the prediction landed (weakly)

`results/lens_13.json`. Curves fitted on the same four training sequences from
the same cache; only the evaluation set changed.

## B1 — the recommendation holds, unanimously

On the real Aria lens, a foreign rim-compressing curve beats the lens's own on
**13 of 13** recordings.

| | thirteen recordings | seq136 |
|---|---|---|
| the real lens's own curve | −14.9% ± 3.2 | −21.1% |
| best rim-compressing curve | −18.9% ± 4.4 | −26.8% |
| **advantage** | **+4.03 ± 1.33 pts** | +5.75 pts |

So §4.6's advice stands and its value is **about 4 points, not 5–6**. Never
negative on any recording, which is what makes it worth recommending at all.

**B3 passes**: row means keep the ordering — `orthographic` −26.5%,
`equisolid` −25.2%, `aria_kb4` −18.4%.

## B2 — the prediction was right, and one test is weak evidence

H29 ended with an observation I stated rather than measured: *seq136 responds
unusually strongly to interventions.* H30 pre-registered the prediction that
seq136's advantage would sit **more than 1 σ above** the thirteen-recording
mean.

It does: **+1.29 σ**.

**That single result is weak.** For a recording drawn at random, P(z > 1) ≈ 16%,
so this test alone is p ≈ 0.16 and would not be worth quoting. What makes the
observation credible is the campaign, not this run:

| intervention measured on seq136 | z |
|---|---|
| H25 radial curve | **+5.74** |
| H26 adapter | +2.60 |
| H27 GT-trained arm | +3.12 |
| H29 border, DA3-Small | +3.32 |
| H29 border, DA3-Large | **+5.96** |
| H30 lens advantage | +1.29 |

| unperturbed measurement | z |
|---|---|
| H28 rim ÷ centre | −0.11 |
| H28 roll ±20° | +0.07 |
| H28 roll ±30° | +0.52 |

**Six interventions, all positive** — a sign test alone gives p ≈ 0.016, before
counting that two of them exceed 5 σ. **Three unperturbed measurements, all at
zero.** The split is clean and it is what the observation rests on. H30's own
contribution is one more point on the correct side, pre-registered, which is the
most a single run can be.

Still an observation and **not a mechanism**. It says *what* seq136 does, not
*why*, and this line has already had two mechanism guesses refuted.

## The practical consequence

**Any intervention measured only on seq136 should be assumed inflated.** That now
covers every published in-room figure in this line, all of which have been
re-measured, and it is a standing instruction for anything measured in future.

The three unperturbed measurements suggest the converse is safe: a *descriptive*
number measured on seq136 has been representative every time it was checked. But
that rests on three data points and should not be leaned on.
