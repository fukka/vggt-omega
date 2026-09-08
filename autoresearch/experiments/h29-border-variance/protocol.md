# H29 — does the border warning hold across recordings?

## The claim being checked

§4.2 calls one result "**the most transferable warning here**":

> DA3-Large is the best of the four on a clean frame and the worst with a border
> — a **7.4× collapse** (+636% against DA3-Small's +106%) from a change that
> touches no scored pixel.

That is the strongest practical advice in either report, and it is one recording.
H25 showed exactly this pattern — a strong claim on seq136 — and found it a
5.74 σ outlier for one method and 2.3–3.1 σ for others. H28 then found seq136
*ordinary* for the phenomenon numbers. Which kind is the border result?

The dose curve is in the same position: "+22% at 3.1% of the frame, +6.6% at
39.5%" is the evidence for **cost is set by proximity, not area**, and it too is
one recording.

## Design

Two existing runners, unchanged, on the same thirteen recordings, restricted to
`--angles 0` since both published comparisons are at the level camera.

| arm | runner | published number being checked |
|---|---|---|
| `models` | `roll_boundary.py --models da3:small,da3:large --angles 0` | +106% and +636% |
| `dose` | `border_dose.py --angles 0` | the 0→100 px dose curve |

## Bars, locked before running

* **B1 — is +636% representative?** DA3-Large's border cost across the thirteen
  has sd under one third of its mean, and seq136 sits within **2 sd**.
  **Falsified beyond 2 sd**, in which case the headline warning is restated on
  the thirteen-recording mean.
* **B2 — does the ordering survive?** DA3-Large's border cost exceeds
  DA3-Small's on a **majority** of the thirteen. **This is the claim that
  actually matters**: failing it would not restate the warning, it would
  withdraw it.
* **B3 — is it proximity rather than area?** The 70 px border (39.5% of the
  frame) costs **less** than the 5 px border (3.1%) on a majority of the
  thirteen. That inversion is the whole evidence for the proximity reading.

## What each outcome means

* **B2 and B3 pass** → the warning stands as advice regardless of what B1 does,
  because the advice is "do not put a border next to what you care about", not a
  particular percentage.
* **B2 fails** → the strongest practical recommendation in either report is a
  one-recording artefact and must be withdrawn, not softened.
* **B3 fails** → "proximity, not area" loses its evidence, and §4.2's headline
  becomes just "borders are expensive", which is much weaker and much less
  useful.

## Cost

Evaluation only, at a single camera angle: 2 models × 13 recordings for the
first arm, 7 widths × 13 for the second. No training.
