# H40 — border × resampling, at a matched angle, in one 2×2

## The caveat this exists to remove

§03z's headline is that **the border was ~95% of de-rotation's price**: H38
charged 64.1% for a rotation *with* a border, H39b measured 3.03% for the
resampling alone, ratio 0.047.

Its own callout names the axis that was never varied: **the rotation angle**.
H38's 64.1% is at ±30°; H39b's 3.03% is at the real head roll (median 4–7°),
and H39b's P2 showed the resampling price *grows* with the angle. So 0.047 is
an upper bound on the advantage, not an angle-matched ratio, and I wrote in the
report that isolating it at 30° "cannot be done on this data" because ADT's
maximum real roll is ~28°.

**That was true of that construction, not of the question.** A different view
geometry isolates it directly.

## The construction

Aria's cone is 54.83°. The 89° square view used throughout this line has a
corner ray of 62.9°, so it *already* runs off the cone — which is why rotating
it creates black corners. A **60° square view has a corner ray of 42.4°**, well
inside the cone, so rotating it about its optical axis **never creates a
border**. Its inscribed disc is at 30°, so scoring is capped at θ ≤ 29° and is
invariant under the rotation.

That makes a clean 2×2 possible at a single, freely chosen angle:

| arm | rotation | border | what it isolates |
|---|---|---|---|
| `direct` | none | none | baseline |
| `rt` | −δ then +δ | none | **resampling alone** |
| `masked` | none | inscribed disc | **border alone** |
| `rt_masked` | −δ then +δ | inscribed disc | both |

δ = 30°, matching H38 exactly. The content of `rt` is the same content as
`direct` up to two resamplings — no roll is removed or added, and nothing
leaves the cone.

## Bars, locked before running

* **B1 — is the border really the dominant term?** At a matched 30°, the border
  alone costs at least **4×** what the resampling alone costs.
  **Falsified under 4×**, in which case §03z's "the border was ~95% of the
  price" is an artefact of the angle mismatch and must be restated as a
  real-roll-only claim.
* **B2 — how much of 0.047 was the angle?** The resampling price at 30° exceeds
  H39b's real-roll 3.03%. This is expected (P2 showed it grows with angle); the
  point is to publish the matched number instead of an upper bound.
  **Falsified** if the 30° resampling price is *below* the real-roll one, which
  would mean H39b's P2 was measuring something else.
* **B3 — do the two terms add?** `rt_masked` is within **±30% (relative)** of
  `rt + masked` treated additively on the error ratio. Not a bar on the claim —
  a bar on whether "price = border + resampling" is even the right sentence. If
  it fails, the interaction is reported and the decomposition is described as
  approximate.

## Sanity check (not a bar)

At δ = 0 the `rt` arm must be bit-equivalent to `direct`, and the fraction of
black pixels inside the 60° view must be **zero at every rotation angle**. Both
are printed. If any black appears, the construction has failed and the run is
void — the entire point is a border-free rotation.

## Cost

Evaluation only: 4 arms × 2 backbones × 13 recordings × 20 frames, at one
angle. Both GPUs, split by backbone.
