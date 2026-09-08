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

## The construction — first version VOID, repaired

**The first construction was wrong and its own sanity check said so**, before a
single number was interpreted. I reasoned that a 60° view has a corner ray of
42.4°, inside Aria's 54.83° cone, so rotating it creates no border. That
confuses two different things: the view's **rays** stay inside the cone, but
rotating a square **raster** by 30° always throws its corners outside the
raster. 15.3% of the frame went black, and the arms it produced (DA3-Small
"resampling" +80%) were measuring a border again — exactly the quantity the
experiment exists to separate out.

**Repair: rotate on a larger canvas and crop afterwards.** Render a big view of
side VS ≥ vs·√2 at the same focal length. Its inscribed disc, radius VS/2,
contains every corner of the rotated vs×vs crop, so the crop stays fully
populated under any rotation. With vs = 630 and 60°, VS = 896 and the canvas
spans 78.5° with its own corners at 49.1° — still inside the 54.83° cone, so
nothing is black anywhere at all. The model always sees the same 630 px, 60°
crop; only the number of times it was resampled changes. Scoring is capped at
θ ≤ 28°, inside the crop's 30° inscribed disc with a margin so the crop edge
cannot bleed in.

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

The fraction of black pixels **inside the crop** must be zero after the double
rotation. It is printed for every backbone and recording. If any appears, the
construction has failed and the run is void — the entire point is a border-free
rotation. This check has already voided one design; it stays.

## Cost

Evaluation only: 4 arms × 2 backbones × 13 recordings × 20 frames, at one
angle. Both GPUs, split by backbone.
