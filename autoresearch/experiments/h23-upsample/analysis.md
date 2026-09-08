# H23 — B1 and B2 refuted. It is not upsampling loss.

`results/src504.json`, `results/src1008.json`. Two arms, identical supervision,
only the resolution the source frames are read at differs.

## The result

Row-mean gain per fitting geometry, same room:

| fitted on | rim area | source @504 | source @1008 | change |
|---|---|---|---|---|
| orthographic | 68.3% | −28.4% | −27.0% | +1.4 |
| equidistant | 75.0% | −25.8% | −24.2% | +1.6 |
| the real lens | 79.4% | −20.1% | −18.1% | +2.0 |
| rectilinear | 86.6% | −10.5% | −8.6% | +1.9 |
| **spread** | | **18.0 pts** | **18.5 pts** | **+0.5** |

Rearranged room: spread 11.7 → 13.6 pts.

**B1 refuted.** The bar was "spread shrinks by at least 30%, falsified below
10%". It did not shrink at all — it grew slightly on both recordings (−2.9% and
−16.3%).

**B2 refuted.** `rectilinear` was predicted to gain most. It did not: on the same
room every geometry drifted by a near-identical +1.4 to +2.0 points, and on the
rearranged room `orthographic` moved the *other* way (−1.7) while `rectilinear`
barely moved (+0.2). The difference between them is −0.5 and −1.9 points against
a bar of +3.

**B3 passes.** `orthographic` moved 1.4 / 1.7 points, small as designed, so the
two arms really do differ only in how much real detail reaches the rim.

## What this kills

**The hypothesis in H22's `analysis.md` is withdrawn.** Rim-expanding geometries
do not lose because their extra rim pixels are interpolated. Feeding the warp a
source with twice the real detail — detail that was sitting in the file all
along, since ADT frames are natively 1408×1408 — changes the ordering not at all
and the spread not at all.

That is now **two mechanisms ruled out** for the same effect:

1. "More rim pixels make a better fitting set" — H22, refuted; it is the exact
   opposite.
2. "Rim-expanding geometries lose because the extra pixels are invented" — H23,
   refuted.

## What survives, and what it now has to explain

Whatever orders these geometries is **not about pixel-level detail**, because
doubling the available detail does nothing. It has to be about *where the
angular content sits in the frame* — the same content, the same rays, the same
underlying sharpness, arranged differently across the image radius.

That is a narrower and more interesting residual than before, and it rhymes with
§4.2's border result, where what mattered was position in the frame rather than
how much of it was affected. But I have now had two mechanism guesses refuted on
this one effect, so **no third guess is offered here.**

## The recommendation is unaffected, and gains a robustness check

H22's row effect was measured against fixed targets, and H23 reproduces the same
monotone ordering at both source resolutions with the same slope. So "fit on a
rim-compressing geometry" is not an artefact of reading the source at 504. It
replicates at 1008.

One incidental: at 1008 every geometry is ~1.5–2 points *worse* on the same room
and mixed on the rearranged one. A uniform offset, not a differential — the
sharper input is slightly further from what the 504-derived teacher cache
represents. Worth remembering if anyone re-derives the cache at higher
resolution, but it is not the effect under study.

## What would actually decide it

Separating "where content sits in the image" from "how much angular range each
pixel covers" needs a probe that moves one without the other. Scaling the same
lens into a smaller disc moves content inward but introduces a border, and §4.2
says a border is catastrophic — so that particular design is confounded and
should not be run as-is. Recorded as an open problem rather than a queued
experiment.
