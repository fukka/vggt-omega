# H24 — the ordering generalises. Most of the headroom does not.

`results/da3_large.json`. Six lenses, identical to H22 in every respect except
the depth model.

## The bars

**B1 passes.** Spearman(rim area share, row-mean gain) = **+0.943** on the same
room and **+1.000** on the rearranged one, against a bar of +0.8. The single
inversion is `orthographic` and `equisolid` swapping by 0.2 points. So the
ordering is a property of the task, not of DA3-Small.

**B2 passes.** On the real Aria lens, the best foreign curve beats the lens's own
by **+9.1 points** (same room: equisolid −17.2% against its own −8.1%) and
**+4.3** (rearranged: orthographic −2.2% against its own +2.1%). Both clear the
3-point bar, and the same-room margin is larger than DA3-Small's 5.8.

**B3 fails on one cell, in both recordings.** `global` was supposed to be worse
than `radial` on every diagonal. For `rectilinear` it is better: radial +4.6% vs
global +1.9% (same room), +9.9% vs +5.6% (rearranged). Both are positive — this
is the cell where the method has collapsed entirely for this model, so the
sanity check fails exactly where nothing works. Recorded, not explained away.

## The finding the bars nearly hide

| fitted on | rim area | DA3-Small | DA3-Large | Small (rearranged) | Large (rearranged) |
|---|---|---|---|---|---|
| orthographic | 68.3% | −30.5% | −13.5% | −15.6% | **−0.7%** |
| equisolid | 73.5% | −29.0% | −13.7% | −14.1% | +0.8% |
| equidistant | 75.0% | −27.7% | −11.3% | −13.3% | +1.0% |
| stereographic | 77.9% | −24.7% | −8.1% | −11.1% | +1.0% |
| the real lens | 79.4% | −21.8% | −5.4% | −9.3% | +2.6% |
| rectilinear | 86.6% | −11.3% | +3.6% | −2.4% | +5.0% |

**On the harder recording, DA3-Large's whole method sits at zero.** The best
geometry manages −0.7%; every other one makes things worse. The ordering is
perfectly monotone (ρ = +1.000) — around nothing.

So B2 "passing" on that recording means −2.2% against +2.1%: *slightly helps*
versus *actively hurts*. That is a relative bar cleared while the absolute
numbers hover at zero, and it is exactly the kind of pass that reads better than
it is.

## Reading it honestly

Three separable claims, at three different strengths:

1. **Which geometry to fit on generalises.** ρ = +0.943 / +1.000 on a model whose
   framing sensitivity is wildly different (a border alongside the scored region
   costs DA3-Large +636% against DA3-Small's +106%). Strong.
2. **Preferring a foreign rim-compressing curve over the lens's own generalises.**
   +9.1 and +4.3 points. Solid.
3. **The size of the prize does not.** DA3-Large gets less than half of
   DA3-Small's gain on the easy recording and essentially none on the hard one.

An observation, offered as headroom arithmetic rather than as a mechanism —
this effect has already eaten two of my mechanism guesses. DA3-Large starts with
**half the rim error** of DA3-Small (0.191 against 0.389 in the four-model table),
so there is less systematic radial error available to remove. A correction that
removes a systematic component should do less when there is less of it.

## What both reports must now say

The §4.6 recommendation cannot stand unqualified. It generalises **as a rule
about which geometry to pick**, and it does not generalise **as a promise about
how much you will gain**. On a stronger depth model the ordering holds and the
prize largely evaporates — and one geometry, the most rim-expanding, turns
actively harmful.

Anyone applying this should fit the curve and check it against ground truth on
their own model before shipping it, because "it helped a lot on DA3-Small" does
not carry.
