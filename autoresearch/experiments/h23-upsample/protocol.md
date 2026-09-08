# H23 — is the rim-density effect upsampling loss, or the angular distribution itself?

## What H22 left open

H22 found a perfectly monotone relation (Spearman +1.000): the lens geometry
that gives the rim the fewest pixels makes the best fitting set. Two
explanations survive it, and its `analysis.md` says so rather than picking one.

* **A — upsampling loss.** Rim-*expanding* geometries stretch the source's rim
  over more pixels. The extra pixels are interpolated from the same underlying
  samples, so they carry no new information and add blur. Rim-*compressing*
  geometries downsample instead: fewer pixels, each a genuine average of real
  ones. On this reading the fit is limited by rim pixel **quality**, not count —
  the same shape of explanation as H20's motion result.
* **B — generic resampling damage, or the angular distribution itself.** The row
  and column effects in H22 are ordered the same way, so the whole ranking may
  simply be "how much this rendering was degraded", hurting fitting and scoring
  alike.

The practical recommendation survives either way, because H22's row effect was
measured against fixed targets. The mechanism does not.

## The test

ADT frames are natively **1408×1408** and everything so far has read them at
504, so a rim-expanding warp at 504 is inventing detail that exists in the file.

Two arms, identical in every other respect:

* `src504` — source read at 504, warped to each lens at 504. H22's setting.
* `src1008` — source read at 1008, warped to each lens at 504. Now the rim of a
  rim-expanding rendering is drawn from real detail rather than interpolated.

**The supervision is byte-identical between arms.** The cached teacher target and
the coverage mask are always warped from the 504 grid; only the *input image*
changes. So this isolates "how much real detail reaches the rim of the warped
input" from everything else.

Four lenses spanning H22's range: `orthographic` (68.3% rim area share),
`equidistant` (75.0%), `aria_kb4` (79.4%), `rectilinear` (86.6%).

## Bars, locked before running

Let *spread* be the range of row-mean gain across the four lenses — 19.2 points
at 504 in H22 (orthographic −30.5 to rectilinear −11.3).

* **B1 — A predicts the spread collapses.** Spread under `src1008` is at least
  **30% smaller** than under `src504`. **Falsified if it shrinks by less than
  10%**, which would leave B as the explanation.
* **B2 — A predicts it is the expanders that gain.** `rectilinear`'s row mean
  improves by more than `orthographic`'s, by at least 3 points of difference.
* **B3 — sanity.** `orthographic` under `src1008` is no worse than under
  `src504` by more than noise. A compressing geometry downsamples either way, so
  a large change there would mean the two arms differ in something other than
  rim detail.

## What each outcome means

* **B1 and B2 pass** → the effect is upsampling loss. "Squeeze before you fit"
  is really "do not ask the warp to invent rim pixels", and the better advice
  becomes *render the fitting set from the highest-resolution source you have*,
  which is cheaper and more general than choosing a lens shape.
* **B1 fails** → A is wrong. The ranking is about the angular distribution or
  generic resampling damage, the hypothesis in H22's analysis gets withdrawn,
  and "squeeze before you fit" stays an empirical rule with no mechanism.

Either way H22's recommendation stands; what changes is what we can say about
why, and whether a cheaper substitute exists.

## Cost

Two arms x four lenses x 360 frames ≈ 2,880 DA3-Small forwards. No new data, no
teacher inference. The runner gains `--src-size`, defaulting to `--size` so
H21/H22 remain reproducible.
