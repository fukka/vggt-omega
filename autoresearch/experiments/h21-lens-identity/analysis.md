# H21 — "a property of the lens" is wrong. B2 refuted.

`results/lens_identity.json`. Four lenses over the Aria cone, every arm warped
once from the real Aria camera so resampling blur is common to all.

## The transfer matrix (seq136, near-rim gain)

| fitted on ↓ / applied to → | aria_kb4 | equidistant | stereographic | equisolid |
|---|---|---|---|---|
| **aria_kb4** | −21.0% | −25.8% | −23.3% | −26.9% |
| **equidistant** | −26.0% | −32.8% | −28.6% | −34.9% |
| **stereographic** | −23.6% | −29.1% | −25.9% | −30.6% |
| **equisolid** | **−26.8%** | −34.2% | −29.4% | −36.6% |
| `global` (diagonal) | +5.2% | +3.5% | +4.7% | +2.3% |

## Bars

**B3 passes.** `global` is worse than doing nothing on every diagonal (+2.3 to
+5.2%) against `radial`'s −21 to −36.6%. The warped setting is sound and the
radial structure is real, so B1 and B2 mean something.

**B2 is refuted — the pre-registered falsification.** Cross-lens minus same-lens
is **+0.84 points** (seq136) and **+0.47** (dec132), both inside the ±1.0
draw-to-draw sd H20 measured. Matching the lens buys nothing.

**B1 is marginal and should not be leaned on.** Mean pairwise |Δa| is 0.0783
against a 0.069 noise floor, and two of the six pairs sit *below* the floor
(aria↔stereographic 0.0224, equidistant↔equisolid 0.0459). The four curves are
close to the same shape.

## The dominant effect is which geometry you FIT on

Row means (the lens the curve was fitted on) span **7.5 points** on seq136:
aria_kb4 −24.2, stereographic −27.3, equidistant −30.6, equisolid −31.7. The
diagonal-versus-off-diagonal difference is 0.84. **The fit-geometry effect is
about 9x the lens-matching effect.**

And the ordering is consistent: `equisolid`'s curve is the best available curve
for **every** target lens, including the real one, where it beats the real
lens's own curve by **5.8 points** (−26.8% against −21.0%) on seq136 and **4.4**
on dec132 (−14.3% against −9.9%).

**Correction, 2026-09-08.** This section originally guessed that `equisolid`
"allocates more image radius to high θ ... so its fitting set contains more rim
pixels". **That is backwards**, and it is checkable with no data at all: every
lens is normalised so `r(θ_max) = R_disc`, so what matters is the shape of
`r(θ)/r(θ_max)`. `equisolid` gives the outer half of the angle range **73.5%** of
the disc area, against equidistant's 75.0% and stereographic's 77.9% — *fewer*
rim pixels, not more.

The ranking above is therefore monotone in the **opposite** direction to the
original guess: the geometry with the fewest rim pixels made the best fitting
set. Why that would be is now genuinely open. H22 tests the corrected direction
by extending the family at both ends (`orthographic` compresses further,
`rectilinear` expands far past everything here). The guess was labelled "likely
mechanical" rather than measured, but it was stated in the wrong direction and
committed, so it is corrected rather than quietly dropped.

## What this changes

**"A property of the lens" is wrong** and appears in both reports. The correction
is essentially the same object across four quite different radial mappings; what
differs between them is how well each *estimates* it. The honest restatement:
it is a property of how the backbone's error grows with incidence angle, and it
is largely independent of how the lens distributes pixels across that angle.

That also makes calibration a genuine one-off, which is the better deployment
story the protocol flagged in advance.

**And it produces a new, actionable recommendation:** resample the footage to a
rim-stretching geometry *before* fitting, then apply the resulting curve to the
real images. On this data that is worth 5.8 points on the real Aria lens, for
free.

## Two limits, stated rather than discovered later

**These are synthetic re-renderings, not four real cameras.** All four share the
same footage, cone, optics, MTF, vignetting and native resolution. So
lens-independence is established for **radial mapping**, which is what the curve
is indexed by — not for everything the word "lens" covers.

**These coefficients are not comparable to the earlier fit.** Every arm here went
through a resample, so a(θ) runs 1.6–2.1 where the unwarped H18.2 fit ran
1.34–1.47, and the gains are correspondingly larger. Compare only within this
experiment.
