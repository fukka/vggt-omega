# H22 — I got H21's mechanism backwards. Which direction is it really?

## The error this starts from

H21's `analysis.md` says, of why `equisolid` was the best geometry to fit on:

> equisolid allocates more image radius to high θ than Aria's KB4, so a fitting
> set rendered through it contains more rim pixels

**That is backwards, and it is checkable without any data.** Every lens in the
family is normalised so `r(θ_max) = R_disc`, so what decides how much of the disc
goes to the rim is the *shape* of `r(θ)/r(θ_max)`. Share of disc area given to
the outer half of the angle range:

| lens | r/R at 27.4° | area share of the outer half-angle |
|---|---|---|
| orthographic | 0.563 | **68.3%** |
| equisolid | 0.515 | 73.5% |
| equidistant | 0.500 | 75.0% |
| stereographic | 0.470 | 77.9% |
| rectilinear | 0.365 | **86.6%** |

`equisolid` gives the rim **fewer** pixels than equidistant or stereographic, not
more. The speculation was flagged "likely mechanical" rather than measured, but
it was stated in the wrong direction and is committed, so it gets corrected here
and in H21's analysis regardless of what this run finds.

## What the corrected direction predicts

H21's ranking of fitting geometries, best first: `equisolid` −31.7,
`equidistant` −30.6, `stereographic` −27.3, `aria_kb4` −24.2.

Against the table above, that is **monotone in the opposite direction to my
explanation**: the geometry with the *fewest* rim pixels made the best fitting
set. Three points is not much of a trend, so extend it at both ends.

* `orthographic` compresses the rim further than `equisolid`.
* `rectilinear` expands it far past everything else.

## Bars, locked before running

* **B1 — is it monotone?** Across all six lenses, the row-mean gain is monotone
  in rim area share, with Spearman ρ ≥ +0.8 (less rim area → more negative gain
  → better). Falsified if the ordering is not monotone.
* **B2 — does it keep going?** `orthographic` beats `equisolid` as a fitting
  geometry. **Falsified if it does not**, which would place the optimum at or
  near `equisolid` and make the relationship turn over rather than continue.
* **B3 — the other end.** `rectilinear` is the worst fitting geometry, worse
  than `aria_kb4`.

## What each outcome means

* **B1 and B2 pass** → "fit on a rim-compressing geometry" is a knob with a
  direction, and §4.6's recommendation should name `orthographic`, not
  `equisolid`. The mechanism is the opposite of what H21 guessed.
* **B2 fails, B1 passes** → there is an optimum near `equisolid`, which is a
  more useful result than a direction: it names a specific geometry to use.
* **B1 fails** → rim area share is not what orders these at all, H21's ranking
  is driven by something else, and the recommendation in both reports should be
  narrowed to "equisolid helped here" with no mechanism attached.

All three are worth having. The one thing already settled is that H21's stated
mechanism was wrong.

## Cost

Six lenses instead of four: ~2,160 DA3-Small forwards. Same runner, which now
also reports rim area share so the correlation is in the artifact rather than a
side calculation.
