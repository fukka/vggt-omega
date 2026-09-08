# H34 — mirror the input and see whether the asymmetry flips

## Correcting something H33 said

H33's analysis ended: *"no third design is offered — that would need the upright
convention itself varied, which is invasive."*

**That under-called what is available.** There is a cheaper and cleaner
discriminator: **flip the source image left-right** and re-run the same sweep.
It needs one flag, no change to any convention, and it separates the two
candidate carriers directly.

## Why a mirror separates them

A roll of **+θ applied to mirrored content** is, in content terms, **−θ on the
original**. So:

* if the asymmetry is carried by the **scene** — the wearer's hands and the desk
  sitting low and to one side — mirroring the content **flips its sign**;
* if it is carried by something **fixed in image space** — our principal-point
  offset (4.5 px, and *not* mirrored here), the quarter-turn upright convention,
  or a handedness the model learned in image coordinates — it **survives
  unchanged**.

The camera model is deliberately left untouched, so the principal-point offset
stays on the same side of the frame while the content moves to the other. That
is what makes the two cases distinguishable rather than both flipping together.

## Design

`roll_boundary.py --mirror`, `--models da3:small`, `--angles=-30,-20,0,20,30`,
on **the same eight recordings H33 used**, same frame count. H33's un-mirrored
`da3:small` numbers are the paired control — same runner, same angles, same
frames, same recordings — so this is a within-recording comparison, not a
re-measurement.

`da3:small` because H33 found it has by far the largest asymmetry
(**+22.84 ± 25.58** points, 7/8) and therefore the most signal to flip.

## Bars, locked before running

* **B1 — does it flip?** The mirrored difference `(+30°) − (−30°)` has the
  **opposite sign** to the un-mirrored one on a **majority** of the eight
  recordings, and its mean is negative.
* **B2 — magnitude preserved?** |mean mirrored difference| is within a factor of
  two of the un-mirrored **+22.84**. If the asymmetry flips but shrinks to
  nothing, the mirror has destroyed the effect rather than reflected it, and
  neither reading is supported.
* **B3 — sanity.** The level (0°) error is close between mirrored and
  un-mirrored. A large gap would mean mirroring alone damages the input, and the
  comparison would be confounded.

## What each outcome means

* **Flips (B1 passes)** → the asymmetry travels with the scene, not with our
  geometry. H32's finding survives as a statement about content-plus-model, and
  the "it might be our rendering" caveat in both reports can be dropped.
* **Does not flip** → it is fixed in image space, which points at our own
  principal-point offset or upright convention. Then H32's asymmetry is largely
  **our artefact** and should be withdrawn from both reports rather than
  narrowed.
* **B3 fails** → the test is void and says nothing either way.

## Cost

One backbone, 5 angles, 8 recordings, 20 frames. Minutes.
