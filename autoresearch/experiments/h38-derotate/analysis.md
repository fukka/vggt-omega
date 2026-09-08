# H38 — you can read the roll. Do not rotate the picture.

**Status: B2 and B3 PASS. B1 FAILS, informatively.**

Six recordings, four backbones, three arms per angle. `null` is the
untransformed-baseline bar: the same two resamplings and the same black corners
as `derot`, with **no roll removed**.

## The table

Percentages of each backbone's own error on the rolled view. The two signs are
reported separately — H32's rule — and they differ.

| backbone | angle | roll costs | de-rotating vs raw | price alone (`null`) | vs a level camera |
|---|---|---|---|---|---|
| `vggt_omega` | +30° | +6.0% ± 5.8 | **−0.1% ± 3.6** (helps 3/6) | +6.8% | +6.0% |
| `vggt_omega` | −30° | +1.7% ± 4.5 | +3.6% ± 8.8 (2/6) | +8.0% | +5.3% |
| `vggt` | +30° | +12.3% ± 8.2 | +26.8% ± 30.0 (0/6) | +42.5% | +41.8% |
| `vggt` | −30° | +2.5% ± 1.8 | +39.0% ± 25.9 (0/6) | +39.3% | +42.2% |
| `da3:small` | +30° | +45.4% ± 20.1 | **−12.0% ± 10.7 (6/6)** | +58.1% | +26.3% |
| `da3:small` | −30° | +27.8% ± 11.2 | −1.9% ± 6.2 (2/6) | +70.1% | +25.3% |
| `da3:large` | +30° | +25.2% ± 17.1 | +31.9% ± 21.5 (1/6) | +268.6% | +63.9% |
| `da3:large` | −30° | +25.0% ± 19.6 | +35.9% ± 7.4 (0/6) | +201.0% | +69.8% |

## B3 — the border cost reproduces on a completely different border shape

The price of the operation, `null`/`raw` − 1, against H37's annulus border cost
measured on thirteen recordings with a different script:

| backbone | H38 corners | H37 annulus |
|---|---|---|
| `vggt_omega` | +7.4% | +6.0% |
| `vggt` | +40.9% | +48.4% |
| `da3:small` | +64.1% | +59.0% |
| `da3:large` | +234.8% | +268.9% |

Four small corner triangles cost **the same as a full annulus**. Same ordering,
same magnitudes, different shape, different area, different script, different
recordings.

This is the third independent route to "**cost is proximity, not area**". H17.5
got it from one border, H29's dose curve got it from sweeping the width, and
this gets it from changing the *shape* while holding proximity fixed. The dose
curve is the weakest of the three (non-monotone); this is the cleanest.

## B2 — de-rotation does not get you back to a level camera

For `da3:large`, de-rotating leaves it **+63.9% / +69.8%** above where it would
be with a level camera, against a falsification line of +5%. It is also worse
than doing nothing (+31.9% / +35.9% against `raw`).

Even `vggt_omega`, the most border-tolerant backbone in existence here, nets
**−0.1%** at +30° and **+3.6%** at −30°. Its prize (+6.0%) and its price
(+6.8%) are the same size. There is nothing to win.

## B1 fails, and the failure is the asymmetry again

De-rotation genuinely removes roll: for `da3:small` at +30° it helps on **6 of
6** recordings, −12.0% ± 10.7. At −30° it helps on only 2 of 6.

That is not inconsistent — it is H32's asymmetry showing up in a third place.
At +30° `da3:small` pays +45.4% for the roll and at −30° only +27.8%, so there
is less to recover on the negative side while the price is, if anything, higher
(+70.1% vs +58.1%). **The one cell in this whole experiment where de-rotation is
a real win is the single most roll-damaged (backbone, sign) pair.**

B1 asked for a majority at *both* signs and did not get it. Recorded as a fail
rather than softened, because the pre-registered claim was that the operation
works, and it works in one cell out of eight.

## What to actually do with the gravity vector

Not nothing — the opposite conclusion from "roll is unfixable".

`raw` at 0° in this experiment **is** the deployable operation: a view rendered
level directly from the fisheye, by folding the roll into the resampling grid.
It has no border, loses no content, and is exactly the level-camera reference
every de-rotated arm fails to reach. Any egocentric pipeline that already
rectifies a view out of the fisheye can fold in a known roll for free.

The rule that comes out of this:

> **Use the gravity vector in the resampling, not after it.** Rotating the
> sampling grid costs nothing; rotating the picture creates a border, and a
> border beside the scored region costs between +7% and +235%.

The caveat is real: this only applies if your pipeline resamples the fisheye at
all. Feed the raw fisheye frame straight to the model and there is no warp to
fold the roll into — and then the only available move, rotating the image, is a
net loss for three of the four backbones.

## Limits

* **Six recordings, not thirteen.** sd is reported for every cell. The B3
  ordering is stable enough that it reproduces H37's thirteen-recording
  numbers, but the individual gains are six-recording means.
* **A synthetic roll, not a real one.** The rig rolls the virtual camera; it
  does not replay the wearer's actual head roll frame by frame. H17.1 measured
  the real distribution and H35 integrated against it; this measures the
  mechanism at a fixed angle.
* **No mechanism for why `vggt` is hurt more than `da3:small` here** (+26.8%
  against −12.0% at +30°) despite a lower border price. It has less roll to
  recover (+12.3% vs +45.4%), which is arithmetic, not an explanation.
