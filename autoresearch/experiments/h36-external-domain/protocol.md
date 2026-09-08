# H36 — does the rim penalty exist outside Aria at all?

## The exposure this addresses

Every result in this line comes from **one camera family (Aria), one apartment,
one small office**. The opening claim — "the rim is about 2.2× worse than the
centre, in every model" — has been verified across thirteen *recordings* and
four *backbones*, but never across a **different fisheye domain**.

The cross-room gap is blocked on a data decision. **This one is not**:
`SynWoodScape` has been sitting extracted on `/netapp` at 102 GB — synthetic
automotive surround-view fisheye, **outdoor**, ~190° lens, four cameras, with
depth ground truth and calibration. It is about as far from an indoor Aria
recording as a fisheye dataset gets.

The `depthfisheye/` package in this repo already has a loader, a
`PolynomialFisheye` camera and a checked depth convention for it. **Used
read-only** — that package belongs to a different work stream and is not
modified here.

## The judgement call, stated up front

Aria's cone is **54.83°**; SynWoodScape's is roughly **95°** half-angle. The
Aria zone thresholds (rim ≥38°, centre ≤11°) are **absolute angles** and mean
completely different things on a 190° lens — 38° there is barely off-axis.

So zones are defined as **fractions of `theta_max`**, taken from the Aria
definition: **rim = outer 30.7% of the cone** (38/54.83), **centre = inner
20.1%** (11/54.83). That keeps "the outer ring versus the middle" comparable
rather than keeping a number comparable.

The Aria zones also carry a **depth band** (≤2 m), which encodes "the wearer's
hands". There is no automotive equivalent, so the primary statistic is computed
**over all valid depths**, with no band. Stated because it is a choice that
moves the answer.

## Bars, locked before running

* **B1 — does the rim penalty exist here at all?** rim ÷ centre AbsRel is
  **> 1.5**. **Falsified at ≤ 1.2**, which would mean the penalty is a property
  of Aria-like indoor fisheye and the report's opening claim needs a domain
  qualifier it currently does not have.
* **B2 — is it quantitatively similar?** The ratio falls inside the Aria range
  H28 measured, **1.4–2.7**. Failing this while passing B1 means the effect is
  real but domain-dependent in size.
* **B3 — sanity.** Whole-image AbsRel is under **0.6**. Above that the frozen
  model is not producing usable depth on this domain and the ratio means
  nothing.

## What each outcome means

* **B1 and B2 pass** → the line's opening claim generalises past its own
  dataset, which nothing in H25–H35 could establish because they all re-measured
  *within* Aria.
* **B1 fails** → the rim penalty is Aria-specific and every headline in both
  reports needs "on indoor head-worn fisheye" attached.

## Cost

One frozen model, a few hundred images, no training. Data already on disk.
