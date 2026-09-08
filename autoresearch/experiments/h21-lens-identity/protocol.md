# H21 — is the correction curve a property of the lens, or largely lens-independent?

## The claim being tested is one I already published

Both reports say the transferable part is "a recalibration of depth against
viewing angle — **a property of the lens**". H18.2 read that off a curve fitted
on one device transferring to another; H19 and H20 then showed the LiteOffice
failure was motion, not lens mismatch.

But **nobody has ever changed the lens with everything else held fixed.** Real
devices confound lens with room, scene, wearer and motion. So "property of the
lens" is an interpretation, not a measurement.

The H20 candidate said this needs a third device. It does not. H15's lens family
builds many lenses over **one fixed cone filling the same disc**, so a warp
between any two is a pure radial re-distribution of the *same rays*: no void, no
extrapolation, same content. That holds room, scene, wearer, motion, teacher and
backbone weights fixed and varies only the radial mapping — a cleaner contrast
than a third device could give.

## Why this costs no teacher inference

The cache stores `depth_convention: "range"`, and `grid_between` resamples by
**ray**. Range along a ray is invariant when the lens changes at a fixed cone.
So the teacher target under lens L is an exact `nearest` resample of the cached
target — never a re-derivation. Only the frozen prediction has to be recomputed,
because the network genuinely sees a differently warped image, which is the
whole point.

## Arms

Four lenses over the Aria cone (θ_max 54.83°): `aria_kb4`, `equidistant`,
`stereographic`, `equisolid`. Different `r(θ)`; same rays, same disc.

**Control detail that decides the reading:** every arm — including `aria_kb4` —
is warped once from the real Aria camera. If the aria arm alone were left
un-warped, resampling blur would be confounded with lens shape. All four carry
the same interpolation cost.

Fit on the four Apartment training sequences (240 frames, camera spread 1.77 m,
which H20 says is enough motion). Evaluate on seq136 and dec132, each rendered
through each lens. Full 4×4 transfer matrix, plus the 2-parameter `global`
control in every cell.

## Bars, locked before running

* **B1 — do the curves differ at all?** Mean pairwise |Δa(θ)| between the four
  fitted curves exceeds **0.069**, the draw-to-draw noise floor H20 measured on
  the high-motion arm. **Falsified if the four curves agree within that**, which
  would say the curve is not lens-specific.
* **B2 — does the difference matter?** Applying lens A's curve to lens B loses
  materially against B's own curve — more than the ±1.0 point draw-to-draw sd
  H20 measured. **Falsified if cross-lens ≈ same-lens.**
* **B3 — sanity, not a finding.** `radial` beats `global` on the diagonal in
  every lens. If it does not, something is wrong with the warped setting and
  B1/B2 mean nothing.

## The outcome that would matter most

**If B1 and B2 both fail, "a property of the lens" is wrong** and has to be
restated in both reports as something weaker and more interesting: a property of
how the backbone errs against incidence angle, largely independent of how the
lens distributes pixels across that angle. That would also make calibration a
genuine one-off rather than a per-camera step, which is a much better deployment
story than anything claimed so far.

Both outcomes change what the reports say. That is why it is worth running.

## Cost

~1,440 DA3-Small forwards at 504². No training, no teacher inference.
