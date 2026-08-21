# H12 — Lens-Jacobian conditioning

**Status:** primitive built and verified CPU-side (10 tests, no torch, <2 s).
Pilot not yet run.

## Why, in one paragraph

Three rim-targeted interventions have now failed against their own controls:
H5's rim-weighted losses lose to plain LoRA (near-rim −80.6 % vs **−83.5 %** on
seq136, −33.3 % vs −33.5 % on decoration_seq132, and plain wins on pose too),
H6's rim-restricted KV loses to the all-token control (−52.2 % vs **−75.9 %**),
and the centre/rim dual-expert MoE was killed by H7+F2+F4. Meanwhile the
interventions that *do* work are global lens operations: `rect_derect` beats
native fisheye on slambench, and plain LoRA beats everything on ADT.

Read together, those say **the rim deficit is not a region-shaped problem.** It
is a symptom of a global lens-prior mismatch — the backbone's features were
learned on near-pinhole statistics, and a fisheye's local image formation
departs from that prior by an amount that grows with incidence angle. The
departure is loudest at the rim, which is why it *looks* like a rim problem. But
adding capacity where the symptom is loudest has now failed three times.

So: stop telling the network **where** to try harder, and hand it the geometry
it currently has to infer from content.

## The quantity

A radial lens maps a ray at incidence θ to image radius `r(θ) = f·d(θ)`. In the
local orthonormal frame on the sphere the map is diagonal:

    m_rad = r'(θ)                 m_tan = r(θ)/sin(θ)

Both → f as θ → 0, so normalising kills `f` and leaves a pure **shape**
descriptor:

    log_area  = log( d'(θ)·d(θ)/sin θ )
    log_aniso = log( d'(θ)·sin θ/d(θ) )

Both are exactly 0 on axis for every lens. `f` cancelling is the whole transfer
argument: the field encodes the **lens**, not the sensor, so a model conditioned
on it has some hope of working on a lens it never saw.

## The measurement that motivates it

Computed on Aria's calibration of record (`test_aria_field_is_NOT_monotone_and_the_anisotropy_changes_sign`):

| θ (deg) | 0 | 10 | 20 | 30 | 40 | 49.8 | 54.8 |
|---|---|---|---|---|---|---|---|
| `log_area` | 0 | 0.049 | 0.168 | 0.317 | 0.482 | **0.580** | 0.456 |
| `log_aniso` | 0 | 0.016 | 0.047 | 0.069 | 0.077 | **0.003** | **−0.191** |

**`log_area` peaks at 48.9° and then falls. `log_aniso` peaks near 40°, crosses
zero near 50°, and goes strongly negative at the rim.** Inside ~50° the lens
stretches radially more than tangentially; outside it, the other way round. The
"rim" band is different in **kind**, not merely in degree.

That is a concrete mechanism for why H5 failed: **a scalar rim weight is
monotone in θ by construction, so it cannot represent a quantity that turns over
and changes sign inside the very band it is weighting.** It is also why θ alone
is not a substitute for the field — `test_theta_alone_cannot_stand_in_for_the_jacobian`
shows two θ more than 10° apart sharing one `log_aniso` value.

## The pilot, and how it dies

One question, on ADT, before anything touches ScanNet++:

> Does Jacobian conditioning beat the **same architecture and parameter count**
> conditioned on a **shuffled** field?

Shuffled, not absent: a model that improves with a scrambled geometry field has
gained from extra capacity, not from geometry, and we would have re-run H5 with
more steps. If real ≤ shuffled, **stop** — do not proceed to ScanNet++, and
write it up as the fourth controlled negative, which would nail the reframing
above rather than weaken it.

## What is NOT established

* Whether the non-monotone `log_area`/`log_aniso` near the rim is **physical**
  or an artefact of the KB4 fit. Aria's turnover is at 62.33° and θ_max is
  54.83°, so `r(θ)` is monotone over the field — but `d'(θ)` near the rim is
  still a property of the fit. `docs/research/scannetpp-camera-reference.md`
  is the standing warning about reading fits as lenses.
* That conditioning helps at all. Nothing has been trained.
* Anything about transfer. That is the *reason* for the design, not a result.
