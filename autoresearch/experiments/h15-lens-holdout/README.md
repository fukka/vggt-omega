# H15 — lens conditioning, decided on a lens the model never saw

**Status:** code written, CPU tests written. Nothing trained. Pre-registration
below is written *before* the first run.

## Why H12's answer was not an answer

H12 pre-registered a clean criterion — the real lens-Jacobian field must beat a
**position-shuffled** copy of itself at equal capacity — and it fired: `jac`
was the worst of three arms on near-rim on both held-out sequences. The
criterion was honest and the run was clean.

The **setting** could not decide it. H12 trained and tested on **one lens**, and
on one lens the two arms are information-equivalent by construction:

> with a single lens the field is a fixed function of token position, so a
> fixed permutation of it is an equally learnable fixed function of token
> position.

Nothing in that setting separates "the network used the geometry" from "the
network memorised a per-position modulation". The margin H12 measured — ~5%
relative, one seed, no error bars — is what that degeneracy looks like from
outside.

And H12's claim was never about one lens. `log_area` and `log_aniso` are
normalised so the **focal length cancels**: the field describes the *lens*, not
the sensor, and the entire point of that was transfer to a lens the model never
saw. H12 stopped one step before the only experiment that could test it.

`test_the_real_field_is_consistent_across_lenses_and_a_shuffle_is_not` states
the argument as a measurement rather than as prose: across two lenses the real
fields correlate > 0.9 at matching token positions, per-lens shuffled ones
< 0.2. On one lens that difference is invisible.

## The other half of the post-mortem, and how this design removes it too

The Mac post-mortem of H12 (`data/h12_gradient_and_field_sensitivity_2026-08-24.md`)
found a second reason the field could lose that is not the identifiability
argument above: **`log_aniso` is 10-40x more sensitive to KB4 coefficient
perturbation at the rim than mid-field** (~31% vs ~5% relative swing). The
network was handed a quantity that is least determined exactly where the method
needed it, and a confidently-wrong input can be worse than a scrambled one.

That is a property of reading a *fitted* KB4 near the edge of the field it was
fitted on, and this family largely does not have it: `equidistant`,
`equisolid`, `stereographic`, `orthographic` and `rectilinear` have
**closed-form, exact** `d` and `d'`, so their fields carry no fit uncertainty
at all. **Both held-out lenses are analytic ones.** The three KB4 members
(`aria_kb4`, `kb4x0.5`, `kb4x1.5`) keep a fitted lens in the training mix,
which is realistic, but the decider does not rest on one.

So the two live explanations for H12's null -- single-lens
non-identifiability, and an ill-determined rim field -- are removed by the same
experiment, and the arms still differ only in what the conditioner is shown.

## The setting

Every frame is warped into a lens drawn from a family that all image **exactly
the same cone into exactly the same disc** — `f = R_disc / d(theta_max)`, so
the warp is a pure radial re-distribution of the same rays.

Three consequences that do the work:

* **No void.** Every destination pixel has a source ray. Warping to a wider
  lens would leave a hole (3.3% median / 21.6% worst on the ScanNet++ route);
  to a narrower one would discard the rim, the region under study.
* **Planar z is invariant** — same ray, same theta — so GT is resampled with
  `nearest` and **converted not at all**. Checked numerically, not asserted in
  a comment.
* **The augmentation is exactly the conditioned quantity.** `(log_area,
  log_aniso)` *is* the derivative of this re-distribution. Nothing else about
  the image changes.

| | lenses |
|---|---|
| train | `aria_kb4`, `equidistant`, `orthographic`, `rectilinear`, `kb4x0.5`, `kb4x1.5` |
| **held out** | `stereographic`, `equisolid` |

The held-out pair is **interpolative** in field space — their rim
`(log_area, log_aniso)` sit inside the training family's range, asserted by
`test_held_out_lenses_are_interpolative_in_field_space` — so the decider is
transfer, not extrapolation.

## Arms

| arm | what the conditioner is shown |
|---|---|
| `jac` | the real field of the lens the frame is currently in |
| `mismatched` | a real field of a **different** lens: smooth, plausible, wrong |
| `shuffled` | this lens's own field, **per-lens** position permutation |
| `none` | no conditioner — plain LoRA, the standing baseline |

Two controls, not one. `shuffled` is H12's and it is distribution-matched but
destroys smoothness, so a win over it alone could be "a smooth field is easier
to fit". `mismatched` closes that hole: equally smooth, equally structured,
and describing the wrong lens. The permutation is **per lens** because one
shared permutation preserves the value↔geometry correspondence up to a single
relabelling the network can simply learn — a control that is not a control.

## Pre-registered predictions

**P1 (the decider).** On the **held-out lenses**, `jac` beats **both**
`mismatched` and `shuffled` on `near_rim`, on both held-out sequences.

**P2 (the reconciliation, and the reason this is not just a retry).** On the
**training** lenses, `jac ≈ shuffled` — H12's null reproduced. If P1 holds and
P2 holds, H12's negative is explained rather than contradicted: the
conditioning is real and single-lens experiments cannot see it. **If P1 holds
but P2 fails**, something other than geometry differs between the arms and
nothing is claimed until it is found.

**P3 (does it buy anything at all).** `jac` beats `none` on the held-out lens.
Losing to plain LoRA would mean the conditioning is not worth its 25.5k
parameters even where it is identifiable — a fifth controlled negative, and a
sharper one than H12's because it would be measured in the setting the method
was designed for.

**P5 (the post-mortem's reading, tested rather than assumed).** The H12
post-mortem found `jac - control` monotone in theta and reversing sign at
38-45 deg: real geometry helped on axis and hurt at the rim, hence "geometric
conditioning is a CENTRE tool on this lens". If that survives lens diversity --
`jac` wins the centre on the held-out lens but still loses the rim -- then the
centre-tool reading is confirmed independently of the two explanations above,
and it is a finding rather than a consolation. If instead `jac`'s advantage
extends to the rim once the lens varies, the centre-only shape was an artefact
of the single-lens setting.

**P4 (centre, as always).** No arm may damage `near_center` more than `none`
does. Centre collateral killed H2.1 and is invisible in pooled zones, so the
full (θ × depth) joint table is read.

## What would make a win uninteresting

* **Resampling density.** Each lens resamples the rim differently. Every arm
  sees the **identical** warped images, so this cannot separate the arms — but
  it does mean absolute numbers across lenses are not comparable, and only
  within-lens, across-arm differences are read.
* **The native lens getting a free pass.** `aria_kb4` goes through the same
  resampler as every other lens (identity warp), so it carries the same
  interpolation blur. `test_the_identity_warp_is_the_identity` pins it.
* **One seed per arm.** As in H12. Direction plus pre-registration licenses a
  stop; a claim needs the paired frame bootstrap.

## Files

| file | what |
|---|---|
| `code/lens_family.py` | the lens shapes, the fixed-cone construction, the warps, the per-lens token field |
| `code/arms.py` | what each arm is shown — shared by trainer and evaluator so they cannot diverge |
| `code/test_lens_family.py` | 20 CPU tests, no weights, no data |
| `code/train_multilens.py` | the four arms |
| `code/eval_lens.py` | every arm on every lens, held-out lens flagged |

Reused unchanged: `../h12-lens-jacobian/code/jacobian.py` (the field),
`film.py` (the conditioner), `../h5-rim-finetune/code/{losses,lora,train}.py`.

## The external replication, if this lands

SynWoodScape is already on `lambda_63` and has **four real cameras with four
different polynomial calibrations** — a real multi-lens dataset, not a
synthetic family, and the `depthfisheye/` reimplementation already loads it.
A held-out-camera split there is the natural next row, and it is also where
DepthFisheye's own camera embedding should be identifiable, which makes the two
experiments read each other.
