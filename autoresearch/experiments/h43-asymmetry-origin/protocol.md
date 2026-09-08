# H43 — is the roll asymmetry an artefact of the origin?

## The one thing in this line still open, and four points that disagree

H32 found it: at ±30°, −30° costs +24.4% and +30° costs +56.5% on DA3-Small,
**2.3× more expensive on the positive side**, with +30° costlier on 11/13.
H33 could not attribute it (all four backbones positive-mean, but DA3-Large at
chance and magnitudes differing 6×). H34 was void trying. H42 added a fourth
point that does not line up: against a **gravity-aligned** origin `vggt`
*inverts*, 1.093 at −22° against 1.023 at +22°, where H33 had it 8/8 the other
way.

Every one of those measurements shares one thing: **its zero is the
device-aligned render.**

## The arithmetic that might explain all of it

Rendering at `roll_deg = a` leaves residual roll ψ − a. So a device-origin
sweep measures

    cost(+a) − cost(−a) = g(ψ − a) − g(ψ + a)

where g is the residual-roll curve H41/H42 measured. **If ψ has a negative
signed median, this is positive even when g is perfectly symmetric** — because
|ψ − a| > |ψ + a| for a > 0, so the "+a" arm is sampled further out on a convex
curve than the "−a" arm.

And H17.1 measured exactly that: the signed median roll is **negative**
(−2.70° pooled; per-sequence +0.44° to −7.50°). H32's own §03u note recorded
this as "a coincidence worth writing down but not over-reading" — the wearer's
typical tilt landing near the curve's cheapest angle — and H41 already showed
that "coincidence" is an identity about coordinates.

**So the published asymmetry may be the same identity, one step further on.**

## Method — no GPU, no new data

Purely arithmetic on artefacts already committed:

* g per recording per backbone: `h42-curve-four-backbones/results/` (and
  `h41-residual-curve/results/` for `da3:small`), 15 residual angles.
* ψ per frame: the same JSONs carry `roll_deg`.

For each recording *r* and backbone *m*, with μ_r the signed median ψ:

    Δ_grav(m, r) = g(+22) − g(−22)                  measured about the true zero
    Δ_dev(m, r)  = g(μ_r − 22) − g(μ_r + 22)        what a device-origin sweep sees

Δ_dev needs g outside ±22°, so it is computed **twice** — linear extrapolation
from the outer two points, and a quadratic fit to the outer three — and both are
reported. Any conclusion that depends on which is used is not a conclusion.

## Bars, locked before running

* **B1 — does the offset manufacture asymmetry?** Δ_dev > Δ_grav on a majority
  of the 4 × 13 cells, under **both** extrapolations. **Falsified** otherwise:
  the offset would then not be inflating anything and the published asymmetry
  is about the model.
* **B2 — the sign flip.** For `vggt`, Δ_grav is negative (H42 measured it) but
  Δ_dev is **positive** on a majority of recordings, under both extrapolations.
  That is the specific contradiction H42 raised, and this is the only
  explanation on the table. **Falsified** if Δ_dev stays negative.
* **B3 — does it scale with the offset?** Across all cells, Δ_dev − Δ_grav
  correlates negatively with μ_r (more negative median roll → more manufactured
  asymmetry), Spearman ≤ −0.4. **Falsified above −0.2**, which would mean the
  inflation is not coming from the offset at all.

## What each outcome means

* **All three pass** → the asymmetry that has cost three experiments is
  substantially an artefact of measuring against a device-aligned zero on
  footage whose median tilt is not zero. That does not make it entirely
  artefact — Δ_grav is not zero — but it means **no published asymmetry number
  in this line survives as a statement about models**, and the residual
  asymmetry is the small one H42 measured.
* **B1 or B3 fails** → the offset is not the mechanism and the asymmetry stays
  where H33 left it: undecided, with nothing leaning on it.
* **B2 fails while B1 passes** → the offset inflates but does not flip, so
  H42's contradiction with H33 stands unexplained and must be reported as such.

## What this cannot do

It cannot *prove* the published numbers were artefacts, because it predicts them
from a curve measured on the same footage rather than re-running H32's sweep
against a gravity origin. That re-run is the confirmatory version and is one GPU
pass; it is written down here as the follow-up rather than folded in, so this
stays what it is — **an arithmetic prediction, made before looking.**
