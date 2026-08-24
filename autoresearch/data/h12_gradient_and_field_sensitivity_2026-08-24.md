# H12 post-mortem, Mac side (2026-08-24)

Two analyses the GPU delivery did not run, both on the delivered artefacts.
Source: `results` da38331 (`eval_{jac,shuffled,theta}_seq{136,132}.json`) and
`autoresearch/experiments/h12-lens-jacobian/code/jacobian.py` (pure numpy, runs
on this Mac: 10 tests, 0.86 s).

**Neither analysis reverses the verdict.** The pre-registered criterion was
"jac must beat shuffled on near-rim" and it did not, on either sequence. H12
stays stopped. What follows explains *why* it lost, and the explanation is
usable.

## 1. The advantage of the real field is monotone in eccentricity, and reverses

Per joint cell (8 theta rings x 5 depth bins, cells with >=500 px), the
difference `jac - control` in AbsRel (negative = real geometry better):

| | vs shuffled | vs theta-only |
|---|---|---|
| seq136, count-weighted corr(diff, theta) | **+0.240** | +0.230 |
| seq136, mean diff theta<30 / theta>=38 | −0.0010 / **+0.0040** | +0.0015 / +0.0037 |
| seq132, count-weighted corr(diff, theta) | **+0.656** | +0.646 |
| seq132, mean diff theta<30 / theta>=38 | **−0.0200** / **+0.0045** | −0.0143 / +0.0044 |

On seq132's nearest depth column the ordering across the 8 theta rings is
**perfectly monotone** (7/7 increasing steps):

    -0.092  -0.071  -0.048  -0.034  -0.013  -0.000  +0.013  +0.019
     3.4deg  10.3    17.1    24.0    30.8    37.7    44.5    51.4

So the real Jacobian field **helps most on axis, helps less the further out you
go, and actively hurts at the rim** — same sign on both held-out sequences and
against BOTH controls. Beating `theta-only` at the centre matters: the centre
gain is the Jacobian *content*, not merely "some smooth radial field".

This sharpens the GPU's own unregistered observation ("it protects the centre")
from a two-number comparison into a shape.

## 2. The field is far less determined at the rim than in the mid-field

`log_aniso` under a +-1% perturbation of each KB4 coefficient in turn
(Aria calibration of record, k = 0.3852, -0.4442, 0.5591, -0.3254):

| perturbed | swing at 30deg | at 45deg | at 54.83deg |
|---|---|---|---|
| k1 | 0.00164 | 0.00260 | 0.00632 |
| k2 | 0.00107 | 0.00306 | 0.01329 |
| k3 | 0.00056 | 0.00286 | **0.02304** |
| k4 | 0.00012 | 0.00109 | 0.01627 |

Against field values of +0.069 at 30deg and −0.191 at 54.83deg, worst-case
single-coefficient relative swing is ~2% at 30deg and ~12% at the rim; summed
over the four coefficients, ~5% vs ~31%. **The quantity the network is being
handed becomes an order of magnitude less determined exactly where the method
needed it to work.**

Caveat, stated plainly: 1% is an arbitrary probe, not Aria's published
coefficient uncertainty, so this is a conditioning-style sensitivity and not a
calibrated error bar. It does not establish that the rim turnover is a fit
artefact — that question (flagged in the H12 README) stays open. It does
establish that rim sensitivity is 10-40x mid-field sensitivity.

## 3. Reading

The two results line up: conditioning pays where the field is well-determined
and costs where it is not, and the crossover sits near the same 45-50deg band
where `log_aniso` turns over. A confidently-wrong input is worse than a
scrambled one, which is what the `shuffled` control measured.

Consequence for the line, not a rescue: **geometric conditioning is a centre
tool on this lens.** Any future use of it should be scoped to the well-determined
band and must not be sold as a rim fix.

## Gap to close (cheap, GPU)

`eval_cond.py` writes no `per_frame` key, so no bootstrap is possible on H12 —
unlike #35/#36, whose per-frame values gave the other two kills real error bars.
The checkpoints exist; a re-emit is minutes. Worth it because the centre effect
(+0.31% vs +16.73% on seq132) is the largest margin in the whole experiment and
is currently a single unbarred number.
