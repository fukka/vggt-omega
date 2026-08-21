# Addendum, 2026-08-22 — the oracle null is not a single number

The radial artefact published earlier today quoted **one** null per dataset
(1.10× aea, 1.26× nymeria) and repeated the ticket's explanation that the
residual is "the coarseness of five distance strata, not a residual finding".

That explanation is **incomplete**, and the sweep in `strata/` shows why.

## 1. More strata helps on one dataset and plateaus on the other

Standardised spread of the oracle (`--oracle-noise 0.15`), by strata count:

| strata | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| aea | 1.126 | 1.098 | 1.098 | **1.103** |
| nymeria | 1.295 | 1.217 | 1.164 | **1.129** |

nymeria falls monotonically and is still falling at 10 — that part is strata
coarseness, as the ticket said. **aea plateaus at ~1.10 and stops falling**, so
on aea the residual is not coarseness. Refining the binning does not drive
either to 1.0.

## 2. The null grows with the size of the injected error

At fixed 8 strata, varying only `--oracle-noise` (metres):

| σ | 0.05 | 0.15 | 0.30 |
|---|---|---|---|
| aea | 1.068 | 1.098 | **1.169** |
| nymeria | 1.087 | 1.164 | **1.283** |

**The null is a function of the error magnitude**, so quoting one null for all
models is wrong. Mechanism: AbsRel is `|err|/GT`, the rim looks at nearer
surfaces, and a distance stratum is a band rather than a point — so within a
stratum the rim's GT is still slightly nearer, and a larger metric error
amplifies that residual. A model with big errors is entitled to a bigger null.

## 3. The matched-null read — and the conclusion survives

Each model's own implied metric error `σ = Σ AbsRel·GT·n / Σ n`, with the null
interpolated to that σ:

| dataset | model | σ (m) | spread | matched null | ratio |
|---|---|---|---|---|---|
| aea | vggt_omega | 0.204 | 1.835 | 1.123 | **1.63×** |
| aea | vggt_1b | 0.269 | 1.803 | 1.154 | **1.56×** |
| aea | da3_large | 0.286 | 1.589 | 1.162 | **1.37×** |
| aea | da3_small | 0.345 | 1.559 | 1.169 | **1.33×** |
| aea | dav2_large | 0.229 | 1.414 | 1.135 | **1.25×** |
| nymeria | vggt_1b | 0.243 | 2.072 | 1.237 | **1.67×** |
| nymeria | vggt_omega | 0.199 | 1.962 | 1.202 | **1.63×** |
| nymeria | da3_large | 0.282 | 1.681 | 1.268 | **1.33×** |
| nymeria | dav2_large | 0.259 | 1.568 | 1.250 | **1.25×** |
| nymeria | da3_small | 0.348 | 1.566 | 1.283 | **1.22×** |

**All five models still exceed their own matched null on both datasets.** The
headline stands — the rim degradation is real on real egocentric footage — but
the margins are smaller and honestly derived: 1.22–1.67×, not the 1.25–1.67×
implied by the single-null table, and `da3_small` on nymeria is the thinnest at
1.22×.

## What this changes for anyone using this harness

Do not quote a null without saying at what error magnitude it was measured, and
prefer a null matched to the model under test. The single-null table in the
original `ANALYSIS.txt` should be read through this addendum.

## Not claimed

* σ for `da3_small` (0.345/0.348) is past the swept range, so its null is
  **clamped** at the σ=0.30 value rather than interpolated. Its true matched
  null is slightly higher and its ratio slightly lower than shown.
* Why aea plateaus at 1.10 while nymeria keeps falling is not established.
* The σ estimate is a count-weighted mean over bins, not a per-pixel fit.
