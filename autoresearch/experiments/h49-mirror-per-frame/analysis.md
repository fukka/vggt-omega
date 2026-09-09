# H49 — the Aria mirror effect is carried by nearly every frame

**EXPLORATORY.** Post-hoc analysis of per-frame data committed by H45, H46 and
H47. No bars were locked before looking; it is labelled that way and no claim
here is presented as confirmatory.

| set | backbone | frames | worse | >1.5× | median | p10 | p90 |
|---|---|---|---|---|---|---|---|
| **Aria 60°, border-free** | `da3:small` | 113 | **96%** | 88% | 2.91 | 1.31 | 5.44 |
| | `da3:large` | 113 | **98%** | 90% | 3.63 | 1.52 | 8.19 |
| | `vggt` | 113 | 86% | 45% | 1.43 | 0.96 | 2.29 |
| | `vggt_omega` | 113 | 73% | 24% | 1.13 | 0.76 | 2.33 |
| **Aria 89°** | `da3:small` | 232 | **99%** | 94% | 3.14 | 1.69 | 6.25 |
| | `da3:large` | 232 | **100%** | 95% | 3.67 | 1.83 | 7.98 |
| | `vggt` | 232 | 92% | 54% | 1.59 | 1.05 | 2.49 |
| | `vggt_omega` | 232 | 89% | 50% | 1.50 | 1.00 | 2.28 |
| **ScanNet++ rectified 89°** | `da3:small` | 260 | 53% | 16% | 1.05 | 0.79 | 1.63 |
| | `da3:large` | 260 | 56% | 3% | 1.03 | 0.78 | 1.25 |
| | `vggt` | 260 | 30% | 0% | 0.94 | 0.61 | 1.12 |
| | `vggt_omega` | 260 | 57% | 0% | 1.01 | 0.93 | 1.10 |

## What it rules out

**The subset form of the content explanation.** If particular scene
configurations carried the effect — a hand in frame, one furniture layout, a
close surface — the distribution would be tail-driven or bimodal. It is neither:
on Aria, **DA3-Large degrades on 232 of 232 frames at 89°**, and DA3-Small's
*least affected tenth* still degrades by **31–69%**. The floor of the
distribution is above 1, not just its mean.

On ScanNet++ the same models over 260 frames are at a coin flip (30–57% worse,
median ratio ≈ 1.0, and **0–16% of frames past 1.5×**).

## What it does not rule out

**A pervasive property of egocentric imagery.** "Content" can also mean
something true of every Aria frame and no ScanNet++ frame — everything is close,
the floor sits in a fixed place, the wearer's body occupies the same region.
That form of the content explanation predicts exactly this shape. **H49 narrows
the question; it does not decide it, and H48 is still the experiment that
would.**

## The correlation nobody asked for

`corr(baseline error, mirror ratio)` is **negative everywhere** — −0.17 to −0.43
on Aria, −0.25 to −0.82 on ScanNet++. Frames the model finds *easy* degrade
proportionally *more*.

Consistent with a roughly additive corruption: a fixed amount of added error is
a larger ratio on a frame that started accurate. Offered as consistency, not
mechanism — and it is one more reason the ratio of means is the citable
statistic, since per-frame ratios are largest exactly where the denominator is
smallest.

## Cost

None. Arithmetic on committed JSONs, on the Mac, while `lambda_63` was
unreachable.

---

## Addendum — is the effect additive or multiplicative? It does not separate.

H49's negative `corr(baseline, ratio)` suggested a roughly additive corruption,
so the obvious next step was to fit both forms per backbone and compare
residuals: `mirror = normal + c` against `mirror = k · normal`.

**It does not separate.** Across all twelve (dataset, backbone) cells the two
RMSEs are within **10%** of each other, and three cells tie outright:

| set | backbone | additive c | rmse | mult k | rmse | affine slope, intercept |
|---|---|---|---|---|---|---|
| Aria 60° | `da3:small` | 0.185 | 0.133 | 2.14 | 0.140 | 1.59, +0.114 |
| | `da3:large` | 0.134 | 0.121 | 2.59 | 0.141 | 1.44, +0.111 |
| | `vggt` | 0.033 | 0.043 | 1.28 | 0.048 | **0.92, +0.039** |
| | `vggt_omega` | 0.017 | 0.046 | 1.18 | 0.048 | **0.98, +0.018** |
| Aria 89° | `da3:small` | 0.283 | 0.185 | 2.84 | 0.187 | 1.96, +0.156 |
| | `da3:large` | 0.169 | 0.132 | 3.18 | 0.150 | 1.56, +0.137 |
| ScanNet++ | `da3:small` | 0.024 | 0.072 | 1.08 | 0.074 | 0.75, +0.078 |
| | `vggt` | −0.028 | 0.072 | 0.71 | 0.056 | 0.05, +0.132 (degenerate) |

**Recorded as a null so nobody spends a tick re-running it.** A residual
difference under 10% between two one-parameter fits on 113–260 points, with no
held-out split, distinguishes nothing.

**The one thing worth noticing, and not building on:** on Aria the two VGGT
variants fit an affine with **slope ≈ 1 and a small positive intercept**
(0.92/+0.039 and 0.98/+0.018) — the shape of a nearly pure offset — while the
DA3 variants have slope 1.4–2.0 *and* an intercept near 0.11–0.16. If that held
up it would say the reflection costs the roll-robust models a fixed amount and
the others a fixed amount *plus* a scaling. It is one fit per cell with no
cross-validation, so **it is written down as an observation and nothing here
depends on it.**
