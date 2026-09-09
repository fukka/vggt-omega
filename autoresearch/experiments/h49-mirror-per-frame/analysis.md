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
