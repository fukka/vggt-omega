# H33 — the test did not separate the two explanations. The claim narrows.

Four backbones, identical rendering, upright convention and scene content, eight
recordings. Primary statistic per the amendment: `(+30°) − (−30°)` in points.

| | mean difference | sd | +30° costlier on |
|---|---|---|---|
| `da3:small` | **+22.84** | 25.58 | 7/8 |
| `da3:large` | +4.60 | 20.13 | **4/8** |
| `vggt` | +8.38 | 5.39 | **8/8** |
| `vggt_omega` | +3.67 | 5.16 | 7/8 |

## B1 fails: the family split is not there

Single-image pair **+13.72 ± 24.15**, multi-view pair **+6.03 ± 5.65**. They
differ by 7.69 points against a within-family spread of 24.15 — the between-
family gap is smaller than the scatter inside one family. **B1 fails.**

**B2 passes**: `da3:small` reproduces the asymmetry, 7/8, consistent with H32's
11/13.

**The falsification did not trigger either.** All four do not agree within
noise: the magnitudes span 6× (+22.8 against +3.7) and `da3:large` shows no
consistent direction at all (4/8, a coin flip).

## So the experiment came out undecided, and that is the finding

I designed a clean either/or — model property or setup artefact — and the data
gave neither.

**What points at the setup:** all four backbones have a *positive* mean
difference. A purely model-specific effect would be expected to vanish or invert
somewhere, and it does not.

**What points against a pure artefact:** the magnitudes differ 6× across models
that see literally identical pixels, and `da3:large` is at chance while `vggt` is
8/8. A rendering asymmetry should hit all four equally.

**The defensible reading** is that both are present: a small common asymmetry of
roughly **+4 to +8 points** that every model shows and that the setup could
account for, plus a **much larger and much noisier DA3-Small-specific component**
(+22.8 ± 25.6). This experiment cannot separate them, and no third design is
offered here — that would need the upright convention itself varied, which is
invasive and was not part of this run.

## What this does to H32's published claim

H32 reported "**+30° costs 2.3× what −30° costs**" as a property of the roll
penalty. **That measurement was DA3-Small only** — and DA3-Small turns out to be
the model with the largest and by far the noisiest asymmetry of the four.

The claim is **narrowed, not withdrawn**:

* it holds for **DA3-Small** (7/8 here, 11/13 in H32);
* **DA3-Large shows no consistent asymmetry** (4/8), so it is not a property of
  the DA3 family;
* all four have a small positive mean, part of which may come from our own
  rendering, and this run cannot say how much.

Anyone quoting the 2.3× should say "on DA3-Small" and should not present it as a
fact about depth models.

## A caution recorded for the next person

The original protocol asked for the +30/−30 **ratio**. On the multi-view models
the −30° baseline is 0.6–3.7% and in one cell −0.2%, so ratios explode and one
was already `nan`. Quoted uncritically that would have read "VGGT is 5×
asymmetric" — a headline generated entirely by dividing by noise. The amendment
that replaced it with the difference is recorded in `protocol.md` as a change
made **after** seeing partial data, with well-definedness as the stated reason.
