# H29 — the warning survives 13/13. Its numbers were the worst outliers yet.

Two runners unchanged on the same thirteen recordings, at `--angles 0`.

## The two bars that matter both pass, unanimously

**B2 passes 13 of 13.** DA3-Large's border cost exceeds DA3-Small's on **every
single recording**. The ordering — which is what the advice rests on — is not a
one-recording artefact.

**B3 passes 13 of 13.** A 70 px border, blacking out **39.5%** of the frame,
costs **less** than a 5 px border blacking out **3.1%**, on every recording. The
non-monotonic dip that is the entire evidence for *proximity, not area* is
completely robust.

| border | % of frame | thirteen recordings | seq136 |
|---|---|---|---|
| 5 px | 3.1% | +19.5% ± 13.0 | +22.0% |
| 12 px | 7.5% | +14.7% ± 10.2 | +13.2% |
| 25 px | 15.2% | +18.1% ± 11.8 | +17.1% |
| 45 px | 26.5% | +21.8% ± 15.6 | +16.8% |
| **70 px** | **39.5%** | **+7.3% ± 4.9** | +6.6% |
| 100 px | 53.4% | +35.6% ± 20.9 | +32.2% |

## B1 fails, and by the largest margin in this whole campaign

| | thirteen recordings | seq136 | |
|---|---|---|---|
| DA3-Small border cost | +59.0% ± 14.1 | **+105.7%** | **+3.32 σ** |
| DA3-Large border cost | +268.9% ± 61.5 | **+635.5%** | **+5.96 σ** |
| ratio, Large ÷ Small | **4.56×** | 7.4× | |

**+5.96 σ is the largest deviation found anywhere in H25–H29**, beating H25's
5.74 σ. Both published border figures are extreme outliers, and the headline
"7.4× collapse" is really **4.56×**.

## The refinement this forces on H28's asymmetry

H28 concluded that seq136 "flattered what we built, not what we measured" — it
was ordinary for the rim/centre ratio (−0.11 σ) and the roll penalties (+0.07,
+0.52 σ). But the border test is also a *measurement*, not a method, and here
seq136 is +3.3 and +6.0 σ.

The sharper statement, offered as an observation and **not** a mechanism — this
project has had two mechanism guesses refuted already:

> **seq136 responds unusually strongly to interventions.** Corrections applied to
> it help unusually much (H25–H27); perturbations applied to it hurt unusually
> much (H29). Its *unperturbed* structure is ordinary (H28).

That covers all five experiments without special pleading, and it predicts that
any future intervention measured on seq136 alone will read high.

## What changes and what does not

**The advice does not change at all.** "Never put a hard black border next to the
region you care about" and "cost is set by proximity, not area" are both 13/13.
If anything they are better supported than when published, because they now rest
on thirteen recordings rather than one.

**Three numbers change.** +106% → **+59% ± 14**, +636% → **+269% ± 62**, and the
7.4× collapse → **4.56×**. A 4.6× collapse from a change that touches no scored
pixel is still the strongest practical warning in either report.

**One phrase should go.** "DA3-Large is the best of the four on a clean frame and
the worst with a border" was checked against four backbones on one recording.
H29 only re-measured two of them. The claim that DA3-Large is worst *of the four*
is now supported for the two tested and untested for VGGT and VGGT-Omega.
