# H28 — the foundation holds. Two published numbers do not.

Two runners unchanged on the same thirteen recordings, plus seq136.

## The asymmetry that is the main result

H25–H27 found seq136 inflated **every method** by ×1.33 to ×2.36. H28 finds it
was **ordinary for the phenomenon**:

| | thirteen recordings | seq136 | |
|---|---|---|---|
| rim ÷ centre, DA3-Small | 2.171 ± 0.382 | 2.13 | **−0.11 σ** |
| rim ÷ centre, DA3-Large | 2.127 ± 0.476 | 2.04 | −0.19 σ |
| raw rim error, DA3-Small | 0.396 ± 0.056 | 0.389 | ordinary |
| roll penalty ±20° | +12.4% ± 6.7 | +12.8% | +0.07 σ |
| roll penalty ±30° | +40.4% ± 11.1 | +46.3% | +0.52 σ |
| roll penalty ±40° | +91.7% ± 16.8 | **+130.6%** | **+2.32 σ** |

**That recording flattered what we built, not what we measured.** Worth stating
plainly: it is a different kind of luck than the correction arc found, and it
means the opening sections of both reports were resting on sounder ground than
the distillation sections were.

## Bars

**B2 passes cleanly** (−0.11 σ). **B3 passes on both halves** at ±30°:
sd/mean = 0.275 against a bar of ⅓, and seq136 at +0.52 σ.

**B1 is marginal**: sd/mean = 0.176 for DA3-Small against a pass bar of 0.15 and
a falsification bar of 0.25 (DA3-Large: 0.224). Neither outcome fires, and it
should be reported as undecided rather than rounded either way.

## Two numbers that must be restated

**"The rim is 2.0–2.6× worse than the centre, in every model."** The interval was
never a spread across models — it was four numbers, each from *one* recording,
that happened to land close together. Across thirteen recordings DA3-Small runs
**1.39–2.65** and DA3-Large **1.14–2.98**. The *mean* survives intact (2.17 and
2.13 against the published 2.13 and 2.04); the tight interval does not.

Honest form: **about 2.2× on average, with recording-to-recording spread from
roughly 1.4 to 2.7.**

**"+131% at ±40°."** The representative value is **+92% ± 17**. seq136 is 2.3 σ
high here specifically — the only place in the orientation section where it is
not representative.

## What is unaffected

The qualitative conclusions all survive, and two get firmer:

* the rim really is about twice as bad as the centre, on every recording tried;
* the roll penalty grows steeply with angle — +12% → +40% → +92% is still a
  steep curve, and the DA3-vs-VGGT gap that H17 turned on was measured as a
  *ratio between models on the same frames*, so a shift in the common baseline
  does not touch it;
* **±20° remains cheap**, +12.4% on average, which is what H17's "98.5% of real
  frames sit inside the model's comfortable range" depends on.

One caveat that the spread does add: at ±20° the sd is 6.7 on a mean of 12.4
(sd/mean 0.54), and the range reaches 21.3%. So "±20° is nearly free" is true on
average and not on every recording.

## A gotcha worth keeping

The roll arm failed on all fourteen sequences at the first attempt with
`argument --angles: expected one argument`. A value beginning with `-` is read
by argparse as a flag. The runner's *default* `-40,-30,…` works because it never
passes through argparse; passing the same string explicitly does not. Use
`--angles=-40,…`, never a space. This will bite anyone scripting these runners.
