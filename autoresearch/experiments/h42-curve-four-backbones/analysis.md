# H42 — the family split is a property of the whole curve, and two published claims move

**Status: B1, B2 and B3 all PASS.** Thirteen recordings, 15 residual angles,
each backbone normalised by its own level error.

| residual | `da3:small` | `da3:large` | `vggt` | `vggt_omega` |
|---|---|---|---|---|
| −22° | 1.3068 | 1.2259 | 1.0928 | 1.0605 |
| −15° | 1.1577 | 1.0859 | 1.0523 | 1.0299 |
| −11° | 1.0859 | 1.0775 | 1.0356 | 1.0292 |
| −8° | 1.0499 | 1.0408 | 1.0355 | 1.0136 |
| −4° | 1.0147 | 1.0381 | 1.0209 | 1.0091 |
| **0°** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| +4° | 1.0341 | 1.0227 | 1.0109 | 1.0058 |
| +8° | 1.0642 | 1.0518 | 1.0084 | 1.0220 |
| +11° | 1.0987 | 1.0887 | 1.0140 | 1.0281 |
| +15° | 1.1883 | 1.1391 | 1.0118 | 1.0392 |
| +22° | 1.3538 | 1.3035 | 1.0229 | 1.0518 |

## B1 — the pretraining-family split survives as a whole curve

At ±22° the two families do not overlap: **every DA3 value is ≥ 1.226 and every
VGGT value is ≤ 1.093.** H17.2's claim — *roll sensitivity is a property of the
pretraining data, not of depth estimation* — rested on one number per backbone
on seq136, the recording H25–H31 showed amplifies every intervention. It now
holds as a statement about the entire curve, on thirteen recordings, against a
per-frame gravity-aligned origin.

## B2 — VGGT-Omega's own curve predicts its own null

H39 found nothing recoverable on that backbone and B2 there refused to call the
−1.04% a result. From its curve alone, at ψ = 11°:

| | predicted | H39b measured |
|---|---|---|
| prize A | **+2.52%** | +0.56% |
| price S | **−0.21%** | −0.48% |

Both inside the locked windows (|A| < 5%, |S| < 3%). **A flat curve is why
nothing was recoverable**, and that is now the second independent closing of
this loop — H41 did it for DA3-Small with a steep curve, H42 does it for
VGGT-Omega with a flat one.

*Stated with the honesty the magnitudes require*: these are differences of
order 1–3% against a recording-to-recording sd of several percent. Agreement in
sign and magnitude is all that is claimed.

## B3 — the origin correction is about the reference, not about DA3-Small

Every backbone's curve takes its minimum at residual 0 (VGGT-Omega at −2°,
inside the grid spacing). So H41's explanation of §03u's dip — that the device
zero sits a median 4° off level — is a property of **the reference**, and
nothing about DA3-Small in particular.

## Two published claims move

**1. "Capacity does not substitute — DA3-Large is MORE roll-sensitive" does not
survive.** H17.2 measured +52% for DA3-Large against +46% for DA3-Small at 30°,
on seq136. Here, on thirteen recordings against a gravity-aligned origin,
**DA3-Large is flatter on both sides** (1.226/1.304 against 1.307/1.354). H35
had already found their integrated expected costs identical (1.85% against
1.80%).

Honest summary across all three measurements: **within the DA3 family, capacity
makes little difference to roll robustness in either direction.** The
"more sensitive" half is withdrawn; the part that matters — that *changing the
pretraining family* buys 4× — is untouched and is now stronger.

**2. The asymmetry gets murkier, not clearer.** Three of four backbones are
costlier at positive residual, matching H32 and H33. **`vggt` inverts**:
g(−22°) = 1.093 against g(+22°) = 1.023. H33 had `vggt` at 8/8 with +30°
costlier by +8.38 points.

Different angle, different origin, different construction — so this is **not** a
contradiction that can be adjudicated here, and no mechanism is offered. H33
left the asymmetry undecided and H34 was void trying to attribute it; **this
adds a fourth data point that does not line up, and the honest conclusion is
that the asymmetry remains unresolved and is now less clean than when H32 found
it.**

## Limits

* ±22° is the outer edge; the frames reach 28°.
* 20 frames per recording, 15–20 surviving the timestamp join.
* The curves are normalised per backbone, so they compare **shape**, not
  accuracy. VGGT-Omega's level error is ~3× smaller than DA3-Small's in absolute
  terms; that is measured elsewhere and is not visible here by construction.
