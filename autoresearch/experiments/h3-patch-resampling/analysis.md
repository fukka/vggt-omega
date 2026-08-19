# H3 analysis — zero-parameter patch undistortion

## run_014 (2026-08-22, INVALID — implementation bug)

First implementation used an arbitrary tangent basis (up×d0), which rotates
every patch's content by its local azimuth — caught by the protocol's
mandated visual sanity check (the resampled frame shattered into pinwheels;
`to_human/assets/` example from the buggy run superseded). Numbers discarded.

## run_014b (2026-08-22, CONFIRMATORY — corrected Jacobian resampling)

Correct local linearization (projection differential mapped to pixel axes;
identity to first order at each patch center, right orientation and
anisotropy). Result: **the joint table is unchanged to the third decimal in
essentially every cell** (worst cell 0.350→0.356).

## Why — the mechanism, quantified

The corrected warp's own magnitude explains it: within a 14-px patch on Aria
KB4 at 504², the deviation of the exact fisheye map from its linearization is

| θ band | mean | p95 | max |
|---|---|---|---|
| 0–20° | 0.019 px | 0.051 px | 0.087 px |
| 20–40° | 0.024 px | 0.057 px | 0.086 px |
| 40–50° | 0.026 px | 0.060 px | 0.174 px |
| 50–54.8° | 0.036 px | 0.097 px | 0.209 px |

**Within-patch distortion on this lens is ≤ 0.2 px even at the rim.** There is
nothing for patch-content undistortion to fix; the distortion lives BETWEEN
patches — in token geometry and positional encoding — which is where
RayTun3R's ablation found its gains (PE residual ≫ patch resampling; our
measurement supplies the quantitative reason) and where our compression
diagnosis + feature head already operate.

**H3 verdict: REFUTED, with the mechanism.** Axis N2's patch-content branch
(including DarSwin-style content resampling at this granularity, for this
lens class) is closed: it attacks a ≤0.2 px term. Distortion-adaptive design
effort belongs at the inter-patch level. Caveat: this is a statement about
110° KB4 at patch 14/504²; wider lenses (170°+ DSLR, 200° TUM-VI) have larger
within-patch curvature and the number should be recomputed before reusing
the conclusion there.
