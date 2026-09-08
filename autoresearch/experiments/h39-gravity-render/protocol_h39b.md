# H39b — splitting H39's two gravity arms into resampling and roll

**EXPLORATORY. Labelled that way because the two pooled numbers this
decomposition acts on were already seen** when H39's analysis ran. Nothing here
is a confirmatory test of those; the per-frame dose checks (P2, P3) are the only
parts that could still fail, and they are stated before being computed.

## The idea

H39 rendered both rotation signs. That was done to identify the convention, but
it also hands over a decomposition for free, because the two arms differ in
exactly one thing:

    grav_p = S − A          A = roll actually removed
    grav_m = S + A          S = everything that is the same for both signs

Rotating the sampling grid by +ψ and by −ψ resamples the same amount and
removes opposite amounts of roll. So the **symmetric part S** is the price of
the operation — resampling, interpolation, a different set of source pixels —
and the **antisymmetric part A** is the roll it buys back. The operation pays
exactly when **A > S**.

This is H38's `null`-arm decomposition, obtained without a `null` arm.

## Why it matters more than a tidier restatement

H38 measured the price of de-rotation *with a border*: +58.1% for DA3-Small,
+6.8% for VGGT-Omega. H39's renders have **no border** — the fisheye cone
covers the rotated view, which is the entire reason the recommendation is
"fold it into the warp". So S is a direct measurement of **what the resampling
alone costs**, and comparing S against H38's price says how much of that price
was the border rather than the resampling.

If S turns out to be a large fraction of H38's price, then the H38→H39
recommendation is weaker than stated: the win would be mostly about the sign of
the rotation, not about avoiding the border.

## What is predicted, before computing

* **P1 — S ≪ H38's price.** The resampling alone should cost a small fraction
  of what H38 charged for the same rotation *plus a border*. Falsified if S
  exceeds half of H38's price for either backbone, which would mean the border
  was not the dominant cost and §03x's headline is overstated.
* **P2 — S grows with |ψ|.** A bigger rotation resamples through a bigger
  angle. Per-frame, S on frames with |ψ| ≥ 8° exceeds S on frames with
  |ψ| < 3°. Falsified otherwise, in which case S is not resampling and the
  whole reading is wrong.
* **P3 — A grows with |ψ|, faster than S does.** That is what makes the
  operation pay on the frames that need it. Falsified if A's slope is not
  steeper than S's.

## Method

Pure re-analysis of `autoresearch/experiments/h39-gravity-render/results/`.
No GPU, no new data. Per recording and per frame:

    S = (grav_p + grav_m) / 2
    A = (grav_m − grav_p) / 2

both as percentages of that frame's `device` error.
