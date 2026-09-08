# H43 — the origin explains the contradiction, but not the whole asymmetry

**Status: B2 PASSES decisively. B1 passes marginally. B3 FAILS as locked.**

CPU only, on committed artefacts: H41's and H42's curves (15 residual angles,
13 recordings, 4 backbones) and their per-frame roll.

## The arithmetic

A device-origin sweep at ±a measures `g(ψ − a) − g(ψ + a)`. With a negative
signed median ψ that is positive **even for a perfectly symmetric g**, because
the "+a" arm is sampled further out on a convex curve. The signed median in
these 13 recordings is **−3.10°** (H17.1 reported −2.70° over all frames).

| backbone | Δ about the true zero | Δ about the device zero |
|---|---|---|
| `da3:small` | **+4.70%** | +12.49% / +13.52% |
| `da3:large` | **+7.77%** | +8.74% / +19.39% |
| `vggt` | **−6.99%** | **+9.29% / +8.69%** |
| `vggt_omega` | **−0.87%** | +2.92% / +6.35% |

(two numbers = linear / quadratic extrapolation past ±22°)

## B2 — the contradiction H42 raised is explained

H42 found `vggt` **inverting** against a gravity-aligned origin (1.093 at −22°
against 1.023 at +22°) where H33 had it **8/8 the other way** at ±30° against a
device origin. That looked like a straight contradiction.

It is not. The same curve gives **Δ_grav = −6.99%** and **Δ_dev = +9.29%**,
positive on 10/13 recordings (12/13 under the quadratic). **One curve, two
origins, opposite signs.** No property of the model changed; the zero did.

## B1 and B3 — the mechanism is real but does not carry the whole effect

**B1 passes marginally**: Δ_dev exceeds Δ_grav on 31/52 and 32/52 cells, a 60%
majority against a bar of "a majority". That bar was badly chosen: two
recordings have μ ≥ 0 (seq136 +0.02°, seq140 +1.04°), where the offset must
*deflate* rather than inflate, so a per-cell majority was never the right test.

**B3 fails as locked.** The correlation between μ and the inflation is
**−0.345** (linear) and **−0.441** (quadratic), against a bar of ≤ −0.4 under
*both*. Right direction, moderate size, short of the pre-registered strength.

So the honest reading is the middle one, and it is not the tidy one:

> **The device origin demonstrably flips the sign for `vggt`, which dissolves
> the specific contradiction. It does not cleanly account for the asymmetry
> across all cells. The published asymmetry is part origin artefact and part
> something else, and this experiment cannot say how much of each.**

## What must change in the reports regardless

Δ_grav is **not zero** for any backbone, so the origin is not the whole story —
but Δ_dev differs from Δ_grav for every backbone, in one case by a sign. That is
enough to retire the published numbers as statements about models:

* H32's "+30° costs 2.3× what −30° costs" is a device-origin number on footage
  with a −2.7° median tilt. Its direction is exactly what the offset
  manufactures.
* H33's per-backbone differences are the same measurement and inherit the same
  problem, including its "da3:large is at chance" — which is now doubly odd,
  since `da3:large` has the **largest** true-zero asymmetry of the four
  (+7.77%).

**No asymmetry number in this line should be quoted as a fact about a model.**
The one that is currently defensible is the small one measured about the true
zero, per backbone, in the table above — and even that is at ±22° on one lens.

## The confirmatory version, and why this is not it

This predicts the published numbers from a curve measured on the same footage.
It does not re-run H32's ±30° sweep against a gravity-aligned origin, which is
the direct test and is one GPU pass. That run was launched while this was being
written (deltas 0, ±26, ±30, four backbones, 13 recordings) — both to make the
comparison angle-matched with H32 and to **remove the extrapolation** that
weakened B1 and B3, since ±22° was the outer edge of the measured grid and every
Δ_dev above depends on guessing beyond it.

Recorded plainly: **B1 and B3 were evaluated on extrapolated values. B2 was
not** — `vggt`'s Δ_grav is measured, and its Δ_dev is positive under both
extrapolations and on 10 and 12 of 13 recordings, which is why it is the one
result quoted here.
