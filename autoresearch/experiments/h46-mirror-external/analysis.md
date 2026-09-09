# H46 — the mirror effect does not replicate on ScanNet++

**Status: B1 PASSES (plumbing exact). B2 FAILS on both arms. B3 passes on
margins too small to mean anything.**

Thirteen ScanNet++ scenes, 20 frames each, four backbones, three arms, run two
ways: the captured frame as-is, and a rectified 89° co-axial pinhole view — the
same construction H45 used on Aria.

| backbone | baseline AbsRel | mirror cost | sd | range | **H45 (Aria, 89°)** |
|---|---|---|---|---|---|
| **raw captured frame** | | | | | |
| `da3:small` | 0.391 | +20.7% | 44.9 | −36 … +90 | **+254%** |
| `da3:large` | 0.265 | +27.0% | 34.5 | −16 … +110 | **+350%** |
| `vggt` | 0.289 | +0.0% | 32.2 | −52 … +63 | **+67%** |
| `vggt_omega` | 0.218 | +3.2% | 16.3 | −21 … +36 | **+58%** |
| **rectified 89° view** | | | | | |
| `da3:small` | 0.217 | +13.5% | 28.7 | −19 … +77 | **+254%** |
| `da3:large` | 0.150 | +3.3% | 11.6 | −27 … +17 | **+350%** |
| `vggt` | 0.168 | −9.3% | 20.7 | −65 … +13 | **+67%** |
| `vggt_omega` | 0.119 | +1.6% | 6.1 | −7 … +16 | **+58%** |

**The plumbing bar is exact on all 104 cells: 0.000000%.** Whatever this is, it
is not a broken flip.

**The rectified arm is the matched one**, and it was added *before* seeing more
than three scenes: the raw arm's baseline (0.22–0.39 AbsRel) is 2–4× worse than
H45's Aria baseline, because a raw ~115° fisheye is out of distribution for these
models, and a mirror cost measured on a broken baseline is not comparable. The
rectified arm brings the baseline to 0.12–0.22, in H45's range.

Under that matched construction the effect is **10–100× smaller** than on Aria
and, for two of the four backbones, indistinguishable from zero. B3 "passes"
only because +3.3% exceeds +1.6%; with sd's of 12 and 6 that is not an ordering.

## What this establishes and what it does not

It establishes that **H45's numbers are not a general property of these
backbones.** The same models, the same three arms, the same rectification, the
same exact plumbing check — and the effect is absent.

It does **not** establish that H45 was wrong about Aria. That question is H47's,
and H47 answers it: the Aria effect survives removing the black corners.
