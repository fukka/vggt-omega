# BENCH analysis

## Center-PH local anchor (2026-08-19, seq131 odd frames, DA3-S, 504px)

Protocol addendum row + one EXPLORATORY control (vanilla fisheye restricted
to the identical covered mask, same per-frame scale_shift procedure). Visual
sanity check passed (straight edges straight, no dead corners) before any
number was read.

Coverage (P2, P3): 75.2% of cone pixels; near_rim zone 49.6%, far zone
75.1%. The discarded band is exactly the theta>=45-54.7deg region where
H5/H6 claim their gains — Center-PH simply has no answer there.

Like-for-like on identical pixels (AbsRel, covered mask, 504px):

| zone | vanilla fisheye | Center-PH | delta |
|---|---|---|---|
| near_rim covered sliver (38-45deg + diagonal wedges) | 1.026 | 0.792 | -23% |
| near_center (<=2m, <=11deg) | 0.428 | **0.695** | **+62% WORSE** |
| center (<=11deg) | 0.304 | 0.300 | flat |
| far (>=3m) | 0.211 | 0.179 | -15% |

Verdict on locked predictions: P2, P3 confirmed. **P1 half-refuted**: the
rectified input does NOT buy the center on ADT — whole-center is flat and
the near-field center (the egocentric workspace: tabletop, objects at arm's
length) degrades 62%. RayTun3R's ScanNet++ finding (Center-PH wins depth
2.5x) does not transfer to egocentric near-field data. Mechanism unproven;
candidates: (a) rectification magnification pushes close-range content off
the backbone's scale prior, (b) the per-frame affine re-fits differently
once the compressed rim is cropped away (alignment coupling). Marked
EXPLORATORY until reproduced on a held-out scene on the box.

Consequences:
- The "just crop the problem away" control is now measured on ADT and it is
  NOT a free win even where it looks strongest elsewhere — strengthens the
  case that egocentric fisheye needs rim-aware methods rather than cropping.
- Cross-resolution caution: these are 504px numbers; the H5 CPU pilot table
  is 252px. Never quote them in the same row. The paper's main-table
  comparison waits for #35 (504px, held-out scenes).
- Held-out-scene Center-PH rows: same script, add to the box queue AFTER
  #35-#38 drain (do not flood).

Provenance: results/centerph_seq131_odd.json (row),
results/centerph_seq131_odd_vanillacovered.json (control),
code/centerph_row.py + scratch vanilla_covered.py (to be promoted when the
held-out ticket is written).

## Loader reconciliation: #37 vs #38 (2026-08-19) — RESOLVED

The 3.6x near_rim gap (da3_small seq136: 0.149 in #37 vs 0.542 in #38
vanilla) was a double depth conversion in raytun3r_row.py: ADTSequence
already returns euclidean range; the script divided by cos(theta) again,
inflating rim GT by up to 1.73x (and interacting with the per-frame affine).
Frame selection (first-60), alignment, and conventions were otherwise
identical between the two harnesses.

Fix verified on local seq131 (28 frames, 504px, vanilla DA3-S): the fixed
row reproduces the diagnosis-era signature (near rows rising with theta to
~2.0-2.2 at the rim; far rows flat) — results/rt3r_seq131_vanilla_fixed.json.
#38 reopened; 4-row re-run requested with adapters unchanged. v1 numbers
quarantined (never entered findings as claims beyond the direction, which is
itself now unverified).

Residual #37-vs-ours difference is scene content (near-rim mass), documented
above — not protocol.
