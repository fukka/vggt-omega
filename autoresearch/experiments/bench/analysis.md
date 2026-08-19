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
