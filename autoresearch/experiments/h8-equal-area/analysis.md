# H8 analysis — Probe A REFUTES the hypothesis (2026-08-19, seq131, 28 frames, 504px)

Visual check passed before numbers were read. Both arms share frames,
masks (cone ∩ remap-coverage=1.0), and per-frame scale_shift alignment.

| zone | plain fisheye | equisolid remap | delta |
|---|---|---|---|
| near_rim (<=2m, >=38 deg) | 1.061 | 1.394 | +31.4% |
| near_center (<=2m, <=11 deg) | 0.422 | 0.493 | +16.8% |
| center (<=11 deg) | 0.306 | 0.356 | +16.4% |
| far (>=3m) | 0.214 | 0.254 | +18.7% |

Verdict: P2-and-worse. Equalizing per-pixel solid angle does not flatten
the radial field — it degrades EVERY zone at identical token count, worst
exactly where it was supposed to help. The radial compression is therefore
NOT a sampling-density artifact. The efficiency claim (Probe B: 0.6x tokens
at error parity) is dead a fortiori — parity already fails at 1.0x tokens.
H8 closed; no Probe B, no GPU spend.

Pattern across today's three refutations + Center-PH:
- H3 (patch undistortion): no-op — within-patch distortion is sub-pixel.
- Center-PH (rectify-and-crop): near-field center +62%.
- H8 (equal-area resample): everything +16-31%.
- H7 (theta-conditioning the adapter): redundant — PE already conditions.
**Input-space geometric surgery hurts a frozen FM; conditioning the adapter
on geometry is redundant; the interventions that work live behind the
encoder (feature readout, rung 1) or in the objective (rung 2) or add
evidence (rung 3).** This "where to intervene" gradient is a paper-level
organizing result, each rung with its own measured refutation.

Provenance: results/probe_a_seq131.json; code/probe_a.py (protocol locked
pre-run at 2088720's successor; visual check in scratch h8_visual.png).
