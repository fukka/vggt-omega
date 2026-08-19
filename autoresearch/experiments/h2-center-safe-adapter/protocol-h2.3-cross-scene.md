# H2.3 — Does one feature head transfer across scenes?

Locked before running (code prepared CPU-side; the run needs the box's
six-sequence split, so execution goes through a GPU ticket after #29 reports).

**Hypothesis.** A single feature head (same architecture as run_011) trained
on N−1 of the six ticket-024 sequences improves the held-out sequence's
near-field rim by ≥ half of the within-scene gain (#29's per-scene numbers
are the reference), without damaging its center — i.e. the correction is a
property of the (backbone, lens) pair, not of one room's furniture.

**Method.** Leave-one-scene-out over the six-sequence real-stream split:
`cross_scene.py --train-seqs <5 seqs> --eval-seqs <1 seq>`, six folds.
Fit and eval otherwise identical to run_011 (patch residual targets with
per-frame level removed; protocol-of-record joint-table eval; full tables
reported). Reference points per fold: uncorrected, the within-scene head
(#29), and the 48-param table fit cross-scene the same way.

**Refutation.** Cross-scene gain < half the within-scene gain on most folds ⇒
the head is memorizing scene content, and the honest product is a per-scene
online adapter (minutes of fitting per sequence — still useful, weaker claim);
check whether the table's cross-scene share is as high (if the table
transfers equally well, the features add nothing across scenes).

**Not claimed.** Cross-device or cross-lens transfer; VGGT-Ω (separate
protocol once the hook exists).
