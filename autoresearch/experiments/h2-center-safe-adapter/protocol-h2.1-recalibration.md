# H2.1 — Is the near-field-rim failure a fixed, transferable miscalibration?

Locked before running (and before run_009's outcome is known; H2.0b's
refutation note already establishes this rung is worth testing under either
outcome — it just changes the wording of what the table fixes).

**Hypothesis.** A tiny multiplicative correction table
`d′(u) = d(u) · exp(c[i, j])`, indexed by θ-bin `i` (8 bins) × **predicted**
log-depth bin `j` (6 bins over [0.3, 10] m) — **48 parameters, zero-init,
post-hoc, features untouched, pose untouched by construction** — fit on a
training subset of seq131 frames, improves near-field-rim depth on HELD-OUT
frames without hurting center or far cells.

**Why this rung matters.** It is the strongest *trivial* baseline: if 48
fitted numbers transfer, the "disaster zone" is a stable miscalibration and
any learned adapter must beat this table to justify itself. If they do not
transfer, the failure is content-dependent and the adapter needs real
capacity. Either outcome shapes H2.

**Method.**
- Fit: `c[i, j]` = median over training pixels in cell (i, j) of
  `log(GT_range) − log(pred)` minus the training-frame's own median residual
  (the run_009 bias construction, re-indexed by PREDICTED depth so the table
  is applicable at test time with no GT).
- Cells with < 2000 training pixels: c = 0 (identity fallback, zero-init
  spirit).
- Splits: (a) even/odd frames (14/14, interleaved — optimistic, adjacent
  frames correlated); (b) first-half/second-half (harder). Report both.
  Cross-scene transfer is the real test and needs GPU sequences (future
  ticket if this rung passes).
- Eval: identical to run_008b (protocol of record: range domain, scale_shift
  per frame frozen before binning, joint θ×GT-depth table) on held-out frames,
  corrected vs uncorrected. Success = near-field-rim cells (0–2 m, θ ≥ 38°)
  improve by more than the split-to-split noise; center column (θ ≤ 10°) and
  far rows (3–10 m) within noise of uncorrected.

**Refutation.** Held-out near-rim cells unimproved (or center/far damaged) ⇒
the miscalibration is not fixed/transferable at this granularity; H2 moves to
learned, input-conditioned corrections (and the ticket-024B multi-frame
routing idea).

**Not claimed.** Metric scale (eval re-aligns per frame); cross-scene or
cross-device transfer; anything about pose (untouched by construction).
