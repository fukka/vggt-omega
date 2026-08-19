# H2.2 — Can frozen image features disambiguate what output-indexing cannot?

Locked before running. Direct test of run_010's diagnosis: the compression
makes predicted depth many-to-one in true depth, so a correction indexed by
(θ, predicted depth) hurts the minority population in each bin; disambiguation
needs input evidence. The cheapest input evidence is the backbone's own frozen
patch features.

**Hypothesis.** A tiny head reading DA3-Small's FROZEN final patch tokens —
per-token MLP [C → 64 → 1], zero-init last layer, θ supplied as an extra
input channel; ~25k params; backbone untouched, pose untouched by
construction — predicts a per-patch log-depth correction that, on HELD-OUT
frames, (a) beats the H2.1 48-param table in the near-rim zone AND (b) does
not damage the near-center cells the table damaged (the specific cells 0–2 m,
θ ≤ 11° must stay within noise of uncorrected).

**Why this design and not PE surgery first.** H1 established the rim features
feed the pose path; a readout-side head provably cannot move pose. If frozen
features suffice, the minimal safe adapter exists. If they fail, that is
evidence the features themselves lack the information — the measured
justification for input-side surgery (RayTun3R PE residuals / tokens) as the
next rung, with the pose-stability metric then mandatory.

**Method.**
- Features: final-layer patch tokens of the frozen DA3-Small forward already
  cached per frame (hook; cache to scratchpad; record layer name and shape).
- Head input per token: [token features (C=384, layer-normed), sin/cos of θ at
  the patch center, log predicted depth at the patch center]. Output: one
  scalar log-correction per patch, bilinearly upsampled to pixels;
  d′ = d · exp(upsampled correction). Zero-init last layer ⇒ identity at init.
- Training: same splits as run_010 (even/odd and halves). Loss: L1 on
  (log GT_range − log d − frame median residual) per pixel, inside the cone,
  GT ≤ 10 m. Adam 1e-3, ≤300 epochs on cached tokens, early stop on train
  plateau (no val leakage: held-out frames never touched until eval).
- Eval: identical to run_010 — protocol-of-record joint table, BEFORE/AFTER,
  both splits, PLUS the H2.1 table's numbers as the mandatory comparison. Read
  the full table, not zone pools (run_010's lesson); zones reported for
  continuity.
- Also report: correction magnitude map (where does the head act), and
  head-vs-table wins per cell.

**Refutation.** (a) fails ⇒ frozen features do not carry the disambiguating
signal at the near rim — escalate to input-side adaptation (H2.3) with pose
stability instrumented. (b) fails ⇒ the head inherits the conflation; inspect
whether it ignored features (‖feature weights‖ ≈ 0) before concluding.

**Not claimed.** Cross-scene transfer (one scene; GPU ticket if this passes);
metric scale; anything about VGGT-Ω yet (highest-headroom target per ticket
024, but DA3-Small is the CPU-testable representative).
