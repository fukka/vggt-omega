# H2.4 — Does the feature head transfer to VGGT-Ω (the highest-headroom backbone)?

Locked before running. VGGT-Ω has the largest distance-controlled rim penalty
of the five models in ticket 024A (1.81×), no camera-input channel, and is the
repo's own backbone — if the head works here, the "backbone-agnostic readout
fix" claim is complete; if it fails, the failure localizes what DA3's features
had that VGGT-Ω's lack.

**Hypothesis.** The same head recipe (per-patch MLP on frozen final aggregator
patch tokens + sinθ/cosθ/log-pred-depth, zero-init output, minutes of
fitting) improves VGGT-Ω's near-field-rim depth on held-out frames of each
sequence by a relative margin comparable to DA3-Small's (−21%…−75% across the
six sequences), with center within noise.

**Method.**
- Model: VGGT-Ω-1B-512 (box checkpoint), frozen, `enable_camera` on (pose
  untouched by construction — the head reads tokens, never writes).
- Tokens: `aggregated_tokens_list[-1][:, patch_token_start:]` (2·embed_dim);
  depth from `dense_head` — **planar z, converted once to range via cos θ of
  the calibrated KB4 camera** (convention discipline; VGGT couples depth to
  its own FoV estimate, which is exactly part of what the head may fix).
- Input 512² (patch 16 ⇒ 32×32 grid); camera scaled accordingly.
- Same fit/eval as run_011 / ticket #29: per-scene, both splits, six
  sequences; protocol-of-record joint tables; near-rim/center/far zones; the
  DA3 numbers as the comparison column.
- CPU-side validation before the ticket: the full code path exercised with a
  RANDOM-INIT small config (structure only — no numbers claimed from it).

**Refutation.** Near-rim gain far below DA3's on most sequences ⇒ VGGT-Ω's
final tokens do not expose the needed evidence (plausible: its single dense
head + FoV coupling may bake the miscalibration in earlier) — the follow-up
would probe earlier blocks, which is a new protocol, not this one.

**Not claimed.** Cross-scene for VGGT-Ω (only after per-scene); anything
about VGGT-Ω's pose quality (untouched, stated by construction).
