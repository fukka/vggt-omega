# H6 — Peripheral cross-frame attention (method track 2: efficient fisheye video)

Locked before any training run.

## Hypothesis

A single zero-initialized cross-frame attention block whose QUERIES are only
the rim tokens (θ_patch > 35°) of frame t, with keys/values from the full
token set of neighboring frames, breaks the "context buys the center" wall
(ticket 024B): with the module on, multi-frame input reduces near-field-rim
depth error on held-out scenes beyond the single-frame H5 result, at a
fraction of full temporal attention's cost.

## Why this design is dictated by the measurements

- The rim is fusion-starved: 3/5/10-frame context improves overall depth but
  the rim/center penalty stays flat or worsens (024B) — the evidence exists
  and is spent on the center.
- The rim is alignment-rich: the model's cross-frame pose signal lives there
  (runs 004–007), so rim tokens are exactly the ones whose cross-frame
  correspondences are already encoded.
- Efficiency is principled, not ad hoc: rim tokens are ~40% of the cone's
  tokens, and the center provably does not need temporal help — so
  restricting queries to the rim is where the compute belongs.

## Method

- Module: one attention block (C=384 for DA3-Small, ~0.6 M params), inserted
  after the frozen backbone's final block: `rim_tokens_t += ZeroInitProj(
  Attn(Q=rim_tokens_t, KV=all_tokens_{t-1}))`. Zero-init output projection ⇒
  identity at start; center tokens pass through untouched by construction.
- Pose-safety premise TO VERIFY AT IMPLEMENTATION: DA3's camera path
  (cam_enc/cam_dec) is a separate trunk that does not consume the ViT's
  final-layer tokens; if true, the pose output is bit-identical with the
  module present, same as H5's LoRA-disabled guarantee. If false, the module
  moves behind the depth-head hook only, and pose stability becomes an
  empirical (measured) claim.
- Training: H5's three losses on frame pairs (the multi-frame term is the
  natural teacher here); module-only first, then optionally + H5 LoRA
  (stacking ablation).
- Comparisons: single-frame frozen; H5 (single-frame finetune); full
  temporal attention over ALL tokens at the same insertion point (the
  efficiency control — same params, all-token queries); frame-count scaling
  (1/2/3 frames) with and without the module.
- Eval: three-axis protocol on the two held-out scenes + pen_ctl trend vs
  frame count + FLOPs/latency of the module vs the all-token control.

## Refutation

Rim depth unimproved with the module on ⇒ the fusion starvation is not an
attention-routing problem (maybe the DPT head, maybe the compression is
upstream of the final block) — probe earlier insertion points before
abandoning; if the all-token control matches rim-query at equal params, the
efficiency story dies but the routing story survives.
