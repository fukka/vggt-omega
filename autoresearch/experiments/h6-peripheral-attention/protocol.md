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
- Pose-safety premise VERIFIED 2026-08-24, and it is FALSE as first stated:
  DA3's `_process_camera_estimation(feats, ...)` consumes the backbone
  feats, so modifying them would touch pose. RESOLUTION (better than the
  planned fallback): the depth head and the camera head are PARALLEL
  readouts of `feats`, so the module writes its rim-token update into a
  copy consumed ONLY by `_process_depth_head`; the camera path reads the
  originals. Structural pose safety is preserved. (cam_enc, despite the
  name, is input camera-conditioning, not the pose estimator.)
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

## Addendum 2026-08-19: H6.1 KV-compression probe (locked before run)

Trigger: human asked to study Spark3R (arXiv:2605.06270) — queries are
compression-sensitive (view-specific requests), KV tolerates aggressive
pruning (shared context), layer-adaptive factors, training-free. Our module
already embodies the query half (rim-only queries, geometry-selected); its
KV side is untouched: prev-frame KV = ALL 1296 grid tokens, of which ~323
are dead-corner tokens outside the imaged cone (vignette black — noise).

Probe (CPU, delivered #36 rim checkpoint, seq131 dense frames — a TRAINING
scene, so this is an EXPLORATORY mechanism probe, not a held-out claim):
run the module with KV restricted to (a) full 1296 [as trained], (b) cone
only (~973), (c) cone rim-only, (d) cone center-only, comparing zone AbsRel
of the after-arm and the output drift |d_b - d_a|.

Predictions (locked):
- P1: dropping dead-corner KV (b) changes near-rim AbsRel by <2% relative —
  the trained attention already ignores vignette tokens; if so, cone-KV
  becomes the default (25% attention-FLOPs saving, free).
- P2: rim-only KV (c) degrades little for SMALL motions (temporal
  neighbours see the same rim band) but center-only KV (d) hurts more —
  the rim queries fetch context from where content overlaps, i.e. mostly
  the rim band itself. Whichever way (c)/(d) lands, it tells us where the
  cross-frame evidence lives, feeding the H6.2 design below.

H6.2 (design note, no run yet): temporal KV pyramid — rolling memory where
KV from t-k is pruned/merged with a factor growing in k (t-1 cone-full,
t-2 2x-merged, t-4 4x-merged): multi-frame peripheral memory at ~constant
cost. Spark3R's asymmetry finding + our geometry = merge KV along the ring
where solid angle says tokens are redundant (the H8 refutation killed
INPUT remapping; KV-side merging is context compression, which Spark3R
shows is the safe side).

## Addendum 2026-08-19b: H6.2 training-free multi-frame KV probe (locked)

The module was trained with t-1 KV only, but MHA is permutation-invariant
over the KV set — concatenating rim-KV from older frames needs no
retraining. Variants (rim queries fixed): KV = rim(t-1) [H6.1 winner],
rim(t-1)+rim(t-2), rim(t-1)+rim(t-2)+rim(t-4). seq131 dense 20 frames,
EXPLORATORY (training scene).

Predictions (locked):
- P1: adding rim(t-2) does not hurt (near_rim within +2% of t-1-only);
  any improvement >2% licenses H6.2 (trained pyramid) as a GPU ticket.
- P2: marginal value decays with frame distance (t-4's addition changes
  less than t-2's did) — the pyramid's decreasing-budget schedule is then
  the right shape.
