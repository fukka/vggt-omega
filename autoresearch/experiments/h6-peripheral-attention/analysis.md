# H6 analysis

## CPU pilot (2026-08-26, EXPLORATORY — and it did NOT confirm the direction)

Same reduced setting as the H5 pilot (seq131 even/odd frame split, 252 px,
10 epochs). Result: near rim +2.1% (the target zone, unimproved), near
center −22.8%, center −4.0%, far +2.8%.

Two honest reads, both consequential:

1. **The pilot violates the module's premise.** "Adjacent" pilot frames are
   100 capture-frames (~3.3 s) apart — enormous viewpoint change; cross-frame
   attention has little usable correspondence to route. This is a weak test,
   not a refutation. BUT the same trap was about to hit #36 on the box: the
   trainer subsampled frames UNIFORMLY to --max-frames, recreating sparse
   pairs on full sequences. Fixed: Seq gains dense=True (contiguous middle
   block, video-rate neighbors); the H6 trainer and eval now use it (H5
   keeps uniform sampling — its mv term is auxiliary; scene diversity
   matters more there).
2. **"Center untouched" is token-level, not output-level.** The DPT head
   mixes tokens spatially, so rim-token updates CAN move center outputs
   (here they helped: −22.8% near center; elsewhere they could hurt).
   Protocol wording corrected: structural safety claims for H6 cover the
   POSE path (reads original feats) and rim-query locality at the token
   level; center-depth safety is empirical, same as H5.

The GPU run (#36, dense windows) is the real test; the pilot's job was to
catch exactly this kind of setup error before burning box time — it did.

## H6.1 KV-compression probe (2026-08-19, EXPLORATORY — seq131 is a training scene)

Delivered #36 rim checkpoint, 20 dense frames, 504px, training-free KV
restriction at eval (results/probe_kv_seq131.json):

| KV set | tokens | near_rim | center | far |
|---|---|---|---|---|
| full (as trained) | 1296 | 0.685 | 0.380 | 0.211 |
| cone only | 975 | 0.695 | 0.372 | 0.214 |
| **rim only** | **627** | **0.684** | 0.374 | 0.212 |
| center only | 348 | 0.808 | 0.367 | 0.225 |

P1 confirmed: dead-corner KV droppable (+1.5% rel near_rim, inside the
locked <2% bar). P2 confirmed sharply: rim-only KV matches full exactly
(0.684 vs 0.685) while center-only costs +18% — the cross-frame evidence
rim queries consume lives in the previous frame's rim band (ring-to-ring
content overlap under egocentric motion). Attention cost: all-token
1296x1296 -> rim-query 627x1296 (0.48x) -> rim-query+rim-KV 627x627
(**0.23x, ~4.3x cheaper than all-token**), training-free.

Caveats: training scene; small inter-frame motion (dense windows); zone
metrics are alignment-normalized (mean drift up to 0.26 m for cone-KV shows
raw outputs do move — the affine absorbs it). Held-out confirmation added
as an addendum to #36. H6.2 (temporal KV pyramid) design note in
protocol.md — now licensed by this measurement.

## H6.2 training-free multi-frame KV probe (2026-08-19, EXPLORATORY)

Same setup as H6.1, frames pos>=4 so all variants share the eval set
(results/probe_kv_multi_seq131.json):

| KV | near_rim | center | far |
|---|---|---|---|
| rim(t-1) | 0.641 | 0.370 | 0.210 |
| + rim(t-2) | 0.629 (-1.9%) | 0.368 | 0.210 |
| + rim(t-4) | 0.630 | 0.365 | 0.209 |

P1 marginal (improvement just under the 2% ticket bar), P2 confirmed
(t-4 adds nothing beyond t-2 — the decreasing-budget pyramid shape is
right, but the free effect saturates at one extra frame). Verdict: small
free win; do NOT spend GPU on a trained pyramid now. Revisit only if the
#36 held-out evals show the pairwise module strongly positive. Paper: one
sentence in the efficiency/extension paragraph, clearly labeled
training-free and exploratory.
