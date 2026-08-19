# Ticket 037 — ScanNet++ 3f15 pose anchor (external published reference)

STATUS: DRAFT — do not start until the #35/#36 eval batches and #38 v2 rows
are delivered. Will be posted as a gpu-labelled issue then.

## Why (one paragraph)
3f15a9266d is the ScanNet++ sequence RayTun3R's paper names in its per-
sequence tables (Tab. 5, DA3-S + GT calib: R° 0.40, t° 2.2, d_reproj 1.7).
Running our in-repo raytun3r reproduction on the same sequence gives the
paper an EXTERNAL anchor: if our reproduction's numbers sit near the
published ones, every comparison we make against RayTun3R on ADT inherits
credibility (verify-don't-fit: we compare against an independent published
reference, we do not tune toward it). Dense-depth eval on ScanNet++ stays
blocked (render_depth absent upstream), so this is a POSE ticket only.

## Runs (box, full 3f15 sequence from the ScanNet++ download)
1. Vanilla DA3-S pose on consecutive pairs (their protocol): R°, t° medians.
2. RayTun3R fit (defaults: 30 windows / 300 iters / seq_len 3, GT calib)
   then the same pose eval. Compare (1)->(2) against published Tab. 5.
3. Same eval for our H5-full LoRA checkpoint (results/autoresearch-h5-train/
   full/lora_last.pt) loaded on DA3-S — CROSS-LENS probe (trained on Aria
   KB4; ScanNet++ DSLR KB4 115°). Expect degradation; any non-collapse is
   informative. Label the row cross-lens, never mix with same-lens rows.
4. JSONs + a 5-line summary comment; push to results/scannetpp-anchor/.

## Acceptance
- Protocol facts logged: frame count, pair spacing, calib source (GT json).
- If our reproduction lands far from Tab. 5 (>2x on R°), STOP and comment
  with the gap — that is a reproduction finding, not a tuning target.
