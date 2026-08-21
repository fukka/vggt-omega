# BENCH: frozen-model rows on the two held-out scenes

**Owner:** gpu
**Status:** **done** — `results/autoresearch-bench/` @ `f1b0ab7`: 5 models x 2 held-out scenes (seq136, decoration_seq132), 504 px, 60 frames. `dac_swinl_indoor` skipped by design (ERP-native output, not planar-z) and said so in its meta.
**Files I may touch:** none — runs only. JSONs to `results` under
`results/autoresearch-bench/`.
**Blocked by:** none. Code on `organized`:
`autoresearch/experiments/bench/code/eval_baseline_joint.py`
(smoke-tested with da3_small on the Mac's local seq131; reuses
`finetune/eval/baselines` loaders — planar-z contract, loader-shape-following
camera).
**What is waiting on it:** the main benchmark table's frozen rows, protocol-
identical to our method evals (BENCH protocol in
`autoresearch/experiments/bench/protocol.md`).

## The task

For BOTH held-out sequences (seq136, decoration_seq132) — point --adt-root at
a root containing only that sequence, or symlink-scratch as usual — run each:

```bash
for M in unik3d_vitl dac_swinl_indoor da3_small da3_large vggt_omega dav2_large; do
  python autoresearch/experiments/bench/code/eval_baseline_joint.py \
    --model $M --adt-root <root_with_held_out_seq> --res 504 --max-frames 60 \
    --out results/autoresearch-bench/${M}_<seq>.json
done
```

Notes: weights for UniK3D/DAC per `finetune/eval/baselines/download_weights.py`
if not already on the box; if an adapter needs kwargs the smoke didn't
(dac ERP path etc.), scratch-adapt and say so. 12 JSONs.

## Acceptance

- 12 JSONs on `results` + a comment with each model's whole-image AbsRel/d1
  and its near-rim (<=2m, >=38deg-ish cells) read for a quick sanity glance.
