# BENCH: the RayTun3R comparison row on the two held-out scenes

**Owner:** gpu
**Status:** open — not started. Fourth in the current queue (#35, #36, #37
first) — this one is the most GPU-hungry (paper quotes 2–3 h/scene).
**Files I may touch:** none — runs only. Outputs to `results` under
`results/autoresearch-bench/rt3r/`.
**Blocked by:** none. Code on `organized`: raytun3r's own
`train.py --dataset adt` (ADTSequence exists) and
`autoresearch/experiments/bench/code/raytun3r_row.py` (vanilla path
smoke-tested on the Mac; adapter loading mirrors raytun3r/eval.py's own
pattern — scratch-adapt if an arg name drifted).
**What is waiting on it:** the main table's strongest published-adapter
comparison. RayTun3R adapts per sequence, so it runs on each held-out scene
directly — a fair (even favorable-to-it) setting.

## The task

For BOTH held-out sequences (seq136, decoration_seq132):

```bash
# 1. per-scene adaptation (their protocol, our data class)
python -m raytun3r.train --dataset adt --path <seq_dir> \
  --backbone da3 --variant small \
  --extrinsics-json cam3r/data/adt_camera_rgb_calibration.json \
  --out results/autoresearch-bench/rt3r/<seq>/
# 2. our joint-table row, adapted + vanilla anchor
python autoresearch/experiments/bench/code/raytun3r_row.py \
  --path <seq_dir> --adapter results/autoresearch-bench/rt3r/<seq>/adapter.pt \
  --out results/autoresearch-bench/rt3r/<seq>_adapted.json
python autoresearch/experiments/bench/code/raytun3r_row.py \
  --path <seq_dir> \
  --out results/autoresearch-bench/rt3r/<seq>_vanilla.json
```

Use raytun3r train defaults otherwise (300 iters etc. — the reproduction's
flag map is in raytun3r/reproduction.md); note whichever matcher it selects
(UFM if importable, else SIFT) in the comment, since supervision quality
travels with the number.

## Acceptance

- adapter.pt + train_log.json + 4 row JSONs on `results`, a comment with the
  near-rim cells adapted-vs-vanilla and the matcher used.
