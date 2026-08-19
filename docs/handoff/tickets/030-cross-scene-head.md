# Cross-scene head: leave-one-scene-out over the six-sequence split

**Owner:** gpu
**Status:** open — not started.
**Files I may touch:** none — runs only. Results to `results` under
`results/autoresearch-h23-crossscene/`.
**Blocked by:** none. Code on `organized`:
`autoresearch/experiments/h2-center-safe-adapter/code/cross_scene.py`
(smoke-tested CPU-side; protocol
`protocol-h2.3-cross-scene.md` locked). #29's caches are reusable if you
point `H2_CACHE` at a shared root — the script now uses
`$H2_CACHE/<seq_basename>/` per sequence on its own.
**What is waiting on it:** #29 established the per-scene result on all six
sequences (−21%…−75% near-rim, near-center undamaged). This ticket decides
whether that is ONE head or six: train on five sequences, evaluate on the
held-out sixth, all six folds.

## The task

For each fold (six total), with the same six sequences as #29:

```bash
python autoresearch/experiments/h2-center-safe-adapter/code/cross_scene.py \
    --train-seqs <five seq dirs, comma-separated> \
    --eval-seqs <the held-out seq dir> \
    --max-frames 60 \
    --out results/run_012_fold_<heldout>.json
```

Note the #29 discovery that `videos_rgb` is `*.png` on this data — if
`AriaLocalPairs` globs only `*.jpg`, patch the glob in your scratch runner the
same way you did for #29 (or symlink), and say which in the comment.

Success gate (from the locked protocol): cross-scene near-rim gain ≥ half of
that fold's within-scene gain (#29's halves numbers) on most folds, center
undamaged. Also run the 48-param table cross-scene the same way on one fold
(any) as the "do features add anything across scenes" control if time allows.

## Acceptance

- 6 fold JSONs on `results` + a comment: per-fold near-rim before→after vs
  #29's within-scene number, and one sentence: one head or six?
