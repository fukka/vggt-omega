# Feature head on the six-sequence split: does run_011 hold beyond seq131?

**Owner:** gpu
**Status:** open — not started.
**Files I may touch:** none — runs only. Results as JSONs to the `results`
branch under `results/autoresearch-h22-sixseq/`.
**Blocked by:** none. Code is on `organized`:
`autoresearch/experiments/h2-center-safe-adapter/code/feature_head.py`.
**What is waiting on it:** autoresearch H2.2 measured, on seq131 held-out
frames, that a ~25k-param head on frozen DA3-Small patch tokens cuts
near-field-rim AbsRel by 51–67% (vs 18–25% for a 48-param lookup table)
without the table's near-center damage
(`autoresearch/experiments/h2-center-safe-adapter/analysis.md`, runs 010–011).
One scene is not a result; six are a section.

## The task

For each of the six sequences of the ticket-024 split (the same
`fovbench-joint` split, real stream), run both splits of the existing script:

```bash
python autoresearch/experiments/h2-center-safe-adapter/code/feature_head.py \
    --seq <seq_dir> --split halves   --out results/run_011_<seq>_halves.json
python autoresearch/experiments/h2-center-safe-adapter/code/feature_head.py \
    --seq <seq_dir> --split even_odd --out results/run_011_<seq>_even_odd.json
```

Notes:
- The script discovers `videos_rgb/`, `depth_npy/`, `groundtruth/
  aria_trajectory.csv` under `--seq`; it is CPU-only but runs fine on the box
  (one frozen DA3-Small forward per frame, cached under $H2_CACHE — set
  `H2_CACHE=/tmp/h2cache_<seq>` per sequence to keep caches separate).
- If a sequence has many more frames than seq131's 28, cap wall time by
  subsampling to ~40 frames evenly (say so in the comment if you do).
- Copy the printed BEFORE→AFTER joint tables into the issue comment along
  with the four zone lines per run; push the JSONs to `results`.

## Acceptance

- 12 JSONs on `results` + a comment with the zone table across sequences and
  one sentence: does the near-rim improvement hold (order −30% or better) with
  near-center not damaged, on how many of the six?
