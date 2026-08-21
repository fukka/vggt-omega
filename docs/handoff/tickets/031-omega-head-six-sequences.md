# VGGT-Omega feature head: the highest-headroom backbone, six sequences

**Owner:** gpu
**Status:** **done** — `results/autoresearch-h24-omega` (meta `ticket: 31`). Transfers to VGGT-Omega on all 6 sequences and both splits (−19.3% to −40.6%), smaller effect size than DA3-Small.
**Files I may touch:** none — runs only (scratch runners fine, as in #29).
Results to `results` under `results/autoresearch-h24-omega/`.
**Blocked by:** none. Code on `organized`:
`autoresearch/experiments/h2-center-safe-adapter/code/omega_head.py`
(protocol `protocol-h2.4-vggt-omega.md` locked; full path exercised CPU-side
with `--random-init` — structure only).
**What is waiting on it:** VGGT-Omega has the largest distance-controlled rim
penalty of the five ticket-024 models (1.81×) and is this repo's own
backbone. DA3-Small's head held on all six sequences (#29). If the same
recipe works on VGGT-Omega, the backbone-agnostic claim is complete; if not,
the failure localizes what its final tokens lack.

## The task

For each of the six #29 sequences, both splits:

```bash
python autoresearch/experiments/h2-center-safe-adapter/code/omega_head.py \
    --seq <seq_dir> --split halves --weights <the VGGT-Omega-1B-512 checkpoint
    fovbench's vggt_omega backend loads> --out results/run_013_<seq>_halves.json
# and --split even_odd
```

Notes:
- `VGGTOmega()` is instantiated at default width (1024). If
  `load_state_dict(strict=True)` fails on the fovbench checkpoint, load with
  strict=False and put the missing/unexpected key lists in the comment —
  aggregator + dense_head must load fully for the run to count.
- Depth is converted planar-z → range through the calibrated KB4 camera inside
  the script (convention discipline; don't re-convert).
- Same *.png-glob caveat as #29 if it applies; scratch-runner it.
- GPU makes the 1B forwards cheap; caches land under $H2_CACHE/<seq>/.

## Acceptance

- 12 JSONs on `results` + a comment: per-sequence near-rim before→after next
  to #29's DA3-Small numbers, and one sentence: does the recipe transfer to
  the highest-headroom backbone?
