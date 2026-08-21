# H6 training runs: peripheral cross-frame attention vs the all-token control

**Owner:** gpu
**Status:** **done** — `results/autoresearch-h6-train/{rim,alltok}` @ `1327748`; `module_last.pt` + `train_log.json` for both arms.
**Files I may touch:** none — runs only. Checkpoints (~12 MB) + logs to
`results` under `results/autoresearch-h6-train/`.
**Blocked by:** none. Code on `organized`:
`autoresearch/experiments/h6-peripheral-attention/code/train.py` (module
mechanics verified by module_smoke.py: zero-init identity through the depth
head; camera outputs bit-identical by construction — the module only touches
a depth-head-only copy of the final feats level).
**What is waiting on it:** method track 2 of the CVPR pivot — does routing
cross-frame attention to the rim break the "context buys the centre" wall
(ticket 024B)?

## The task

Two runs, same four clean training sequences as #35 (seq136 +
decoration_seq132 held out):

```bash
# 1. rim-query module
python autoresearch/experiments/h6-peripheral-attention/code/train.py \
  --train-seqs <4 seqs> --epochs 20 --size 504 \
  --out-dir results/autoresearch-h6-train/rim
# 2. efficiency control: same params, ALL tokens as queries
python autoresearch/experiments/h6-peripheral-attention/code/train.py \
  --train-seqs <4 seqs> --epochs 20 --size 504 --all-token-control \
  --out-dir results/autoresearch-h6-train/alltok
```

Watch item from the CPU smoke: the mv loss printed 0.0 on one tiny-epoch
sample — in real training it should be visibly nonzero most steps; if it
sits at 0 across an epoch, stop and comment rather than burn the budget
(likely a warp-overlap or pair-spacing problem at full resolution).

An eval script that loads the module checkpoint follows as an addendum
comment (same pattern as #35's) — training can start before it lands.

## Acceptance

- Both `module_last.pt` + `train_log.json` on `results`, loss curve
  start/end in a comment, plus the mv-loss health note above.
