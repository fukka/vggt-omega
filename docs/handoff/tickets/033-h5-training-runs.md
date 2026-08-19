# H5 training runs: rim-targeted LoRA vs the plain-LoRA control

**Owner:** gpu
**Status:** open — not started.
**Files I may touch:** none — runs only. Checkpoints (~500 KB each) + train
logs to `results` under `results/autoresearch-h5-train/`.
**Blocked by:** none. Code on `organized`:
`autoresearch/experiments/h5-rim-finetune/code/train.py` (mechanics
smoke-verified CPU-side: teacher = LoRA-disabled path is bit-identical, base
weights never move; protocol in the same directory).
**What is waiting on it:** method track 1 of the CVPR pivot. Evaluation runs
CPU-side from the tiny checkpoints — this ticket is training only.

## The task

Two runs on the four clean training sequences (seq131, seq133, seq134,
seq135 of the ticket-024 split; seq136 + decoration_seq132 are HELD OUT — do
not train on them):

```bash
# 1. full method
python autoresearch/experiments/h5-rim-finetune/code/train.py \
  --train-seqs <seq131>,<seq133>,<seq134>,<seq135> \
  --epochs 20 --size 504 \
  --out-dir results/autoresearch-h5-train/full
# 2. plain-LoRA control (same budget, none of our losses)
python autoresearch/experiments/h5-rim-finetune/code/train.py \
  --train-seqs <same> \
  --epochs 20 --size 504 \
  --depth-alpha 0 --lambda-f 0 --lambda-m 0 \
  --out-dir results/autoresearch-h5-train/plain
```

Same *.png glob workaround as before if needed. If an epoch is slower than
~10 min on the box, subsample --max-frames 40 and say so.

## Acceptance

- `lora_last.pt` + `train_log.json` for both runs on `results`, and a comment
  with the two loss curves' start/end values. No evaluation needed — the
  checkpoints come back here.
