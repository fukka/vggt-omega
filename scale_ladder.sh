#!/bin/bash
# The data ladder at CONSTANT gradient steps.
#
# The earlier ladder (60/240/600 frames per sequence, all at 20 epochs) varied
# data AND compute together: 4.8k -> 19.2k -> 48k steps. So "more data helped"
# and "more steps helped" were not separable. Here every rung takes ~48k steps,
# so the only thing that changes is how many DISTINCT frames those steps saw.
#
#   60/seq   x 200 epochs = 48,000   240 distinct frames, seen 200x
#   600/seq  x  20 epochs = 48,000   2,400 distinct
#   1200/seq x  10 epochs = 48,000   4,800 distinct
#   2880/seq x   4 epochs = 46,080   11,520 distinct  (every frame on disk)
#
# The 600 rung already exists (results/autoresearch-followups/scale_600) and is
# reused rather than re-run.
set -uo pipefail
REPO=/user/f.zhang2/projects/vggt-omega-organized
A=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
OUT=$REPO/results/autoresearch-scale-ladder
GPU=${GPU:-0}
RUNS=${RUNS:?set RUNS, e.g. "d2880:2880:4 d1200:1200:10"}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
cd "$REPO" || exit 2
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"

TRAIN_SEQS=""
for s in seq131 seq133 seq134 seq135; do
  TRAIN_SEQS="${TRAIN_SEQS:+$TRAIN_SEQS,}$A/Apartment_release_clean_${s}_M1292"
done
HELD="Apartment_release_clean_seq136_M1292 Apartment_release_decoration_seq132_M1292"

step () {
  local tag=$1 log=$2; shift 2
  local t0=$(date +%s); echo "[$tag] START $(date -Is)"
  ( CUDA_VISIBLE_DEVICES=$GPU "$@"; echo "MARKER_$tag=$?" ) > "$log" 2>&1
  echo "[$tag] $(grep -o "MARKER_$tag=[0-9]*" "$log" | tail -1) after $(( $(date +%s)-t0 ))s"
}

for spec in $RUNS; do
  tag=${spec%%:*}; rest=${spec#*:}; frames=${rest%%:*}; epochs=${rest##*:}
  [ -s "$OUT/$tag/lora_last.pt" ] || \
    step "train_$tag" "$OUT/$tag.log" \
      python autoresearch/experiments/h14-rect-distill/code/train_student.py \
        --arm gt --train-seqs "$TRAIN_SEQS" --max-frames "$frames" \
        --epochs "$epochs" --seed 0 --out-dir "$OUT/$tag"
  for seq in $HELD; do
    case "$seq" in *seq136*) sq=seq136;; *) sq=seq132;; esac
    [ -s "$OUT/${tag}_${sq}.json" ] || \
      step "eval_${tag}_${sq}" "$OUT/${tag}_${sq}.log" \
        python autoresearch/experiments/h5-rim-finetune/code/eval_lora.py \
          --seq "$A/$seq" --lora "$OUT/$tag/lora_last.pt" \
          --out "$OUT/${tag}_${sq}.json"
  done
done
echo "ALL_DONE_$GPU"
