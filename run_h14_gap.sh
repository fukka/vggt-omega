#!/usr/bin/env bash
# Two things at once.
#
# 1. RETRAIN gt. The reported labelled upper bound was trained for 8 epochs,
#    not 20: the first (failed) launch of run_h14.sh died on a missing cache
#    for the two distillation arms but NOT for gt, which needs no cache, so it
#    trained for ~3 minutes before the session was killed -- and the relaunch's
#    "skip if a checkpoint exists" guard then skipped it. Its loss was still
#    dropping 27.6% per 5 epochs when it stopped, so every "fraction of what
#    labels buy" figure computed against it is an OVERSTATEMENT of the
#    label-free arms.
#
# 2. Measure the train/test gap directly. Every number so far is on the two
#    held-out sequences; nothing says how much of the gain is this apartment.
#    Evaluating the same checkpoints on TRAINING sequences gives the gap.
set -uo pipefail
REPO=/user/f.zhang2/projects/vggt-omega-organized
A=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
CACHE=/netapp/datasets/f.zhang2/h14_teacher_cache
OUT=$REPO/results/autoresearch-h14-rectdistill
GPU=${GPU:-0}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
cd "$REPO" || exit 2
export PYTHONUNBUFFERED=1

TRAIN_SEQS=""
for s in seq131 seq133 seq134 seq135; do
  TRAIN_SEQS="${TRAIN_SEQS:+$TRAIN_SEQS,}$A/Apartment_release_clean_${s}_M1292"
done

step () {
  local tag=$1 log=$2; shift 2
  local t0=$(date +%s); echo "[$tag] START $(date -Is)"
  ( CUDA_VISIBLE_DEVICES=$GPU "$@"; echo "MARKER_$tag=$?" ) > "$log" 2>&1
  echo "[$tag] $(grep -o "MARKER_$tag=[0-9]*" "$log" | tail -1) after $(( $(date +%s)-t0 ))s"
}

rm -rf "$OUT/gt" "$OUT/eval_gt_seq136.json" "$OUT/eval_gt_seq132.json"
step train_gt20 "$OUT/gt.log" \
  python autoresearch/experiments/h14-rect-distill/code/train_student.py \
    --arm gt --train-seqs "$TRAIN_SEQS" --epochs 20 --seed 0 --out-dir "$OUT/gt"

# held-out, again, against the properly trained upper bound
for seq in Apartment_release_clean_seq136_M1292 Apartment_release_decoration_seq132_M1292; do
  case "$seq" in *seq136*) sq=seq136;; *) sq=seq132;; esac
  step "eval_gt_$sq" "$OUT/eval_gt_$sq.log" \
    python autoresearch/experiments/h5-rim-finetune/code/eval_lora.py \
      --seq "$A/$seq" --lora "$OUT/gt/lora_last.pt" --out "$OUT/eval_gt_$sq.json"
done

# the gap: the same checkpoints on sequences they were TRAINED on
for arm in rect rect_ring gt; do
  for sq in seq131 seq133; do
    tag="evalTRAIN_${arm}_${sq}"
    [ -s "$OUT/$tag.json" ] && { echo "[$tag] skip"; continue; }
    step "$tag" "$OUT/$tag.log" \
      python autoresearch/experiments/h5-rim-finetune/code/eval_lora.py \
        --seq "$A/Apartment_release_clean_${sq}_M1292" --lora "$OUT/$arm/lora_last.pt" \
        --out "$OUT/$tag.json"
  done
done

echo "=== H14_GAP_DONE $(date -Is) ==="
