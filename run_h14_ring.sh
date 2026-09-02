#!/usr/bin/env bash
# H14.2: does a teacher that can SEE the whole rim make a better student than
# one that sees 70% of it much better?
#
#   arm            near-rim zone the teacher can answer for   teacher gain there
#   rect            70.4%                                      -35.3%
#   rect_ring       99.3%                                      -16.0%
#
# The ring adds the outermost annulus, which is the hardest part of the zone --
# so its average gain is lower while its coverage is nearly complete. Which
# makes the better student is exactly what this measures. roundtrip_ring exists
# so the control carries the ring's resampling budget rather than the single
# view's.
set -uo pipefail
REPO=/user/f.zhang2/projects/vggt-omega-organized
A=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
CACHE=/netapp/datasets/f.zhang2/h14_teacher_cache
OUT=$REPO/results/autoresearch-h14-rectdistill
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-20}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
cd "$REPO" || exit 2
export PYTHONUNBUFFERED=1
mkdir -p "$OUT" "$CACHE"

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

for arm in rect_ring roundtrip_ring; do
  for s in seq131 seq133 seq134 seq135; do
    d=$A/Apartment_release_clean_${s}_M1292; n=$(basename "$d")
    [ -s "$CACHE/$arm/$n/manifest.json" ] && { echo "[cache_${arm}_$s] skip"; continue; }
    step "cache_${arm}_$s" "$CACHE/${arm}_${s}.log" \
      python autoresearch/experiments/h14-rect-distill/code/cache_teacher.py \
        --arm "$arm" --layout ring --seq "$d" --out "$CACHE/$arm/$n" --score-teacher
  done
done

echo "=== RING PRE-CHECK (teacher vs raw, same pixels, 4 training sequences) ==="
grep -ahA 5 "PRE-CHECK" "$CACHE"/rect_ring_*.log 2>/dev/null

for arm in rect_ring roundtrip_ring; do
  [ -s "$OUT/$arm/lora_last.pt" ] && { echo "[train_$arm] skip"; continue; }
  step "train_$arm" "$OUT/$arm.log" \
    python autoresearch/experiments/h14-rect-distill/code/train_student.py \
      --arm "$arm" --train-seqs "$TRAIN_SEQS" --epochs "$EPOCHS" \
      --seed 0 --out-dir "$OUT/$arm" --cache-root "$CACHE"
done

for arm in rect_ring roundtrip_ring; do
  for seq in Apartment_release_clean_seq136_M1292 Apartment_release_decoration_seq132_M1292; do
    case "$seq" in *seq136*) sq=seq136;; *) sq=seq132;; esac
    tag="eval_${arm}_${sq}"
    [ -s "$OUT/$tag.json" ] && { echo "[$tag] skip"; continue; }
    step "$tag" "$OUT/$tag.log" \
      python autoresearch/experiments/h5-rim-finetune/code/eval_lora.py \
        --seq "$A/$seq" --lora "$OUT/$arm/lora_last.pt" --out "$OUT/$tag.json"
  done
done

echo "=== H14_RING_DONE $(date -Is) ==="
