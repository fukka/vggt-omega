#!/usr/bin/env bash
# H14: rect-teacher -> fisheye-student distillation.
#
# Order matters. The teacher cache carries a PRE-CHECK (--score-teacher) that
# can kill the premise before any training: if the rect teacher is not better
# at the near rim than the raw model on this backbone, 024A does not transfer
# here and no student trained on it can help. Read the cache logs before
# reading anything else.
#
# Split of record: train on the four clean sequences, hold out seq136 and
# decoration_seq132 -- the same split as #35/#36 and H12, so the eval lands
# beside them. Eval is h5's eval_lora.py verbatim, which also reports pose.
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
  d=$(ls -d "$A"/Apartment_release_clean_${s}_M1292 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "[h14] missing $s"; exit 2; }
  TRAIN_SEQS="${TRAIN_SEQS:+$TRAIN_SEQS,}$d"
done

step () {  # step <tag> <logfile> <cmd...>
  local tag=$1 log=$2; shift 2
  local t0=$(date +%s)
  echo "[$tag] START $(date -Is)"
  ( CUDA_VISIBLE_DEVICES=$GPU "$@"; echo "MARKER_$tag=$?" ) > "$log" 2>&1
  echo "[$tag] $(grep -o "MARKER_$tag=[0-9]*" "$log" | tail -1) after $(( $(date +%s)-t0 ))s"
}

# ---- 1. teacher caches, both arms, four training sequences ----------------
for arm in rect roundtrip; do
  for s in seq131 seq133 seq134 seq135; do
    d=$A/Apartment_release_clean_${s}_M1292
    # Keyed by the FULL sequence name, because that is what Seq.name gives the
    # trainer to look the cache up by. A short key here builds a cache the
    # student cannot find, which is how the first launch died.
    n=$(basename "$d")
    [ -s "$CACHE/$arm/$n/manifest.json" ] && { echo "[cache_${arm}_$s] skip"; continue; }
    step "cache_${arm}_$s" "$CACHE/${arm}_${s}.log" \
      python autoresearch/experiments/h14-rect-distill/code/cache_teacher.py \
        --arm "$arm" --seq "$d" --out "$CACHE/$arm/$n" --score-teacher
  done
done

echo "=== PRE-CHECK (rect teacher vs raw fisheye, same pixels) ==="
grep -ahA 6 "PRE-CHECK" "$CACHE"/rect_*.log 2>/dev/null | head -40
grep -ah "PREMISE NOT CONFIRMED" "$CACHE"/rect_*.log 2>/dev/null

# ---- 2. the three arms ----------------------------------------------------
for arm in rect roundtrip gt; do
  [ -s "$OUT/$arm/lora_last.pt" ] && { echo "[train_$arm] skip"; continue; }
  extra=(); [ "$arm" != gt ] && extra=(--cache-root "$CACHE")
  step "train_$arm" "$OUT/$arm.log" \
    python autoresearch/experiments/h14-rect-distill/code/train_student.py \
      --arm "$arm" --train-seqs "$TRAIN_SEQS" --epochs "$EPOCHS" \
      --seed 0 --out-dir "$OUT/$arm" "${extra[@]}"
done

# ---- 3. held-out eval, h5's eval_lora.py verbatim (depth zones + pose) ----
for arm in rect roundtrip gt; do
  for seq in Apartment_release_clean_seq136_M1292 Apartment_release_decoration_seq132_M1292; do
    case "$seq" in *seq136*) sq=seq136;; *) sq=seq132;; esac
    tag="eval_${arm}_${sq}"
    [ -s "$OUT/$tag.json" ] && { echo "[$tag] skip"; continue; }
    step "$tag" "$OUT/$tag.log" \
      python autoresearch/experiments/h5-rim-finetune/code/eval_lora.py \
        --seq "$A/$seq" --lora "$OUT/$arm/lora_last.pt" --out "$OUT/$tag.json"
  done
done

echo "=== H14_DONE $(date -Is) ==="
