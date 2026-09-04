#!/usr/bin/env bash
# Re-run everything that ran on sideways input.
#
# The backbone had been fed frames in ADT's native sensor orientation, a quarter
# turn off upright, in every run of the H1/H5/H12/H14/H15/H9 line. Frozen
# DA3-Small on seq136 measures -64% whole-image AbsRel and -71% near-rim once
# the turn is applied, and the pose error more than halves (12.07 -> 5.77 deg),
# so every absolute number and the rim penalty itself have to be re-established.
#
# Everything below is the SAME code and the SAME protocol as before; the only
# change is autoresearch/experiments/common/upright.py at the backbone boundary.
# Old outputs are moved aside rather than deleted -- they are the sideways
# record, and the comparison between the two is itself a result.
set -uo pipefail
REPO=/user/f.zhang2/projects/vggt-omega-organized
A=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
CACHE=/netapp/datasets/f.zhang2/h14_teacher_cache_upright
GPU=${GPU:-0}
STAGE=${STAGE:-all}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
cd "$REPO" || exit 2
export PYTHONUNBUFFERED=1

TRAIN_SEQS=""
for s in seq131 seq133 seq134 seq135; do
  TRAIN_SEQS="${TRAIN_SEQS:+$TRAIN_SEQS,}$A/Apartment_release_clean_${s}_M1292"
done
HELD="Apartment_release_clean_seq136_M1292 Apartment_release_decoration_seq132_M1292"

step () {
  local tag=$1 log=$2; shift 2
  [ -n "${SKIP_IF:-}" ] && [ -s "$SKIP_IF" ] && { echo "[$tag] skip"; return; }
  local t0=$(date +%s); echo "[$tag] START $(date -Is)"
  ( CUDA_VISIBLE_DEVICES=$GPU "$@"; echo "MARKER_$tag=$?" ) > "$log" 2>&1
  echo "[$tag] $(grep -o "MARKER_$tag=[0-9]*" "$log" | tail -1) after $(( $(date +%s)-t0 ))s"
}

# ---------------------------------------------------------------- H14 + H14.2
if [ "$STAGE" = all ] || [ "$STAGE" = h14 ]; then
OUT=$REPO/results/autoresearch-h14-upright
mkdir -p "$OUT" "$CACHE"

echo "### H14 teacher caches (the pre-check is the premise, re-measured upright)"
for arm in rect roundtrip; do
  for s in seq131 seq133 seq134 seq135; do
    d=$A/Apartment_release_clean_${s}_M1292; n=$(basename "$d")
    SKIP_IF="$CACHE/$arm/$n/manifest.json" step "cache_${arm}_$s" "$CACHE/${arm}_${s}.log" \
      python autoresearch/experiments/h14-rect-distill/code/cache_teacher.py \
        --arm "$arm" --seq "$d" --out "$CACHE/$arm/$n" --score-teacher
  done
done
echo "### PRE-CHECK, upright"
grep -ahA 5 "PRE-CHECK" "$CACHE"/rect_*.log 2>/dev/null

for arm in rect_ring roundtrip_ring; do
  for s in seq131 seq133 seq134 seq135; do
    d=$A/Apartment_release_clean_${s}_M1292; n=$(basename "$d")
    SKIP_IF="$CACHE/$arm/$n/manifest.json" step "cache_${arm}_$s" "$CACHE/${arm}_${s}.log" \
      python autoresearch/experiments/h14-rect-distill/code/cache_teacher.py \
        --arm "$arm" --layout ring --seq "$d" --out "$CACHE/$arm/$n" --score-teacher
  done
done

for arm in rect rect_ring roundtrip roundtrip_ring gt; do
  extra=(); [ "$arm" != gt ] && extra=(--cache-root "$CACHE")
  SKIP_IF="$OUT/$arm/lora_last.pt" step "train_$arm" "$OUT/$arm.log" \
    python autoresearch/experiments/h14-rect-distill/code/train_student.py \
      --arm "$arm" --train-seqs "$TRAIN_SEQS" --epochs 20 --seed 0 \
      --out-dir "$OUT/$arm" "${extra[@]}"
done

for arm in rect rect_ring roundtrip roundtrip_ring gt; do
  for seq in $HELD; do
    case "$seq" in *seq136*) sq=seq136;; *) sq=seq132;; esac
    SKIP_IF="$OUT/eval_${arm}_${sq}.json" step "eval_${arm}_${sq}" "$OUT/eval_${arm}_${sq}.log" \
      python autoresearch/experiments/h5-rim-finetune/code/eval_lora.py \
        --seq "$A/$seq" --lora "$OUT/$arm/lora_last.pt" --out "$OUT/eval_${arm}_${sq}.json"
  done
done
# the train/test gap, upright
for arm in rect gt; do
  for sq in seq131 seq133; do
    SKIP_IF="$OUT/evalTRAIN_${arm}_${sq}.json" step "evalTRAIN_${arm}_${sq}" "$OUT/evalTRAIN_${arm}_${sq}.log" \
      python autoresearch/experiments/h5-rim-finetune/code/eval_lora.py \
        --seq "$A/Apartment_release_clean_${sq}_M1292" --lora "$OUT/$arm/lora_last.pt" \
        --out "$OUT/evalTRAIN_${arm}_${sq}.json"
  done
done
echo "=== H14_UPRIGHT_DONE $(date -Is) ==="
fi

# ------------------------------------------------------------------------ H15
if [ "$STAGE" = all ] || [ "$STAGE" = h15 ]; then
OUT=$REPO/results/autoresearch-h15-upright; mkdir -p "$OUT"
for arm in jac mismatched shuffled none; do
  SKIP_IF="$OUT/$arm/cond_last.pt" step "train_$arm" "$OUT/$arm.log" \
    python autoresearch/experiments/h15-lens-holdout/code/train_multilens.py \
      --arm "$arm" --train-seqs "$TRAIN_SEQS" --epochs 40 --seed 0 --out-dir "$OUT/$arm"
done
for arm in jac mismatched shuffled none; do
  for seq in $HELD; do
    case "$seq" in *seq136*) sq=seq136;; *) sq=seq132;; esac
    SKIP_IF="$OUT/eval_${arm}_${sq}.json" step "eval_${arm}_${sq}" "$OUT/eval_${arm}_${sq}.log" \
      python autoresearch/experiments/h15-lens-holdout/code/eval_lens.py \
        --seq "$A/$seq" --ckpt "$OUT/$arm/cond_last.pt" --out "$OUT/eval_${arm}_${sq}.json"
  done
done
echo "=== H15_UPRIGHT_DONE $(date -Is) ==="
fi

# ------------------------------------------------------------------------- H9
if [ "$STAGE" = all ] || [ "$STAGE" = h9 ]; then
OUT=$REPO/results/autoresearch-h9-upright; mkdir -p "$OUT"
for seq in Apartment_release_clean_seq136_M1292 Apartment_release_decoration_seq132_M1292 \
           Apartment_release_clean_seq131_M1292 Apartment_release_clean_seq133_M1292 \
           Apartment_release_clean_seq134_M1292 Apartment_release_clean_seq135_M1292; do
  sq=$(echo "$seq" | sed 's/.*release_//; s/_M1292//')
  SKIP_IF="$OUT/$sq.json" step "h9_$sq" "$OUT/$sq.log" \
    python autoresearch/experiments/h9-raycal-tta/code/run_h9.py --seq "$A/$seq" --out "$OUT/$sq.json"
  grep -aE "anchors from|LOCKED BAR" "$OUT/$sq.log" 2>/dev/null | sed 's/^/    /'
done
echo "=== H9_UPRIGHT_DONE $(date -Is) ==="
fi
echo "=== RERUN_ALL_DONE $(date -Is) ==="
