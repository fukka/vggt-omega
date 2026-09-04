#!/usr/bin/env bash
# Two follow-ups that the upright re-run makes worth doing.
#
# A. CROSS-ROOM. Every number this project has is from one apartment. LiteOffice
#    is ADT's other scene -- another room, another device, the same dense GT --
#    and it is scored through its OWN lens via camera.json. This is the direct
#    test of "did the method learn the fisheye or the apartment".
#
# B. DATA SCALING. 17,027 frames carry both RGB and depth across the six
#    sequences and every run so far used 360 of them, 2.1%. The 60-per-sequence
#    cap came from H5 and has never been chosen against anything. If held-out
#    performance is still improving at 600, then "use the data we have" is a
#    cheaper answer to the overfitting question than "get another scene".
set -uo pipefail
REPO=/user/f.zhang2/projects/vggt-omega-organized
A=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
LITE=/user/f.zhang2/Documents/adt_liteoffice_extracted
H14=$REPO/results/autoresearch-h14-upright
OUT=$REPO/results/autoresearch-followups
GPU=${GPU:-1}
STAGE=${STAGE:-all}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
cd "$REPO" || exit 2
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"

TRAIN_SEQS=""
for s in seq131 seq133 seq134 seq135; do
  TRAIN_SEQS="${TRAIN_SEQS:+$TRAIN_SEQS,}$A/Apartment_release_clean_${s}_M1292"
done

step () {
  local tag=$1 log=$2; shift 2
  [ -n "${SKIP_IF:-}" ] && [ -s "$SKIP_IF" ] && { echo "[$tag] skip"; return; }
  local t0=$(date +%s); echo "[$tag] START $(date -Is)"
  ( CUDA_VISIBLE_DEVICES=$GPU "$@"; echo "MARKER_$tag=$?" ) > "$log" 2>&1
  echo "[$tag] $(grep -o "MARKER_$tag=[0-9]*" "$log" | tail -1) after $(( $(date +%s)-t0 ))s"
}

if [ "$STAGE" = all ] || [ "$STAGE" = cross ]; then
  echo "### A. cross-room: the apartment-trained students, on LiteOffice"
  for d in "$LITE"/*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    for arm in rect gt; do
      SKIP_IF="$OUT/cross_${arm}_${n}.json" step "cross_${arm}_${n}" "$OUT/cross_${arm}_${n}.log" \
        python autoresearch/experiments/h5-rim-finetune/code/eval_lora.py \
          --seq "$d" --lora "$H14/$arm/lora_last.pt" --out "$OUT/cross_${arm}_${n}.json"
      grep -aE "near_rim|near_center|center\(|far\(" "$OUT/cross_${arm}_${n}.log" 2>/dev/null | sed "s/^/    /"
    done
  done
  echo "=== CROSS_DONE $(date -Is) ==="
fi

if [ "$STAGE" = all ] || [ "$STAGE" = scale ]; then
  echo "### B. data scaling: the labelled arm at 60 / 240 / 600 frames per sequence"
  for nf in 60 240 600; do
    SKIP_IF="$OUT/scale_$nf/lora_last.pt" step "train_scale_$nf" "$OUT/scale_$nf.log" \
      python autoresearch/experiments/h14-rect-distill/code/train_student.py \
        --arm gt --train-seqs "$TRAIN_SEQS" --epochs 20 --seed 0 \
        --max-frames "$nf" --out-dir "$OUT/scale_$nf"
    for seq in Apartment_release_clean_seq136_M1292 Apartment_release_decoration_seq132_M1292; do
      case "$seq" in *seq136*) sq=seq136;; *) sq=seq132;; esac
      SKIP_IF="$OUT/scale_${nf}_${sq}.json" step "eval_scale_${nf}_${sq}" "$OUT/scale_${nf}_${sq}.log" \
        python autoresearch/experiments/h5-rim-finetune/code/eval_lora.py \
          --seq "$A/$seq" --lora "$OUT/scale_$nf/lora_last.pt" --out "$OUT/scale_${nf}_${sq}.json"
    done
  done
  echo "=== SCALE_DONE $(date -Is) ==="
fi
echo "=== FOLLOWUPS_DONE $(date -Is) ==="
