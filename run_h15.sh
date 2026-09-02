#!/usr/bin/env bash
# H15: lens conditioning decided on a lens the model never saw.
#
# Four arms trained over a family of lenses that all image the same cone into
# the same disc; the decider is stereographic + equisolid, held out. The eval
# scores every arm on every lens (training lenses included) because the
# in-domain row is load-bearing: H12 predicts jac == shuffled there, and
# reproducing that tie is what turns H12 from an unexplained negative into a
# measured statement about identifiability.
set -uo pipefail
REPO=/user/f.zhang2/projects/vggt-omega-organized
A=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
OUT=$REPO/results/autoresearch-h15-lensholdout
GPU=${GPU:-1}
EPOCHS=${EPOCHS:-40}
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
  local t0=$(date +%s)
  echo "[$tag] START $(date -Is)"
  ( CUDA_VISIBLE_DEVICES=$GPU "$@"; echo "MARKER_$tag=$?" ) > "$log" 2>&1
  echo "[$tag] $(grep -o "MARKER_$tag=[0-9]*" "$log" | tail -1) after $(( $(date +%s)-t0 ))s"
}

for arm in jac mismatched shuffled none; do
  [ -s "$OUT/$arm/cond_last.pt" ] && { echo "[train_$arm] skip"; continue; }
  step "train_$arm" "$OUT/$arm.log" \
    python autoresearch/experiments/h15-lens-holdout/code/train_multilens.py \
      --arm "$arm" --train-seqs "$TRAIN_SEQS" --epochs "$EPOCHS" \
      --seed 0 --out-dir "$OUT/$arm"
done

for arm in jac mismatched shuffled none; do
  for seq in Apartment_release_clean_seq136_M1292 Apartment_release_decoration_seq132_M1292; do
    case "$seq" in *seq136*) sq=seq136;; *) sq=seq132;; esac
    tag="eval_${arm}_${sq}"
    [ -s "$OUT/$tag.json" ] && { echo "[$tag] skip"; continue; }
    step "$tag" "$OUT/$tag.log" \
      python autoresearch/experiments/h15-lens-holdout/code/eval_lens.py \
        --seq "$A/$seq" --ckpt "$OUT/$arm/cond_last.pt" --out "$OUT/$tag.json"
  done
done

echo "=== H15_DONE $(date -Is) ==="
