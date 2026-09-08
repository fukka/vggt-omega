#!/usr/bin/env bash
# H38 — de-rotation arms. $1 = comma list of backbones, $2 = tag, $3 = GPU id.
set -u
cd /user/f.zhang2/projects/vggt-omega-organized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r

MODELS="${1:?models}"; TAG="${2:?tag}"; GPU="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

CLEAN=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
EXTRA=/netapp/datasets/f.zhang2/adt_apartment_extra
OUT=autoresearch/experiments/h38-derotate/results
mkdir -p "$OUT"

SEQS="$CLEAN/Apartment_release_clean_seq136_M1292"
for n in 137 138 140 141 142; do
  SEQS="$SEQS $EXTRA/Apartment_release_clean_seq${n}_M1292"
done

for s in $SEQS; do
  tag=$(basename "$s" | sed -E 's/.*_(seq[0-9]+)_.*/\1/')
  f="$OUT/${TAG}_${tag}.json"
  if [ -s "$f" ]; then echo "[h38] skip $tag ($TAG)"; continue; fi
  echo "[h38] === $TAG $tag ==="
  python autoresearch/experiments/h38-derotate/code/derotate.py \
    --seq "$s" --models "$MODELS" --angles=0,30,-30 --max-frames 20 \
    --out "$f" || echo "[h38] FAILED $TAG $tag"
done
echo "[h38] $TAG done: $(ls "$OUT"/${TAG}_*.json 2>/dev/null | wc -l) of 6"
