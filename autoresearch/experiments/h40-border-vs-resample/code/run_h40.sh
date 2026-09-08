#!/usr/bin/env bash
# H40 — border x resampling 2x2. $1 = backbones, $2 = tag, $3 = GPU.
set -u
cd /user/f.zhang2/projects/vggt-omega-organized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
MODELS="${1:?models}"; TAG="${2:?tag}"; GPU="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
CLEAN=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
EXTRA=/netapp/datasets/f.zhang2/adt_apartment_extra
OUT=autoresearch/experiments/h40-border-vs-resample/results
mkdir -p "$OUT"
SEQS="$CLEAN/Apartment_release_clean_seq136_M1292"
for n in 137 138 140 141 142 143 144 145 146 147 148 149 150; do
  SEQS="$SEQS $EXTRA/Apartment_release_clean_seq${n}_M1292"
done
for s in $SEQS; do
  tag=$(basename "$s" | sed -E 's/.*_(seq[0-9]+)_.*/\1/')
  f="$OUT/${TAG}_${tag}.json"
  if [ -s "$f" ]; then echo "[h40] skip $tag ($TAG)"; continue; fi
  echo "[h40] === $TAG $tag ==="
  python autoresearch/experiments/h40-border-vs-resample/code/border_vs_resample.py \
    --seq "$s" --models "$MODELS" --max-frames 20 --out "$f" \
    || echo "[h40] FAILED $TAG $tag"
done
echo "[h40] $TAG done: $(ls "$OUT"/${TAG}_*.json 2>/dev/null | wc -l) files"
