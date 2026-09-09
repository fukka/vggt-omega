#!/usr/bin/env bash
# H48 reciprocal arm — Aria content through ScanNet++'s lens. $1 = tag, $2 = GPU.
set -u
cd /user/f.zhang2/projects/vggt-omega-organized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
TAG="${1:?tag}"; GPU="${2:-0}"; NUMS="${3:-137 138 140 141 142 143}"
export CUDA_VISIBLE_DEVICES="$GPU"
CLEAN=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
EXTRA=/netapp/datasets/f.zhang2/adt_apartment_extra
# One ScanNet++ scene, used ONLY for its camera. Picked here rather than
# interpolated by the caller — see run_h48.sh's `auto` note.
LENS=/netapp/datasets/f.zhang2/scannetpp/data/$(ls /netapp/datasets/f.zhang2/scannetpp/data | head -1)
OUT=autoresearch/experiments/h48-content-vs-pipeline/results
mkdir -p "$OUT"
SEQS="$CLEAN/Apartment_release_clean_seq136_M1292"
for n in $NUMS; do SEQS="$SEQS $EXTRA/Apartment_release_clean_seq${n}_M1292"; done
for s in $SEQS; do
  tag=$(basename "$s" | sed -E 's/.*_(seq[0-9]+)_.*/\1/')
  f="$OUT/${TAG}_${tag}.json"
  [ -s "$f" ] && { echo "[h48r] skip $tag"; continue; }
  echo "[h48r] === $tag ==="
  python autoresearch/experiments/h48-content-vs-pipeline/code/mirror_aria2sca.py \
    --seq "$s" --lens-scene "$LENS" --max-frames 20 --out "$f" \
    || echo "[h48r] FAILED $tag"
done
echo "[h48r] $TAG done: $(ls "$OUT"/${TAG}_*.json 2>/dev/null | wc -l) files"
