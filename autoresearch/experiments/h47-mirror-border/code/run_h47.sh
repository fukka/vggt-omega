#!/usr/bin/env bash
# H47 — H45's mirror test on a 60 deg Aria view, which has no black corners.
set -u
cd /user/f.zhang2/projects/vggt-omega-organized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
TAG="${1:?tag}"; GPU="${2:-0}"; NUMS="${3:-137 138 140 141 142 143}"
export CUDA_VISIBLE_DEVICES="$GPU"
CLEAN=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
EXTRA=/netapp/datasets/f.zhang2/adt_apartment_extra
OUT=autoresearch/experiments/h47-mirror-border/results
mkdir -p "$OUT"
SEQS="$CLEAN/Apartment_release_clean_seq136_M1292"
for n in $NUMS; do SEQS="$SEQS $EXTRA/Apartment_release_clean_seq${n}_M1292"; done
for s in $SEQS; do
  tag=$(basename "$s" | sed -E 's/.*_(seq[0-9]+)_.*/\1/')
  f="$OUT/${TAG}_${tag}.json"
  if [ -s "$f" ]; then echo "[h47] skip $tag"; continue; fi
  echo "[h47] === $tag ==="
  python autoresearch/experiments/h44-mirror-asymmetry/code/mirror_curve.py \
    --seq "$s" --calib cam3r/data/adt_camera_rgb_calibration.json \
    --deltas=0 --max-frames 20 --view-fov 60 --common-theta-deg 28 \
    --models da3:small,da3:large,vggt,vggt_omega --out "$f" \
    || echo "[h47] FAILED $tag"
done
echo "[h47] $TAG done: $(ls "$OUT"/${TAG}_*.json 2>/dev/null | wc -l) files"
