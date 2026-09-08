#!/usr/bin/env bash
# H41 — residual-roll curve. $1 = deltas (must include 0), $2 = tag, $3 = GPU.
set -u
cd /user/f.zhang2/projects/vggt-omega-organized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
DELTAS="${1:?deltas}"; TAG="${2:?tag}"; GPU="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
CLEAN=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
EXTRA=/netapp/datasets/f.zhang2/adt_apartment_extra
CALIB=cam3r/data/adt_camera_rgb_calibration.json
OUT=autoresearch/experiments/h41-residual-curve/results
mkdir -p "$OUT"
SEQS="$CLEAN/Apartment_release_clean_seq136_M1292"
for n in 137 138 140 141 142 143 144 145 146 147 148 149; do
  SEQS="$SEQS $EXTRA/Apartment_release_clean_seq${n}_M1292"
done
for s in $SEQS; do
  tag=$(basename "$s" | sed -E 's/.*_(seq[0-9]+)_.*/\1/')
  f="$OUT/${TAG}_${tag}.json"
  if [ -s "$f" ]; then echo "[h41] skip $tag ($TAG)"; continue; fi
  echo "[h41] === $TAG $tag ==="
  python autoresearch/experiments/h41-residual-curve/code/residual_curve.py \
    --seq "$s" --calib "$CALIB" --deltas="$DELTAS" --max-frames 20 \
    --out "$f" || echo "[h41] FAILED $TAG $tag"
done
echo "[h41] $TAG done: $(ls "$OUT"/${TAG}_*.json 2>/dev/null | wc -l) files"
