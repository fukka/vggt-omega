#!/usr/bin/env bash
# H39 — gravity-aligned rendering. $1 = backbones, $2 = tag, $3 = GPU id.
set -u
cd /user/f.zhang2/projects/vggt-omega-organized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r

MODELS="${1:?models}"; TAG="${2:?tag}"; GPU="${3:-0}"
# $4 = space-separated sequence numbers from adt_apartment_extra.
# Default is the six-recording roll-spread set H39 was run on.
NUMS="${4:-136 138 142 144 145 149}"
export CUDA_VISIBLE_DEVICES="$GPU"

CLEAN=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
EXTRA=/netapp/datasets/f.zhang2/adt_apartment_extra
CALIB=cam3r/data/adt_camera_rgb_calibration.json
OUT=autoresearch/experiments/h39-gravity-render/results
mkdir -p "$OUT"

# Chosen for roll spread (H17.1 per-sequence): 145 (7.50 deg median |roll|,
# 29.6% beyond 10) and 144 (6.61, 26.6) and 142 (6.46, 20.4) at the high end,
# 136 (2.69, 8.7), 138 (2.94, 5.3) and 149 (2.92, 3.6) at the low end.
# seq136 is the only one under the clean tree; the rest are in the extras.
SEQS=""
for n in $NUMS; do
  if [ "$n" = "136" ]; then
    SEQS="$SEQS $CLEAN/Apartment_release_clean_seq136_M1292"
  else
    SEQS="$SEQS $EXTRA/Apartment_release_clean_seq${n}_M1292"
  fi
done

for s in $SEQS; do
  tag=$(basename "$s" | sed -E 's/.*_(seq[0-9]+)_.*/\1/')
  f="$OUT/${TAG}_${tag}.json"
  if [ -s "$f" ]; then echo "[h39] skip $tag ($TAG)"; continue; fi
  echo "[h39] === $TAG $tag ==="
  python autoresearch/experiments/h39-gravity-render/code/gravity_render.py \
    --seq "$s" --calib "$CALIB" --models "$MODELS" --max-frames 60 \
    --out "$f" || echo "[h39] FAILED $TAG $tag"
done
echo "[h39] $TAG done: $(ls "$OUT"/${TAG}_*.json 2>/dev/null | wc -l) files"
