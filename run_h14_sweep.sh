#!/usr/bin/env bash
# H14 diagnostic: how wide can we rectify before the teacher gets WORSE?
#
# The pre-check fired at 110 deg -- the cone-covering pinhole -- with the
# teacher worse than the raw model in every zone and +107% at the near centre.
# Two explanations are live and this sweep separates them, because the corner
# ray of a square pinhole is atan(sqrt(2) tan(fov/2)) and Aria's cone ends at
# 54.68 deg:
#
#   fov  on-axis reach  corner ray  frame content   cone coverage
#    85     42.3          52.1       100% real       partial
#    89     44.5          54.2       100% real       partial
#    95     47.5          57.0       small black     partial
#   110     55.0          63.7       21.5% black     FULL
#
# So the sweep runs from "filled frame, cannot see the rim" to "sees the whole
# rim, large black vignette". If the teacher only degrades once the corners go
# black, the problem is an unseen image statistic and a MULTI-VIEW teacher
# fixes it. If it degrades smoothly with FOV from 85 upward, then wide
# rectilinear rendering is itself the problem and H14 needs the multi-view form
# or nothing.
#
# Levels are NOT comparable across rows (each FOV scores a different pixel
# set); only teacher-minus-raw within a row is.
set -uo pipefail
REPO=/user/f.zhang2/projects/vggt-omega-organized
A=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
OUT=$REPO/results/autoresearch-h14-rectdistill/sweep
GPU=${GPU:-0}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
cd "$REPO" || exit 2
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"

for seq in Apartment_release_clean_seq131_M1292 Apartment_release_clean_seq136_M1292; do
  case "$seq" in *seq131*) sq=seq131;; *) sq=seq136;; esac
  for fov in 85 89 95 110; do
    tag="fov${fov}_${sq}"
    [ -s "$OUT/$tag/manifest.json" ] && { echo "[$tag] skip"; continue; }
    t0=$(date +%s)
    ( CUDA_VISIBLE_DEVICES=$GPU python \
        autoresearch/experiments/h14-rect-distill/code/cache_teacher.py \
        --arm rect --seq "$A/$seq" --out "$OUT/$tag" \
        --teacher-fov "$fov" --score-teacher --precheck-only \
        --allow-partial-coverage
      echo "MARKER_$tag=$?" ) > "$OUT/$tag.log" 2>&1
    echo "[$tag] $(grep -o "MARKER_$tag=[0-9]*" "$OUT/$tag.log" | tail -1) after $(( $(date +%s)-t0 ))s"
    grep -aE "pinhole frame is|cone coverage|near_rim|near_center|center\(|far\(" "$OUT/$tag.log" | sed 's/^/    /'
  done
done

# The resampling cost on its own: same two resamplings, no change of projection.
for seq in Apartment_release_clean_seq131_M1292; do
  tag="roundtrip_seq131"
  [ -s "$OUT/$tag/manifest.json" ] || {
    ( CUDA_VISIBLE_DEVICES=$GPU python \
        autoresearch/experiments/h14-rect-distill/code/cache_teacher.py \
        --arm roundtrip --seq "$A/$seq" --out "$OUT/$tag" \
        --score-teacher --precheck-only
      echo "MARKER_$tag=$?" ) > "$OUT/$tag.log" 2>&1
    echo "[$tag] $(grep -o "MARKER_$tag=[0-9]*" "$OUT/$tag.log" | tail -1)"
    grep -aE "near_rim|near_center|center\(|far\(" "$OUT/$tag.log" | sed 's/^/    /'
  }
done

echo "=== H14_SWEEP_DONE $(date -Is) ==="
