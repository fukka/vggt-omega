#!/usr/bin/env bash
# The teacher-FOV sweep, re-measured on upright input.
# Same code, same protocol, same sequences as the original sweep; only the
# frame the backbone sees has changed. Levels are NOT comparable across rows
# (each FOV scores a different pixel set); read teacher-minus-raw within a row.
set -uo pipefail
REPO=/user/f.zhang2/projects/vggt-omega-organized
A=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
OUT=$REPO/results/autoresearch-h14-upright/sweep
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

tag="roundtrip_seq131"
[ -s "$OUT/$tag/manifest.json" ] || {
  ( CUDA_VISIBLE_DEVICES=$GPU python \
      autoresearch/experiments/h14-rect-distill/code/cache_teacher.py \
      --arm roundtrip --seq "$A/Apartment_release_clean_seq131_M1292" --out "$OUT/$tag" \
      --score-teacher --precheck-only
    echo "MARKER_$tag=$?" ) > "$OUT/$tag.log" 2>&1
  echo "[$tag] $(grep -o "MARKER_$tag=[0-9]*" "$OUT/$tag.log" | tail -1)"
  grep -aE "near_rim|near_center|center\(|far\(" "$OUT/$tag.log" | sed 's/^/    /'
}
echo "=== SWEEP_UPRIGHT_DONE $(date -Is) ==="
