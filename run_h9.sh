#!/usr/bin/env bash
# H9 RayCal-TTA on every ADT sequence.
#
# This is test-time adaptation, so there is no train/test split to respect:
# every sequence adapts to itself from its own video, with NO depth labels. The
# two held-out sequences are still reported first because that is where the
# rest of the project's numbers live.
#
# What to read: not the AbsRel, the GAP. The locked bar from the hypothesis
# registration is that the scale_shift <-> frozen-affine gap must COLLAPSE --
# ~82% of the near-rim penalty was measured to be the eval affine's placement,
# so a method that merely lets the per-frame affine sit better has changed
# nothing about the geometry.
set -uo pipefail
REPO=/user/f.zhang2/projects/vggt-omega-organized
A=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
OUT=$REPO/results/autoresearch-h9-raycal
GPU=${GPU:-1}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
cd "$REPO" || exit 2
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"

for seq in Apartment_release_clean_seq136_M1292 \
           Apartment_release_decoration_seq132_M1292 \
           Apartment_release_clean_seq131_M1292 \
           Apartment_release_clean_seq133_M1292 \
           Apartment_release_clean_seq134_M1292 \
           Apartment_release_clean_seq135_M1292; do
  sq=$(echo "$seq" | sed 's/.*release_//; s/_M1292//')
  [ -s "$OUT/$sq.json" ] && { echo "[$sq] skip"; continue; }
  t0=$(date +%s)
  ( CUDA_VISIBLE_DEVICES=$GPU python autoresearch/experiments/h9-raycal-tta/code/run_h9.py \
      --seq "$A/$seq" --out "$OUT/$sq.json"
    echo "MARKER_$sq=$?" ) > "$OUT/$sq.log" 2>&1
  echo "[$sq] $(grep -o "MARKER_$sq=[0-9]*" "$OUT/$sq.log" | tail -1) after $(( $(date +%s)-t0 ))s"
  grep -aE "anchors from|^\[h9/|LOCKED BAR" "$OUT/$sq.log" | sed 's/^/    /'
done

echo "=== H9_DONE $(date -Is) ==="
