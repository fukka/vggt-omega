#!/usr/bin/env bash
# H51 — the mirror test on real vs rendered frames of the SAME recordings.
# $1 = GPU. Both arms run back to back per sequence so a partial run still has
# matched pairs rather than five real arms and no synthetic ones.
set -u
cd /user/f.zhang2/projects/vggt-omega-organized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
C=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
OUT=autoresearch/experiments/h51-imagery-vs-capture/results
mkdir -p "$OUT"
# The five sequences that ship Blender renders; verified three-way overlap of
# videos_synthetic / videos_rgb / depth_npy (protocol.md).
for n in 131 133 134 135 136; do
  s=$C/Apartment_release_clean_seq${n}_M1292
  for arm in videos_rgb videos_synthetic; do
    tag=$([ "$arm" = "videos_rgb" ] && echo real || echo synth)
    f="$OUT/${tag}_seq${n}.json"
    [ -s "$f" ] && { echo "[h51] skip $tag seq$n"; continue; }
    echo "[h51] === $tag seq$n ==="
    python autoresearch/experiments/h51-imagery-vs-capture/code/mirror_synth.py \
      --seq "$s" --calib cam3r/data/adt_camera_rgb_calibration.json \
      --rgb-subdir "$arm" --max-frames 20 --out "$f" \
      || echo "[h51] FAILED $tag seq$n"
  done
done
echo "[h51] done: $(ls "$OUT"/*.json 2>/dev/null | wc -l) files"
