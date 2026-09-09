#!/usr/bin/env bash
# H50 — H48's 2x2 on HELD-OUT data. $1 = cell (aa|r|a|ss), $2 = GPU.
# Scene and sequence lists are built HERE, never interpolated by the caller --
# run_h48.sh's note about three levels of quoting over ssh applies unchanged.
set -u
cd /user/f.zhang2/projects/vggt-omega-organized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
CELL="${1:?cell}"; GPU="${2:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
ROOT=/netapp/datasets/f.zhang2/scannetpp/data
EXTRA=/netapp/datasets/f.zhang2/adt_apartment_extra
OUT=autoresearch/experiments/h50-retention-replication/results
mkdir -p "$OUT"

# Held out: H48 used ADT seq136-143 and the FIRST EIGHT scenes with depth.
SEQS=""
for n in 144 145 146 147 148 149 150; do
  SEQS="$SEQS $EXTRA/Apartment_release_clean_seq${n}_M1292"
done
SCENES=$(for s in $(ls "$ROOT"); do
           [ -d "$ROOT/$s/dslr/render_depth" ] && echo "$s"
         done | sed -n '9,16p' | tr '\n' ' ')

case "$CELL" in
  r)   for s in $SEQS; do
         tag=$(basename "$s" | sed -E 's/.*_(seq[0-9]+)_.*/\1/')
         f="$OUT/r_${tag}.json"; [ -s "$f" ] && { echo "[h50] skip r $tag"; continue; }
         echo "[h50] === r $tag ==="
         LENS=$ROOT/$(echo $SCENES | awk '{print $1}')
         python autoresearch/experiments/h48-content-vs-pipeline/code/mirror_aria2sca.py \
           --seq "$s" --lens-scene "$LENS" --max-frames 20 --out "$f" \
           || echo "[h50] FAILED r $tag"
       done ;;
  aa)  for s in $SEQS; do
         tag=$(basename "$s" | sed -E 's/.*_(seq[0-9]+)_.*/\1/')
         f="$OUT/aa_${tag}.json"; [ -s "$f" ] && { echo "[h50] skip aa $tag"; continue; }
         echo "[h50] === aa $tag ==="
         python autoresearch/experiments/h44-mirror-asymmetry/code/mirror_curve.py \
           --seq "$s" --calib cam3r/data/adt_camera_rgb_calibration.json \
           --deltas=0 --max-frames 20 --view-fov 60 --common-theta-deg 28 \
           --models da3:small,da3:large,vggt,vggt_omega --out "$f" \
           || echo "[h50] FAILED aa $tag"
       done ;;
  a)   for s in $SCENES; do
         f="$OUT/a_${s}.json"; [ -s "$f" ] && { echo "[h50] skip a $s"; continue; }
         echo "[h50] === a $s ==="
         python autoresearch/experiments/h48-content-vs-pipeline/code/mirror_sca2aria.py \
           --scene "$ROOT/$s" --max-frames 20 --out "$f" || echo "[h50] FAILED a $s"
       done ;;
  ss)  for s in $SCENES; do
         f="$OUT/ss_${s}.json"; [ -s "$f" ] && { echo "[h50] skip ss $s"; continue; }
         echo "[h50] === ss $s ==="
         python autoresearch/experiments/h46-mirror-external/code/mirror_scannetpp.py \
           --scene "$ROOT/$s" --max-frames 20 --out "$f" || echo "[h50] FAILED ss $s"
       done ;;
  *)   echo "unknown cell: $CELL"; exit 2 ;;
esac
echo "[h50] $CELL done: $(ls "$OUT"/${CELL}_*.json 2>/dev/null | wc -l) files"
