#!/usr/bin/env bash
# H37 — border cost for vggt and vggt_omega on the same fourteen recordings
# H29 used for da3:small and da3:large. Evaluation only, level camera.
set -u
cd /user/f.zhang2/projects/vggt-omega-organized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r

CLEAN=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
EXTRA=/netapp/datasets/f.zhang2/adt_apartment_extra
OUT=autoresearch/experiments/h37-border-four-backbones/results
mkdir -p "$OUT"

SEQS="$CLEAN/Apartment_release_clean_seq136_M1292"
for n in 137 138 140 141 142 143 144 145 146 147 148 149 150; do
  SEQS="$SEQS $EXTRA/Apartment_release_clean_seq${n}_M1292"
done

for s in $SEQS; do
  tag=$(basename "$s" | sed -E 's/.*_(seq[0-9]+)_.*/\1/')
  f="$OUT/models_${tag}.json"
  if [ -s "$f" ]; then echo "[h37] skip $tag (exists)"; continue; fi
  echo "[h37] === $tag ==="
  python autoresearch/experiments/h17-roll-prior/code/roll_boundary.py \
    --seq "$s" --models vggt,vggt_omega --angles 0 --max-frames 20 \
    --out "$f" || echo "[h37] FAILED $tag"
done
echo "[h37] done: $(ls "$OUT"/models_*.json 2>/dev/null | wc -l) of 14"
