#!/usr/bin/env bash
# H46 — mirror test on ScanNet++. $1 = scene list, $2 = tag, $3 = GPU.
set -u
cd /user/f.zhang2/projects/vggt-omega-organized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
SCENES="${1:?scenes}"; TAG="${2:?tag}"; GPU="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
ROOT=/netapp/datasets/f.zhang2/scannetpp/data
OUT=autoresearch/experiments/h46-mirror-external/results
mkdir -p "$OUT"
for s in $SCENES; do
  f="$OUT/${TAG}_${s}.json"
  if [ -s "$f" ]; then echo "[h46] skip $s"; continue; fi
  if [ ! -d "$ROOT/$s/dslr/render_depth" ]; then echo "[h46] no depth for $s"; continue; fi
  echo "[h46] === $s ==="
  python autoresearch/experiments/h46-mirror-external/code/mirror_scannetpp.py \
    --scene "$ROOT/$s" --max-frames 20 --out "$f" || echo "[h46] FAILED $s"
done
echo "[h46] $TAG done: $(ls "$OUT"/${TAG}_*.json 2>/dev/null | wc -l) files"
