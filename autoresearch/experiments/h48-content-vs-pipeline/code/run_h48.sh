#!/usr/bin/env bash
set -u
cd /user/f.zhang2/projects/vggt-omega-organized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
SCENES="${1:?scenes}"; TAG="${2:?tag}"; GPU="${3:-0}"
# `auto` = pick the scenes here rather than have the caller interpolate a list
# into a nested quoted string. SPACE_CONTAINER.md's "never build a long command
# over this link" applies to lambda_63 too: a scene list expanded inside a
# tmux command inside an ssh command is three levels of quoting and it broke on
# the first dry run of resume_after_outage.sh.
if [ "$SCENES" = "auto" ]; then
  SCENES=$(ls /netapp/datasets/f.zhang2/scannetpp/data | head -8 | tr '\n' ' ')
fi
export CUDA_VISIBLE_DEVICES="$GPU"
ROOT=/netapp/datasets/f.zhang2/scannetpp/data
OUT=autoresearch/experiments/h48-content-vs-pipeline/results
mkdir -p "$OUT"
for s in $SCENES; do
  f="$OUT/${TAG}_${s}.json"
  [ -s "$f" ] && { echo "[h48] skip $s"; continue; }
  [ -d "$ROOT/$s/dslr/render_depth" ] || { echo "[h48] no depth $s"; continue; }
  echo "[h48] === $s ==="
  python autoresearch/experiments/h48-content-vs-pipeline/code/mirror_sca2aria.py \
    --scene "$ROOT/$s" --max-frames 20 --out "$f" || echo "[h48] FAILED $s"
done
echo "[h48] $TAG done: $(ls "$OUT"/${TAG}_*.json 2>/dev/null | wc -l) files"
