#!/usr/bin/env bash
# Assemble a dedicated --adt-root for ticket 024 part B out of what the pod has
# and what was just staged, and then GATE ON THE DIGEST.
#
# A dedicated root rather than repairing the pod's existing trees, for two
# reasons. The pod's projectaria_tools_adt_data_clean holds 18 sequences of
# which 17 have no synthetic stream; a root with exactly the six the #019 split
# uses cannot silently acquire a seventh later. And nothing here mutates a tree
# another project reads -- every entry is a symlink.
#
# Sources differ per sequence, which is the whole reason this script exists:
#   clean 131        all three streams already on the pod
#   clean 133-136    depth+rgb on the pod, synthetic staged from lambda
#   decoration 132   rgb lives in the OTHER pod tree (…_adt_data, not …_clean),
#                    depth and synthetic staged from lambda
set -uo pipefail
CLEAN=/group-volume/Fengjia/data/projectaria_tools_adt_data_clean
RAW=/group-volume/Fengjia/data/projectaria_tools_adt_data
STAGE=/group-volume/Fengjia/data/adt-024-stage
ROOT=/group-volume/Fengjia/data/adt-024-6seq
D132=Apartment_release_decoration_seq132_M1292
WANT=601fcb22767e

rm -rf "$ROOT"; mkdir -p "$ROOT"
link () { ln -sfn "$1" "$2"; }

d=Apartment_release_clean_seq131_M1292
mkdir -p "$ROOT/$d"
for s in depth_npy videos_rgb videos_synthetic; do link "$CLEAN/$d/$s" "$ROOT/$d/$s"; done

for n in 133 134 135 136; do
  d=Apartment_release_clean_seq${n}_M1292
  mkdir -p "$ROOT/$d"
  link "$CLEAN/$d/depth_npy"        "$ROOT/$d/depth_npy"
  link "$CLEAN/$d/videos_rgb"       "$ROOT/$d/videos_rgb"
  link "$STAGE/$d/videos_synthetic" "$ROOT/$d/videos_synthetic"
done

mkdir -p "$ROOT/$D132"
link "$STAGE/$D132/depth_npy"        "$ROOT/$D132/depth_npy"
link "$RAW/$D132/videos_rgb"         "$ROOT/$D132/videos_rgb"
link "$STAGE/$D132/videos_synthetic" "$ROOT/$D132/videos_synthetic"

echo "=== per-sequence file counts"
for d in "$ROOT"/*/; do
  echo "  $(basename "$d"): depth=$(ls "$d/depth_npy" 2>/dev/null | wc -l) rgb=$(ls "$d/videos_rgb" 2>/dev/null | wc -l) synth=$(ls "$d/videos_synthetic" 2>/dev/null | wc -l)"
done

echo "=== digest gate"
source /group-volume/Fengjia/envs/vggt360-py312/bin/activate
cd /group-volume/Fengjia/projects/vggt-omega-024
ADT="$ROOT" python - <<'PY'
import os
from fovbench.split import build_split
sp = build_split(os.environ["ADT"], n_frames=50, verbose=False)
print("digest", sp.digest, len(sp.frames))
for s in sorted({f.seq for f in sp.frames}):
    print("   ", s, sum(1 for f in sp.frames if f.seq == s))
PY
