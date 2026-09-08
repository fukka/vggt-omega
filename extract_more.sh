#!/bin/bash
# Extract the Apartment sequences whose depth VRS is on disk but never unpacked.
#
# The downloader left 13 more `clean` sequences (seq137-150) with video.vrs and
# depth/depth_images.vrs but no depth_npy. `tools/extract_adt_sequence.py` wants
# the raw download layout (everything flat in one directory), while these live
# in the repo's `_clean` layout, so a directory of symlinks is built per
# sequence -- no copying, and the provider reads the real VRS.
#
# Verified before running: the extractor reproduces seq131's existing extraction
# byte-for-byte through this same symlink layout (`--verify-against`, 8/8).
#
# stride 4 keeps ~715 frames per sequence (30 fps -> 7.5 fps, one frame every
# 0.13 s) at ~6.3 MB/frame; 13 sequences is ~58 GB, which is why this writes to
# /netapp rather than the box's root filesystem (185 GB free).
set -uo pipefail
REPO=/user/f.zhang2/projects/vggt-omega-organized
C=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
DEST=${DEST:-/netapp/datasets/f.zhang2/adt_apartment_extra}
STRIDE=${STRIDE:-4}
SEQS=${SEQS:-"seq137 seq138 seq140 seq141 seq142 seq143 seq144 seq145 seq146 seq147 seq148 seq149 seq150"}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myadt
cd "$REPO" || exit 2
export PYTHONUNBUFFERED=1
mkdir -p "$DEST"

for s in $SEQS; do
  n=Apartment_release_clean_${s}_M1292
  src=$C/$n
  [ -f "$src/depth/depth_images.vrs" ] || { echo "[$s] no depth vrs, skip"; continue; }
  [ -d "$DEST/$n/depth_npy" ] && [ "$(ls "$DEST/$n/depth_npy" | wc -l)" -gt 100 ] && { echo "[$s] done, skip"; continue; }
  R=/tmp/adt_raw_$s; rm -rf "$R"; mkdir -p "$R"
  ln -s "$src/video.vrs" "$R/video.vrs"
  ln -s "$src/depth/depth_images.vrs" "$R/depth_images.vrs"
  for f in "$src"/groundtruth/*; do ln -s "$f" "$R/$(basename "$f")"; done
  [ -d "$src/mps" ] && ln -s "$src/mps" "$R/mps"
  t0=$(date +%s)
  echo "[$s] START $(date -Is)"
  python tools/extract_adt_sequence.py --seq "$R" --out "$DEST/$n" --stride "$STRIDE" \
    > "$DEST/$n.log" 2>&1
  echo "[$s] exit=$? frames=$(ls "$DEST/$n/depth_npy" 2>/dev/null | wc -l) after $(( $(date +%s)-t0 ))s"
  rm -rf "$R"
done
echo "EXTRACT_DONE"
df -h /netapp/datasets | tail -1
