#!/usr/bin/env bash
# Ticket 024 part B: give the pod the four A100s can only earn if it can rebuild
# the #019 split (digest 601fcb22767e) from its own --adt-root.
#
# WHAT THE POD IS ACTUALLY MISSING. It already holds depth_npy + videos_rgb for
# the five `clean` sequences and videos_rgb for `decoration_seq132`. What it has
# not got is:
#   clean 133/134/135/136   videos_synthetic   ~575 MB each
#   decoration_132          videos_synthetic   512 MB
#   decoration_132          depth_npy          11 GB whole -- see below
#
# WHY THE DEPTH IS SENT AS A SUBSET AND WHY THAT IS NOT A SHORTCUT.
# The split is the intersection of the three streams' frame ids, so a depth
# frame whose id is absent from videos_synthetic can never enter it. Sending
# only the depth files whose id appears in decoration_132's 400 synthetic frames
# is therefore intersection-preserving by construction, and takes 11 GB down to
# ~1.6 GB. It is not a judgement call: the digest gate on the pod either returns
# 601fcb22767e or the runs do not start.
#
# Bulk data goes over `space storage`, not scp: the SSH chain to the pod drops
# mid-command (#023), and one dropped 14 GB copy costs more than every retry
# here put together. One invocation per directory, so a drop costs one directory.
set -uo pipefail
SRC=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
TMP=/data/f.zhang2/adt024-stage
DST="file://groups/SR-TORAIC-IVU/Fengjia/data/adt-024-stage"
D132=Apartment_release_decoration_seq132_M1292

echo "=== building the decoration_132 depth subset"
mkdir -p "$TMP/$D132/depth_npy"
python3 - <<PY
import os, re, glob, shutil
src = "$SRC/$D132"
dst = "$TMP/$D132/depth_npy"
rx  = re.compile(r"frame[_-]?(\d+)")
def fid(p):
    m = rx.search(os.path.basename(p))
    return str(int(m.group(1))) if m else None
want = {fid(p) for p in glob.glob(os.path.join(src, "videos_synthetic", "*.png"))}
want.discard(None)
n = 0
for p in glob.glob(os.path.join(src, "depth_npy", "*.npy")):
    if fid(p) in want:
        q = os.path.join(dst, os.path.basename(p))
        if not os.path.exists(q):
            shutil.copy2(p, q)
        n += 1
print(f"  synthetic ids: {len(want)}  depth files kept: {n}")
PY

up () {  # up <local dir> <remote subpath>
  echo "=== $2  ($(du -sh "$1" | cut -f1), $(ls "$1" | wc -l) files)"
  space storage upload file "$1" "$DST/$2" --max-request-processes 16 --attempts 23 \
    || { echo "!!! FAILED: $2" >&2; return 1; }
}

fail=0
for s in 133 134 135 136; do
  d=Apartment_release_clean_seq${s}_M1292
  up "$SRC/$d/videos_synthetic" "$d/videos_synthetic" || fail=$((fail+1))
done
up "$SRC/$D132/videos_synthetic" "$D132/videos_synthetic" || fail=$((fail+1))
up "$TMP/$D132/depth_npy"        "$D132/depth_npy"        || fail=$((fail+1))

echo
echo "=== verifying file counts on the far side (a truncated dir passes silently otherwise)"
bad=0
check () {  # check <local dir> <remote subpath>
  local want got
  want=$(ls "$1" | wc -l)
  got=$(space storage list file "$DST/$2" 2>/dev/null | grep -c "| file  *|")
  if [ "$want" = "$got" ]; then echo "  ok   $2  $got"
  else echo "  SHORT $2  want $want got $got"; bad=$((bad+1)); fi
}
for s in 133 134 135 136; do
  d=Apartment_release_clean_seq${s}_M1292
  check "$SRC/$d/videos_synthetic" "$d/videos_synthetic"
done
check "$SRC/$D132/videos_synthetic" "$D132/videos_synthetic"
check "$TMP/$D132/depth_npy"        "$D132/depth_npy"
echo "=== [stage024] exit uploads_failed=$fail short_dirs=$bad"
