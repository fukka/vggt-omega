#!/usr/bin/env bash
# Fetch every experiment's results from lambda_63 to this machine.
#
# WHY THIS EXISTS
# ---------------
# The pull-before-launch habit on the box is `git stash push -u`, and that
# sweeps up UNTRACKED result JSONs — files that exist nowhere else. On
# 2026-09-08 it swept the same thirteen files twice in one work session. Both
# times they were recoverable from the stash's third parent
# (`git checkout "stash@{0}^3" -- <path>`), and both times the only reason it
# was noticed is that an analysis script honestly reported "no complete scenes"
# instead of averaging over half the data.
#
# The lesson was written down after the first time and violated after the
# second, because the stash line was buried inside a long launch command. So
# the fix belongs in a command, not in a log: RUN THIS BEFORE ANY STASH OR
# PULL ON THE BOX. It costs seconds and makes the box's working tree
# disposable.
set -u
HOST="${HOST:-lambda_63}"
REMOTE="${REMOTE:-/user/f.zhang2/projects/vggt-omega-organized}"
LOCAL="$(cd "$(dirname "$0")/../.." && pwd)"

n=0
for d in $(ssh -o ConnectTimeout=25 "$HOST" \
             "ls -d $REMOTE/autoresearch/experiments/*/results 2>/dev/null"); do
  rel="${d#$REMOTE/}"
  mkdir -p "$LOCAL/$rel"
  if rsync -q "$HOST:$d/"*.json "$LOCAL/$rel/" 2>/dev/null; then
    c=$(ls "$LOCAL/$rel"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "  $rel -> $c json"
    n=$((n + 1))
  fi
done
echo "[fetch] $n result directories synced from $HOST"
