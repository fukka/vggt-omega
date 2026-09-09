#!/usr/bin/env bash
# One command to restart work on lambda_63 after an outage, in the order that
# does not lose anything.
#
# WHY THE ORDER MATTERS
# ---------------------
# Every step here is a mistake already made in this session:
#
#   1. FETCH FIRST. `git stash push -u` sweeps untracked result JSONs — files
#      that exist nowhere else. It swept the same thirteen twice on 2026-09-08.
#      Recoverable from the stash's third parent, but only if you notice.
#   2. Stash, then pull. The box's tree collects untracked outputs that collide
#      with what was committed from the Mac; a plain pull aborts.
#   3. RESTORE from `stash@{0}^3` — that is where `-u` puts untracked files.
#   4. Only then launch, and launch into a NAMED TMUX session with the marker
#      and the command wrapped in a subshell before the pipe:
#         ( CMD; echo MARKER=$? ) 2>&1 | tee log
#      `CMD | tee log; echo $?` writes the marker only to the pane and reports
#      tee's status, so a crashed job looks like a clean one.
#
# Usage:  bash autoresearch/tools/resume_after_outage.sh [--dry-run]
set -u
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
HOST=lambda_63
REMOTE=/user/f.zhang2/projects/vggt-omega-organized
LOCAL="$(cd "$(dirname "$0")/../.." && pwd)"

run() { if [ "$DRY" = 1 ]; then echo "DRY: $*"; else eval "$@"; fi; }

echo "== 1. reachable?"
if [ "$DRY" = 0 ] && ! ssh -o ConnectTimeout=25 "$HOST" 'echo up' 2>/dev/null | grep -q up; then
  echo "   $HOST still down — nothing else is safe to do."; exit 1
fi

echo "== 2. fetch results BEFORE touching the stash"
run "bash '$LOCAL/autoresearch/tools/fetch_results.sh'"

echo "== 3. stash + pull + restore untracked from stash@{0}^3"
run "ssh -o ConnectTimeout=25 $HOST 'cd $REMOTE && \
  git stash push -u -m resume-\$(date +%s) >/dev/null 2>&1; \
  git pull --ff-only 2>&1 | tail -1; \
  git checkout \"stash@{0}^3\" -- autoresearch/experiments/ 2>/dev/null; \
  git log --oneline -1'"

echo "== 4. launch H48 (both directions of the 2x2) on the two GPUs"
run "ssh -o ConnectTimeout=25 $HOST 'cd $REMOTE && \
  tmux new-session -d -s h48a -c $REMOTE \"( bash autoresearch/experiments/h48-content-vs-pipeline/code/run_h48.sh auto A 0; echo MARKER_H48A=\\\$? ) 2>&1 | tee /tmp/h48a.log; exec bash\" && \
  tmux ls | grep h48'"

echo
echo "Then: check /tmp/h48a.log for the per-scene lines, and remember the"
echo "reciprocal arm (Aria content -> ScanNet++ lens) still needs writing —"
echo "the protocol has it, the script does not."
