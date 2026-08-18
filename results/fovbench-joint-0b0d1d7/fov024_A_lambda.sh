#!/usr/bin/env bash
# Ticket 024 Part A -- the joint (incidence x GT depth) table on the #019 split.
#
# WHY THIS RUNS ON LAMBDA AND NOT THE POD, WHICH HAS FOUR IDLE A100s:
# part A's done-when is "byte-identical to #019's non-joint payload", and #019
# (results/fovbench-rectfix-393cab9) was produced on THIS box. #021 measured a
# box/dtype delta of up to 0.51% AbsRel, which is small but is not zero, and
# zero is what this checkbox asks for. The pod gets part B, whose contract is a
# digest and not a bit pattern.
#
# --workers 1 is not a preference: fovbench deadlocks at dav2_large above one
# thread on this box (#023, 754 threads in futex_wait, GPU 0%). _ordered_map
# pools in split order so serial is bit-identical to threaded anyway.
set -uo pipefail
REPO=/user/f.zhang2/projects/vggt-omega-organized
ADT=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
OUT=$REPO/eval_out/fovbench-joint
WANT=601fcb22767e
cd "$REPO" || exit 2
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate raytun3r || exit 2
export VGGT_OMEGA_CKPT=$REPO/checkpoints/VGGT-Omega-1B-512/model.pt
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

# Gate on the digest BEFORE half an hour of forward passes, not after. The
# ticket says stop if it is not 601fcb22767e; this makes stopping cost seconds.
D=$(python - <<'PY'
from fovbench.split import build_split
import os
sp = build_split(os.environ["ADT"], n_frames=50, verbose=False)
print(sp.digest, len(sp.frames))
PY
)
echo "[A] split: $D"
case "$D" in
  "$WANT "*) : ;;
  *) echo "[A] DIGEST MISMATCH: got '$D', want '$WANT'. Not running."; exit 3 ;;
esac

t0=$(date +%s)
echo "[A] === start $(date -Is) gpu=$CUDA_VISIBLE_DEVICES workers=1 ==="
python -m fovbench.run --adt-root "$ADT" --protocols radial \
  --models vggt_1b,vggt_omega,dav2_large,da3_large,da3_small \
  --n-frames 50 --workers 1 --out "$OUT/partA_6seq" > "$OUT/partA.log" 2>&1
rc=$?
echo "=== [fov024_A] exit $rc after $(( $(date +%s) - t0 ))s ==="
