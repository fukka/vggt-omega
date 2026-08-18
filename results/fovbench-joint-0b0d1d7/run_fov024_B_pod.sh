#!/usr/bin/env bash
# Ticket 024 part B -- the context arms on the six-sequence split, one arm per
# A100. Usage: run_fov024_B_pod.sh <context_frames> <gpu>
#
# WHY THE POD. Part B is ~6.3 h of forward pass on one card and the arms are
# independent, so four idle A100-80GB turn it into one arm's wall clock. Part A
# stayed on lambda because its done-when is a byte-identical payload against a
# run made there; part B's contract is the digest, and the digest is now proven
# equal here (601fcb22767e) with all 900 underlying files sha256-identical to
# lambda's. Nothing about this half asks for the same bit pattern as a lambda
# run -- it asks for one curve measured consistently.
#
# WHICH IS WHY N=1 IS AN ARM HERE. #024 says not to re-run N=1 because #019's
# partA_6seq already is it and two N=1 curves that differ are worse than one.
# That reasoning holds ON ONE BOX. Plotting lambda's N=1 against A100 N=3/5/10
# would put a box change inside the curve, and #021 measured that step at up to
# 0.51% AbsRel -- small against the 3-40% context effects in
# fovbench-ctx-d351d94, but the same size as its smallest cells (vggt_1b real
# rect N=1 -> 5c is +0.2%). So this box measures its own N=1, the curve is drawn
# from THIS box's four points, and lambda's N=1 becomes the cross-box control
# rather than a point on the plot. Which N=1 the deck plots is stated in
# meta.json, not left to whoever opens the directory.
#
# --workers 1 with OMP_NUM_THREADS=16: this box has 247 threads across two NUMA
# sockets and unpinned BLAS makes it ~7x SLOWER than lambda, and fovbench
# deadlocks above one worker on lambda (#023). _ordered_map pools in split
# order, so one worker is bit-identical to sixteen anyway.
set -uo pipefail
N="${1:?usage: run_fov024_B_pod.sh <context_frames> <gpu>}"
GPU="${2:?usage: run_fov024_B_pod.sh <context_frames> <gpu>}"
REPO=/group-volume/Fengjia/projects/vggt-omega-024
ADT=/group-volume/Fengjia/data/adt-024-6seq
OUT=$REPO/eval_out/fovbench-joint
WANT=601fcb22767e
TAG="partB_6seq_${N}s"

source /group-volume/Fengjia/envs/vggt360-py312/bin/activate || exit 2
cd "$REPO" || exit 2
export HF_HOME=/group-volume/Fengjia/hf-cache
export VGGT_OMEGA_CKPT=/group-volume/Fengjia/projects/vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT"

# Gate before the forward passes, not after them: a wrong pool here is the one
# failure that makes every number in the arm unpublishable.
D=$(ADT="$ADT" python - <<'PY'
import os
from fovbench.split import build_split
sp = build_split(os.environ["ADT"], n_frames=50, verbose=False)
print(sp.digest)
PY
)
if [ "$D" != "$WANT" ]; then
  echo "[$TAG] DIGEST MISMATCH: got '$D', want '$WANT'. Not running."; exit 3
fi
echo "[$TAG] split $D ok, gpu=$GPU"

# The N=1 control takes stride 1, not 10: a stride is meaningless with one
# frame, and this arm exists to be #019's configuration on this box, not a
# variation on it.
if [ "$N" = 1 ]; then STRIDE=1; TAG=partB_6seq_1; else STRIDE=10; fi

t0=$(date +%s)
python -m fovbench.run --adt-root "$ADT" --protocols radial \
  --models vggt_1b,vggt_omega,da3_large,da3_small \
  --n-frames 50 --workers 1 --context-frames "$N" --context-stride "$STRIDE" \
  --out "$OUT/$TAG" > "$OUT/partB_${N}s.log" 2>&1
rc=$?
echo "=== [$TAG] exit $rc after $(( $(date +%s) - t0 ))s ==="
