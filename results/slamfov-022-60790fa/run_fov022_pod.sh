#!/usr/bin/env bash
# Ticket 022 -- the FOV question asked of real MPS SLAM points, on the A100 pod.
#
# WHY THE POD. Step 3 (window) is 17 windows a frame, ~17x the headline radial
# pass, and the arms shard cleanly by model. Two idle A100-80GB turn that into
# one lane's wall clock. Nothing in this ticket asks for a bit-identical payload
# against a lambda run -- its contract is #020's manifest, and the manifest
# fixes the frames rather than the box.
#
# OMP_NUM_THREADS=16 is mandatory here: 247 threads across two NUMA sockets make
# unpinned BLAS ~7x SLOWER than lambda (SPACE_CONTAINER.md section 4).
set -uo pipefail

REPO=/group-volume/Fengjia/projects/vggt-omega-023
VENV=/group-volume/Fengjia/envs/vggt360-py312
OUT=$REPO/eval_out/slamfov-022
MANIFEST=$REPO/results/slambench-020-143686a/step2/manifest.pod.json

source "$VENV/bin/activate" || exit 2
cd "$REPO" || exit 2

export EGOSYNTH=/group-volume/Fengjia/data/ego-synth-5b
export EGOSYNTH_CALIB=/group-volume/Fengjia/data/ego-synth-5b-calib
export VGGT_OMEGA_CKPT=/group-volume/Fengjia/projects/vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt
export HF_HOME=/group-volume/Fengjia/hf-cache
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

MODELS=vggt_1b,vggt_omega,dav2_large,da3_small,da3_large
COMMON=(--manifest "$MANIFEST" --calib-root "$EGOSYNTH_CALIB"
        --egosynth-root "$EGOSYNTH" --datasets aea,nymeria
        --baselines raw --omega-checkpoint "$VGGT_OMEGA_CKPT")

run () {  # run <tag> <gpu> <args...>
  local tag=$1 gpu=$2; shift 2
  if [ -f "$OUT/$tag/results.json" ]; then echo "[$tag] already done, skipping"; return 0; fi
  local t0=$(date +%s)
  echo "[$tag] START gpu=$gpu $(date -Is)"
  ( CUDA_VISIBLE_DEVICES=$gpu python -m slambench.run_fov "$@" --out "$OUT/$tag"
    echo "MARKER_$tag=$?" ) > "$OUT/$tag.log" 2>&1
  echo "[$tag] END $(grep -o "MARKER_$tag=[0-9]*" "$OUT/$tag.log" | tail -1) after $(( $(date +%s) - t0 ))s"
}

# ---- step 1: preflight, CPU only, no weights -------------------------------
python -m pytest tests slambench/tests -q 2>&1 | tail -3
run smoke 0 --egosynth-root "$EGOSYNTH" --calib-root "$EGOSYNTH_CALIB" \
  --models analytic --baselines raw --protocols radial,window \
  --datasets aea,nymeria --takes 1 --n-frames 8 --tilts 0,20,40 --device cpu
grep -qE "MARKER_smoke=0" "$OUT/smoke.log" || { echo "PREFLIGHT FAILED -- stopping"; exit 3; }
echo "=== preflight ok ==="

# ---- step 3 lane 1 (GPU 1), backgrounded: window, the four non-vggt_1b -----
( run window-lane1 1 "${COMMON[@]}" --models vggt_omega,dav2_large,da3_small,da3_large \
    --protocols window --window-fov 40 --tilts 0,10,20,30,40 --azimuths 0,90,180,270
) &
LANE1=$!

# ---- GPU 0: step 2 headline, step 4 control, then step 3's vggt_1b lane ----
run radial 0 "${COMMON[@]}" --models "$MODELS" --protocols radial \
  --context-frames 1 --theta-edges 0,10,20,30,40,50,55,60
run oracle 0 "${COMMON[@]}" --models oracle --oracle-noise 0.15 --protocols radial \
  --context-frames 1 --theta-edges 0,10,20,30,40,50,55,60
run window-lane0 0 "${COMMON[@]}" --models vggt_1b \
  --protocols window --window-fov 40 --tilts 0,10,20,30,40 --azimuths 0,90,180,270

wait $LANE1
echo "=== FOV022_ALL_DONE $(date -Is) ==="
for f in "$OUT"/*.log; do echo "$(basename $f): $(grep -o 'MARKER_[a-z0-9-]*=[0-9]*' $f | tail -1)"; done
