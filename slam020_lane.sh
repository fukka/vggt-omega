#!/usr/bin/env bash
# Ticket 020 -- slambench steps 2 and 3, sharded by MODEL across the two GPUs.
#
# One lane = one GPU. Jobs run sequentially inside a lane so a card never holds
# two model loads at once, and the two lanes carry disjoint halves of the model
# list. slambench.run intersects the support inside its per-model loop, so a
# model never sees another model's arms and this split is exact, not an
# approximation -- tools/merge_slambench_shards.py re-checks that claim (same
# digest, same config) before it staples the shards together.
#
# Usage: slam020_lane.sh <gpu> <lane>
set -uo pipefail

GPU="$1"; LANE="$2"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
cd /user/f.zhang2/projects/vggt-omega-organized

export EGOSYNTH=/data/f.zhang2/ego-synth-5b
export EGOSYNTH_CALIB=/data/f.zhang2/ego-synth-5b-calib
export VGGT_OMEGA_CKPT=/user/f.zhang2/projects/vggt-omega-organized/checkpoints/VGGT-Omega-1B-512/model.pt
export CUDA_VISIBLE_DEVICES="$GPU"

OUT=eval_out/slambench-020
mkdir -p "$OUT"

# The shards. Balanced on the raw run's own per-model timings (vggt_1b 177 s,
# vggt_omega 85, dav2_large 82, da3_large 70, da3_small 57 over 600 frames):
# step 2 splits 234 s against 237 s. Step 3 drops dav2_large -- it is monocular
# and predict_stack raises rather than scoring the target alone -- and puts one
# heavy transformer plus one DA3 in each lane, which is the partition least
# sensitive to how badly VGGT's global attention scales with a 10-frame window.
if [ "$LANE" = "0" ]; then
  S2=vggt_1b,da3_small
  S3=vggt_1b,da3_small
else
  S2=vggt_omega,dav2_large,da3_large
  S3=vggt_omega,da3_large
fi

run () {  # run <name> <models> [extra flags...]
  local name="$1" models="$2"; shift 2
  local dir="$OUT/$name.g$LANE"
  if [ -f "$dir/results.json" ]; then echo "[lane$LANE] SKIP $name"; return 0; fi
  echo "[lane$LANE] START $name ($models) $(date -Is)"
  python -m slambench.run \
    --egosynth-root "$EGOSYNTH" --calib-root "$EGOSYNTH_CALIB" \
    --datasets aea,nymeria --baselines raw,rect_derect \
    --models "$models" --takes 8 --n-frames 25 \
    "$@" --out "$dir" > "$OUT/$name.g$LANE.log" 2>&1
  echo "[lane$LANE] END $name rc=$? $(date -Is)"
}

run step2    "$S2"
run step3-s1 "$S3" --context-frames 1,3,5,10 --context-stride 1
run step3-s10 "$S3" --context-frames 1,3,5,10 --context-stride 10

echo "[lane$LANE] ALL DONE $(date -Is)"
