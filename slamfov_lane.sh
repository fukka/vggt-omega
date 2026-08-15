#!/usr/bin/env bash
# Ticket 022 -- the FOV question on ego-synth SLAM points, sharded by MODEL.
#
# One lane = one GPU, jobs sequential inside a lane so a card never holds two
# model loads at once. slambench.run_fov accumulates each model's table inside
# its own per-model loop and clears it before the next, so a model never sees
# another model's arms and this split is exact rather than an approximation.
#
# The partition is #021's conclusion, not #020's original: vggt_1b does not pay
# for a window at the same rate as the others, so it gets a lane to itself.
# After the bf16 fix it is ~2.9x cheaper than the run that measured that, which
# makes the imbalance smaller but not zero.
#
# Usage: slamfov_lane.sh <gpu> <lane> [radial|window]
set -uo pipefail

GPU="$1"; LANE="$2"; STAGE="${3:-radial}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
cd /user/f.zhang2/projects/vggt-omega-organized

export EGOSYNTH=/data/f.zhang2/ego-synth-5b
export EGOSYNTH_CALIB=/data/f.zhang2/ego-synth-5b-calib
export VGGT_OMEGA_CKPT=/user/f.zhang2/projects/vggt-omega-organized/checkpoints/VGGT-Omega-1B-512/model.pt
export CUDA_VISIBLE_DEVICES="$GPU"

# Bin exactly the frames #020 scored, so the FOV tables and the accuracy tables
# are two views of one set of points rather than two samples of a dataset.
MANIFEST=results/slambench-020-143686a/manifest.json
OUT=eval_out/slamfov-022
mkdir -p "$OUT"

if [ ! -f "$MANIFEST" ]; then
  echo "no $MANIFEST -- fetch the results branch first:"
  echo "  git fetch origin results && git checkout origin/results -- results/"
  exit 1
fi

if [ "$LANE" = "0" ]; then
  MODELS=vggt_1b
else
  MODELS=vggt_omega,dav2_large,da3_small,da3_large
fi

common=(--manifest "$MANIFEST"
        --calib-root "$EGOSYNTH_CALIB"
        --datasets aea,nymeria
        --models "$MODELS"
        --omega-checkpoint "$VGGT_OMEGA_CKPT"
        --device cuda)

case "$STAGE" in
  radial)
    # One forward pass per frame per model -- about step 2 of #020's cost.
    # Runs to 60 deg because this is raw-only: adding rect_derect would
    # intersect the support and truncate every arm at ~55.
    python -m slambench.run_fov "${common[@]}" \
      --baselines raw --protocols radial --context-frames 1 \
      --theta-edges 0,10,20,30,40,50,55,60 \
      --out "$OUT/radial-lane$LANE" 2>&1 | tee "$OUT/radial-lane$LANE.log"
    ;;
  window)
    # 1 + 4x4 = 17 windows a frame. dav2_large is monocular, which is fine at
    # --context-frames 1; the window arm does not ask for a context.
    python -m slambench.run_fov "${common[@]}" \
      --protocols window --context-frames 1 \
      --window-fov 40 --tilts 0,10,20,30,40 --azimuths 0,90,180,270 \
      --out "$OUT/window-lane$LANE" 2>&1 | tee "$OUT/window-lane$LANE.log"
    ;;
  oracle)
    # Step 4 of the ticket, and the cheapest thing here. The oracle answers per
    # point from the ground truth with a known affine and NO field dependence,
    # so both its curves must be flat. A pooled curve that slopes while the
    # standardised one does not is the distance confound caught in the act, on
    # the real data rather than in a unit test -- keep that output.
    python -m slambench.run_fov \
      --manifest "$MANIFEST" --calib-root "$EGOSYNTH_CALIB" \
      --datasets aea,nymeria --models oracle --oracle-noise 0.15 \
      --baselines raw --protocols radial --context-frames 1 \
      --theta-edges 0,10,20,30,40,50,55,60 \
      --device cpu --out "$OUT/oracle" 2>&1 | tee "$OUT/oracle.log"
    ;;
  *)
    echo "usage: $0 <gpu> <lane> [radial|window|oracle]"; exit 2 ;;
esac
