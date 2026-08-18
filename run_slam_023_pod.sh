#!/usr/bin/env bash
# Ticket 023, the slambench half -- runs B and B3 -- on space-container.
#
# WHY THIS IS NOT ON lambda_63, WHICH IS WHERE EVERY PUBLISHED NUMBER CAME FROM
#
# Both runs were attempted there first and both died the same way, in
# VGGT-360-fisheye/vggt_visfeat/layers/attention.py:95, on the softmax the
# attention fusion keeps:
#
#   slam_B   OOM after ~16 min, at take 6 of 16
#   slam_B3  OOM after 965 s, same take
#
# This is a real capacity shortfall and NOT fragmentation, which matters because
# the usual fix does not apply: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# was already set and only 128 MiB was reserved-but-unallocated at the point of
# failure -- there was nothing left to hand back. Our process held 19.9 GB and
# needed 1.35 GB more; another user's process holds 26.1 GB of card 0's 47.4 GB
# throughout. #020 never hit this because its peak was ~15 GB: the vggt360 arm
# is what pushed the requirement past what the card has spare.
#
# This pod is an idle A100-SXM4-80GB. That is the whole reason to be here.
#
# THE SPLIT IS THE PUBLISHED ONE -- VERIFY, DO NOT ASSUME
#
# #020's split digest is 61195914f090 (400 frames, 16 takes, 2 datasets) and
# lambda's slam_B printed the same before it died. This run MUST print
# `split 61195914f090` too. The 16 takes were staged by stage_egosynth_to_pod.sh
# and their clip counts already match lambda exactly (aea 251, nymeria 256), so
# the pod's find_takes `[:8]` selects the same eight -- but the digest is the
# token that says so, and it is one grep. If it differs, STOP: the run is still
# internally valid (slambench compares arms within one model and one run) but it
# is a different split and must not be tabulated beside #020's.
#
# WHAT THE `raw` ARM IS DOING IN RUN B
#
# It is the control for this box. Our numbers come off an A100 and #020's off an
# RTX 6000 Ada, so `raw` here should reproduce #020's `raw` and any drift is the
# measure of what changing silicon cost. Expect it to be small but NOT zero, and
# note that the GPU is not the only thing that moved: #020's vggt_1b column is
# fp32 (see its PROVENANCE.md -- it predates #021's bf16 fix), so a difference of
# up to #021's measured 0.51% AbsRel bound is expected from dtype alone. Beyond
# that bound, something else changed and the vggt360 row should not be read until
# it is understood.
#
# rect_derect is in B3 and deliberately absent from B: a 110 deg pinhole has no
# answer at the rim, so including it truncates the comparison at ~55 deg --
# exactly the field this method exists to cover. B is the headline; B3 is read
# only against itself.
set -uo pipefail

REPO=/group-volume/Fengjia/projects/vggt-omega-023
OUT="$REPO/eval_out/vggt360-023"

# HF_HOME on the persistent mount, not $HOME: everything outside /group-volume
# dies with the pod, and VGGT-1B is a 9.4 GB re-download each time otherwise.
export HF_HOME=/group-volume/Fengjia/hf-cache
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/group-volume/Fengjia/envs/vggt360-py312/bin:$PATH

mkdir -p "$OUT"
cd "$REPO"

run() {  # run <tag> <args...>
  local tag="$1"; shift
  echo "=== [$tag] start $(date -Is) on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  local t0=$SECONDS
  "$@" 2>&1 | tee "$OUT/$tag.log"
  local rc=${PIPESTATUS[0]}
  echo "=== [$tag] exit $rc after $((SECONDS - t0))s"
}

COMMON=(--egosynth-root /group-volume/Fengjia/data/ego-synth-5b
        --calib-root /group-volume/Fengjia/data/ego-synth-5b-calib
        --datasets aea,nymeria --models vggt_1b
        --context-frames 1 --n-frames 25 --takes 8)

# B first: it is the ticket's required run and B3 is explicitly optional, so if
# the pod drops mid-programme the thing that survives is the one that matters.
run slam_B  python -m slambench.run "${COMMON[@]}" \
  --baselines raw,vggt360 --out "$OUT/slam_B"

run slam_B3 python -m slambench.run "${COMMON[@]}" \
  --baselines raw,rect_derect,vggt360 --out "$OUT/slam_B3"

echo "=== SLAM 023 DONE $(date -Is)"
