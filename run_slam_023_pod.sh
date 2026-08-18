#!/usr/bin/env bash
# Ticket 023, the slambench half, on space-container. ONE RUN PER INVOCATION.
#
#   run_slam_023_pod.sh slam_B  0
#   run_slam_023_pod.sh slam_B3 1
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
# the usual fix is already spent: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# was set and only 128 MiB was reserved-but-unallocated at the point of failure.
# Ours held 19.9 GB and needed 1.35 GB more; another user's process holds 26.1 GB
# of card 0's 47.4 GB throughout. #020 never hit this because its peak was
# ~15 GB: the vggt360 arm is what pushes the requirement past the card's spare,
# so this recurs for anyone running that arm there, not just today.
#
# WHY ONE RUN PER INVOCATION, ON ITS OWN CARD
#
# This pod has FOUR idle A100-SXM4-80GB. Check all four -- an earlier
# `nvidia-smi --query-gpu` here reported only index 0 and that reading was
# wrong. B and B3 share nothing but the code and each peaks ~22 GB, so running
# them on separate cards is free and halves the wall clock. An earlier version
# ran them sequentially in one lane; that was leaving a card idle for no reason.
#
# It also fixes a real trap: the lane version kept the tmux pane alive with
# `exec bash` after the work finished, so "is the session alive" could never mean
# "is the job done". Completion is the `=== [tag] exit` line in this run's OWN
# log, which is what a waiter should grep. Do not go back to session-liveness.
#
# THE SPLIT IS THE PUBLISHED ONE -- VERIFY, DO NOT ASSUME
#
# #020's split digest is 61195914f090 (400 frames, 16 takes, 2 datasets) and
# lambda's slam_B printed the same before it died. Confirmed on this pod
# 2026-08-17: it prints `split 61195914f090` with clip counts matching lambda
# exactly (aea 251, nymeria 256), so the staged takes reproduce the same
# find_takes `[:8]` selection. Re-check it per run anyway -- it is one grep, and
# it is the only thing licensing a pod number beside a lambda one. If it ever
# differs, the run stays internally valid (slambench compares arms within one
# model and one run) but must not be tabulated beside #020's.
#
# WHAT THE `raw` ARM IS DOING IN RUN B
#
# It is the control for this box. Our numbers come off an A100 and #020's off an
# RTX 6000 Ada, so `raw` here should reproduce #020's `raw`, and the drift
# measures what changing silicon cost. Expect small but NOT zero -- and note the
# GPU is not the only thing that moved: #020's vggt_1b column is fp32 (see its
# PROVENANCE.md, it predates #021's bf16 fix), so up to #021's measured 0.51%
# AbsRel is expected from dtype before the GPU is blamed for any of it.
#
# rect_derect is in B3 and deliberately absent from B: a 110 deg pinhole has no
# answer at the rim, so including it truncates the comparison at ~55 deg --
# exactly the field this method exists to cover. B is the headline; B3 is read
# only against itself.
set -uo pipefail

TAG="${1:?usage: run_slam_023_pod.sh <slam_B|slam_B3> <gpu>}"
GPU="${2:?usage: run_slam_023_pod.sh <slam_B|slam_B3> <gpu>}"

REPO=/group-volume/Fengjia/projects/vggt-omega-023
OUT="$REPO/eval_out/vggt360-023"

case "$TAG" in
  slam_B)  BASELINES=raw,vggt360 ;;
  slam_B3) BASELINES=raw,rect_derect,vggt360 ;;
  *) echo "unknown tag: $TAG" >&2; exit 2 ;;
esac

# HF_HOME on the persistent mount, not $HOME: everything outside /group-volume
# dies with the pod, and VGGT-1B is a 9.4 GB re-download each time otherwise.
export HF_HOME=/group-volume/Fengjia/hf-cache
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/group-volume/Fengjia/envs/vggt360-py312/bin:$PATH
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$OUT"
cd "$REPO"

echo "=== [$TAG] start $(date -Is) on GPU $GPU: $(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader)"
t0=$SECONDS
python -m slambench.run \
  --egosynth-root /group-volume/Fengjia/data/ego-synth-5b \
  --calib-root /group-volume/Fengjia/data/ego-synth-5b-calib \
  --datasets aea,nymeria --models vggt_1b \
  --baselines "$BASELINES" \
  --context-frames 1 --n-frames 25 --takes 8 \
  --out "$OUT/$TAG" 2>&1 | tee "$OUT/$TAG.log"
rc=${PIPESTATUS[0]}
echo "=== [$TAG] exit $rc after $((SECONDS - t0))s" | tee -a "$OUT/$TAG.log"
