#!/usr/bin/env bash
# Ticket 023 -- the vggt360 row in both published tables, across the two GPUs.
#
# One lane = one GPU, jobs sequential inside a lane so a card never holds two
# model loads at once. The split is by RUN rather than by model, because the
# three runs are independent invocations that share nothing but the code: run A
# is a fovbench grid, run C is a one-model fovbench control, run B is a
# slambench grid. Nothing has to be merged afterwards.
#
# **The two lanes are not balanced, because free memory decided the split rather
# than compute did.** The slambench vggt360 arm peaks near 22 GB (see
# PYTORCH_CUDA_ALLOC_CONF below). Another user's job holds 26 GB of card 0 and
# 33 GB of card 1, leaving 21.3 GB and 13.8 GB -- so every run that touches
# vggt360 with attention fusion fits on card 0 and none of them fits on card 1.
# Lane 0 therefore carries the whole programme serially, and lane 1 gets only the
# fovbench control, which is the one vggt360 run small enough to try there
# (12.4 GB measured) and is cheap to re-run on card 0 if it dies.
#
# The first attempt at this split put run A on card 1 and it was on course to OOM
# the moment it reached vggt360, after spending six minutes on the vanilla
# models. Balance the lanes on what is free, not on what is fast.
#
# Every run tees to its OWN log. Do not pipe a run into `tail` -- the pipe
# buffers and the per-take "ring stops N deg short of it" line, which the tables
# cannot carry, becomes invisible until the run ends.
#
# Usage: vggt360_023_lane.sh <gpu> <lane>
set -uo pipefail

GPU="$1"; LANE="$2"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate raytun3r
cd /user/f.zhang2/projects/vggt-omega-organized

export ADT=/user/f.zhang2/Documents/projectaria_tools_adt_data_clean
export EGOSYNTH=/data/f.zhang2/ego-synth-5b
export EGOSYNTH_CALIB=/data/f.zhang2/ego-synth-5b-calib
export VGGT_OMEGA_CKPT=/user/f.zhang2/projects/vggt-omega-organized/checkpoints/VGGT-Omega-1B-512/model.pt
export CUDA_VISIBLE_DEVICES="$GPU"

# Not a tuning knob -- without it the slambench vggt360 arm does not run on this
# box at all. `--vggt360-fuse attn` is the method, and the fusion reads frame
# attention, so `attention.py` keeps the softmax output of every frame block:
# nine 518 px views is a 1.24 GiB allocation per block and a ~22 GB peak. Another
# user's job (pid 2991964) is parked on 26 GB of card 0 and 33 GB of card 1, so
# 21.3 GB is all that is free on the roomier card. The first attempt died with
# 908 MiB free and 1.29 GiB reserved-but-unallocated; handing that fragmentation
# back is exactly the difference between OOM and a 1m47s smoke.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The runs print things the tables cannot carry -- "ring stops N deg short of it"
# per take, "N% of the imaged cone filled with a constant" -- and python
# block-buffers stdout when it is a pipe, which `tee` is. Without this the log
# arrives in 8 KB bursts and a healthy run is indistinguishable from a wedged one
# for minutes at a time. It cost one false "stalled" diagnosis: the process was
# State: R with the GPU at 98%, and the log had simply not flushed. Liveness is
# the run process's CPU time, never the length of its log.
export PYTHONUNBUFFERED=1

OUT=eval_out/vggt360-023
mkdir -p "$OUT"

# --workers 1, and this costs wall clock to buy a run that finishes at all.
#
# **fovbench deadlocks on this box at dav2_large, reproducibly, at any thread
# count above 1.** Both attempts died in the same place: immediately after
# `Loading weights: 100%` for the third model, with no frame line ever printed,
# main thread and every worker sleeping in futex_wait, GPU at 0%.
#
#     --workers 8 (default)   498 threads   wedged 28 min before I killed it
#     --workers 16            754 threads   wedged 8 min before I killed it
#
# Thread count scaling with --workers is what identifies it: each worker is
# bringing up ~45 threads of its own, so this is a thread-explosion deadlock, not
# a stall waiting on something external. Ruled out by measurement, not by
# argument: 469 GB RAM free and load 2.3 on 64 cores (not starvation);
# dav2_large alone scores a frame in 6 s both online and under HF_HUB_OFFLINE=1
# (not the HF Hub, despite the unauthenticated-requests warning in the log); and
# the 18-frame smoke passed all six models in one process, so it needs a few
# hundred frames of pool churn to appear.
#
# results/fovbench-rectfix-393cab9 was produced 2026-08-14 with --workers 16 over
# the same 300 frames and the same five models without hanging, so something under
# the harness moved in the three days since -- transformers is 5.15.0 here. Worth
# a cpu ticket; not worth blocking this run on.
#
# Serial is the safe escape and not a compromise on the numbers: _ordered_map
# pools rows in split order precisely so the arithmetic is bit-identical to the
# threaded path, and tests/test_end_to_end.py pins that. Expect roughly 3.3x on
# the view stage, per the DEFAULT_WORKERS note in fovbench/run.py.
#
# Do NOT instead reach for OMP_NUM_THREADS=1. Python thread count is guaranteed
# not to move fovbench's numbers; BLAS thread count is not -- the scale_shift fit
# goes through lstsq -- and the reference run was produced under the default. That
# trades a hang for a silent risk of failing the bit-identity self-check.
WORKERS=1

run() {  # run <tag> <args...>
  local tag="$1"; shift
  echo "=== [$tag] start $(date -Is) on GPU $GPU"
  local t0=$SECONDS
  "$@" 2>&1 | tee "$OUT/$tag.log"
  local rc=${PIPESTATUS[0]}
  echo "=== [$tag] exit $rc after $((SECONDS - t0))s"
}

case "$LANE" in
0)
  # Run A. The vanilla models re-run BESIDE vggt360 in one command, so the
  # comparison is within a single run rather than across two.
  #
  # **--n-frames 50, not the ticket's 25, and da3_small is added.** The ticket's
  # self-check -- "the four vanilla cells must match the #019 headline run,
  # because views and protocols are not in the split digest" -- only works at the
  # frame count the reference actually used, and the published #019 headline is
  # results/fovbench-rectfix-393cab9/partA_6seq: 6 sequences x 50 frames, digest
  # 601fcb22767e, five models including da3_small. 25 frames reproduces no
  # published digest on this box (the older 25-frame run, fovbench-main-22c108d
  # / 2ab412af0ccc, is a ONE-sequence run from when only 1 of 20 ADT sequences
  # had all three streams; there are 6 now). So 25 would have put our row beside
  # a table that does not exist and forfeited the self-check. 50 costs about
  # twice as long -- ~10m43s for the whole vanilla five at 300 frames, both
  # views, on the reference run -- which is not the expensive part here.
  #
  # da3_small costs a couple of minutes and turns the self-check from three
  # exact-match rows into four. Adding a model cannot perturb the others:
  # fovbench aligns and scores per model.
  run fov_A python -m fovbench.run --adt-root "$ADT" --n-frames 50 \
    --models vggt_1b,vggt_omega,dav2_large,da3_large,da3_small,vggt360 \
    --views fisheye --protocols radial --workers "$WORKERS" \
    --out "$OUT/fov_A"

  # Run C, the resolution control, and it is not optional. In run A the nine
  # tangent views are cut from ADT's native 1408 frame -- a 0.62x downsample to
  # 518 -- while the vanilla models see a 518 resize, from which the same views
  # would be a 1.69x upsample. So A is not an equal-pixel comparison and it leans
  # our way. `native` says what the method is worth; `view` says what it is worth
  # on the pixels everyone else got; the gap between them is the resolution term
  # rather than the lens term. Reporting A alone makes the number unreadable.
  #
  # It runs here, ahead of the slambench pair, because it OOM'd on card 1 at
  # 13.61 GB with only 170 MiB of reclaimable fragmentation -- it genuinely wants
  # ~14.7 GB and card 1 has 13.8 free. 50 frames to match run A: the native-view
  # gap is only a resolution term if nothing else differs.
  run fov_C python -m fovbench.run --adt-root "$ADT" --n-frames 50 \
    --models vggt360 --views fisheye --protocols radial --workers "$WORKERS" \
    --vggt360-source view \
    --out "$OUT/fov_C"

  # Run B. rect_derect is deliberately absent: arms are scored on the points
  # every arm could answer for, and a 110 deg pinhole has no answer at the rim,
  # so including it truncates the whole comparison at ~55 deg -- exactly the
  # field this method exists to cover. The three-arm version after it is read
  # only against itself.
  run slam_B python -m slambench.run --egosynth-root "$EGOSYNTH" \
    --calib-root "$EGOSYNTH_CALIB" \
    --datasets aea,nymeria --models vggt_1b --baselines raw,vggt360 \
    --context-frames 1 --n-frames 25 --takes 8 \
    --out "$OUT/slam_B"

  run slam_B3 python -m slambench.run --egosynth-root "$EGOSYNTH" \
    --calib-root "$EGOSYNTH_CALIB" \
    --datasets aea,nymeria --models vggt_1b --baselines raw,rect_derect,vggt360 \
    --context-frames 1 --n-frames 25 --takes 8 \
    --out "$OUT/slam_B3"
  ;;
1)
  # Deliberately empty. Card 1 has 13.8 GB free behind another user's llama-server
  # and every run in this ticket needs more: 14.7 GB for the fovbench control,
  # ~22 GB for the slambench arms. Lane 1 existed for one attempt at run C and it
  # OOM'd with 170 MiB of reclaimable fragmentation, so there is nothing left to
  # try. If that server exits, move fov_C here and the queue shortens by ~12 min.
  echo "lane 1: nothing to run -- card 1 has 13.8 GB free and the smallest" \
       "vggt360 run in this ticket needs 14.7 GB. See the comment in this script."
  ;;
*)
  echo "unknown lane: $LANE" >&2; exit 2 ;;
esac

echo "=== LANE $LANE DONE $(date -Is)"
