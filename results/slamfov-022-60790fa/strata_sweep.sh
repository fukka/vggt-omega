#!/usr/bin/env bash
# #022 follow-up: tighten the oracle null.
#
# The published radial half reports a residual standardised spread of 1.10x
# (aea) / 1.26x (nymeria) for a model with NO field dependence, and the ticket
# says that residual is the coarseness of five distance strata, not a finding.
# It is also the DENOMINATOR of every "over null" figure in the artefact, so
# tightening it directly sharpens the headline claim rather than adding a new
# one. Cheap: the oracle arm cost 86 s.
#
# Also sweeps oracle noise, because the null must not depend on the size of the
# error we inject -- if 1.10x moves with --oracle-noise then it is not purely
# strata coarseness and the ticket's explanation is wrong.
set -uo pipefail
REPO=/group-volume/Fengjia/projects/vggt-omega-023
OUT=$REPO/eval_out/slamfov-022/strata
source /group-volume/Fengjia/envs/vggt360-py312/bin/activate || exit 2
cd "$REPO" || exit 2
export EGOSYNTH=/group-volume/Fengjia/data/ego-synth-5b
export EGOSYNTH_CALIB=/group-volume/Fengjia/data/ego-synth-5b-calib
export HF_HOME=/group-volume/Fengjia/hf-cache
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
MAN=$REPO/results/slambench-020-143686a/step2/manifest.pod.json
mkdir -p "$OUT"

for S in 4 6 8 10; do
  for N in 0.15; do
    tag="oracle_s${S}_n${N}"
    [ -s "$OUT/$tag/results.json" ] && { echo "[$tag] skip"; continue; }
    t0=$(date +%s)
    ( python -m slambench.run_fov --manifest "$MAN" --calib-root "$EGOSYNTH_CALIB" \
        --egosynth-root "$EGOSYNTH" --datasets aea,nymeria --baselines raw \
        --models oracle --oracle-noise $N --protocols radial --context-frames 1 \
        --theta-edges 0,10,20,30,40,50,55,60 --depth-strata $S \
        --out "$OUT/$tag"; echo "MARKER_$tag=$?" ) > "$OUT/$tag.log" 2>&1
    echo "[$tag] $(grep -o "MARKER_$tag=[0-9]*" "$OUT/$tag.log"|tail -1) after $(( $(date +%s)-t0 ))s"
  done
done
# noise sweep at the finest strata, to prove the null is a property of the
# binning and not of the injected error size
for N in 0.05 0.30; do
  tag="oracle_s8_n${N}"
  [ -s "$OUT/$tag/results.json" ] && { echo "[$tag] skip"; continue; }
  t0=$(date +%s)
  ( python -m slambench.run_fov --manifest "$MAN" --calib-root "$EGOSYNTH_CALIB" \
      --egosynth-root "$EGOSYNTH" --datasets aea,nymeria --baselines raw \
      --models oracle --oracle-noise $N --protocols radial --context-frames 1 \
      --theta-edges 0,10,20,30,40,50,55,60 --depth-strata 8 \
      --out "$OUT/$tag"; echo "MARKER_$tag=$?" ) > "$OUT/$tag.log" 2>&1
  echo "[$tag] $(grep -o "MARKER_$tag=[0-9]*" "$OUT/$tag.log"|tail -1) after $(( $(date +%s)-t0 ))s"
done
echo "=== STRATA_SWEEP_DONE $(date -Is) ==="
