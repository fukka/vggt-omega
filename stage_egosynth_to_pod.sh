#!/usr/bin/env bash
# Stage exactly the takes ticket 023's slambench runs need onto space-container.
# Runs ON lambda_63. IDEMPOTENT: re-run to fill gaps, it skips takes already there.
#
# WHY THIS ROUTE
#
# `file://groups/SR-TORAIC-IVU/Fengjia` IS the pod's `/group-volume/Fengjia` --
# confirmed by listing it from lambda_63 and seeing the venv this session created
# minutes earlier. So the transfer is lambda -> object storage -> a path the pod
# already has mounted, at S3 speeds, and it needs no pod at all. That sidesteps
# both of the pod's chronic problems: the SSH chain drops mid-transfer, and the
# pod's address goes stale between sessions. It is also ~7x the SSH path, which
# measured 5.4 MB/s pod<->lambda on 2026-08-11.
#
# WHY THESE SIXTEEN TAKES AND NOT A PREFIX OF THE DATASET
#
# slambench picks takes as `sorted(os.listdir(ds_dir))`, filtered to those with
# `sparse_depth/*.npz` matching `meta.json`, then `found[:limit]`
# (slambench/data.py:find_takes). So staging exactly the 8 per dataset that
# `--takes 8` selects on lambda_63 makes the pod's `found[:8]` the SAME eight,
# and the split digest should follow. VERIFY THAT rather than assume it: run
# slambench on the pod at --n-frames 1 first and check the digest against
# lambda's. If it differs, the run is still internally valid -- slambench
# compares arms within one model, never across runs -- but it is not the same
# split as #020's and must not be tabulated beside it.
#
# 7.2 GB total. The full release is 381 GB, which is why this is a take list and
# not an rsync.
#
# STAGE THE CALIBRATION TOO -- the first version of this script did not, and it
# cost a 1965 s run. The pod already had 143 aea calibrations and exactly ONE of
# nymeria's 254, so every aea take passed and the run died on the second nymeria
# take, 33 minutes in. slambench opens a take's `camera_rgb.json` only when it
# reaches that take, so a missing one is invisible until then. The whole calib
# tree is 4.4 MB against the data's 7.2 GB -- there is no reason to be selective,
# and it ships a SHA256SUMS.txt so the copy can be verified rather than trusted.
set -uo pipefail

SRC=/data/f.zhang2/ego-synth-5b
DST="file://groups/SR-TORAIC-IVU/Fengjia/data/ego-synth-5b"

TAKES=(
  aea/loc1_script1_seq1_rec1
  aea/loc1_script1_seq3_rec1
  aea/loc1_script1_seq5_rec1
  aea/loc1_script1_seq6_rec1
  aea/loc1_script1_seq7_rec1
  aea/loc1_script2_seq1_rec1
  aea/loc1_script2_seq1_rec2
  aea/loc1_script2_seq3_rec1
  nymeria/20230614_s0_elizabeth_sandoval_act0_bzf7du
  nymeria/20230614_s1_matthew_harper_act0_e1qur5
  nymeria/20230615_s0_dawn_heath_act1_9s9e13
  nymeria/20230615_s1_vincent_bell_act0_s0kg0n
  nymeria/20230616_s0_kristen_thomas_act0_5edc0f
  nymeria/20230616_s1_michael_griffin_act0_g5u8n8
  nymeria/20230619_s1_eric_martin_act0_oeiu1g
  nymeria/20230620_s0_marie_vasquez_act0_5vd9yy
)

# One invocation per take. Startup is ~1 s, so this is cheap, and it buys
# per-take resumability -- a drop costs one take, not the whole 7.2 GB. Same
# lesson as the 2026-08-11 380 GiB pull.
# Calibration first: it is 4.4 MB and it is what a missing file costs an hour of.
CALIB_SRC=/data/f.zhang2/ego-synth-5b-calib
CALIB_DST="file://groups/SR-TORAIC-IVU/Fengjia/data/ego-synth-5b-calib"
echo "=== calibration ($(du -sh "$CALIB_SRC" | cut -f1), $(find "$CALIB_SRC" -name camera_rgb.json | wc -l) camera models)"
space storage upload file "$CALIB_SRC" "$CALIB_DST" \
  --max-request-processes 16 --attempts 23 \
  || echo "!!! calibration upload FAILED -- runs will die partway through a dataset" >&2

ok=0; fail=0
for t in "${TAKES[@]}"; do
  n_local=$(find "$SRC/$t" -type f 2>/dev/null | wc -l)
  echo "=== $t  ($n_local files)"
  if space storage upload file "$SRC/$t" "$DST/$t" \
       --max-request-processes 16 --attempts 23; then
    ok=$((ok+1))
  else
    echo "!!! FAILED: $t" >&2
    fail=$((fail+1))
  fi
done

echo
echo "=== uploaded $ok take(s), $fail failed"
# Gate on file COUNT per take, not on the transfer's own exit code: a truncated
# take passes silently otherwise, and slambench would then score a take that is
# quietly missing clips.
echo "=== verifying file counts against the source"
bad=0
for t in "${TAKES[@]}"; do
  want=$(find "$SRC/$t" -type f 2>/dev/null | wc -l)
  got=$(space storage list file "$DST/$t" --recursive 2>/dev/null \
        | grep -c "| file  *|")
  if [ "$want" != "$got" ]; then
    echo "  MISMATCH $t: local $want, remote $got"; bad=$((bad+1))
  fi
done
# And assert every take has a camera model, which is the specific gap that bit.
# Counting files is not enough: aea was complete and nymeria was one of 254.
for t in "${TAKES[@]}"; do
  space storage list file "$CALIB_DST/$t/camera_rgb.json" >/dev/null 2>&1 \
    || { echo "  NO CALIBRATION for $t"; bad=$((bad+1)); }
done
[ "$bad" = 0 ] && echo "  all $((${#TAKES[@]})) takes match, and each has a camera model" || echo "  $bad problem(s) -- re-run this script"
echo "=== STAGE DONE"
