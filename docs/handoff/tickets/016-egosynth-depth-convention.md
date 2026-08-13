# slambench: is ego-synth's `d` planar z, or is it range?

**Owner:** gpu
**Status:** **open, and now step 1 of #020** — nothing else in the SLAM evaluation means anything until this is settled.
**Files I may touch:** `tools/fetch_egosynth_mps_points.py` (new),
`slambench/verify_depth_convention.py` (new). **Not** `slambench/data.py`,
`baselines.py` or `metrics.py` — if the answer comes back "range" that is a
second ticket, not this one.
**Blocked by:** none. Independent of the `rect_derect` arm and of ticket 013.

## The question

**Every slambench number is scored against `pts.d` as if it were planar z about
the camera axis. Nothing has ever measured that.**

It is the data card's statement — `docs/data/ego-synth-5b-sparse-depth.md`,
"`d` is **metric camera-frame Z in metres** — planar z, not range", and gotcha 4
repeats it. That is a decent authority and it is probably right. But this repo
has already been burned by exactly this once: CONTEXT.md records that the
ERP-era "depth is range" assumption is what made the VGGT-360 fisheye port score
range against z-GT, for AbsRel 0.146 / δ1 0.79 on a *perfect* prediction.

If `d` is range and we score it as z, every number carries a `1/cos(theta)`
error: **1.00 on axis, 1.36 at 43 deg, 1.74 at 55 deg.** Radial, so the
per-frame affine cannot absorb any of it, and it lands hardest exactly where the
fisheye arm is supposed to have an advantage over `rect_derect`.

## Why it has not been caught

Two things in `slambench/` look like checks of this and are structurally
incapable of being one. Both are now documented in place; do not let either
persuade you the question is settled.

1. **The rectified/fisheye depth agreement** (`baselines.py`). The producer's two
   point sets carry identical `d` for the same points. True, and it is what
   `derectify` needs — but euclidean range is a property of the *ray*, so it is
   just as invariant under a co-axial rectification as planar z is. The
   measurement separates nothing.

2. **`verify_camera`'s 0.29 px twin residual.** `predicted_pixels` builds
   `xyz = d * [(u-c)/f, (v-c)/f, 1]`; the range reading would give `d` times that
   same vector normalised. The two differ by the positive scalar `|ray|`, and
   `Fisheye624.project` starts from `x/z, y/z`. **The predicted pixel is
   bit-identical under both hypotheses** — that residual could go to zero and
   still say nothing here.

## What can decide it

The source MPS semi-dense points, which are the thing ego-synth re-projected.
They give the world position of each point and the trajectory gives the device
pose, so *both* readings are computable and they differ by `sec(theta)`.

`aea` and `nymeria` are the two takes already staged locally and the two in
`camera.VERIFIED_ROTATION`, so use those.

## Steps

1. **Fetch the MPS points and trajectory** for one staged take per dataset.
   `tools/fetch_egosynth_calibration.py` already pulls a single member out of a
   remote zip by ranged GET (`_curl_range` / `_central_directory`) from the same
   per-dataset URL manifest; copy that pattern rather than downloading the 576 MB
   archives. Wanted:

   * `semidense_points.csv.gz` — `uid, px_world, py_world, pz_world,
     inv_dist_std, dist_std`
   * `closed_loop_trajectory.csv` — `tracking_timestamp_us`, and the device pose
     `T_world_device`

   `GROUP`/`MEMBER` in the fetcher name the group and path for each dataset; the
   points live under the MPS *slam* group alongside `online_calibration`.

2. **Line the frames up.** `meta.json -> clips[].source_timestamps_ns` gives each
   ego-synth frame its capture timestamp, so pick the trajectory pose nearest
   that timestamp. `camera_rgb.json` (already fetched, ticket 012) gives
   `T_device_camera`; if the extrinsic is not in that file, take it from the same
   `online_calibration` record rather than assuming identity.

3. **Compute both readings** for each world point visible in the frame:

   ```
   p_cam = T_camera_device @ T_device_world @ p_world
   z     = p_cam[2]
   range = norm(p_cam)
   ```

4. **Match to ego-synth's rows and compare.** Project `p_cam` through
   `camera.Fisheye624` for the take, match to the nearest `fisheye_uvd` row
   within 1 px (the same nearest-neighbour primitive `verify_camera.nn_distances`
   already provides), and regress `d` against each of `z` and `range`.

   **Report the residual of both, binned by theta.** That is the whole point: on
   axis the two agree, so a pooled statistic would be dominated by the centre and
   read as agreement either way. Use `camera.Fisheye624.theta_of` and bin at
   0-15 / 15-30 / 30-45 / 45+ deg.

## Done when

- [ ] `python -m pytest slambench/tests -q` passes
- [ ] the check runs on one `aea` take and one `nymeria` take
- [ ] the per-theta-bin table for both readings is in the issue, with the number
      of matched points per bin
- [ ] a one-line verdict: **z**, **range**, or **undecided** — and if undecided,
      which step was too noisy, not a guess

## What "undecided" costs, so it does not get rounded to "z"

The two hypotheses are 1% apart at 8 deg and 74% apart at 55 deg. If the matched
points do not reach past ~30 deg the check has not separated anything, and the
honest outcome is that the answer is still the card's — which is where we are
today, and is fine, as long as the report says so.

## Not in scope

Changing anything in `slambench/`. If the answer is "range", the fix is a
conversion at the read in `data.read_points` plus a re-run of every number, and
that is a ticket written after this one reports — not a change smuggled in with
the measurement that motivated it.
