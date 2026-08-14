# slambench: is ego-synth's `d` planar z, or is it range?

**Owner:** cpu (it needed no GPU box — see "the route changed" below)
**Status:** **DONE. The verdict is `z`** — planar camera-frame Z, on both
staged datasets, decisively. The data card was right, no published number moves,
and **#020 step 2 is released to run.** Measured 2026-08-14 on the Mac; see
"The answer" below. Issue
[#21](https://github.com/fukka/vggt-omega/issues/21), reported and closed;
code on `organized` @ `8573cf0`.
**Files I may touch:** `tools/fetch_egosynth_mps_points.py` (new),
`slambench/verify_depth_convention.py` (new),
`slambench/tests/test_depth_convention.py` (new — see the note on scope below).
**Not** `slambench/data.py`, `baselines.py` or `metrics.py`, and none of them
was touched.
**Blocked by:** none. Independent of the `rect_derect` arm and of ticket 013.

## The answer: `d` is planar z

**Verdict `z` on both staged takes**, 2026-08-14, on `aea/loc1_script1_seq1_rec1`
and `nymeria/20230614_s0_elizabeth_sandoval_act0_bzf7du`, 4 clips x 121 frames
each. Nothing already published moves, and #020 step 2 may run.

    |d - z| / z  by incidence angle          |d - range| / range
      theta        aea      nymeria            aea    (1-cos)   nymeria  (1-cos)
      0-15      0.0002       0.0002         0.0158   0.0158    0.0185   0.0185
     15-30      0.0002       0.0002         0.0831   0.0831    0.0897   0.0898
     30-45      0.0002       0.0002         0.2097   0.2096    0.2051   0.2051
     45-90      0.0002       0.0002         0.3665   0.3664    0.3590   0.3591

     aea      2 598 579 stored rows -> 261 275 uniquely matched (10.1 %)
     nymeria    539 936 stored rows ->   9 366 uniquely matched  (1.7 %)

Two things make this decisive rather than merely favourable.

**`d - z` is 0.0002 at every angle — flat.** 0.0002 is the float16 quantisation
of the stored `d`, so the residual is not small, it is *absent*: there is nothing
left above the noise floor of what the ground truth can express. And flat is the
part that matters. Any error in this convention is radial by construction, so a
convention error cannot hide in a flat residual.

**The losing hypothesis is wrong by exactly the predicted amount.** This is the
stronger half and it is easy to state wrongly, so: the two readings each predict
the *other's* residual, and they are **different functions**.

    if d is z:      |d - range| / range  =  1 - cos(theta)
    if d is range:  |d - z| / z          =  sec(theta) - 1

Those are not the same curve — at 50 deg one is 0.36 and the other 0.56. The
measured range residual matches **`1 - cos(theta)` to four decimal places in all
eight bins across both datasets**, which is the signature of `d` being z, not
merely of range being wrong. `sec(theta) - 1` is the wrong comparator here and an
earlier draft of the report printed it, which made an exact agreement look
approximate.

**An independent second reading agrees.** Reconstruct each stored row into the
world under each hypothesis and measure how far it lands from the actual cloud —
no matching at all, so it shares no arithmetic with the table above. Distance
under z is 0.0007-0.0016 m and **flat**; under range it is 0.0065-0.0216 m and
**grows with theta**, on both datasets, over ~2.6 M and ~1.3 M points.

**The check can return "range".** The load-bearing test in
`slambench/tests/test_depth_convention.py` builds a synthetic take under each
convention and requires the check to recover that one; on range-built data it
reads range, and the loser's residual matches its predicted function to 10 %. A
check that could only ever say "z" would make agreement with the data card
worthless, and that is the failure mode this ticket was written against.

### The route changed: no GPU box, and no trajectory

Two departures from the Steps below, both of which made the work smaller.

1. **The trajectory does not need fetching.** Steps 1-2 specify
   `closed_loop_trajectory.csv` (101 MB per Nymeria take) plus `T_device_camera`,
   composed here. The release already ships that composition:
   `camera_poses.json` carries `T_world_camera` per frame and states its own
   provenance — MPS closed-loop trajectory @ `T_device_camera(camera-rgb)`,
   looked up at each frame's exact capture timestamp. Re-deriving it would have
   spent 101 MB to reproduce the producer's own arithmetic and introduced two
   fresh ways to be wrong (the extrinsic, the interpolation) inside a check whose
   purpose is to *remove* an unverified assumption. Only the world points were
   fetched: **44.8 MB for aea** (of a 349.1 MB archive) and **161.3 MB for
   nymeria** (of 593.0 MB), by the same ranged-zip read ticket 012 built.
2. **It never needed the box.** Both signed URL JSONs and the staged sample are
   on the Mac, and the sample holds exactly the one `aea` and one `nymeria` take
   this ticket calls for. Owner corrected from `gpu` to `cpu` accordingly.

### On the scope line

`slambench/tests/test_depth_convention.py` is a third new file, beyond the two
this ticket names. It collides with nothing — #020 touches nothing under
`slambench/`, and the file is new — but it is more than the ticket said, and the
falsifiability test is the reason it exists rather than tidiness.

Nothing under `slambench/` that the evaluation runs on was modified:
`data.py`, `baselines.py` and `metrics.py` are untouched, as specified.

### What this does not settle

* **Two takes, one per dataset** — the same limit `verify_camera` has, and for
  the same reason: that is what the local sample holds. `oxford` and `egoexo4d`
  are unmeasured. `oxford` is out of #020 step 2 anyway; if it is ever added,
  this check should run on it first.
* The unique-match rate is low (10.1 % on aea, **1.7 %** on nymeria) because a
  row is dropped whenever a second world point falls inside the angular
  tolerance. That is deliberate — resolving the ambiguity by depth would put a
  hypothesis back into the matching — but it is why nymeria needed all 484
  frames to clear the coverage bar. The independent reconstruction reading has
  no such filter and carries ~1.3 M points on nymeria.

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

- [x] `python -m pytest slambench/tests -q` passes — **79 passed, 1 skipped**
      with `EGOSYNTH_SAMPLE` set. The skip is the pre-existing
      `test_camera.py:187` (a `projectaria_tools` predating the `rescale`
      binding), not anything this ticket added. 9 of the passes are new.
- [x] the check runs on one `aea` take and one `nymeria` take
- [x] the per-theta-bin table for both readings, with the number of matched
      points per bin — above, and in the issue
- [x] a one-line verdict: **z — planar camera-frame Z.** aea reads |rel| 0.0002
      vs 0.2781 beyond 30 deg over 172 831 points; nymeria 0.0002 vs 0.2522 over
      5 089. Not undecided, and the coverage bar (1 000 matched points beyond
      30 deg) was cleared rather than lowered — a 48-frame nymeria run that
      reached only 138 was reported as undecided and re-run longer, not
      re-thresholded.

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
