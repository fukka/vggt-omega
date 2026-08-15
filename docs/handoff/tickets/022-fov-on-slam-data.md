# The FOV question, asked of real SLAM points

**Owner:** gpu
**Files I may touch:** `slambench/fov.py`, `slambench/run_fov.py`,
`slambench/fov_report.py`, `slambench/tests/test_fov.py`, `slamfov_lane.sh`
(all new; nothing in the published `slambench.run` path)
**Blocked by:** nothing. The code is on `organized` and the CPU suites are green.

## Goal

`fovbench/` established that depth degrades toward the rim of a wide lens, on a
synthetic twin with a dense depth map. This asks the same question of the real
thing: **does the same degradation appear on real egocentric footage, measured
against MPS SLAM points?** When it is done there is a
`results/slamfov-022-<sha>` carrying an AbsRel-against-eccentricity curve for
each of the five models, on AEA and Nymeria, with distance held fixed.

## What is already built, and what it does that fovbench does not

`slambench/fov.py` + `run_fov.py` + `fov_report.py`, with
`slambench/tests/test_fov.py` (19 tests, CPU-only, no data, no weights). Two
protocols, as in the original:

* **radial** — one whole frame to the model, per-point error binned by the
  incidence angle of the point's ray. *Where in this image.*
* **window** — a fixed 40 deg pinhole re-aimed across the lens, the model sees
  only that. *Where the camera was pointed.* Width is **held fixed** and only the
  aim moves, for the reason ticket 009 records.

Both inherit fovbench's one non-negotiable rule — one affine per frame, fitted
over every point, frozen before any binning
(`test_per_bin_alignment_would_erase_the_effect` measures the difference: an
effect that reads 0.13 under the shared fit reads under 0.03 per bin).

**The thing that is new is a control, and this run is not interpretable without
it.** On this data distance and eccentricity move together, hard — measured on
the release, 15 frames of one AEA take:

    theta        0-10   10-20   20-30   30-40   40-50   50-55   55-60
    median GT     2.57    2.09    1.10    1.07    0.93    0.80    0.71  m

3.6x across the field. Every metric here is relative, so a bare
error-against-eccentricity curve on this data is partly a distance curve, and
`test_a_depth_gradient_alone_invents_a_field_effect` shows a model with **no**
field dependence producing a sloping curve from that alone. So the primitive is
a two-way table, eccentricity x distance, and the report prints the pooled curve
and the distance-held-fixed curve side by side. On the smoke run the two
disagreed about the *sign* of the effect.

The third row of that table is the price: centre and rim barely overlap in
distance, so the shared strata can describe a small slice of a bin. It is
printed per bin. A low share does not invalidate the number, it narrows it, and
that has to be visible.

### How much this matters, measured on real AEA points

`--models oracle --oracle-noise 0.15` puts a fixed **0.15 m** error on every
point — the same error at every eccentricity and every distance, so a model with
no field dependence whatever. 20 frames of one AEA take, 4 strata:

    theta            0-10   10-20   20-30   30-40   40-50   50-55   spread
    pooled          0.105   0.105   0.113   0.134   0.163   0.194     1.86
    fixed distance  0.134   0.133   0.131   0.134   0.147   0.165     1.26
    share            1.00    1.00    1.00    1.00    1.00    1.00

A **1.86x** "the rim is worse" effect, from a model that has no idea where the
rim is. Holding distance fixed removes most of it, on the full data — the share
is 1.00 in every bin, so nothing was narrowed to get there. The 1.26x left is
the coarseness of four strata, not a residual finding; expect it to fall with
`--depth-strata 6`.

**So the pooled row is not a weaker version of the result, it is a different
claim.** Anyone who reads only the pooled row on this dataset will report a
field effect that is mostly distance.

## Context

* The rectified path was re-audited on 2026-08-15 before this was written.
  `derectify`'s address — fisheye px -> ray -> pinhole px — agrees with the
  producer's own rectified projection at **median 0.28 px**, which is the
  float16 floor of the stored coordinates. Confirmed two ways, the second using
  no depth at all: matching the two clouds by position alone finds 98.9 % / 99.3 %
  of pairs carry bit-identical depths. The sampler is reading the pixel it means
  to.
* At `--rect-fov 110` the pinhole is **100 %** backed by real fisheye pixels
  (the 896 frame reaches 56.1 deg at its edge midpoint and 70.3 deg at its
  corner; the pinhole needs 54.9 / 63.6). Zero ground-truth points have a
  bilinear stencil touching padding. Pinned by
  `test_the_run_settings_leave_no_void_for_the_sampler_to_average_over`, so
  widening `--rect-fov` fails a test rather than quietly averaging in black.
* **Read the run against this, it is not controlled for:** the two arms do not
  resample alike, and the difference runs with eccentricity. `raw` decimates the
  896 frame by a flat 1.73x through `cv2.INTER_AREA`, which filters. The
  rectified arm resamples through `remap`/`INTER_LINEAR`, which does not, by a
  factor running 2.13x on axis to 0.78x at 55 deg. So the rectified image is
  *more* aliased in the middle and *sharper* at the rim before any model sees
  it. That is shaped like the effect being measured. Prefer `--baselines raw`
  for the headline curve.
* Adding `rect_derect` truncates every arm at ~55 deg, because the support is
  intersected across arms and the 110 deg pinhole stops there.

## Steps

Run from `organized` at the sha this ticket is filed with. `run_fov` needs
`--calib-root` on **every** arm, including raw — eccentricity is a question
about the lens.

1. **Preflight, no weights, ~2 min.**

       python -m pytest tests slambench/tests -q          # expect 232 passed
       python -m slambench.run_fov --models analytic --baselines raw \
         --protocols radial,window --datasets aea,nymeria --takes 1 \
         --n-frames 8 --tilts 0,20,40 --device cpu --out /tmp/fovsmoke

   `analytic` reads image intensity, so its scores are meaningless by
   construction — what this checks is that both protocols produce full tables and
   that no window falls under `MIN_IN_CONE_FRAC`.

2. **Radial, the headline. `--manifest` is the point:** pass
   `results/slambench-020-143686a/manifest.json` so this bins *exactly* the
   frames #020 scored and the two runs are the same points seen two ways.

       python -m slambench.run_fov \
         --manifest results/slambench-020-143686a/manifest.json \
         --calib-root $EGOSYNTH_CALIB --datasets aea,nymeria \
         --models vggt_1b,vggt_omega,dav2_large,da3_small,da3_large \
         --baselines raw --protocols radial --context-frames 1 \
         --theta-edges 0,10,20,30,40,50,55,60 \
         --out eval_out/slamfov-022/radial

   Roughly step 2 of #020's cost — one forward pass per frame per model — so
   ~10 min on one card now that #021's bf16 fix has landed. `--theta-edges` runs
   to 60 because this is a raw-only run and the field is not truncated; the
   55-60 bin is edge-and-corner only and its own point count says so.

3. **Window, the second protocol.** 1 + 4x4 = 17 windows a frame, so ~17x step 2.
   Shard by model across the two cards with `slamfov_lane.sh`; give `vggt_1b` a
   lane to itself, as #021 concluded for step 3.

       python -m slambench.run_fov \
         --manifest results/slambench-020-143686a/manifest.json \
         --calib-root $EGOSYNTH_CALIB --datasets aea,nymeria \
         --models <lane's models> --protocols window \
         --window-fov 40 --tilts 0,10,20,30,40 --azimuths 0,90,180,270 \
         --out eval_out/slamfov-022/window-lane<N>

4. **The control that costs nothing and is worth the most.** Re-run step 2 with
   `--models oracle`. The oracle answers per point from the ground truth with a
   known affine and no field dependence, so **both** its curves must be flat. If
   the pooled one slopes and the standardised one does not, that is the confound
   being caught in the act on the real data, and it belongs in the artefact.

5. Publish to `results/slamfov-022-<sha>/` with `meta.json` in #020's shape:
   preflight output, the shard rationale, what is comparable to what, and a
   `not_claimed` list. Put the resampling asymmetry above in `not_claimed`.

## Done when

- [ ] `python -m pytest tests fovbench/tests slambench/tests -q` passes on the box
- [ ] step 1 produces both tables with every window over `MIN_IN_CONE_FRAC`
- [ ] steps 2 and 3 land, on #020's own manifest, same digest
- [ ] step 4's oracle reads flat on the standardised curve
- [ ] `results/slamfov-022-<sha>/` pushed to `results`, issue commented with the sha

## What this ticket deliberately does not do

* **It does not unify the binning with `fovbench`.** `slambench/fov.py` is a
  second implementation of "align once, bin, report" — knowingly. The shared part
  is three lines that already come from `finetune/eval/metrics.py`; everything
  else (sums pooled across frames, quantile strata, the standardisation and its
  share) has no counterpart in `fovbench`, which bins dense 2-D masks. If a third
  caller ever needs it, that is the moment to lift the primitive into
  `finetune/eval/` — the route `tests/test_experiment_separation.py` names.
* **It does not touch `slambench/run.py`.** That driver's contract is that it has
  no eccentricity axis and three published artefacts were produced under it;
  `test_the_published_slam_driver_cannot_reach_the_fov_binning` now enforces it
  structurally.
* It does not control the resampling asymmetry between the two arms. Doing that
  means rendering the raw arm through the same unfiltered path, or the rectified
  arm through a filtered one, and either changes what `rect_derect` means. It is
  recorded, not fixed.
