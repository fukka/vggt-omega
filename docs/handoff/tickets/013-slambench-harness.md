# The SLAM depth evaluation — `slambench/`

**Owner:** cpu (harness) → gpu (the run)
**Files I may touch:** `slambench/**` only. **Nothing under `fovbench/`.**
**Blocked by:** the `rect_derect` baseline needs #17 (ticket 012, the calibration
download). The `raw` baseline is not blocked and is finished.

## This is not the ADT-FOV experiment

`fovbench/` asks *where in the field of view does depth degrade*, on ADT's
synthetic twin, by binning error against eccentricity. `slambench/` asks *how
accurate is each model on real egocentric footage*, against ego-synth 5B's MPS
SLAM points, and has **no eccentricity axis at all**. They share a repository,
some models, and a definition of AbsRel. They share no protocol, no ground truth
and no conclusion.

The two were entangled once — the SLAM data was implemented *inside* the FOV
experiment (`fovbench/datasets_egosynth.py`, ticket 011, issue #16), and by the
time it was caught `fovbench/geometry.py` carried a function kept alive only
because the SLAM path mirrored it, and `fovbench/report.py` branched on the
dataset in eight places. `slambench/tests/test_separation.py` now enforces the
boundary mechanically: no import of `fovbench`, and no FOV vocabulary in the
scoring path.

**Unwinding the entanglement inside `fovbench/` is deliberately not in this
ticket.** It is the owner's call when to do it, and until then the FOV package is
left exactly as it stands.

## Goal

Every model is scored against ego-synth's SLAM ground truth **in the form the
producer captured it**, under two baselines that differ only in how they treat
the lens.

## The shape of it

    harness    raw fisheye frame + native GT points
                 -> baseline.predict(frame) -> one depth per point
                 -> metrics

The harness rectifies nothing, warps nothing and converts no depth convention.
Only `fisheye_uvd` is read; the release's pre-rectified point set is never a
scoring target, because consuming it would bake one baseline's choice into the
measurement.

**The lens is the baseline's business:**

| baseline | inside the baseline | needs a camera model |
|---|---|---|
| `raw` | the frame goes to the model as it is | no |
| `rect_derect` | rectify → model → map the depth back onto the fisheye points | yes, per take |

Both are scored on the points **both** could answer for. A pinhole cannot cover
the whole fisheye cone, so `rect_derect` has no answer at the rim; comparing
aggregates over different point sets would compare the sets. What each gave up to
reach the intersection is reported as `coverage`, beside the numbers and not
inside them.

## Status

**Done, and verified on this machine.** 36 CPU tests, ~4 s, no weights and no
GPU; 6 of them drive the real staged sample end to end.

* the camera model (`camera.py`): FISHEYE624 project/unproject round-trips to
  1e-3 px, rescale on the pixel-centre convention, rotation
* the de-rectification chain (`baselines.py`): a perfect prediction on the
  rectified image, sampled back at the fisheye points, recovers the closed-form
  depth of an analytic slanted plane to a median 2e-3 relative
* co-axial planar-z invariance, so no depth conversion is applied — measured on
  the release itself, where the producer's own two point sets carry identical
  depths for the same points
* the protocol (`metrics.py`): a disparity model is exactly recoverable under a
  disparity-space fit and not under a depth-space one, and `check_protocol`
  refuses the wrong pairing before a run rather than after
* **the closing invariant**: `--models oracle` reads **AbsRel 0.000, δ1 1.000**
  on all four datasets, through decode, sigma filter, gather, alignment and
  report. The harness adds nothing of its own.

* `verify_camera.py`, the acceptance test for the calibration, and it **passes**.
  It replaced the contaminated float16-depth pairing with a point-*cloud*
  comparison: rectified points → rays → predicted fisheye pixels, then
  nearest-neighbour distance to the actual fisheye cloud. No pairing is
  attempted, so no pairing can be wrong. On both staged takes the 90° rotation
  reads **0.29 px median, 96.9 % within 1 px**, against a 80 % bar, a 0.5-2 %
  chance rate, and 4-6 px for the other three quarter turns.

* **The calibration convention, which was the open question and is now closed.**
  The fix is three corrections and **not one fitted parameter** — every number
  is a sensor specification, a published report, or a pixel-centre convention
  the reference implementation already uses:

  | | aea | nymeria |
  |---|---|---|
  | as it was | 6.74 | 6.96 |
  | + the 2816 crop | 1.37 | 1.51 |
  | + rotating the ray, not the model | 1.05 | 0.99 |
  | + the rectified render's centre | **0.31** | **0.29** |

  1. **The 1408 stream is a 2816 centre crop of the 2880 sensor, binned 2x** —
     not a plain downsample. MPS writes the calibration at 2880; `meta.json`
     says `source_width = 1408` on all four datasets. This is
     [projectaria_tools #322](https://github.com/facebookresearch/projectaria_tools/issues/322),
     and that issue's own reported numbers pin the 32 px offset to 0.005 px.
     One `rescale(scale = 896/2816, origin_offset = 32)`, in the library's own
     shift-then-scale order.
  2. **A quarter turn may not be applied to the model.** `p1, p2` and `s1..s4`
     are not isotropic. `projectaria_tools` makes the same call structurally —
     its `rotate_camera_calib_cw90deg` is Linear-model-only. The turn now goes
     on the ray in and the pixel out.
  3. **The rectified render is centred on `(N-1)/2`, not `N/2`.** This one was a
     bug in `verify_camera` itself, not in the camera. Half a pixel there is
     magnified by the ratio of the focal lengths (386.28/313.69) on the way to
     the fisheye, and it left a constant `(+0.71, +0.71)` residual, flat in
     radius, on both datasets.

  `aea` and `nymeria` are now in `camera.VERIFIED_ROTATION`, written by hand.
  **On one take each** — that is what the local sample holds.

* **`rect_derect` runs**, for the first time, on both datasets, at coverage 0.99.

**Not done.**

* `oxford` has never been measured — no calibration staged locally. It is in
  scope and is **not** in `VERIFIED_ROTATION`, so `require_verified` stops it.
* `egoexo4d` is **paused** by the owner's decision; scope is aea, nymeria,
  oxford.
* No weights have run here. **No number from a real network is claimed.**

## Steps remaining

1. **Run the `raw` baseline on the box.** It needs no calibration and does not
   wait on anything below — the fastest route to the evaluation's first real
   numbers.
2. Land #17 (three datasets; Oxford needs a route decision — its MPS is a
   18-24 GB `.tar.gz` that cannot be range-read).
3. Re-run `verify_camera` **across takes**, not the one per dataset staged here,
   and on `oxford` for the first time. Add `oxford` to `VERIFIED_ROTATION` by
   hand if it passes — a measurement that promotes itself is not a check.
4. Then the full `raw,rect_derect` run.

## Done when

- [x] `python -m pytest slambench/tests -q` passes (49 tests)
- [x] `slambench` imports nothing from `fovbench`, enforced by a test
- [x] `--models oracle` reads AbsRel 0 end to end on the staged sample
- [x] `verify_camera.py` exists and returns a decisive verdict
- [x] the `raw` run below is in `results/` — `results/slambench-raw-b1659a0`,
      results branch `b139bef`, issue #18. 0 errors, 0 models skipped; the
      oracle was re-earned on that split and still reads AbsRel 0.000.
- [x] the convention is settled and `rect_derect` runs against a verified camera
- [ ] `verify_camera` has passed on more than one take per dataset, and on oxford

## The GPU run — the `raw` arm is ready now

Two things the box needs that this ticket did not say, both found on the run:

* **`pytest slambench/tests -q` reads 40 passed, 6 skipped on lambda_63**, and
  the skip is silent. `test_end_to_end.py` falls back to a Mac path; export
  `EGOSYNTH_SAMPLE=/data/f.zhang2/ego-synth-5b-sample` and all 46 pass. The six
  that skip are exactly the ones that drive the real release.
* **`da3_small`'s weights are not in the persistent cache.** The registry
  reports `download` because they sit in `~/.cache/huggingface/hub` only;
  copying the 131 MB into `checkpoints/hf/hub` makes all five ready.

Smoke it first, no weights, no camera, one dataset:

```bash
python -m slambench.run --egosynth-root /data/f.zhang2/ego-synth-5b \
  --models analytic --baselines raw --datasets aea --takes 1 --n-frames 3 \
  --device cpu --out eval_out/slambench_smoke
```

Then the real thing. **`raw` only** — not because `rect_derect` is broken (it
works now), but because it needs #17's calibrations on the box first, and the
`raw` arm should not wait for them:

```bash
python -m slambench.run \
  --egosynth-root /data/f.zhang2/ego-synth-5b \
  --datasets aea,nymeria,oxford \
  --models vggt_1b,vggt_omega,dav2_large,da3_small,da3_large \
  --baselines raw \
  --out eval_out/slambench_raw 2>&1 | tee eval_out/slambench_raw.log
```

`--takes` defaults to 8 per dataset; the release is 1 611 takes, and the cap
enters the split digest so a capped run can never be compared with a fuller one
by accident.

Once #17 has landed the calibrations, add the second arm — and re-run the
verifier first, because so far it has only seen one take per dataset:

```bash
python -m slambench.verify_camera \
  --egosynth-root /data/f.zhang2/ego-synth-5b \
  --calib-root /data/f.zhang2/ego-synth-5b-calib --takes 8
```

```bash
python -m slambench.run \
  --egosynth-root /data/f.zhang2/ego-synth-5b \
  --calib-root /data/f.zhang2/ego-synth-5b-calib \
  --datasets aea,nymeria --baselines raw,rect_derect \
  --models vggt_1b,vggt_omega,dav2_large,da3_small,da3_large \
  --out eval_out/slambench_both 2>&1 | tee eval_out/slambench_both.log
```

`oxford` is deliberately absent from that second command: it is not in
`VERIFIED_ROTATION`, so `rect_derect` would refuse it anyway. Add it after the
verifier has passed on it.

## The environment

`projectaria_tools` is what owns the FISHEYE624 model and the rectification, and
`slambench` uses it wherever it is importable — `camera.project/unproject` and
`baselines.rectify` both prefer it, with numpy/cv2 fallbacks that the tests pin
against it. It needs **Python >= 3.9**, and the `rescale` binding this depends on
arrived later still, so:

```bash
pip install projectaria-tools
```

Without it nothing breaks and no test fails — the fallbacks carry it, and
`test_camera.py` skips the two checks that need the library rather than
pretending. With it, `_rescale_params` is verified bit-identical to
`CameraCalibration.rescale` and the two rectifiers agree to interpolation
rounding (median 0, max 3 levels on uint8).
