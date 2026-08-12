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

**Not done.**

* `rect_derect` has never run against a real camera model, because none is
  downloaded yet (#17).
* The **sensor-to-upright rotation is unverified** for every dataset.
  `camera.VERIFIED_ROTATION` is empty and `require_verified` refuses to hand out
  a camera, so `rect_derect` currently stops rather than producing a number. A
  quarter-turn error does not make a score worse — it scores a different part of
  the image — which is why this is a hard stop and not a warning.
* No weights have run here. **No number from a real network is claimed.**

## Steps remaining

1. Land #17 so per-take `camera_rgb.json` exists.
2. Write `slambench/verify_camera.py`: the acceptance test for the orientation.
   Correspondences must **not** be matched on exact float16 depth — that was
   tried, and two different points sharing a depth get paired, leaving ~90 % of
   pairs false and p05 at 1.4 px when some true pairs land at 0.20–0.50 px. Match
   through the model itself and iterate, or use the frames where depth is
   unambiguous. Sub-pixel median is the bar (`camera.ORIENTATION_TOL_PX`).
3. Add the passing datasets to `camera.VERIFIED_ROTATION` **by hand** — a
   measurement that promotes itself is not a check.
4. Then the GPU run.

## Done when

- [x] `python -m pytest slambench/tests -q` passes
- [x] `slambench` imports nothing from `fovbench`, enforced by a test
- [x] `--models oracle` reads AbsRel 0 end to end on the staged sample
- [ ] `rect_derect` runs against a verified camera on at least one dataset
- [ ] the GPU run below is in `results/`

## The GPU run, once the above lands

```bash
python -m slambench.run \
  --egosynth-root /data/f.zhang2/ego-synth-5b \
  --calib-root    /data/f.zhang2/ego-synth-5b-calib \
  --models vggt_1b,vggt_omega,dav2_large,da3_small,da3_large \
  --baselines raw,rect_derect \
  --out eval_out/slambench_main 2>&1 | tee eval_out/slambench_main.log
```

Smoke it first, exactly as #13 did — one dataset, no weights, no camera:

```bash
python -m slambench.run --egosynth-root /data/f.zhang2/ego-synth-5b \
  --models analytic --baselines raw --datasets aea --takes 1 --n-frames 3 \
  --device cpu --out eval_out/slambench_smoke
```

`--takes` defaults to 8 per dataset; the release is 1 611 takes, and the cap
enters the split digest so a capped run can never be compared with a fuller one
by accident.
