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

* `verify_camera.py`, the acceptance test for the calibration, and it **runs**.
  It replaced the contaminated float16-depth pairing with a point-*cloud*
  comparison: rectified points → rays → predicted fisheye pixels, then
  nearest-neighbour distance to the actual fisheye cloud. No pairing is
  attempted, so no pairing can be wrong. Discrimination is wide — chance of a
  random point landing within 1 px is 0.5-2 % against the ~100 % a correct model
  must give.

**Not done, and one of these is now a finding rather than a gap.**

* **The calibration does not describe ego-synth's frames.** `verify_camera`'s
  verdict on both staged takes is `UNVERIFIED`: ~4 px median, ~5 % within 1 px.
  Ruled out, each measured — all four 90° rotations; a continuous roll swept at
  2°; the resolution (implied sensor size swept 1000-4200 px, best ~2820-2840
  still 1.4-1.9 px median); the ~38.7° device-to-camera extrinsic; and this
  package's own projection, now the reference `projectaria_tools` one.
  Still open: a **crop** before the resize, a rectification axis tilted rather
  than rolled, or an online calibration for a different stream.
* `rect_derect` therefore still cannot produce a number, and
  `camera.require_verified` stops it rather than letting it.
* `egoexo4d` is **paused** by the owner's decision; scope is aea, nymeria,
  oxford.
* No weights have run here. **No number from a real network is claimed.**

## Steps remaining

1. **Run the `raw` baseline on the box.** It needs no calibration and is
   finished — this is the fastest route to the evaluation's first real numbers
   and does not wait on anything below.
2. Land #17 (three datasets; Oxford needs a route decision — its MPS is a
   18-24 GB `.tar.gz` that cannot be range-read).
3. Settle the convention. The next move is a **joint search over scale and
   principal point** — only the scale has been swept, and a crop before the
   resize would move the centre too. Validate on held-out takes, not on the
   takes the search used.
4. Add the passing datasets to `camera.VERIFIED_ROTATION` **by hand** — a
   measurement that promotes itself is not a check.
5. Then `rect_derect`, and the full run.

## Done when

- [x] `python -m pytest slambench/tests -q` passes (46 tests)
- [x] `slambench` imports nothing from `fovbench`, enforced by a test
- [x] `--models oracle` reads AbsRel 0 end to end on the staged sample
- [x] `verify_camera.py` exists and returns a decisive verdict
- [ ] the `raw` run below is in `results/`
- [ ] the convention is settled and `rect_derect` runs against a verified camera

## The GPU run — the `raw` arm is ready now

Smoke it first, no weights, no camera, one dataset:

```bash
python -m slambench.run --egosynth-root /data/f.zhang2/ego-synth-5b \
  --models analytic --baselines raw --datasets aea --takes 1 --n-frames 3 \
  --device cpu --out eval_out/slambench_smoke
```

Then the real thing. **`raw` only** — `rect_derect` will refuse until the
calibration is verified, and that refusal is deliberate:

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

Adding `--baselines raw,rect_derect --calib-root …` is the run to make **after**
step 3 above, not before.
