# Score depth against the ego-synth 5B sparse SLAM GT

**Owner:** cpu
**Files I may touch:** `fovbench/datasets_egosynth.py` (new), `fovbench/split.py`,
`fovbench/run.py`, `fovbench/tests/test_egosynth.py` (new). Nothing under
`raytun3r/`, nothing under `finetune/`.
**Blocked by:** none. The data is already on lambda_63 — 1 611 takes, 24 931
clips, 380 GiB at `/data/f.zhang2/ego-synth-5b/`, verified complete.

## Goal

`fovbench` can score against ego-synth sparse SLAM depth as well as ADT, on both
the raw fisheye and the 110° rectified pinhole, using the same frozen-fit
protocol it already uses.

## Context

New GT, four Aria datasets, 1 611 takes, 24 931 clips. Format, conventions and
the ten gotchas are in [`docs/data/ego-synth-5b-sparse-depth.md`](../../data/ego-synth-5b-sparse-depth.md)
— read that first; it is short and it is the thing that will save the run.

**You do not need the box to write this.** A 260 MB sample — one take per
dataset, four clips each, every member — is staged at
`/data/f.zhang2/ego-synth-5b-sample` on lambda_63 and on the phone at
`/sdcard/data/ego-synth-5b-sample`, with a `read_sample.py` that exercises every
path. Move it out of band the way `raytun3r/experiments/make_local_sample.py`
describes: this is licensed data and must not be committed to this public repo.

Three facts drive the design:

* GT is **sparse points**, not a depth map. `fovbench.geometry.bin_by` and the
  metric path in `finetune.eval.metrics` are written against dense arrays with a
  validity mask, so scattering the points into an 896² array plus mask looks like
  the cheap adapter. **It is not — do not do it.** Pixel coordinates are float16
  and quantise to half a pixel, so ~20 % of points collide onto a shared pixel
  and the last write wins (5 292 points → 4 150 distinct pixels on the frame
  measured). Gather the prediction at the point list instead; the metrics are
  per-point anyway, so this is both lossless and less code.
* **Both variants exist for the same frame**, which is what ADT could only fake
  by re-rendering. `rect` vs `fisheye` becomes like-for-like by construction.
* **θ is computable on `rectified` and not on `fisheye`** — no fisheye camera
  model ships with this data. Radius binning works on both. Do not silently emit
  a θ column for the fisheye stream; `fovbench/README.md` is already explicit
  that radius and θ are not interchangeable, and this is that distinction with
  teeth.

## Steps

1. Loader that yields, per (take, clip, frame, variant): the 896² RGB frame
   decoded from the mp4, `u,v,d` as float32, and the per-point σ columns. Clip
   the rounded pixel index to 895 — `rint(895.5)` is 896 and `u` really does
   reach 895.5.
2. A per-point metric path: gather `pred[v, u]`, fit the affine over all of a
   frame's points, freeze it, then bin. The frozen-fit rule in
   `fovbench/README.md` is unchanged; only the support changes from a dense mask
   to a point list. Assert on a synthetic case that a known affine bias is
   recovered, the way `tests/test_end_to_end.py` does for ADT.
3. Wire `--egosynth-root` into `fovbench/run.py` alongside `--adt-root`; refuse
   θ binning on `fisheye` with a clear message rather than a wrong number.
4. Filter: drop points with `inv_dist_std` above a stated threshold and say what
   the threshold is in `results.json`. The GT ships unfiltered on purpose.
5. Let bins be empty. A single frame can carry as few as ~1 300 points and can
   populate no bin at all within 30° of the axis, so per-frame binning is not
   safe — aggregate over frames first, and report a missing bin as missing.

## Done when

- [ ] `python -m pytest fovbench/tests -q` passes
- [ ] `python -m pytest raytun3r/tests -q` passes
- [ ] a `--models analytic` run over a staged sample recovers a known `+0.6·θ²`
      bias on `rectified`, to the same tolerance the ADT path holds
- [ ] pushed to `organized`, issue commented with the sha

## Needs a GPU run afterwards?

**yes** → relabel `gpu`. On lambda_63:

```bash
git -C /user/f.zhang2/projects/vggt-omega-organized pull --ff-only origin organized
python -m fovbench.run \
  --egosynth-root /data/f.zhang2/ego-synth-5b \
  --datasets aea,nymeria,egoexo4d,oxford \
  --models vggt_1b,vggt_omega,dav2_large,da3_large \
  --out eval_out/egosynth_fov 2>&1 | tee eval_out/egosynth_fov.log
```

Smoke one dataset with `--models analytic --n-frames 3` first, exactly as #8 did.
