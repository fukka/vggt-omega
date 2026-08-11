# Score depth against the ego-synth 5B sparse SLAM GT

**Owner:** cpu
**Files I may touch:** `fovbench/datasets_egosynth.py` (new), `fovbench/split.py`,
`fovbench/run.py`, `fovbench/tests/test_egosynth.py` (new). Nothing under
`raytun3r/`, nothing under `finetune/`.
**Blocked by:** none for the loader. The GPU run at the bottom is blocked on the
copy landing on lambda_63.

## Goal

`fovbench` can score against ego-synth sparse SLAM depth as well as ADT, on both
the raw fisheye and the 110° rectified pinhole, using the same frozen-fit
protocol it already uses.

## Context

New GT, four Aria datasets, 37 143 clips. Format, conventions and the nine
gotchas are in [`docs/data/ego-synth-5b-sparse-depth.md`](../../data/ego-synth-5b-sparse-depth.md)
— read that first; it is short and it is the thing that will save the run.

Three facts drive the design:

* GT is **sparse points**, not a depth map. `fovbench.geometry.bin_by` and the
  metric path in `finetune.eval.metrics` are written against dense arrays with a
  validity mask. A sparse point set is the *easy* case — a boolean mask with
  ~5 k true pixels out of 896² — so the cheapest correct move is to scatter the
  points into an 896² array plus mask and change nothing downstream. Do that
  before considering a sparse code path.
* **Both variants exist for the same frame**, which is what ADT could only fake
  by re-rendering. `rect` vs `fisheye` becomes like-for-like by construction.
* **θ is computable on `rectified` and not on `fisheye`** — no fisheye camera
  model ships with this data. Radius binning works on both. Do not silently emit
  a θ column for the fisheye stream; `fovbench/README.md` is already explicit
  that radius and θ are not interchangeable, and this is that distinction with
  teeth.

## Steps

1. Loader that yields, per (take, clip, frame, variant): the 896² RGB frame
   decoded from the mp4, `u,v,d` as float32, and the per-point σ columns.
2. Scatter to an 896² depth array + validity mask; feed the existing metric path
   unchanged. Assert on a synthetic case that a known affine bias is recovered,
   the way `tests/test_end_to_end.py` does for ADT.
3. Wire `--egosynth-root` into `fovbench/run.py` alongside `--adt-root`; refuse
   θ binning on `fisheye` with a clear message rather than a wrong number.
4. Filter: drop points with `inv_dist_std` above a stated threshold and say what
   the threshold is in `results.json`. The GT ships unfiltered on purpose.

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
