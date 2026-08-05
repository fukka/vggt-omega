# Re-run the reproduction properly: DA3-Small, full ScanNet++, fixed conventions

**Owner:** gpu
**Files I may touch:** nothing under `raytun3r/` — this is runs only. Results go
to the `results` branch.
**Blocked by:** none. Pull `organized` first; everything below is already in it.

## Why this supersedes the earlier runs

Three things changed since `bfdb47e`, two of which invalidate the numbers on the
`results` branch:

1. **The depth convention was inconsistent between methods.** `baselines.py`
   converted planar z → euclidean range for the virtual-pinhole baselines, and
   ADT ground truth was converted too, but the direct fisheye path
   (`vanilla`, `param_free`, `raytun3r`) handed the head's raw planar z straight
   to `backproject(convention="range")`. So the pinhole baselines were the only
   methods scored in the right convention. On ScanNet++ geometry that mismatch is
   worth **~0.99 px of `d_reproj`**, against a measured method-to-method spread of
   0.10 px — an order of magnitude larger than the effect under study. Fixed:
   backbones now declare `native_depth`, `install(depth_convention=...)` converts
   once at the boundary, and consumers call `Prediction.require_convention`.

   `R°`/`t°` are **not** affected (pose comes from the camera head), so the
   rotation findings still stand. `d_reproj` and `AbsRel` must be re-measured.

2. **`da3` never worked.** It is the paper's primary backbone and Tab. 1/4/7b are
   DA3-Small numbers. Four bugs, all now fixed and covered by a CPU test against
   the real `depth_anything_3` 0.1.1: the public `forward` is wrapped in
   `torch.no_grad()` (so the adapter could never have trained), the DPT hook
   targeted `_apply_pos_embed` instead of `_add_pos_embed`, the helper import path
   was wrong, and `load()` fetched GIANT regardless of `--variant`.

3. **One scene is not a result.** Everything so far is a single ScanNet++ scene.

## What to run

Pull first:

```bash
git -C /user/f.zhang2/projects/vggt-omega-organized pull --ff-only origin organized
```

Install the paper's backbone into the run env (Apache-2.0, 0.08B):

```bash
pip install depth-anything-3
```

### Run 1 (primary) — DA3-Small over all of ScanNet++

```bash
python -m raytun3r.experiments.scannetpp_all --backbone da3 --variant small --weights pretrained --root /netapp/datasets/scannetpp/data --out runs/rt3r/snpp-all-da3s --workers 4
```

Sanity-check the plan on two scenes first with `--limit 2`; a full scene is
~3 min of fit plus eval, and the driver is one scene per GPU.

The protocol defaults already match the paper — Adam 1e-3, clip 1.0, 300 iters,
30 three-frame windows, 2 px flow filter, 504 max side, UFM — with one deliberate
deviation: **`--stride 10`, not 1**. At stride 1 on this data the baseline is
~1.1 cm against ~3 m of depth, translation direction is unobservable (MAGSAC++ is
itself 11.1° off ground truth) and `d_reproj` stops depending on depth. The paper
does not specify a stride. Do not change it back without saying so in `meta.json`.

**On "the same number of GPUs as the paper":** the paper does not state one, and
the quantity is not meaningful for this method — RayTun3R fits ~10k parameters on
one short segment from one camera in about three minutes, and its selling point
is exactly that cheapness. There is nothing to shard *within* a scene, and a
single adapter shared *across* scenes would be a different method. The parallel
axis is therefore scenes, which is what `--workers` does. If you find a GPU count
stated in the paper, put it in `meta.json` and tell CPU-Claude.

### Run 2 — VGGT-1B over all of ScanNet++

Same command, `--backbone vggt`. This makes the earlier single-scene VGGT run
comparable and separates "the method does not reproduce" from "VGGT is not the
paper's backbone".

### Run 3 — the FOV sweep (the diagnostic that matters most)

```bash
python -m raytun3r.experiments.fov_sweep --backbone da3 --variant small --weights pretrained --dataset scannetpp --path /netapp/datasets/scannetpp/data/3f15a9266d --out runs/fov-sweep/3f15a9266d-da3s
```

The open question: Center-PH beat RayTun3R ~5× on `R°`, the reverse of Tab. 1.
The leading explanation is that ScanNet++'s DSLR is a ~170° full-frame fisheye,
not the 115° the paper states, so re-projecting to a 110° pinhole discards the
hard pixels rather than handling them. `--max-fov` narrows Ω only — the images are
untouched — so this isolates where a method is *scored*. If the gap closes as the
cone narrows, FOV is the answer; if it is flat, look at the backbone or the
reproduction.

### Run 4 (cheap) — `--backbone vggt_omega` on one scene

Expected to barely help: it is DINOv3, RoPE-only, and the adapter has 20
parameters there. It is the direct test of the paper's own Tab. 7(b) "RoPE only"
row. A *large* improvement would contradict the paper and would be the most
interesting result available.

### Also worth doing

* **The queued `s10-adt-seq131` run**, now that conventions are fixed.
* **Find ScanNet++ `render_depth/`.** It was absent, so `AbsRel`/`δ₁.₂₅` (Tab. 3)
  have never been measured on ScanNet++ at all. If it is not in the download,
  say so and Tab. 3 stays out of scope for this dataset.

## Recording

Per POLICY.md, results to the `results` branch, JSON and trimmed logs only. Two
corrections to how `meta.json` was written last time:

* **Pin the commit each run actually used**, not one commit for the batch. Last
  time all three runs were labelled `bfdb47e`, but the two stride-1 runs predate
  it — they lack the `coverage` key it introduced and carry the 117 px
  `center_ph` `d_reproj` that it fixed. Record `git rev-parse HEAD` per run.
* Record `depth-anything-3` and `torch` versions alongside the matcher, and the
  `--stride` / `--max-fov` actually used.

## Done when

- [ ] `summary.json` for DA3-Small over all ScanNet++ scenes, with per-scene rows
      and the aggregate (mean ± s.e.m. over scenes)
- [ ] the same for VGGT
- [ ] `fov_sweep` summary for at least one scene
- [ ] pushed to `results`, this issue commented with the branch sha
- [ ] hand back to `cpu` so the README's measured-results section is rewritten
      against real numbers instead of the current provisional one

## Needs CPU-Claude afterwards?

yes — README rewrite, and interpretation of whatever the FOV sweep shows.
