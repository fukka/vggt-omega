# Re-run the reproduction properly: DA3-Small, full ScanNet++, fixed conventions

**Owner:** gpu
**Files I may touch:** nothing under `raytun3r/` — this is runs only. Results go
to the `results` branch.
**Blocked by:** none. Pull `organized` first; everything below is already in it.

> **Read [`raytun3r/PAPER.md`](../../../raytun3r/PAPER.md), not the PDF.** The whole
> paper — every hyperparameter, all five result tables, all four ablation tables,
> the named sequences, and the seven things it never specifies — is condensed
> there. It was written from the actual PDF (2026-08-05) precisely so these runs
> never need it open.

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

**On "the same number of GPUs as the paper" — now settled from the PDF.** The
paper never states a GPU count, and Appendix D says why it would not matter:
experiments ran on single **RTX A6000 / A4000** cards "with independent scenes,
sequences, and baselines executed in parallel when possible", ~180–250 GPU-hours
total. So the paper's setting *is* one GPU per scene with scenes in parallel —
exactly what `--workers` does. Nothing to shard within a scene; a single adapter
shared across scenes would be a different method. This question is closed.

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

### Run 5 (new, rank alongside Run 3) — is 300 iterations simply too few?

```bash
python -m raytun3r.experiments.iters_sweep --backbone da3 --variant small --weights pretrained --dataset scannetpp --path /netapp/datasets/scannetpp/data/3f15a9266d --out runs/iters-sweep/3f15a9266d-da3s
```

Reading the PDF turned up the one hyperparameter Sec. 4.3 never states: **the
number of optimisation steps**. We guessed `--iters 300` (~3 min). Appendix D
quotes **2–3 h per ScanNet++ scene** for a full train-and-evaluate run. Even
allowing that their figure covers full-sequence evaluation and every baseline,
three minutes of fitting does not sit comfortably inside two hours — so we may be
reporting an undertrained adapter and calling it a failed reproduction.

Sweeps 300 → 10000 (~2.3 GPU-h for the scene, points independent so they can be
spread over GPUs). If `raytun3r`'s `R_deg` is still falling at 300, that is the
answer and every other run here needs redoing at the elbow — **so run this before
Runs 1 and 2 if GPU time is tight.** It is orthogonal to Run 3: FOV asks what the
methods are *scored on*, this asks how long ours was *fitted*.

### Also worth doing

* **The queued `s10-adt-seq131` run**, now that conventions are fixed.
* **Find ScanNet++ `render_depth/`.** It was absent, so `AbsRel`/`δ₁.₂₅` (Tab. 3)
  have never been measured on ScanNet++ at all. If it is not in the download,
  say so and Tab. 3 stays out of scope for this dataset.

## Do this first — it is two minutes and may reframe everything

Comparing our one real run against Tab. 2 (same scene `3f15…`, same VGGT
backbone) shows our *unadapted* numbers are far **better** than the paper's, and
only our *adapted* one is worse: vanilla `R°` 2.379 vs their 7.21, Center-PH 0.378
vs 2.45, RayTun3R 1.858 vs 0.93. Every `d_reproj` of ours is 14–30× smaller than
theirs. That is the signature of an evaluation that is *easier* than the paper's,
not of a broken adapter — there is only 2.4° of error available to remove where
they had 7.2°.

So before any long run, print the geometry of the pairs actually being scored:

```bash
python -c "
from raytun3r.data import ScanNetPPFisheye
import torch
s = ScanNetPPFisheye('/netapp/datasets/scannetpp/data/3f15a9266d')
P = [s.pose(i) for i in range(len(s))]
C = [(-R.T@t) for R,t in P]
d = torch.tensor([ (C[i+1]-C[i]).norm() for i in range(len(C)-1) ])
print('frames', len(s), 'consecutive baseline: median', d.median().item(), 'mean', d.mean().item())
"
```

We measured ~1.1 cm at stride 1 against ~3 m of scene depth, which is why
`--stride 10` exists. But ScanNet++ DSLR is a *sparse* handheld capture of a few
hundred images per scene — 1.1 cm between consecutive DSLR shots is not what that
should look like. If the number comes back that small, check whether
`transforms.json` is being read in capture order and whether we are picking up a
denser image set than the DSLR one. **If our pairs are much closer together than
the paper's consecutive-pair protocol, that alone could explain both the too-good
vanilla and the inverted ordering**, and it is cheaper to fix than either sweep.

## How to read the numbers (from the PDF — avoids two false alarms)

* **RayTun3R is not supposed to win `d_reproj`.** In the paper's own Tab. 1 it
  loses that column to Center-PH or Multi-PH on **4 of 5 datasets** (ScanNet++:
  RayTun3R 4.16 vs Multi-PH 1.63). Center-PH also wins depth on ScanNet++ outright
  (AbsRel 0.066 vs 0.108, Tab. 3). The claim is about **pose** — `R°` and `t°`. A
  `d_reproj` or AbsRel loss is a reproduction, not a failure.
* **The target for `R°` on ScanNet++ `3f15…`** is Tab. 2: VGGT vanilla 7.21 →
  RayTun3R **0.93**, with Center-PH at 2.45. So RayTun3R should beat Center-PH by
  ~2.6×. Our run had Center-PH winning by ~5× — inverted, which is the real anomaly
  and what Runs 3 and 5 are for.
* Tab. 1 and Tab. 3 are aggregates over unnamed scenes; only Tab. 2 is
  single-sequence. Match the sequence before comparing.

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
