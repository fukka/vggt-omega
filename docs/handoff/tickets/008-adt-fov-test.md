# Run the ADT-FOV test — four vanilla models vs position in the field of view

**Owner:** gpu
**Status:** **done** — `results/fovbench-main-22c108d`. Note: the `drift` column it reports was removed from the experiment in #017; read the AbsRel/delta1 tables only.
**Files I may touch:** nothing under `fovbench/` — runs only. Results to `results`.
**Blocked by:** none. Independent of #4–#11 (different subproject, different data).

## Goal

Produce the first real ADT-FOV table: how VGGT-1B, VGGT-Omega, DAv2-Large and
DA3-Large degrade with **where in the field of view** the content sits, on
rectified perspective *and* raw fisheye, from synthetic *and* real ADT input.

Nothing here trains anything. Four models, off the shelf, forward passes only.

## What is already done, and what is not

`fovbench/` is complete and committed (`organized` @ fa7e601). Read
`fovbench/README.md` — it is the protocol, and it is short.

**Verified on the Mac:** 72 CPU tests (~5 s, no weights, no data), plus the whole
pipeline run end to end on real ADT frames with `--models analytic`, a stand-in
that bends depth by a known `+0.6·θ²`. The harness recovers that bias to within
0.3% of its analytic value. So the view construction, the ground-truth
convention warp, the alignment, the binning and the summary are measured to be
faithful.

**Not verified:** any number from any real network. No weights have run against
this code at all. You are the first.

## Setup — do this before the long run

DA3 needs a pip install and VGGT-Omega's weights are gated, so both report an
instruction rather than a traceback. The run **refuses to start** if any
requested model is unavailable, precisely so you cannot end up with a two-model
report that looks like a four-model one.

```bash
python -m finetune.eval.baselines.benchmark_adt --list | head -8
```

```bash
pip install --no-deps depth-anything-3 && pip install omegaconf addict einops
export VGGT_OMEGA_CKPT=$OMEGA_CKPT
python -m finetune.eval.baselines.benchmark_adt --download --models vggt_1b,da3_large,dav2_large
```

If DA3 cannot be installed on the box, say so and run the other three with
`--skip-unavailable` — the report and `results.json` will name what was left out.
Do not silently drop it.

## The commands

Smoke first — one model, one protocol, 3 frames. If this does not produce a
`report.txt` with numbers in it, stop and comment rather than starting the grid:

```bash
python -m fovbench.run --adt-root "$ADT" --n-frames 3 \
  --models vggt_1b --protocols radial \
  --out eval_out/fovbench_smoke 2>&1 | tee eval_out/fovbench_smoke.log
```

Then the run:

```bash
python -m fovbench.run --adt-root "$ADT" --n-frames 25 \
  --models vggt_1b,vggt_omega,dav2_large,da3_large \
  --out eval_out/fovbench_main 2>&1 | tee eval_out/fovbench_main.log
```

`--n-frames` is **per sequence** and spread evenly across it, not a prefix.
Models load once each and every frame/stream/view/window goes through them, so
cost is roughly `4 models × n_frames × n_seqs × 2 streams × 2 views × 18 forward
passes`. Launch it detached and do not sit and poll it.

## What to report back

The `report.txt` in full is fine — it is designed to be pasted. Plus one line
answering each of these:

1. **Does error grow with eccentricity, and by how much?** Read the `drift`
   column, not `pen` (see below).
2. **Does rectifying first help?** `rect` vs `fisheye` on the same model and
   stream.
3. **Synthetic vs real** — how much of the degradation is the lens and how much
   is the sensor.
4. **Do the four models differ, or is this a family-wide trait?** VGGT-1B is
   DINOv2 + absolute PE; VGGT-Omega is DINOv3, RoPE only. If they diverge, that
   is a positional-encoding result, not a checkpoint result.

## How to read it, and the trap in it

`pen` (AbsRel outer ÷ inner) is a **residual after one global affine fit**, and a
least-squares affine picks the radius at which it is right. A cleanly monotone
radial error therefore comes out **U-shaped**, and `pen` can read ≈1.0 for a
model that is wrong by 50% at the rim — measured, on the analytic stand-in:
AbsRel `0.175 0.153 0.112 0.047 0.081 0.172`, `pen` 0.98.

`drift` (median gt/pred, **unaligned**, inner ÷ outer) has no fit in the way and
stays monotone. **`drift` is the answer; `pen` is how it feels to a downstream
user who fits a scale.** Quote both.

Absolute AbsRel is **not** comparable across models — DAv2 is scored under a
disparity-space affine and the depth heads under a depth-space one, because those
are the protocols they were built for. `pen` and `drift` are within-model ratios,
so they *are* comparable. Do not rank the models on AbsRel.

## The confounds it reports rather than hides — quote these, do not omit them

* **The rectified arm cannot reach the rim.** A ~85° rectified pinhole has no
  pixels past 42.3° off-axis except in its corners (52.6° at most). Its outer
  bins will be thin or empty. That is geometry, and it is itself a cost of
  rectifying — not model failure, and not a fair "rect wins at the rim".
  The COVERAGE table prints the per-bin pixel counts; paste it.
* **The real stream is blurred *and* imperfectly registered** to the GT. A
  synthetic-vs-real gap is sensor reality plus registration, not blur alone.
* **Render size differs.** VGGT-Omega takes 512, the rest 518, because each view
  is rendered at the model's own token grid rather than resampled. Per-bin pixel
  counts differ slightly between them; the COVERAGE table now has a row per size.
* **These are multi-view models run single-view.** Deliberate — the comparison is
  against monocular nets — but say so in any writeup.

## What makes it able to fail

This is not a run that can only succeed. Three outcomes that would mean something
is wrong rather than something was found:

* **All four models show near-identical `drift`.** Four different
  architectures agreeing to three digits is a harness signature, not a model
  property. Say so rather than reporting it as a finding.
* **`drift` ≈ 1.000 everywhere on every model.** Either off-the-shelf nets are
  genuinely flat across a 55° cone — surprising, and worth stating plainly if
  true — or the GT is being warped into the prediction's own error. The
  convention tests pin the warp against analytic scenes, so lean toward the
  former, but flag it.
* **A bin's `n_frames` is 0 where you expected pixels.** Look at COVERAGE before
  concluding anything about that bin.

The honest prior is that pinhole-trained networks read the periphery as less
inclined than it is, so `drift` > 1 (depth over-predicted at the rim). **Do not
tune anything toward that.** If it comes out below 1, report it below 1 — this
repo has already been burnt once by fitting an unstated setting until a published
number appeared.

## Recording

Push `eval_out/fovbench_main/{results.json,results.csv,report.txt,manifest.json}`
and the smoke log to `results`. **Not** the figures — they regenerate from the
JSON:

```bash
python -c "import json,fovbench.report as R; R.write_all(json.load(open('eval_out/fovbench_main/results.json')),'eval_out/fovbench_main')"
```

`manifest.json` carries the split `digest`; every table is keyed to it, and two
runs are comparable only if their digests match.

## Done when

- [ ] `--list` output pasted, so availability is on the record
- [ ] `report.txt` pasted in full
- [ ] one line each on the four questions above
- [ ] the COVERAGE table quoted, not omitted
- [ ] anything skipped named explicitly
- [ ] pushed to `results`; hand back to `cpu`

## Needs CPU-Claude afterwards?

yes — interpreting the table, and deciding whether the VGGT-1B vs VGGT-Omega gap
(if any) is a positional-encoding result worth its own writeup.
