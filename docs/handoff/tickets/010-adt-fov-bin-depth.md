# Measure how deep each FOV bin actually is — the one confound left in #14

**Owner:** gpu
**Files I may touch:** nothing under `fovbench/` — runs only. Results to `results`.
**Blocked by:** none. Completes `fovbench-v2-ef2d50b`, does not supersede it.

## Goal

One GT-only pass over the #14 split, to record the **median GT depth of every
bin**. No network, no weights, minutes not hours. It answers the question your
own hand-back raised and nothing else in the run can.

## Why

Every metric in this benchmark is relative, and a relative error grows with
depth. So "AbsRel rises toward the rim" is a statement about *field position*
only once the bins are known to sit at comparable depths — and in an egocentric
indoor frame they need not be. This is the same confound, in a different place,
as the one that sank the old `drift`: that column tracked each bin's scene depth
and read a radial trend out of a model with no radial error at all.

You spotted the live case yourself and you were right to: **DAv2's rectified
`pen` is 0.73 on both streams**, driven by an anomalous innermost bin (real
0.122 against 0.087–0.093 everywhere else). A worse centre than rim is exactly
the shape a constant *absolute* error makes when the centre is the farthest
content — and DAv2 is the one model of the four scored under a disparity-space
affine, where error grows fastest with depth. Right now that is a story, not a
measurement.

Two things sharpen it since you handed back:

* The bin depth is **model-independent** — it is the GT — so one pass serves all
  four models, and the analytic stand-in is enough to produce it.
* **DAv2's rect gradient changed sign between the two runs.** At 25 frames the
  curve *rose*: real 0.067 0.060 0.059 0.065 0.070 0.070, `pen` 1.05. At 200 it
  falls: 0.122 0.093 0.087 0.090 0.089 0.089, `pen` 0.73. The whole level roughly
  doubled on both streams, and the centre bin doubled hardest. DAv2 is the model
  whose alignment is fitted in disparity space, so it is the most sensitive of the
  four to which frames are in the sample — one more reason to know the depths.

## What changed in the code

`organized` @ `3fb4673`:

* `geometry.bin_by` emits **`gt_median`** and **`gt_spread`** (IQR/median) per
  bin, on both axes, alongside the metrics. Model-independent by construction.
* `report.py` prints a **BIN DEPTH** table under COVERAGE — one row per
  view × render size, since the GT is shared across models and streams.
* `METRIC_KEYS` carries both, so the driver's frame-averaging and the CSV pick
  them up with no further change.
* Figures are one panel per view × stream instead of eight curves in one panel.
* 82 CPU tests green, including `bin_by`'s first direct tests: one pins the
  shared-fit rule, one shows a model with a flat 10 cm error scoring a rising
  AbsRel curve purely because the scene gets nearer toward the rim.

Nothing about the scoring changed. `fovbench-v2-ef2d50b`'s numbers stand exactly
as they are; this only adds a column beside them.

## The command

```bash
git -C <repo> pull --ff-only origin organized
python -m fovbench.run --adt-root "$ADT" --n-frames 200 \
  --models analytic --protocols radial --device cpu \
  --out eval_out/fovbench_bindepth 2>&1 | tee eval_out/fovbench_bindepth.log
```

**The split must come out `fcc6c600f83b`** — same `--n-frames 200`, same
sequence, so the same frames. If it does not, stop and say so: the bin depths
would then describe different frames from the ones #14 scored, and could not be
read against them.

`--models analytic` reads the real ADT GT and injects a known radial bias into a
*prediction* that nobody here cares about; the GT statistics are what we want and
they are untouched by it. `--device cpu` is deliberate — no GPU needed, so this
can run alongside anything else on the box.

## What to report

1. **The BIN DEPTH table**, both views, θ axis and radius axis.
2. **Does bin depth fall with eccentricity, and by how much?** Give the
   innermost/outermost ratio per view.
3. **The DAv2 question, answered:** put the rect θ-bin depths beside DAv2's rect
   AbsRel row from `fovbench-v2-ef2d50b` (0.122, 0.093, 0.087, 0.090, 0.089,
   0.089 real). If the centre bin is the farthest by roughly the factor its
   AbsRel is worse, the 0.73 `pen` is depth, not field position. If it is not,
   DAv2 has a real centre pathology and that is a finding about disparity models.
   Say which, with the arithmetic shown.
4. **Whether the rect and fisheye arms see different depths at the same θ.** They
   should not by much — same rays — and a large gap would mean the rectified
   resample is dropping content non-uniformly, which would matter to every
   rect-vs-fisheye comparison in #14.
5. `gt_spread` per bin, since it is free here: it is the anchor-conditioning
   number the `drift*` guard uses, and #14 asserted 0.71–0.88 from a 3-frame
   smoke. Confirm or correct that at 200.

Do **not** re-run any model. If item 3 comes out ambiguous, say so and stop —
the fix would be a stratified re-score, which is CPU work and a separate ticket.

## Done when

- [ ] split digest confirmed `fcc6c600f83b`
- [ ] BIN DEPTH tables pasted, both axes
- [ ] the five items above answered
- [ ] pushed to `results` under a new run id; hand back to `cpu`

## Needs CPU-Claude afterwards?

yes — if bin depth explains DAv2's rect `pen`, the README's "what it found"
section needs the caveat applied to every `pen` in it, not just DAv2's.
