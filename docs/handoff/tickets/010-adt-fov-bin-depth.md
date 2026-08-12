# Measure how deep each FOV bin actually is — the one confound left in #14

**Owner:** gpu
**Files I may touch:** nothing under `fovbench/` — runs only. Results to `results`.
**Blocked by:** none. Completes `fovbench-v2-ef2d50b`, does not supersede it.

## Goal

Re-score the #14 split so the depth confound is **removed**, not just recorded.

> **This ticket grew after you were handed it, and got more expensive — read
> this even if you already read the first version.** It was a GT-only pass,
> minutes on CPU, that would only have *reported* each bin's depth. Since then
> the correction itself is implemented (`standardise_by_depth`), and applying it
> needs the models' predictions, which are not saved. So this is a **full
> re-run, ~4 h, same shape as #14** — and it returns the GT-only answers as a
> by-product, so nothing is lost by folding them together.

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

`organized` @ HEAD. **Nothing about the existing scoring changed** —
`fovbench-v2-ef2d50b`'s numbers stand exactly as they are; these are new columns
beside them.

*Recording the confound:*

* `geometry.bin_by` emits **`gt_median`** and **`gt_spread`** (IQR/median) per
  bin, on both axes. Model-independent by construction.
* `report.py` prints a **BIN DEPTH** table under COVERAGE — one row per
  view × render size, since the GT is shared across models and streams.

*Removing it:*

* **`standardise_by_depth`** — direct standardisation, the oldest trick for the
  job. Cut the frame's valid GT at its own quartiles, score every
  (bin × stratum) cell, and average a bin's cells with the weight each stratum
  has in **the frame** rather than in that bin. Quartile cuts make those weights
  uniform, so it is the plain mean over strata. Lands as **`AbsRel_ds`**,
  **`delta1_ds`**, **`ds_strata`** per bin, and **`pen_ds`** next to `pen` in
  every table.
* **It is a reduction, not a removal, and the code says so.** On a scene whose
  entire radial penalty is depth, the share still standing afterwards is 100% at
  1 stratum, 44% at 2, **25% at 4**, and undefined beyond — because the bins stop
  overlapping in depth. Four is the last value that standardises every bin.
  **So `pen_ds` clearly above 1.0 is a real effect; `pen_ds` near 1.0 means
  "mostly depth", not "nothing there".**
* A bin that misses a depth stratum gets **`—`**, never a partial average over
  whichever strata happened to be populated — that would reintroduce the very
  bias being removed. If the rim never sees far depth, what it would score at
  the centre's depth is not in this data, and no arithmetic can supply it.
* Figures: one panel per view × stream instead of eight curves in one, plus
  standardised versions of the AbsRel and δ₁ charts on both axes.

*Two more things, added after the first version of this ticket:*

* **CONTINUOUS profiles on both coordinates**, at **1°** of incidence angle and
  **0.025** half-widths of radius — beside the six-bin tables, not replacing
  them, and off the same frozen alignment fit. Six bins can tell you the rim is
  worse; they cannot tell you *where* it starts, whether it is a ramp or a knee,
  or whether the four models turn over in the same place. They can also invent
  a shape, since the edges are arbitrary. `geometry.fine_profile` accumulates
  sums per frame with one `bincount` and `pool_profiles` divides once at the
  end. Lands as `profiles.theta` / `profiles.radius` per radial run, and as
  `figures/profile_{AbsRel,delta1}_{theta,radius}.png`.
  **These are pooled over frames (pixel-weighted); the coarse tables are
  averaged per frame.** That is deliberate — a fine bin holds a handful of
  pixels in one frame and thousands in another — but it means the two are
  different estimators and must not be quoted as one number at two resolutions.
* **The window `pen` no longer spans a clipped aim.** A 40° *square* window has
  a 27.2° half-diagonal, so aiming it 40° off-axis puts its corners outside the
  54.83° cone: your own run measured `in_cone_frac` **0.842** there and 1.000
  everywhere else, and `pen` was therefore comparing two windows that differ in
  dead area as well as in aim. Clipped cells are still scored and printed, now
  flagged `t40!` and ringed, but excluded from the ratio. Over the clean aims
  the rectified window is **flat** (`pen` 0.90–1.21, median 1.04) while the raw
  one still climbs (1.14–1.67, median 1.35) — a sharper "rectifying helps" than
  the radial arm can give, and it was buried. A new WINDOW GEOMETRY table prints
  `in_cone_frac` and `src_px_per_out_px` per aim.
* 119 CPU tests green (Python 3.8+). `bin_by` had no direct test at all despite being the
  driver's entry point; it now has several, including one that pins the
  shared-fit rule and one that pins the strata table above.

Cost: **+164 ms** per frame per (view, stream) cell for the standardised
columns and **+54 ms** for both profiles, measured at 518px — together about
**5%** on top of #14's 3 h 53 m, and ~100 KB more `results.json`. Everything
else is unchanged.

## The command

```bash
git -C <repo> pull --ff-only origin organized
python -m fovbench.run --adt-root "$ADT" --n-frames 200 \
  --models vggt_1b,vggt_omega,dav2_large,da3_large \
  --out eval_out/fovbench_v3 2>&1 | tee eval_out/fovbench_v3.log
```

Same models, same 200 frames, same box, same env as #14 — **the point is that
only the scoring columns are new**, so every old number must reproduce and any
that does not is a bug worth stopping for.

**The split must come out `fcc6c600f83b`.** If it does not, stop and say so: the
new columns would then describe different frames from the ones #14 scored and
could not be read against them.

Smoke first, as you did last time — `--models vggt_1b --protocols radial
--n-frames 3` — and check `AbsRel_ds` is populated rather than `—` on real data
before committing four hours. If real ADT bins turn out **not** to overlap in
depth, the standardised column is `—` everywhere, and that is itself the answer:
report it and do not run the full grid.

## What to report

0. **That #14's numbers reproduced.** Same split, same frames, same models — the
   plain `AbsRel`/`delta1`/`pen`/`drift*` columns should come back bit-similar.
   Say so, and if anything moved, that first.
1. **`pen` against `pen_ds`, for all sixteen radial cells.** This is the headline
   the run exists for. Read against the rule above: clearly above 1.0 after
   standardising = a real field-position effect; near 1.0 = the raw `pen` was
   mostly depth. The v2 headline to test is fisheye synthetic `pen` **1.97 / 1.83
   / 1.79** (da3 / omega / vggt_1b) — how much of that survives?
2. **The BIN DEPTH table**, both views, θ axis and radius axis, and whether bin
   depth falls with eccentricity — give the innermost/outermost ratio per view.
3. **The DAv2 question, answered:** put the rect θ-bin depths beside DAv2's rect
   AbsRel row from `fovbench-v2-ef2d50b` (0.122, 0.093, 0.087, 0.090, 0.089,
   0.089 real), then give its `pen_ds`. If the centre bin is the farthest by
   roughly the factor its AbsRel is worse, the 0.73 `pen` is depth, not field
   position, and `pen_ds` should come back near 1.0 or above. If `pen_ds` is
   still 0.73, DAv2 has a real centre pathology and that is a finding about
   disparity models. Say which, with the arithmetic shown.
4. **How many bins could be standardised at all** — the `ds_strata` column, and
   any `—`. A `—` is not a failure, it is the finding that those bins do not
   overlap in depth, which would mean the confound can only ever be *reported*
   on this data. Say plainly which case we are in.
5. **Whether the rect and fisheye arms see different depths at the same θ.** They
   should not by much — same rays — and a large gap would mean the rectified
   resample is dropping content non-uniformly, which would matter to every
   rect-vs-fisheye comparison in #14.
6. `gt_spread` per bin, since it is free here: it is the anchor-conditioning
   number the `drift*` guard uses, and #14 asserted 0.71–0.88 from a 3-frame
   smoke. Confirm or correct that at 200.

7. **The continuous profiles.** Paste the four `profile_*.png`, and answer in
   words what they show that the six bins do not: **at what angle (and what
   radius) does each model's error actually start to rise**, is it a ramp or a
   knee, and do the four models turn over in the same place? Also say where each
   curve's *minimum* sits — that is where the single global affine chose to be
   right, and it is a property of the fit rather than of the lens.
8. **Whether the profile and the table agree** where they should. Re-aggregate
   the fine bins inside one coarse bin and compare; they will not match exactly,
   because the profile is pixel-weighted and the table is frame-weighted, and a
   *large* gap would mean the frames differ a lot in how much of each bin they
   fill — worth knowing either way.

Do not tune anything toward any of these outcomes; all of them are publishable
and the `—` case is the most interesting of the three.

## Done when

- [ ] split digest confirmed `fcc6c600f83b`
- [ ] #14's plain columns confirmed reproduced
- [ ] `pen` vs `pen_ds` table pasted for all sixteen radial cells
- [ ] BIN DEPTH tables pasted, both axes
- [ ] the nine items above answered
- [ ] the four profile_*.png pasted
- [ ] pushed to `results` under a new run id; hand back to `cpu`

## Needs CPU-Claude afterwards?

yes, and substantially. If `pen_ds` comes back far below `pen`, the README's
"what it found" section is overstated everywhere it says the periphery is worse,
not just for DAv2, and the headline becomes a smaller and more careful claim. If
`pen_ds` holds up, the result gets stronger than it currently is and can be
stated without the depth caveat.
