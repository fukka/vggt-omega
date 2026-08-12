# Re-score ADT-FOV: three pictures, and the real GT depth behind them

**Owner:** gpu
**Files I may touch:** nothing under `fovbench/` — runs only. Results to `results`.
**Blocked by:** none. Supersedes `fovbench-v2-ef2d50b`'s radial arm; its window
arm stands and is not re-run.

> **This ticket was rewritten and is now much smaller.** Earlier versions asked
> for nine reported items, a `drift*` column and a full 4 h grid. The owner has
> since scoped the experiment: **AbsRel and δ₁ against position in the field, for
> the four baselines, plus the GT depth those metrics were divided by.** Ignore
> anything you remember from the previous versions.

## Goal

One command, ~15 minutes, producing three figures and the JSON behind them.

## What the experiment asks

How does each baseline's depth accuracy vary with **where in the field of view**
the content sits — on the raw fisheye and after rectifying, on synthetic and on
real pixels? Two metrics, two readings of "where":

| | |
|---|---|
| `AbsRel`, `delta1` | the metrics |
| `theta` | incidence angle off the optical axis — the ray direction, and the **only** axis on which `rect` and `fisheye` mean the same thing |
| `radius` | distance from the optical centre in the image plane, in half-widths — where in the *picture*. Each view measures it in its own plane, so the same radius is a different direction in each |

Both axes at two resolutions off **one** alignment fit per frame: six coarse bins
for the tables, and a continuous profile (1°, 0.025 half-widths) for the curves.

## The command

```bash
git -C <repo> pull --ff-only origin organized
python -m fovbench.run --adt-root "$ADT" --n-frames 200 --protocols radial \
  --models vggt_1b,vggt_omega,dav2_large,da3_large \
  --out eval_out/fovbench_v3 2>&1 | tee eval_out/fovbench_v3.log
```

**`--protocols radial` is the whole cost story.** The window sweep is 3 400
forward passes per (model, stream, view) against radial's 200 — **94.4% of your
3 h 53 m** — and it gains nothing here: every new column (`gt_median`,
`gt_std`, the profiles, `AbsRel_ds`) is produced inside `bin_by`, which only the
radial path calls. `_score_window` goes straight to `depth_metrics`. So the
window numbers in `fovbench-v2-ef2d50b` stand as they are and re-running them
would be pure duplication.

Expect **~15 minutes**, not four hours.

**The split must come out `fcc6c600f83b`** — same sequence, same 200 frames, so
every old radial number must reproduce. If it does not, stop and say so.

## What is new in the output

`organized` @ HEAD. Nothing about the existing scoring changed.

* **`gt_median` and `gt_std` per bin, and `gt_mean`/`gt_std` along the
  continuous profile** — the real GT depth of the frames actually scored,
  identical for every model and both streams. This is the item the owner asked
  for by name, and it must come from the data, not from any model of a room.
* **Continuous profiles** on both axes, pooled over frames (pixel-weighted).
  The coarse tables are averaged per frame, so the two are different estimators
  of the same thing and should not be quoted as one.
* **`pen_ds`** beside `pen`: the same outer/inner AbsRel ratio after each bin is
  re-scored at the frame's own depth mix (quartile strata). A **reduction** of
  the depth confound, not a removal — ~25% of a purely-depth penalty still
  stands at four strata — so `pen_ds` clearly above 1.0 is a real
  field-position effect and `pen_ds` near 1.0 means "mostly depth". `—` means a
  bin missed a depth stratum in at least half the frames, which is itself the
  answer: on this data the confound could then only be reported, not corrected.
* **`drift*` is gone** from every table, legend, CSV and figure. It is still in
  `results.json` because `datasets_egosynth` cross-checks against it; ignore it.

116 CPU tests green, **Python 3.8+** (the ego-synth code uses the walrus
operator, so the default `python3` on some boxes will not import the package).

## What to report

Short. Four things.

1. **The three figures** — `figures/AbsRel.png`, `figures/delta1.png`,
   `figures/gt_depth.png`. Paste all three. Each already carries every model,
   both views, both streams and both axes; no extra assembly.
2. **That the old radial numbers reproduced.** Same split, same frames — say so,
   and if anything moved, that first.
3. **What `gt_depth.png` shows**, in words and numbers: does the measured GT
   depth fall with eccentricity, on each view and each axis, and by what ratio
   innermost to outermost? Give the per-bin medians. *This is the question the
   owner asked; do not substitute an argument for the measurement.*
4. **`pen` against `pen_ds`** for the eight radial cells, and how many bins could
   be standardised at all (`ds_strata`, and any `—`).

Do not tune anything toward any outcome. If `pen_ds` collapses the effect, that
is the result; if it survives, that is the result.

## Done when

- [ ] split digest confirmed `fcc6c600f83b`
- [ ] the old radial numbers confirmed reproduced
- [ ] the three figures pasted
- [ ] items 3 and 4 answered with numbers
- [ ] pushed to `results` under a new run id; hand back to `cpu`

## Needs CPU-Claude afterwards?

Yes — folding the measured depth into `fovbench/README.md`, which currently
carries a *modelled* estimate of the depth trend that should be replaced with
the real one, and scaling back any `pen` claim that `pen_ds` does not support.
