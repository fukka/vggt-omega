# The ADT-FOV test

*Does monocular depth get worse toward the edge of a wide field of view, and
does rectifying the pixels first change the answer?*

Four vanilla, off-the-shelf networks on ADT's Aria 214-1 fisheye — no
fine-tuning, no adapters, no lens model given to any of them:

| key | model | emits | alignment |
|---|---|---|---|
| `vggt_1b` | VGGT-1B (DINOv2 + RoPE) | up-to-scale depth | affine in depth |
| `vggt_omega` | VGGT-Omega 1B/512 (DINOv3, RoPE only) | up-to-scale depth | affine in depth |
| `dav2_large` | Depth-Anything V2 Large | relative disparity | affine in **disparity** |
| `da3_large` | Depth-Anything 3 Large | up-to-scale depth | affine in depth |

## The grid

```
model × stream × view × protocol
```

* **stream** — `synthetic` (ADT's re-rendered `videos_synthetic`, sharp and
  pixel-registered to the GT) and `real` (`videos_rgb`, the actual sensor: motion
  blur, rolling shutter, real photometry). Both are scored on the **same frames**;
  see [`split.py`](split.py).
* **view** — `rect` (rectified perspective) and `fisheye` (raw pixels, no
  undistortion). In the window protocol the two are *paired*: the raw window is
  the square containing the rectified window's own source footprint, so it sees
  the same directions.
* **protocol** —
  * `radial`: the model gets the **whole frame once**; per-pixel errors are binned
    by incidence angle. *Where in this image.*
  * `window`: a fixed 40° window is **re-aimed** across the lens (tilt 0–40°,
    four azimuths) and scored on its own. *Where the camera was pointed.*

They can disagree, informatively. A model that is merely bad at image corners
shows a `radial` gradient and a flat `window` curve; one whose error is driven by
ray geometry shows both.

## Running it

```bash
python -m fovbench.run --adt-root $ADT --out eval_out/fovbench
```

No weights and no GPU — exercise the whole pipeline with an analytic model that
bends depth by a *known* function of eccentricity (ADT frames are still read, so
this needs the data; for a run that needs nothing at all, `pytest fovbench/tests`
drives the same path on a synthetic fisheye scene):

```bash
python -m fovbench.run --adt-root $ADT --models analytic --device cpu --n-frames 3
```

Check availability before committing to a run:

```bash
python -m finetune.eval.baselines.benchmark_adt --list
```

## Reading the output

Two summary columns, and they answer different questions.

**`pen`** — AbsRel in the outermost populated bin ÷ AbsRel in the innermost.
How much worse the periphery is, in the metric a downstream user reads.

**`drift`** — `median(gt/pred)` after the model's own affine is fitted **on the
innermost bin alone**, innermost ÷ outermost. Above 1.0 the model over-predicts
depth toward the rim. **`radial` protocol only**: every window is a separate
forward pass of an up-to-scale model, so a window-to-window ratio compares two
arbitrary constants — the column is left blank there.

Why anchored, and not the two obvious alternatives. Fitting *nothing* looks
alignment-free but every model here has an additive degree of freedom, and an
offset makes `median(gt/pred)` track each bin's *scene depth*: on a scene whose
depth falls with eccentricity, a model with **no radial error at all** reports
0.648 or 1.253 depending on the offset's sign, and an affine-invariant disparity
model reports 1.143 — the size of the effect being looked for. Fitting on the
*whole frame* spends the scale and shift partly on the radial trend itself:
correct on the no-distortion cases, but a real `+0.6·θ²` bias then reads 0.965.
Anchoring on a ~10° central band removes the offset without absorbing the trend:
exactly 1.000 on all four no-distortion cases, 1.37 against a true 1.49. It
under-reports by <10% and never invents. The anchor must have depth *spread* to
determine an affine — real ADT bands measure IQR/median 0.71–0.88; a single flat
wall measures 0.00 and is refused.

Report both. Per-bin AbsRel is a *residual after one global fit*, and a
least-squares affine chooses the radius at which it is right — so a cleanly
monotone radial error comes out **U-shaped**, and `pen` can read ≈1.0 for a model
that is wrong by 50% at the rim. Measured on the analytic stand-in with a known
`+0.6·θ²` bias the AbsRel curve comes out a bowl, so `pen` reads ≈1.0 while
`drift` recovers the bias;
[`tests/test_end_to_end.py`](tests/test_end_to_end.py) asserts that against an
independently derived value, with no data needed. `pen` says how it feels;
`drift` says what the model did.

Absolute AbsRel is **not** comparable across models — DAv2 is scored under a
disparity-space affine and the depth heads under a depth-space one, because those
are the protocols they were built for. `pen` and `drift` are within-model ratios,
so the alignment protocol cancels and they *are* comparable.

## Two things that would quietly invalidate this, and how they are held

**Depth conventions.** ADT GT is planar z about the *camera* axis; every depth
head emits planar z about *its own* view axis. For a window at tilt those differ
by a per-pixel `cos(θ_window)/cos(θ_camera)` — up to 2.15× on this lens, radial,
and so not absorbable by any scale-and-shift. GT is converted once, at the warp,
and [`tests/test_geometry.py`](tests/test_geometry.py) pins it against analytic
scenes with closed-form depth.

**The dead-pixel confound.** An earlier sweep in this repo varied per-window FOV
while holding the ring layout fixed, so wider windows overhang the imaged cone and
width and dead area moved together — the 110° arm's AbsRel 0.582 was ~38% black
wedge, not the cost of width. Here the window FOV is **held fixed** and only the
aim moves; `in_cone_frac` is reported per window and anything under 50% is
dropped rather than scored.

## What this does not claim

Nothing here corrects for distortion and no model is given the lens. These are
measurements of how four off-the-shelf networks degrade on an uncorrected
wide-FOV camera — not an attempt to improve them, and not a statement about what
any of them would do fine-tuned.

The `real` stream carries a caveat of its own: `videos_rgb` is only as well
registered to the GT as the digital twin's pose, and it is motion-blurred. A gap
between the streams is *sensor reality plus registration*, not blur alone.

## Layout

| file | holds |
|---|---|
| `split.py` | the frozen frame manifest + digest (the comparability token) |
| `geometry.py` | window rendering, GT convention warp, θ maps, radial binning |
| `models.py` | the four models behind one call + the analytic stand-in |
| `run.py` | the driver (CLI) |
| `report.py` | tables, CSV, figures, `pen`/`drift` |
| `tests/` | 77 CPU tests: no weights, no data, ~7 s |

Model loading, availability and downloads live in
[`finetune/eval/baselines/model_zoo.py`](../finetune/eval/baselines/model_zoo.py);
the fisheye camera model in
[`VGGT-360-fisheye/utils/fisheye_cam.py`](../VGGT-360-fisheye/utils/fisheye_cam.py);
the metric definitions in
[`finetune/eval/metrics.py`](../finetune/eval/metrics.py). This package adds the
FOV-location axis on top of those and duplicates none of them.
