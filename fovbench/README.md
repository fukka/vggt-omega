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
* **binning axis** — every radial run is binned twice off the same fit: by
  **distance from the optical centre** (in half-widths of that view's own frame;
  1.0 = middle of a frame edge, √2 = a corner) and by **incidence angle**. They
  are not interchangeable: on the raw fisheye radius is near-proportional to θ,
  on the rectified pinhole it goes as `tan θ`, so **a given radius is a different
  direction in the two views**. Radius answers *where in the picture*; θ answers
  *which ray*, and `rect` vs `fisheye` is like-for-like only on θ.
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

The `results` branch carries JSON and logs, never images
([`POLICY.md`](../docs/handoff/POLICY.md)), so redraw the figures from any run's
`results.json` — no data, no GPU, no re-scoring:

```bash
python -c "import json;from fovbench import report;report.write_figures(json.load(open('results.json')),'figs')"
```

That writes exactly three pictures — `AbsRel.png`, `delta1.png`, `gt_depth.png` —
each carrying every model, both views, both streams and **both axes**. The line
is the continuous 1° profile and the dots are the six binned values; **both are
pooled over frames and weighted by pixels**, so they are one estimator at two
resolutions and any gap between them is the bin width. `gt_depth.png` adds a
second row: the pixel count behind every part of the curve, as a density so the
bars and the profile are the same quantity. A run that never measured depth gets
no depth page, rather than an empty one.

## The scoring protocol

**The scale (and shift) is fitted once per frame, over every valid pixel, and
then frozen. Binning is a masking step applied afterwards to that frozen
prediction.** Both binning axes below read off the same single fit, so they are
two readings of one measurement (`geometry.bin_by`). Fitting per bin would hand
an up-to-scale model a separate scale at every radius and flatten exactly the
effect being looked for.

Every column obeys this — `AbsRel`, `delta1`, `RMSE` and `pen` alike.

## Reading the output

**`pen`** — AbsRel in the outermost populated bin ÷ AbsRel in the innermost.
How much worse the periphery is, in the metric a downstream user reads.

Absolute AbsRel is comparable **only among models that share an alignment
protocol**, and here three of the four do: VGGT-1B, VGGT-Omega and DA3 are all
scored under the same depth-space affine, so their levels can be read against
each other directly. **DAv2 cannot** — it is scored under a disparity-space
affine, because that is the protocol it was built for, and no column reconciles
the two. `pen` is a within-model ratio, so the alignment protocol cancels and it
is comparable across all four.

**`gt_median`** — per bin, and not a score: the median GT depth of what that bin
was looking at. Every metric here is relative and grows with depth, so "the rim
is worse" is a claim about *field position* only once the bins are known to sit
at comparable depths — and in an egocentric indoor frame they need not. A model
with a constant 10 cm error and no radial behaviour whatever scores a rising
AbsRel curve on a scene that gets nearer toward the rim
(`tests/test_geometry.py::test_gt_median_exposes_the_depth_confound_in_absrel`).
This is the same confound that made the withdrawn `raw_scale_ratio` drift read a
radial trend out of a flat model, so it is now carried in the tables rather than
argued about. `report.txt` prints it as the BIN DEPTH table.

**Nothing corrects for it.** An earlier version carried a `pen_ds` column that
re-scored each bin at the frame's own depth mix. It came out `—` in all sixteen
cells of the 200-frame run: the innermost bin is both the smallest (4.8k px
against 36k at the rim) and the *narrowest in depth* (`gt_spread` 0.42 against
0.95), because a 10° cone on the far wall is close to a single depth, so it
misses a depth quartile in 83% of frames. A correction that fails exactly where
every comparison starts is worse than none, so the column is gone and the
measured depth is reported instead. Weigh it yourself.

## What it found

Run `fovbench-v2-ef2d50b`, split `fcc6c600f83b` — 200 frames of one sequence
(`Apartment_release_clean_seq131_M1292`), all four models, both streams, both
views, ~4 h on one RTX 6000 Ada. Numbers in
[`results/fovbench-v2-ef2d50b/`](../results) on the `results` branch.

> **The depth confound, measured.** Every metric here is relative, so a bin that
> is nearer scores worse for that reason alone. Run `fovbench-v3-24b38e1`
> measured the GT depth of the scored frames, per bin, model-independently:
>
> | axis · view | 0–10° | 10–20 | 20–30 | 30–40 | 40–50 | 50–55 | outer/inner |
> |---|---|---|---|---|---|---|---|
> | θ · fisheye | 3.00 | 2.87 | 2.65 | 2.31 | 1.96 | 1.70 | **0.57×** |
> | θ · rect | 3.02 | 2.87 | 2.65 | 2.30 | 1.99 | 1.94 | **0.64×** |
>
> **Depth falls monotonically from the very first bin**, on both views and both
> axes. An earlier version of this section carried a *modelled* empty room
> instead, which got the endpoint right (1.70 m) and the shape wrong — it
> predicted planar z constant out to 35° and then collapsing, whereas seq131 is
> already down 12% by 20–30°. The measurement replaced it; do not reinstate the
> model.
>
> If a model's error were fixed in metres and it had no radial behaviour at all,
> that depth trend alone would raise AbsRel **1.77× (fisheye)** and **1.55×
> (rect)** across 0–55°. So read every raw `pen` below as an upper bound, and
> see item 7 for what survives when the confound is taken out.

**1. AbsRel roughly doubles toward the rim on the raw lens — and most of that
is the depth trend above.** `pen` reaches 1.97 (DA3 synthetic), 1.83 (Omega),
1.79 (VGGT-1B) on fisheye. The measured depth ratio over the same span is 1.77,
so a model with a constant absolute error and *no radial behaviour at all* would
score ~1.77 here. Raw `pen` is therefore close to uninformative about field
position on its own.

**2. Whether any of that is field position is not settled here, on purpose.**
This benchmark reports the two curves and the depth they were divided by, and
stops. It does not attempt a correction: an earlier version re-scored each bin
at the frame's own depth mix, and the run showed that the innermost bin cannot
be re-scored at all — it is both the smallest and the narrowest in depth, so it
misses a depth quartile in 83% of frames. A correction that fails exactly where
the comparison starts is worse than none, so the column was removed and the
measured depth is reported instead.

**3. Rectifying helps, and costs field.** At the honest 40–50° bin, VGGT-1B
synthetic scores 0.074 rect against 0.102 fisheye; across the three depth heads
the rect `pen` is 1.00–1.25 against 1.79–1.97 on the raw lens. The price is in
the COVERAGE table: an ~85° pinhole has nothing past 42.3° except in its corners.

**3b. The window sweep says it more sharply, once its own clipped aim is set
aside.** A 40° *square* window has a 27.2° half-diagonal, so from an aim of 30°
its corners leave the 54.83° Aria cone, and the 40° aim is only **84% imaged** —
that cell differs from the on-axis one in dead area as well as in aim, which is
the very confound the fixed-FOV design exists to avoid, and it is now excluded
from `pen` and flagged `t40!`. Over the fully-imaged aims, 0°→30°:

| | `pen`, 8 model × stream cells | median |
|---|---|---|
| **rect** window | 0.90 – 1.21 | **1.04** |
| **fisheye** window | 1.14 – 1.67 | **1.35** |

A rectified window scores the same wherever it is aimed; a raw one does not.
That is a cleaner statement of "rectifying helps" than the radial arm can make,
because here the two views see the same directions by construction. Including
the clipped aim inflates the rect numbers to 1.12–2.10 and hides it.

**4. The sensor sets the level, the lens sets the slope.** `real` sits well above
`synthetic` at every bin (VGGT-1B fisheye 0.110 vs 0.068 on axis) while the two
curves have nearly the same shape.

**5. The two axes have mirror-image blind spots, which is why both are kept.**
The rectified arm covers radius to the corners but its 50–55° θ bin is 2,939 px
of corner; the fisheye arm reaches 55° with 36,245 px but has *nothing* past
radius 1.0, the image circle being inscribed in the frame. Fisheye's two axes
nearly agree (`pen` 1.79 on θ, 1.71 on radius); rect's do not and can change
direction (1.08 on θ, 0.91 on radius), which is `tan θ` versus `θ`.

**6. 25 frames was not enough, and the run says so plainly.** Against the earlier
25-frame run every cell moved: levels by 15–47%, and in several cells the *shape*
changed, not just the height. VGGT-Omega's rect `pen` went 1.45 → 1.00 (a clear
rim penalty became flat) on a curve whose inner bins barely moved and whose
outermost — the 2,939 px corner sliver — moved 39%. VGGT-1B real fisheye went
`pen` 1.75 → 1.30 with the inner bins rising 17% and the outer falling 12%.
DAv2's anomalous rect *centre* bin does not exist at 25 frames at all: it is 0.058
there and 0.110 here. What is stable is the fisheye synthetic **shape** for the
three depth heads (`pen` within 0.07), even though those levels moved 15–19%
uniformly. So: read the fisheye synthetic shape claims from either run; read
nothing else from the 25-frame one. This also means one sequence at 200 frames is
the floor, not the target — the estimator was wrong at any n, *and* the sample
was thin.

**Still open.** One sequence — 200 frames of `Apartment_release_clean_seq131_M1292`,
so nothing here is an across-scene claim. At ~15 min per sequence radial-only
that is now a cheap gap to close.

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

Holding the FOV fixed is necessary and was not sufficient. A 40° *square* window
still reaches 27.2° along its diagonal, so an aim of 40° puts its corners well
outside the 54.83° cone and the window measures **0.842** imaged — a 16% black
wedge that moves with the swept variable, exactly the shape of the confound above,
just an order smaller. The 50% floor let it through. `pen` now spans
**fully-imaged aims only** (`MIN_CLEAN_CONE_FRAC`), clipped aims are flagged
`t40!` in the tables and ringed in the figures, and `in_cone_frac` is printed per
aim in the WINDOW GEOMETRY table rather than living only in `results.json`.

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
| `report.py` | tables, CSV, the three figures, `pen` |
| `tests/` | 123 CPU tests: no weights, no data, ~9 s (needs Python 3.8+) |

Model loading, availability and downloads live in
[`finetune/eval/baselines/model_zoo.py`](../finetune/eval/baselines/model_zoo.py);
the fisheye camera model in
[`VGGT-360-fisheye/utils/fisheye_cam.py`](../VGGT-360-fisheye/utils/fisheye_cam.py);
the metric definitions in
[`finetune/eval/metrics.py`](../finetune/eval/metrics.py). This package adds the
FOV-location axis on top of those and duplicates none of them.
