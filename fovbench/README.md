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

That writes AbsRel, δ₁ and the unaligned scale ratio against **both** axes
(incidence angle and distance from the optical centre) for the radial protocol,
and against window aim for the window protocol — one panel per view × stream.

## The scoring protocol

**The scale (and shift) is fitted once per frame, over every valid pixel, and
then frozen. Binning is a masking step applied afterwards to that frozen
prediction.** Both binning axes below read off the same single fit, so they are
two readings of one measurement (`geometry.bin_by`). Fitting per bin would hand
an up-to-scale model a separate scale at every radius and flatten exactly the
effect being looked for.

`AbsRel`, `delta1`, `RMSE` and `pen` all obey this. **`drift*` does not, and it
is the only column that does not** — see below.

## Reading the output

Two summary columns, and they answer different questions.

**`pen`** — AbsRel in the outermost populated bin ÷ AbsRel in the innermost.
How much worse the periphery is, in the metric a downstream user reads.

**`drift*`** — **outside the protocol above, deliberately, and marked with the
asterisk everywhere it appears.** `median(gt/pred)` after the model's own affine
is fitted **on the innermost bin alone**, innermost ÷ outermost. Above 1.0 the
model over-predicts depth toward the rim. **`radial` protocol only**: every
window is a separate forward pass of an up-to-scale model, so a window-to-window
ratio compares two arbitrary constants — blank there.

It is kept because it separates *the model bends depth with radius* from *the
model is just noisier at the rim*, and no whole-frame-fitted column can: a global
affine spends its scale and shift on the radial trend itself and reads 0.965 for
a real `+0.6·θ²` bias. Read it as a diagnostic beside the protocol, never as part
of it, and do not quote it in the same breath as the columns above.

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
are the protocols they were built for. `pen` is a within-model ratio, so the
alignment protocol cancels and it *is* comparable. So is `drift*`, but of a
differently-fitted quantity: compare a `drift*` only against another `drift*`.

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

## What it found

Run `fovbench-v2-ef2d50b`, split `fcc6c600f83b` — 200 frames of one sequence
(`Apartment_release_clean_seq131_M1292`), all four models, both streams, both
views, ~4 h on one RTX 6000 Ada. Numbers in
[`results/fovbench-v2-ef2d50b/`](../results) on the `results` branch.

**1. The periphery is noisier, not systematically farther.** On the raw fisheye,
AbsRel roughly doubles from the centre to the 50–55° rim — `pen` 1.97 (DA3
synthetic), 1.83 (VGGT-Omega), 1.79 (VGGT-1B) — while `drift*` over the same
cells stays at 1.02–1.09. The rim error is variance, not a radial scale bend.
An earlier estimator put that bend at 14–19%; it was measuring each bin's scene
depth and is withdrawn.

**2. There *is* a small bend, and it is the lens.** Every one of the eight
model×stream pairs has a higher fisheye `drift*` than its own rect `drift*`, by
3–14 points — 8 for 8, and it survives restricting both views to the same
0–50° span, so it is not the rectified arm's corner sliver. Order of a few per
cent, not double digits.

**3. Rectifying helps, and costs field.** At the honest 40–50° bin, VGGT-1B
synthetic scores 0.074 rect against 0.102 fisheye; across the three depth heads
the rect `pen` is 1.00–1.25 against 1.79–1.97 on the raw lens. The price is in
the COVERAGE table: an ~85° pinhole has nothing past 42.3° except in its corners.

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

**Open, and not answered by this run:** `gt_median` did not exist when it ran, so
DAv2's rect `pen` of 0.73 — a *worse* centre than rim, on both streams — is still
unattributed. It is the shape a constant-error model makes when the centre is the
farthest content, and DAv2 is the one model scored in disparity space, where
error grows fastest with depth. One GT-only pass settles it.

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
| `tests/` | 82 CPU tests: no weights, no data, ~8 s |

Model loading, availability and downloads live in
[`finetune/eval/baselines/model_zoo.py`](../finetune/eval/baselines/model_zoo.py);
the fisheye camera model in
[`VGGT-360-fisheye/utils/fisheye_cam.py`](../VGGT-360-fisheye/utils/fisheye_cam.py);
the metric definitions in
[`finetune/eval/metrics.py`](../finetune/eval/metrics.py). This package adds the
FOV-location axis on top of those and duplicates none of them.
