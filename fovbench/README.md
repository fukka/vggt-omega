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

> **Every published `vggt_1b` column here was produced in fp32 (#021).**
> `raytun3r.backbones.VGGTBackbone.forward` was the one VGGT call site in the
> repo not opening the bf16 autocast the model is written to expect — the other
> three models were already on their own reference precision. Fixed after these
> runs and **deliberately not re-run**, so no current checkout reproduces the
> `vggt_1b` rows in any `results/fovbench-*`. The accuracy cost is ≤0.51% on
> AbsRel with no model ordering flip, measured on the SLAM grid; nothing this
> README concludes depends on a margin that small. Any *timing* comparison
> against the other three models is invalid for these runs.

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

**The harness is CPU-bound, not GPU-bound.** A run of the six-sequence Part A
sat at 25–30% on one GPU with ~2 of 64 cores busy: the per-frame cost is the
1408² decode, the fisheye→pinhole remap, and the binning arithmetic, none of
which is the network. Measured per (frame, stream, view) at 518 px, on a
12-core Mac with the analytic model:

| stage | ms | |
|---|---|---|
| `bin_by` | 162 | 13 `depth_metrics` passes, 2 profiles, 2 anchored fits |
| `full_frame_view` | 78 | remap + resample (was 107 before the maps were memoised) |
| `_load_frame` | 41 | PNG decode + the GT `.npy` |
| forward | — | 4.6 ms analytic; ~120 ms for a real head on the box |

`--workers N` scores frames on N threads — numpy and cv2 release the GIL, so
they parallelise — with the forward pass serialised behind a lock and the rows
pooled **in split order**. Order is the whole safety argument: float addition is
not associative, so pooling in completion order would move the last digits of
every published number. Measured 2.07× end to end at 8 workers on 12 cores,
with a bit-identical checksum at every worker count; `--workers 1` is the
pre-threading path exactly. Default is `min(8, cpu_count)`.

The `results` branch carries JSON and logs, never images
([`POLICY.md`](../docs/handoff/POLICY.md)), so redraw the figures from any run's
`results.json` — no data, no GPU, no re-scoring:

```bash
python -c "import json;from fovbench import report;report.write_figures(json.load(open('results.json')),'figs')"
```

**Both axes are physical and shared, so the panels can be read side by side.**
θ already meant the same thing in both views. Radius did not, and drawing it raw
was worse than incomparable — it **inverted**: radius is measured in each view's
own image plane, the rectified frame runs to √2 in its corners while the fisheye
stops at 1.0, and that reads as the fisheye seeing less field when it sees more.
It is still *binned* in each view's own plane (that is where the pixel sat in the
tensor the network was given) but *drawn* where the ray lands on the raw sensor:

| rect's own radius | = θ | = radius on the raw sensor |
|---|---|---|
| 1.000 (inscribed circle) | 42.2° | **0.73** |
| 1.411 (corner, its axis end) | 52.1° | **0.93** |
| — | 54.8° (fisheye's limit) | **0.98** |

The whole rectified radius axis lives inside 0.93 of the sensor. On the drawn
axis the rectified arm therefore ends *first* on both coordinates, which is the
truth. The conversion is monotone, so no curve is reordered by it, and the bin
widths warp with it — the pixel-count row of `gt_depth.png` is a density per unit
of the plotted x, so it is computed on the converted edges.

Each panel then carries two marks where its own field runs out:

* a **grey band** past the last angle at which the view images a whole ring —
  42.2° on rect, nothing on fisheye. Inside it the curve continues but on four
  corner wedges, a different set of directions from every point to its left.
* a **hatched band** where the view images nothing at all. This is the part a
  shared axis exists to show: rect ends at 52.1° against fisheye's 54.8° on θ,
  and at 0.93 against 0.98 on the sensor radius.

Before the axes were shared, a panel that had run out of camera looked like a
curve that had finished. `report.txt`'s RING COVERAGE table gives the same thing
as numbers — the ring fraction per bin and the angle each bin *actually*
averages, which is not the bin midpoint once the ring is partial.

The temporal-context arm has its own figure, which needs several runs at once
and so is not part of `write_figures`:

```bash
python -c "
import json, collections
from fovbench import report
runs = collections.OrderedDict((lab, json.load(open(f'{d}/results.json')))
                               for lab, d in [('N=1','partA_seq131'),
                                              ('5c','partB_5c'), ('10c','partB_10c'),
                                              ('5s','partB_5s'), ('10s','partB_10s')])
report.write_context_figure(runs, 'figs')"
```

One panel per (model × view × stream), one line per configuration: colour is the
number of frames in the stack, dashed is strided rather than consecutive. Every
payload must carry the same digest and the call refuses otherwise — lines drawn
from different splits would look exactly like a context effect.

### Panels for a paper or a deck

`write_panels` writes **one file per (metric, view, axis, stream), with no
titles** — `AbsRel_fisheye_theta_synthetic.png` and so on — because the caption
belongs wherever the panel is pasted and a title baked into the pixels cannot be
edited there. It takes the same `{label: payload}` mapping, so single-frame
baselines and multi-frame ones land on one axes; there colour carries the
**model** and the dash pattern carries the temporal configuration.

```bash
python -c "
import json, collections
from fovbench import report
runs = collections.OrderedDict(...)   # as above
report.write_panels(runs, 'panels',
                    include=lambda model, n, stride: n == 1 or model == 'vggt_omega')"
```

`include` is not optional decoration. Every model at every configuration is
**sixteen curves on one axes**, told apart by four colours and two dashes, which
is a dump rather than a figure. The selector is how the merge stays a choice.

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

The headline run is `fovbench-rectfix-393cab9`, split `601fcb22767e` — **50
frames each of six ADT sequences**, five models, both streams, both views,
radial only. Its three figures (`AbsRel`, `delta1`, `gt_depth`) are in
[`results/fovbench-rectfix-393cab9/partA_6seq/figures/`](../results) on the
`results` branch, the old/new comparison in `ANALYSIS.txt`.

> **It replaced `fovbench-ctx-d351d94`, whose rect arm predates #018.** That run
> is a mixture — its fisheye column matches the current code and its rect column
> was measured through a lens description a pixel wrong — so a comparison drawn
> from it needs a footnote saying which half is current. Re-scoring cost the rect
> arm up to **−1.65 %** on `pen`, and the fisheye arm **nothing**: it came back
> bit-identical across all 8336 numeric leaves, from a single-threaded run to a
> `--workers 16` one, which is also the threading control #014 wanted.
> Per-sequence tables are still only in the older run's `ANALYSIS.txt`, and
> item 3b below has *not* been re-measured — see its note.

> **Items 1–6 below were measured on seq131 alone, and seq131 is the mildest of
> the six scenes.** It ranks in the two mildest in 13 of 16 (model × view ×
> stream) cells, median `pen` **0.79×** the six-scene median. Every `pen` quoted
> from it is therefore a *low* estimate — see item 7 for the range. The
> direction of all six claims survives; the size does not.

Items 1–6 come from run `fovbench-v2-ef2d50b`, split `fcc6c600f83b` — 200 frames
of one sequence (`Apartment_release_clean_seq131_M1292`), both protocols
including the window sweep, ~4 h on one RTX 6000 Ada. Numbers in
[`results/fovbench-v2-ef2d50b/`](../results).

> **The depth confound, measured.** Every metric here is relative, so a bin that
> is nearer scores worse for that reason alone. Run `fovbench-v3-24b38e1`
> measured the GT depth of the scored frames, per bin, model-independently:
>
> | axis · view | 0–10° | 10–20 | 20–30 | 30–40 | 40–50 | 50–55 | outer/inner |
> |---|---|---|---|---|---|---|---|
> | θ · fisheye | 3.00 | 2.87 | 2.65 | 2.31 | 1.96 | 1.70 | **0.57×** |
> | θ · rect | 3.02 | 2.87 | 2.65 | 2.30 | 1.99 | 1.94 | **0.64×** |
>
> Six sequences later the confound is unchanged, which is itself the point —
> it is the lens and the room, not the room alone (`fovbench-ctx-d351d94`,
> 300 frames): fisheye 2.96 → 1.65 m (**0.56×**), rect 2.96 → 2.02 (**0.68×**).
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

**3. "Rectifying flattens the curve" was mostly the rectified arm being cut
short — the radial version of the claim does not survive a matched comparison.**
It read: rect `pen` 1.00–1.25 against 1.79–1.97 on the raw lens, and at the
40–50° bin VGGT-1B synthetic scoring 0.074 rect against 0.102 fisheye.

The rectified view images a **whole ring only out to 42.2°**. Past that it is
four corner wedges — 55% of the ring across the 40–50° bin and **3%** across
50–55° — and it ends at 52.1°, where the raw lens runs to 54.8°. So `pen`, which
divides the outermost bin by the innermost, was spanning a *shorter and
differently-aimed* arc on rect than on fisheye. Two consequences, both measured
(`geometry.ring_coverage`, pure geometry, no data):

* The corners are exactly the directions that reach furthest, so a bin labelled
  the same in both views is **not the same angle**: rect's 40–50° bin averages
  **43.8°** against fisheye's 45.2°, and its 50–55° bin **50.7°** against 52.4°.
  The inner four bins agree to within 0.2°. The level comparison above is
  therefore flattered by ~1.4° of eccentricity as well.
* Compared over a span both views image completely, the advantage disappears:

| rect `pen` ÷ fisheye `pen`, 8 model × stream cells | median |
|---|---|
| headline span, 50–55° ÷ 0–10° | **0.84** |
| matched bins, 30–40° ÷ 0–10° (both 100% ring) | **1.02** |
| continuous profile, 42° ÷ 5° (both 100% ring) | **1.00** |

**Over the field both views actually cover, the raw lens and the rectified one
degrade at the same rate.** Only VGGT-1B keeps a real advantage from rectifying
(0.87/0.90 matched, 0.82/0.89 on the profile); VGGT-Omega reverses (1.04/1.15).

This does not overturn 3b, which is the *sharper* form of the same claim and is
untouched — a window sweep aims both views at the same directions by
construction, so it has no truncation to hide behind. It overturns the radial
arm's weaker version of it, which is the one that was easiest to quote.

The figures now shade the rect θ panels past 42.2° and the report prints a RING
COVERAGE table, so the next reader does not have to know this to avoid it.

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

> **This item's rect numbers still predate #018 and have not been re-measured.**
> `fovbench-rectfix-393cab9` is `--protocols radial`, so it re-scored item 3's
> rect column and not this one. The lens fix moves rect wherever it appears, so
> treat the 0.90–1.21 above as *unverified*, not as unchanged. The radial arm's
> movement (≤1.65 % on `pen`) is the best available guide to its size, and it is
> far smaller than the rect/fisheye gap this item rests on — but that is an
> argument, not a measurement.

**4. The sensor sets the level, the lens sets the slope.** `real` sits well above
`synthetic` at every bin (VGGT-1B fisheye 0.110 vs 0.068 on axis) while the two
curves have nearly the same shape.

**5. The two axes are two binnings of one field, not two fields — and an earlier
version of this item got that wrong.** It read "the two axes have mirror-image
blind spots": rect covering radius to its corners while the fisheye "has nothing
past radius 1.0". That is a false symmetry. Radius 1.0 on the fisheye *is* its
entire field, and rect's corner at radius 1.411 is only 0.93 of the same sensor —
so the rectified arm is the shorter one on both coordinates, and there is no
respect in which it reaches further. The figures now draw radius on the sensor
for exactly this reason.

What does survive: within a view, θ and radius are different reparametrisations,
so the same data binned by each gives a different `pen`. Fisheye's two nearly
agree (1.79 on θ, 1.71 on radius); rect's do not and can change direction (1.08
on θ, 0.91 on radius), which is `tan θ` against `θ` — the rectified frame
stretches its outer field over more pixels, so equal-width radius bins there cut
the field at very different angles. Both are kept because a per-view radius is
where the pixel sat in the tensor the network saw, which is a real question about
the model; it is only the *cross-view* reading that needs the conversion.

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

**7. Across six scenes the shape holds and the magnitude was understated.**
`pen > 1` in **6 of 6** sequences for all three `scale_shift` heads in every
(view × stream) cell — one exception, VGGT-Omega rect/real at 5 of 6, and the
dissenter there is seq131 itself (0.975). So "the periphery is worse" is not one
apartment's quirk. But seq131 is the *mild* end of the distribution, so the
pooled six-sequence run reads harder than anything above: fisheye `pen`
1.01–2.27, rect 1.08–1.83, against seq131's 1.07–2.10 / 0.92–1.30.

Read the rect column of that comparison against item 3: rect `pen` spans an arc
that is partly corners, so rect and fisheye `pen` are comparable **within** a
view across sequences, and not **between** views at these edges.

| cell | seq131 | 6-seq median | max |
|---|---|---|---|
| VGGT-Omega rect/synthetic | 1.11 | 1.88 | 2.38 |
| DA3 rect/synthetic | 1.24 | 2.08 | 2.36 |
| VGGT-Omega fisheye/real | 1.48 | 2.06 | **2.86** |

**DAv2 is the one model whose *sign* moves with the scene**, and not the way the
seq131 run suggested. Its sub-1 rect `pen` is not seq131-specific: it goes below
1 on seq132 in all four cells (0.74–0.91) and on seq131 in two. Part of the
original 0.73 was also an aggregation artefact — cross-frame pooling is now
pixel-weighted, which moved 8 of 96 cells on identical data (all eight DAv2,
largest `0.1101 → 0.0915`, no depth-head cell moved at all) and took the rect
`pen` to 0.81 / 0.88. Reduced, not removed.

**8. A temporal context does not rescue the periphery — and more of the same
instant makes things worse.** Handing VGGT-1B / VGGT-Omega / DA3 a stack of 5 or
10 frames in one forward pass (`--context-frames`, only the target scored, same
digest `8ca25fd0ebd2` throughout):

* **10 consecutive vs 1** is *worse* in 9 of 12 cells, concentrated on the raw
  lens: +39% VGGT-1B synthetic fisheye, +33% Omega, +19%/+13% on real.
* **10 strided (every 10th frame) vs 10 consecutive** wins in 10 of 12, by up to
  **31%** — DA3 real rect `0.0689 → 0.0478`, which is also 38% better than
  monocular. Ten consecutive ADT frames span 0.30 s and a couple of centimetres,
  so the models can use a second *viewpoint* and not a second sample of the same
  one. Testing only the consecutive arm would have read as "context is useless".
* `pen` moves by ≤ 0.13 across all five configurations. **Context changes the
  level, not the shape.**

**Still open.** Six of ADT's 19 extractable sequences. Nothing was missing — the
`.vrs` files are on disk for all 20 directories; `videos_synthetic/` and
`depth_npy/` are decoded *caches*, and only seq131 had both. Five more were
decoded for this run (400 frames each, 13 GB); the other 13 need `depth_npy`
extracted as well and were left alone.
Part B ran on seq131 only, on a cost estimate that later proved ~60% high — the
harness is CPU-bound (25–30% GPU, ~2 of 64 cores), so the full context grid over
six sequences is affordable and has not been done.

## Two known defects in the data and the harness

**seq131's `videos_synthetic` cache duplicates its last record.** The naming rule
is `frame_<idx>_<ts>.png` with `idx` the record index in the *real* stream and
`ts` that record's capture timestamp; on seq131 synthetic record 0 is omitted and
the last record is written **twice**, so `frame_003190_*` carries pixels 33 ms
older than its name and is scored against the wrong depth map. One frame in 2878
of the candidate set — negligible for any aggregate, but it is a real mispairing
present in every ADT-FOV number measured before `fovbench-ctx-d351d94`. Re-decode
the cache with `extract_synthetic.py` (on the `results` branch) to clear it.

**The analytic stand-in used to depend on when it was called.** Its jitter came
from one generator advanced call by call, so the same frame scored differently
depending on how many frames preceded it and in what order — invisible while
everything was serial, and the first thing threading broke. It is now seeded
from the frame's own content, which also means the harness's fixed point is a
fixed point.

**`--manifest` used to silence `--context-frames`.** A manifest stores the
per-frame context list, so `Split.load` restored a 1-frame context while the
run's own `config` block echoed the flags as given — four context runs came back
bit-identical to the baseline and looked exactly like real ones. The driver now
refuses the disagreement. Note the fix is only a refusal: to freeze a split *and*
vary the context, rebuild it from `--adt-root`, which reproduces the same digest
because the digest excludes the context by design.

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
| `tests/` | 118 CPU tests: no weights, no data, ~14 s (needs Python 3.8+) |

**This package reads ADT and nothing else.** It carried a second ground truth
once — ego-synth 5B, the release `slambench/` was written to score — and that
duplicate reader is what `tests/test_experiment_separation.py` now exists to
prevent: the invariant is which experiment owns which dataset, checked from
outside both, because an import graph cannot see a second *copy*.

Model loading, availability and downloads live in
[`finetune/eval/baselines/model_zoo.py`](../finetune/eval/baselines/model_zoo.py);
the fisheye camera model in
[`VGGT-360-fisheye/utils/fisheye_cam.py`](../VGGT-360-fisheye/utils/fisheye_cam.py);
the metric definitions in
[`finetune/eval/metrics.py`](../finetune/eval/metrics.py). This package adds the
FOV-location axis on top of those and duplicates none of them.
