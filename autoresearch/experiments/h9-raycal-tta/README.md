# H9 — RayCal-TTA: let the rim's parallax pay for the rim's depth

**Status:** code written, 12 CPU tests green, pipeline runs end to end.
Pre-registration below is written before the full run.

## The contradiction this resolves

Two things this project measured, which have been treated as opposites:

| | measured |
|---|---|
| the rim **gives** alignment | widening the admitted field improves rotation on **17/17** pairs (H1.1); the frozen model's pose survives centre deletion untouched and collapses without the rim (runs 004–007) |
| the rim does **not receive** fusion | multi-frame context buys the centre, not the field (024B); rim-restricted cross-frame KV loses to all-token (H6); five more region-targeted interventions lost to their own controls |

Every attempt so far has tried to make the rim **receive**. Six of them failed.
This does not pick a side and does not compromise: it runs the information the
other way. **The rim's parallax surplus triangulates the depth the rim is
missing.**

That also makes it the only line in the project that is label-free by
construction — and label-free is the structural answer to the standing external
validity problem (one apartment, one device), because adaptation then needs
video, not ground truth.

## Method

```
frames t−10, t, t+10 ──▶ dense matches ──▶ triangulate in BEARING space
                                              │   (pose from the device's SLAM)
                                              ▼
                              sparse metric range anchors, 47% of them
                              beyond 38° — where the compression is worst
                                              ▼
                    fit  log(pred) = c(θ) + g(θ)·log(true)  per θ bin
                                              ▼
                    invert it on the dense prediction:  g < 1 IS the compression
```

* **Bearing space, never pixels.** On a fisheye the pixel metric is not the
  angular metric; a triangulation written in pixels is wrong by exactly the
  radial factor this experiment exists to measure.
* **Two gates, both mandatory.** Parallax (adjacent frames have no baseline —
  #22 measured stride 1 buying nothing and stride 10 buying 10–13%), and a
  **motion gate**: a pixel becomes an anchor only if two independent partner
  frames agree on its range to 10% in log space. Moving hands poison
  static-scene triangulation exactly in the worst cells (#28: 80%+ of hand
  pixels beyond 41°, median 0.26–0.94 m), and a moving point cannot agree with
  itself.

## Why this is not H2.1 again

Run 010's 48-parameter table was indexed by `(θ, predicted depth)` and failed
with a mechanism: the compression makes predicted depth many-to-one in true
depth, so an output-indexed correction pushes the majority's fix onto
minorities and the near centre paid. Two things differ:

1. it is fitted against **triangulated truth at the anchors**, not inferred
   from the prediction's own statistics;
2. it is **monotone by construction** (`g > 0` enforced), so two different
   predictions at the same θ can never be sent to the same range —
   `test_the_correction_is_monotone_so_it_cannot_be_many_to_one` holds it.

## Arms

| arm | field | what its result rules out |
|---|---|---|
| `raycal` | `(g, c)` per θ bin | the hypothesis |
| `global` | **one** `(g, c)` for the whole image | **the control that decides.** If a single log-linear recalibration does as well, the *radial* structure is not what is being exploited and the claim collapses to "the range axis is miscalibrated", which is already known |
| `shuffled` | per bin, θ labels permuted | same anchors, same counts, no radial correspondence |
| `none` | identity | the frozen model |

## The locked bar, from the registration and not negotiable here

> **the `scale_shift` ↔ frozen-affine gap must COLLAPSE**, not just the aligned
> number drop.

Because ~82% of seq131's near-rim "penalty" was measured to be the eval
affine's *placement* (refit on near pixels: 1.47 → 0.26), and no single affine
serves near and far. A method that merely lets the per-frame affine sit better
has changed nothing about the geometry. So the report leads with the gap
between per-frame `scale_shift` and **one affine fitted once over every valid
pixel of the sequence** — pooled, not averaged per frame, so the method is not
quietly handed back the per-frame freedom the bar is about.

Collapse is scored as `|gap_raycal| < 0.5 · |gap_none|` on `near_rim`.

## Pre-registered predictions

**P1 (the decider).** `raycal` collapses the near-rim gap relative to `none`.
Failing this, H9 is refuted whatever the AbsRel does.

**P2 (radial structure is real).** `raycal` beats **both** `global` and
`shuffled` on near-rim AbsRel. Losing to `global` means the radial part is
decoration.

**P3 (the centre).** `raycal` must not damage `near_center` more than `global`
does. Centre collateral is what killed H2.1 and it is invisible in pooled zones.

**P4 (it is the anchors, not the pose).** Anchors are gated on two-partner
agreement; the run reports how many survive and what share of them are beyond
38°. If the rim share is small, the field at the rim is extrapolated and the
near-rim number is not citable.

## What this run does NOT establish

* **Pose is taken from the device's SLAM trajectory**, not estimated from the
  images. That is honest for Aria — MPS poses ship with the data and require no
  depth labels — but it is not the same claim as "works from raw video alone".
  The classical-pose arm (H10's matcher feeding MAGSAC) is the next rung and is
  a different experiment.
* One seed, one run per sequence, no error bars.
* Nothing about a lens the model has not seen.

## Files

| file | what |
|---|---|
| `code/anchors.py` | bearing-space triangulation + the two gates |
| `code/raycal.py` | the per-θ log-linear field, its inverse, the four arms |
| `code/test_h9.py` | 12 CPU tests, no weights, no data |
| `code/run_h9.py` | the TTA pipeline and the locked-bar report |
