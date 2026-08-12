# ADT-FOV: re-run the baseline, then add a temporal context

**Owner:** gpu
**Files I may touch:** nothing under `fovbench/` — runs only. Results to `results`.
**Blocked by:** none. Supersedes the radial arm of `fovbench-v3-24b38e1`; the
window arm of `fovbench-v2-ef2d50b` still stands and is not re-run.

> **Rewritten again, and smaller in every part except the grid.** Ignore the
> earlier versions of this file and every comment on issue #15 above the one
> pointing here. `drift*` and the depth-standardisation column (`pen_ds`) have
> both been **removed from the experiment**; do not look for them.

## The question

**How does each baseline's depth accuracy vary with where in the field of view
the content sits — and does giving the multi-view models more frames change it?**

Two metrics, `AbsRel` and `delta1`. Two readings of "where": `theta` (the ray's
incidence angle, the only axis on which `rect` and `fisheye` are comparable) and
`radius` (distance from the optical centre in the image plane). Two views, two
streams, four models. That is the whole experiment.

## Part A — the baseline, re-run

```bash
git -C <repo> pull --ff-only origin organized
python -m fovbench.run --adt-root "$ADT" --n-frames 50 --protocols radial \
  --models vggt_1b,vggt_omega,dav2_large,da3_large \
  --out eval_out/fovA 2>&1 | tee eval_out/fovA.log
```

**First, tell me how many ADT sequences the box has** — sequences with
`depth_npy` *and* both RGB streams, which is what `build_split` requires. Then
run **50 frames from every one of them**. `--n-frames` is per sequence.

Why it must be re-run rather than re-read: the cross-frame pooling changed.
`_mean_metrics` used to average per-frame means with equal weight, so a frame
that contributed a handful of pixels to a bin outvoted one that filled it. It is
now weighted by each frame's pixel count, which makes the binned dots and the
continuous curve the same estimator — a test pins them equal to 1e-9. On
`fovbench-v3-24b38e1` this moves exactly one cell (DAv2 rectified 0–10°, 0.1101
→ 0.0915) and leaves the depth heads untouched, but the change is in the
aggregation, so it cannot be applied to an existing `results.json`.

**Expect ~4 min per sequence** (the v3 run was 14 m 47 s for 200 frames).

## Part B — the temporal context

New: `--context-frames N --context-stride S` hands a multi-view model N frames in
**one forward pass**. Only the split's own frame is scored; the rest are context.
So the measured pixels are identical to Part A and only the evidence moves.

Eight runs, **three models** (`dav2_large` is monocular and is refused — do not
add it), on the **same sequences and the same 50 frames per sequence**:

| N | stride | span at 30 Hz | out |
|---|---|---|---|
| 5 | 1 | 0.13 s | `eval_out/fovB_5c` |
| 10 | 1 | 0.30 s | `eval_out/fovB_10c` |
| 5 | 10 | 1.3 s | `eval_out/fovB_5s` |
| 10 | 10 | 3.0 s | `eval_out/fovB_10s` |

```bash
python -m fovbench.run --adt-root "$ADT" --n-frames 50 --protocols radial \
  --models vggt_1b,vggt_omega,da3_large \
  --context-frames 5 --context-stride 1 --out eval_out/fovB_5c
```

Both spacings, because they answer different halves of one question. Ten
*consecutive* ADT frames span 0.30 s and a head-worn camera moves a couple of
centimetres in that time — almost no parallax. If the consecutive arm shows no
gain, the strided arm is the only thing that says whether that is the model or
the missing baseline.

**The split digest must be identical across every run in Parts A and B.**
Context is deliberately kept out of the digest so that 1-, 5- and 10-frame runs
share one, and that shared digest is the guarantee that the comparison is on the
same pixels. If it differs, stop.

### Measure the cost before committing to the grid

VGGT's global attention is quadratic in total tokens, so 10 frames may be far
more than 10× a single pass. **Time a 3-frame smoke first**:

```bash
python -m fovbench.run --adt-root "$ADT" --n-frames 3 --protocols radial \
  --models vggt_1b --context-frames 10 --context-stride 1 \
  --views fisheye --streams synthetic --out /tmp/fovsmoke
```

Report the seconds per frame against Part A's, and **if 10-frame extrapolates
past ~6 h for the whole grid, run the N=5 rows and the N=10 rows on ONE sequence
only, and say so.** Do not start a job you cannot finish.

## What to report

Five things. Keep it short.

1. **How many sequences the box has**, and which you used.
2. **The three figures from Part A** — `figures/AbsRel.png`, `figures/delta1.png`,
   `figures/gt_depth.png`. Each already carries every model, both views, both
   streams and both axes. `gt_depth.png` now has a second row: the pixel count
   behind every part of the curve.
3. **Whether the curves are the same across sequences.** This is the first run
   with more than one scene, and it is the biggest open question in the result —
   everything so far is one apartment. Per-sequence `pen` for each cell, and
   whether the shape holds.
4. **Part B: does context help, and does spacing matter?** Overall `AbsRel` and
   `pen` per model for 1 / 5c / 10c / 5s / 10s. The two comparisons that matter
   are *N=1 vs N=10 consecutive* (does more of the same instant help) and
   *10 consecutive vs 10 strided* (does parallax help).
5. **The timing**, per model per configuration.

Do not tune toward any outcome. "Context does not help at either spacing" is a
perfectly good result and would be worth knowing.

## Done when

- [ ] sequence count reported, and the same sequences used throughout
- [ ] one split digest across every run; stated
- [ ] the 3-frame timing smoke reported before the grid was launched
- [ ] Part A's three figures pasted
- [ ] items 3–5 answered with numbers
- [ ] pushed to `results` under one run id per configuration; hand back to `cpu`

## Notes

* **Python 3.8+** is required (the ego-synth code uses the walrus operator).
* 119 CPU tests green at the commit you will pull.
* `--context-frames` is radial-only and is refused for the window protocol: a
  window is a crop, and ten crops of ten different instants is not this
  experiment.
* Figures on the `results` branch are fine here even though
  [`POLICY.md`](../POLICY.md) says JSON and logs only — the three pictures *are*
  the deliverable. Keep it to those.
