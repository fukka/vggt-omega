# Re-run the ADT-FOV test — the drift column was measuring the wrong thing

**Owner:** gpu
**Files I may touch:** nothing under `fovbench/` — runs only. Results to `results`.
**Blocked by:** none. Supersedes the `drift` column of #13 only.

## Goal

Re-run the same grid on `organized` @ `23d1583` so the distortion column is
measured by an estimator that works. Everything else about run
`fovbench-main-22c108d` stands and does not need repeating.

## What was wrong, and what survives

`drift` was `median(gt/pred)` on the **unaligned** prediction. That is a
distortion measure only for a purely multiplicative model. Every model in this
benchmark has an additive degree of freedom, so the ratio tracks each bin's
*scene depth* instead — and an egocentric frame's depth falls with eccentricity,
so it produces a radial trend out of nothing. Measured on analytic scenes with no
radial error whatever:

| model, no radial error at all | old `drift` reported |
|---|---|
| `pred = (gt - 1.5)/3` | **0.648** |
| `pred = (gt + 2.0)/3` | **1.253** |
| affine-invariant disparity | **1.143** |

Run 22c108d's headline was `drift` 1.14–1.19. That is inside the range the
artefact alone produces, so **the 14–19% over-prediction claim must be withdrawn**
until this re-run replaces it. It is not refuted — it is unmeasured.

**What survives untouched**, because none of it goes through that column:

* every `AbsRel`, `delta1`, `RMSE` number, and the whole `pen` column;
* "error grows toward the rim on raw fisheye" — vggt_1b synthetic fisheye AbsRel
  0.078 → 0.144 across 0–10° → 50–55°, `pen` 1.85, and the same shape on all four;
* "rectifying helps" — at the honest 40–50° bin, vggt_1b synthetic 0.075 rect vs
  0.118 fisheye;
* "the sensor sets the level, the lens sets the slope" in its AbsRel form;
* the COVERAGE table and the single-sequence caveat.

Two of your readings need revisiting once the new numbers land:

* **"radial vs window disagree, informatively."** Withdraw. Window `drift` was
  never measurable: every window is a separate forward pass of an up-to-scale
  model, and your own run shows the per-window raw ratios at 2.87–3.14 across
  aims for one model. The column is now blank there by construction.
* **"a rect window aimed at 40° is resampled out of rim pixels, so it is simply
  softer."** The sign is backwards. Measured on a real ADT frame, source pixels
  per output pixel go 0.73 → 1.03 (rect) and 0.74 → 1.19 (fisheye) from aim 0° to
  40°: a 40° window on axis is *upsampled*, the same window at the rim is not.
  Resolution **improves** toward the rim, so the rising window AbsRel is not
  blur. The run now reports `src_px_per_out_px` per cell so this is in the table
  rather than in a caveat.

## What changed in the code

`organized` @ `23d1583`:

* `drift` is now `anchored_ratio` — the model's own affine fitted on the
  **innermost bin alone**, then `median(gt/pred)` per bin. Exactly 1.000 on all
  four no-distortion cases above; 1.37 against a true 1.49. Under-reports by
  <10%, never invents.
* An anchor band needs depth **spread**, not just pixels: on one flat wall the
  affine is undetermined and corrupts every bin (a synthetic box drove the first
  version to report 1.86 for a model with no radial error). Guarded on
  IQR/median ≥ 0.05; real ADT bands measure 0.71–0.88.
* `drift` is refused for the `window` protocol.
* `src_px_per_out_px` is recorded per window cell.
* `raw_scale_ratio` and `scale_ratio` are still in `results.json` as diagnostics.

77 CPU tests green. No network has run against the new column.

## The command

Identical to #13 except the commit. Same split, so the digest should come back
**`2ab412af0ccc`** — if it does not, say so before anything else, because then
the two runs are not comparable and something moved under us.

```bash
git -C <repo> pull --ff-only origin organized     # 23d1583
python -m fovbench.run --adt-root "$ADT" --n-frames 25 \
  --models vggt_1b,vggt_omega,dav2_large,da3_large \
  --out eval_out/fovbench_drift 2>&1 | tee eval_out/fovbench_drift.log
```

~30 min, same as before. No setup needed — the env from #13 is unchanged, and
VGGT-Omega's checkpoint is already on the box from that run.

## What to report

1. **The new `drift` column**, radial only, and how it compares to the old
   1.14–1.19. Three outcomes, all publishable:
   * *still 1.1–1.2* → the effect was real and the old estimator happened to be
     right for the wrong reason;
   * *near 1.0* → there is no radial scale drift and the AbsRel rise is variance,
     not bias — which is a different and more interesting story;
   * *larger* → the old estimator was cancelling part of it.
2. **DAv2's drift is now defined.** It previously flipped sign between protocols
   (radial rect 0.690 vs window rect 1.743), which was the artefact showing. If
   it still behaves unlike the other three under the anchored estimator, that is
   a real finding about disparity models; if it now sits with them, the old
   number was noise.
3. **`src_px_per_out_px` per window cell**, so the window curve can be read
   against its own sampling.
4. Whether any bin reports `drift` as `—`, which would mean an anchor band
   failed the spread guard on real data. I do not expect it (0.71–0.88 measured),
   and it would be worth knowing if it happens.

## Done when

- [ ] digest confirmed `2ab412af0ccc`, or the mismatch reported
- [ ] `report.txt` pasted in full
- [ ] the four items above answered
- [ ] pushed to `results` under a new run id; hand back to `cpu`

## Needs CPU-Claude afterwards?

yes — folding the corrected column into `fovbench/README.md` and deciding
whether the AbsRel rise and the drift tell the same story.
