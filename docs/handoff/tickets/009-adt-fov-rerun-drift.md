# Re-run the ADT-FOV test — the drift column was measuring the wrong thing

**Owner:** gpu
**Status:** **done, and its subject is gone** — `results/fovbench-v2-ef2d50b`, still cited by `fovbench/README.md` items 1-6. The `drift`/`anchored_ratio` columns this ticket existed to fix were deleted in #017, so its *instructions* are no longer runnable; its *numbers* stand. History, not a task.
**Files I may touch:** nothing under `fovbench/` — runs only. Results to `results`.
**Blocked by:** none. Supersedes the `drift` column of #13 only.

## Goal

Re-run the grid on `organized` @ HEAD with **three** changes the first run needs:
a distortion estimator that works, **100 frames instead of 25**, and a second
binning axis (distance from the optical centre). Everything else about run
`fovbench-main-22c108d` stands.

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

`organized` @ `915b109`:

* `drift` is now `anchored_ratio`, **printed as `drift*` and labelled OUTSIDE
  THE PROTOCOL** — the protocol being one whole-frame fit per frame with binning
  applied afterwards by masking, which every other column obeys. Quote it as a
  diagnostic, never alongside AbsRel/delta1/pen as if it shared their rules.
  It is the model's own affine fitted on the
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
* **A second binning axis: distance from the optical centre**, in units of the
  frame's half width (1.0 = the middle of a frame edge, sqrt(2) = a corner),
  `--radius-edges 0,0.2,0.4,0.6,0.8,1.0,1.45`. It lands in `results.json` as
  `radius_bins` alongside the existing `bins`, and gets its own figures. Both
  axes are read off **one** alignment fit over the whole valid frame — the scale
  is never fitted per bin, on either axis (`geometry.bin_by`).
  Note the two views use different image planes, so a given radius is a
  *different direction* in each: on the raw fisheye radius is nearly proportional
  to the incidence angle, on the rectified pinhole it goes as `tan(theta)`.
  Radius answers "where in the picture"; theta answers "which ray". Report both,
  compare `rect` against `fisheye` only on theta.

78 CPU tests green. No network has run against the new column or the new axis.

## The command

```bash
git -C <repo> pull --ff-only origin organized
python -m fovbench.run --adt-root "$ADT" --n-frames 100 \
  --models vggt_1b,vggt_omega,dav2_large,da3_large \
  --out eval_out/fovbench_v2 2>&1 | tee eval_out/fovbench_v2.log
```

**`--n-frames 100`, not 25.** 25 frames of a single sequence is too thin to
carry the claim; the sequence has 2939 depth frames and the split spreads the
sample evenly across all of them, so 100 costs nothing but time. Budget ~2 h
(the first run was 30 min at 25 frames and the cost is linear). If you have the
GPU hours, 200 is better still — say which you used.

The split therefore **changes**, and its digest with it: `2ab412af0ccc` was the
25-frame set. Record the new digest; the two runs are not bin-for-bin comparable
and are not meant to be.

No setup needed — the env from #13 is unchanged and VGGT-Omega's checkpoint is
already on the box.

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
5. **The radius-binned tables**, and whether the AbsRel rise looks the same
   against radius as against theta. It should not look identical — the two axes
   are non-linearly related, and the rectified panel stretches very differently.
6. Whether 100 frames moves anything against the 25-frame run beyond noise. If
   the curves are unchanged, that is the sample-size answer and worth stating.

## Done when

- [ ] the new split digest recorded (it will NOT be `2ab412af0ccc` — 100 frames)
- [ ] `--n-frames` actually used, stated
- [ ] `report.txt` pasted in full
- [ ] the six items above answered
- [ ] pushed to `results` under a new run id; hand back to `cpu`

## Needs CPU-Claude afterwards?

yes — folding the corrected column into `fovbench/README.md` and deciding
whether the AbsRel rise and the drift tell the same story.
