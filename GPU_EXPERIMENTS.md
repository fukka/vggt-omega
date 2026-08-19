# GPU run list

Commands only. Run from the repo root on the GPU box. Every flag below was checked
against the scripts' argparse — nothing here is invented.

Running on the **`space-container` A100 pod** rather than `lambda_63`? Read
[`docs/handoff/SPACE_CONTAINER.md`](docs/handoff/SPACE_CONTAINER.md) first — how
to connect, how to rebuild its environment (its default torch is the wrong CUDA
build), how to stage data, and the `OMP_NUM_THREADS=16` setting that is worth a
7x speedup there.

Edit these two lines to match the box, then paste the blocks in order.

```bash
export ADT=/group-volume/Fengjia/data/projectaria_tools_adt_data_clean
export OMEGA_CKPT=../vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt
```

When done: `git add -Af VGGT-360-fisheye/outputs && git commit -m "results" && git push origin HEAD:results`

---

## 0. ADT-FOV test — distortion vs FOV location, four vanilla models

The cross-model experiment: VGGT, VGGT-Omega, DAv2 and DAv3, on rectified
perspective *and* raw fisheye, on synthetic *and* real input, with the error
resolved by where in the field of view it happened. Protocol and how to read the
output: `fovbench/README.md`.

**Check availability first** — DAv3 needs the `depth_anything_3` package and
VGGT-Omega's weights are gated, so both fail with an instruction rather than a
traceback:

```bash
python -m finetune.eval.baselines.benchmark_adt --list | head -8
```

```bash
pip install --no-deps depth-anything-3 && pip install omegaconf addict einops
export VGGT_OMEGA_CKPT=$OMEGA_CKPT
python -m finetune.eval.baselines.benchmark_adt --download --models vggt_1b,da3_large,dav2_large
```

The run itself. `--n-frames` is **per sequence**, spread evenly (not a prefix):

```bash
python -m fovbench.run --adt-root "$ADT" --n-frames 25 \
  --models vggt_1b,vggt_omega,dav2_large,da3_large \
  --out eval_out/fovbench_main 2>&1 | tee eval_out/fovbench_main.log
```

A 30-minute smoke of the same command before committing to the full grid:

```bash
python -m fovbench.run --adt-root "$ADT" --n-frames 3 \
  --models vggt_1b --protocols radial \
  --out eval_out/fovbench_smoke 2>&1 | tee eval_out/fovbench_smoke.log
```

### Optional follow-ups — a *different* axis, not the FOV-location result

The two blocks below do not belong to the main table. The first varies window
**width**, which is a different question from where the window is aimed; it is
here only because the old block-A sweep confounded the two, and `--window-fov` is
held fixed *inside* a run, so sweeping it across runs is the honest way to vary
it. The second adds two small models beyond the four the experiment is about.
Run them only if the main table is already in hand.

```bash
for F in 30 40 60 80; do
  python -m fovbench.run --adt-root "$ADT" --n-frames 10 --protocols window \
    --window-fov $F --models vggt_1b,vggt_omega,dav2_large,da3_large \
    --out eval_out/fovbench_w$F 2>&1 | tee eval_out/fovbench_w${F}.log
done
```

```bash
python -m fovbench.run --adt-root "$ADT" --n-frames 25 --protocols radial \
  --models da3_small,dav2_small --out eval_out/fovbench_small \
  2>&1 | tee eval_out/fovbench_small.log
```

Push `eval_out/fovbench_*/results.json`, `results.csv` and `report.txt` — not the
figures, they regenerate from the JSON with
`python -c "import json,fovbench.report as R; R.write_all(json.load(open('eval_out/fovbench_main/results.json')),'eval_out/fovbench_main')"`.

**Verified before handoff:** 72 CPU tests green, and the whole pipeline run
end-to-end on a real ADT frame with `--models analytic`, which reads a known
injected radial bias back to within 0.3%. **Not** verified: any number from a
real network — no weights ran on this machine.

---

## 0b. Our VGGT-360-fisheye, on both benchmarks

The port as a **row in the two published tables** rather than in its own driver:
one centre view plus an 8-direction ring of 60° tangent views, one VGGT-1B pass
over all nine, fused back onto the fisheye grid. Same pipeline object in both
(`VGGT-360-fisheye/utils/pipeline.py`), and the same one `main_adt.py` runs, so
these two rows and section A's numbers are the same model measured three ways.

"The 60° model" here means `main_adt.py`'s defaults — `--fov 60 --ring-tilt 26
--n-ring 8`, adaptive views on, SA-mask attention on, `--fuse attn`, `--head
depth`, bf16 — with **one deliberate departure**: the tangent views are rendered
at **518**, not `main_adt.py`'s 512. 512 is not a multiple of VGGT's patch size,
so `load_and_preprocess_images` bicubic-resizes every view up by 1.0117× on the
way in; 518 is the token grid, so nothing is resampled between the view and the
network — the same rule the four vanilla models are held to here. Pass
`--vggt360-persp-size 512` to reproduce `main_adt.py` exactly. Every other
`--vggt360-*` flag already defaults to that driver's value, so the commands below
carry none of them; they exist for the ablations at the end.

### The FOV experiment

`vggt360` answers on the **raw fisheye, radial protocol, and nothing else** — it
consumes a fisheye frame plus a lens, so the rectified arm is not an input it
has, and a 40° window is not a 55° cone. Both are refused up front rather than
scored degenerately, which is why this is its own command instead of a fifth
model in section 0.

The four vanilla models are re-run **beside it, in the same command**, so the
comparison is within one run:

```bash
python -m fovbench.run --adt-root "$ADT" --n-frames 25 \
  --models vggt_1b,vggt_omega,dav2_large,da3_large,vggt360 \
  --views fisheye --protocols radial \
  --out eval_out/fovbench_vggt360 2>&1 | tee eval_out/fovbench_vggt360.log
```

**This run has a built-in check on itself.** Views and protocols are not part of
the split digest, so with the same `--n-frames` and streams it carries the *same
digest* as section 0's main run — and its four vanilla fisheye/radial cells must
come back identical to that run's. If they do not, something moved that was not
supposed to, and the `vggt360` row is not readable until that is understood.

A weight-free rehearsal of the refusals and the plumbing, which needs no GPU and
about a second:

```bash
python -m pytest fovbench/tests/test_vggt360.py slambench/tests/test_vggt360.py -q
```

### The SLAM experiment

A third **baseline arm** beside `raw` and `rect_derect` — a lens strategy, in the
slot the other two already occupy. Three things to know before running it:

* it needs `--calib-root`, because it warps through each take's own FISHEYE624
  in both directions and no KB4 is fitted to stand in for it;
* it runs the vendored VGGT-1B itself, so `--models` must be exactly `vggt_1b`
  (refused otherwise — a `dav2_large × vggt360` row would contain no DAv2);
* `--context-frames 1` only. It already hands VGGT a nine-view reconstruction of
  one frame; an N-frame context would be 9N views in one pass.

The headline pairs it against `raw` **without** `rect_derect`, deliberately:
every arm is scored on the points every arm could answer for, and a 110° pinhole
has no answer at the rim, so including it truncates the whole comparison at ~55°
— which is exactly the field this method exists to cover.

```bash
python -m slambench.run --egosynth-root "$EGOSYNTH" --calib-root "$EGOSYNTH_CALIB" \
  --datasets aea,nymeria --models vggt_1b --baselines raw,vggt360 \
  --context-frames 1 --n-frames 25 --takes 8 \
  --out eval_out/slambench_vggt360 2>&1 | tee eval_out/slambench_vggt360.log
```

`aea` and `nymeria` are the two datasets whose sensor-to-upright rotation is in
`camera.VERIFIED_ROTATION`; the other two need `--allow-unverified`, which is for
debugging and not for numbers (a quarter-turn error does not degrade the score,
it scores a different part of the image).

Then the three-arm version, read only against itself:

```bash
python -m slambench.run --egosynth-root "$EGOSYNTH" --calib-root "$EGOSYNTH_CALIB" \
  --datasets aea,nymeria --models vggt_1b --baselines raw,rect_derect,vggt360 \
  --context-frames 1 --n-frames 25 --takes 8 \
  --out eval_out/slambench_vggt360_3arm 2>&1 | tee eval_out/slambench_vggt360_3arm.log
```

### Read the log, not only the table

Both runs print things the tables cannot carry:

* **`lens cone … | layout reaches … | ring stops N deg short of it`**, per take.
  The 60°/tilt-26 layout is sized against Aria's *nominal* 54.83° cone; ego-synth
  calibrations are per take. A ring that stops short leaves the rim to the centre
  view's corners alone, which reads as the method degrading at eccentricity when
  it is the layout not arriving. If this says "short" on most takes, re-run with
  `--vggt360-ring-tilt $(cone - 30)` before quoting anything about the rim.
* **`N% of the imaged cone … filled with a constant`** (fovbench only). Expect
  well under 1%. The FOV harness owns its validity mask and its alignment fit
  cannot take a NaN, so holes are filled; the SLAM harness intersects support
  across arms and gets honest NaN instead.

### Two caveats that belong with these numbers

**The FOV row reads at 1408 and answers at 518, so it is not
resolution-matched to the other four.** The nine tangent views are cut from
ADT's own frame — the resolution the port is designed for, and what `main_adt.py`
uses — and the fused answer is delivered on the harness's 518 scoring grid, so
every metric, mask and eccentricity bin is the one the vanilla models are scored
under. Only the sharpness of what the network saw differs: a 60° view at 518 px
is a **0.62× downsample** of the 1408 source, against **1.69× up** if it were cut
from the 518 view.

That is a real advantage in pixels and it should be stated, not buried — the row
is the method at its designed resolution, not a like-for-like input comparison.
The matched control is one flag:

```bash
python -m fovbench.run --adt-root "$ADT" --n-frames 25 --models vggt360 \
  --views fisheye --protocols radial --vggt360-source view \
  --out eval_out/fovbench_vggt360_matched 2>&1 | tee eval_out/fovbench_vggt360_matched.log
```

Run both and report both. `native` says what the method is worth; `view` says
what it is worth on the same pixels everyone else got. Neither is neutral, and a
gap between them is the resolution term rather than the lens term.

**The SLAM arm is the only one that converts depth conventions**, and it is the
only one that has to: fusion produces euclidean range, `pts.d` is planar z, and
`raw`/`rect_derect` both answer about the same axis the ground truth is measured
about. The conversion is `cos θ` — 1.74× at 55°, radial, unabsorbable by the
affine. That `pts.d` really is planar z is measured (ticket 016, closed
2026-08-14): residual 0.0002 and flat, against a range hypothesis wrong by
exactly `1 − cos θ` in all eight bins on both aea and nymeria.

**On SLAM the vggt360 arm is *not* resolution-matched, and here that favours
it.** It renders nine 518² views out of the 896² frame — a 1.04× resample, so
essentially every one of those 2.4 M pixels carries real detail — against
`raw`'s single 518² resize (4.37 px/deg) and `rect_derect`'s 518² pinhole (3.17
px/deg on axis, 1.04 at 55°). In angular terms that is **7.83 px/deg against
4.37**, a 1.79× sampling advantage, on a 9× pixel budget. Exploiting
resolution by tiling *is* the method, so this is not a flaw in the comparison —
but a reader will otherwise attribute all of the gap to the lens treatment. The
honest control is `--vggt360-fov` widened so fewer, wider views cover the same
cone at `raw`'s density; that run is not in this file and has not been done.

Note the contrast with the FOV row above, where the same nine views come out of a
518² source at a 1.69× **up**sample: same pixel budget, no information behind it.
The two tables therefore lean in opposite directions, and neither is neutral.

### Ablations, once the two rows are in hand

```bash
for ABL in "--vggt360-fuse mean" "--vggt360-no-sa-mask" "--vggt360-no-adaptive"; do
  TAG=$(echo $ABL | tr -d ' -' | tr A-Z a-z)
  python -m fovbench.run --adt-root "$ADT" --n-frames 25 --models vggt360 \
    --views fisheye --protocols radial $ABL \
    --out eval_out/fovbench_v360_$TAG 2>&1 | tee eval_out/fovbench_v360_$TAG.log
done
```

**Verified before handoff:** 262 CPU tests green, including a known field put
round-trip through the FISHEYE624 warp and fusion with no network (mean relative
error 0.0004, and 0.015 when the lens is deliberately 3 px wrong), and the ADT
KB4 geometry checks still passing after the refactor that gave the three drivers
one pipeline. **Not** verified: any number from a real network — no weights ran
on this machine, and neither of the two commands above has been executed.

---

## A. Pipeline metrics vs per-view FoV  — the 60-vs-110 question

Upstream VGGT-360 runs 110°; this port defaults to 60°. These four runs decide
whether that gap explains the results. `--max-frames 100` is the default, so each
run is already multi-frame.

```bash
for FOV in 60 80 100 110; do
  python VGGT-360-fisheye/main_adt.py \
    --adt-root "$ADT" --fov $FOV \
    --eval-domains z range --align-modes scale_shift scale_only \
    --qual-dir VGGT-360-fisheye/outputs/fov_$FOV --n-qual 8 \
    2>&1 | tee VGGT-360-fisheye/outputs/fov_${FOV}.log
done
```

The ring layout is sized for 60° (`tilt 32 + fov/2 = 62.3` = the cone). At 110° the
ring over-covers, so also run 110° with fewer, wider views:

```bash
python VGGT-360-fisheye/main_adt.py \
  --adt-root "$ADT" --fov 110 --n-ring 4 --ring-tilt 25 \
  --eval-domains z range --align-modes scale_shift \
  --qual-dir VGGT-360-fisheye/outputs/fov_110_ring4 --n-qual 8 \
  2>&1 | tee VGGT-360-fisheye/outputs/fov_110_ring4.log
```

## B. Head agreement — depth head vs point head

Same config, both heads. Divergence between them is a geometry-consistency signal
that needs no GT.

```bash
for HEAD in depth point; do
  python VGGT-360-fisheye/main_adt.py \
    --adt-root "$ADT" --fov 100 --head $HEAD \
    --eval-domains z range --align-modes scale_shift \
    --qual-dir VGGT-360-fisheye/outputs/head_$HEAD --n-qual 8 \
    2>&1 | tee VGGT-360-fisheye/outputs/head_${HEAD}.log
done
```

## C. Tiling with per-patch FoV held constant

The current defaults confound tile count with per-patch FoV (1×1 = 100°,
2×2 = 79.7°, 3×3 = 58.2°, 4×4 = 45.3°). These overlap values hold every tiling at
**79.8°** per patch, so only the tile count varies.

```bash
python VGGT-360-fisheye/checks/patch_scale_experiment.py \
  --adt-root "$ADT" --backend vggt_omega --checkpoint "$OMEGA_CKPT" \
  --tilings 1 --total-fov 100 \
  --out VGGT-360-fisheye/outputs/tile_fov80_n1 2>&1 | tee VGGT-360-fisheye/outputs/tile_fov80_n1.log

python VGGT-360-fisheye/checks/patch_scale_experiment.py \
  --adt-root "$ADT" --backend vggt_omega --checkpoint "$OMEGA_CKPT" \
  --tilings 2 --total-fov 100 --overlap 0.403 \
  --out VGGT-360-fisheye/outputs/tile_fov80_n2 2>&1 | tee VGGT-360-fisheye/outputs/tile_fov80_n2.log

python VGGT-360-fisheye/checks/patch_scale_experiment.py \
  --adt-root "$ADT" --backend vggt_omega --checkpoint "$OMEGA_CKPT" \
  --tilings 3 --total-fov 100 --overlap 1.105 \
  --out VGGT-360-fisheye/outputs/tile_fov80_n3 2>&1 | tee VGGT-360-fisheye/outputs/tile_fov80_n3.log

python VGGT-360-fisheye/checks/patch_scale_experiment.py \
  --adt-root "$ADT" --backend vggt_omega --checkpoint "$OMEGA_CKPT" \
  --tilings 4 --total-fov 100 --overlap 1.806 \
  --out VGGT-360-fisheye/outputs/tile_fov80_n4 2>&1 | tee VGGT-360-fisheye/outputs/tile_fov80_n4.log
```

Then the same tile counts at the **default** overlap, as the contrast arm:

```bash
python VGGT-360-fisheye/checks/patch_scale_experiment.py \
  --adt-root "$ADT" --backend vggt_omega --checkpoint "$OMEGA_CKPT" \
  --tilings 1 2 3 4 --total-fov 100 --overlap 0.4 \
  --out VGGT-360-fisheye/outputs/tile_default 2>&1 | tee VGGT-360-fisheye/outputs/tile_default.log
```

## D. Coarse-plus-fine in one pass

```bash
python VGGT-360-fisheye/checks/patch_scale_experiment.py \
  --adt-root "$ADT" --backend vggt_omega --checkpoint "$OMEGA_CKPT" \
  --tilings 1 4 1+4 --total-fov 100 \
  --out VGGT-360-fisheye/outputs/coarse_fine 2>&1 | tee VGGT-360-fisheye/outputs/coarse_fine.log

python VGGT-360-fisheye/checks/patch_scale_experiment.py \
  --adt-root "$ADT" --backend vggt_omega --checkpoint "$OMEGA_CKPT" \
  --tilings 1 4 1+4 --total-fov 100 --center-weight 4 \
  --out VGGT-360-fisheye/outputs/coarse_fine_cw4 2>&1 | tee VGGT-360-fisheye/outputs/coarse_fine_cw4.log
```

## E. FoV sweep on more frames

`outputs/sweep_omega/` is one frame. Repeat on five more so the curve has error bars.

```bash
for F in 2 20 60 120 200; do
  python VGGT-360-fisheye/checks/center_view_sweep.py \
    --adt-root "$ADT" --backend vggt_omega --checkpoint "$OMEGA_CKPT" \
    --frame $F --fovs 40 60 80 100 110 120 --modes tangent raw_roi rectifier \
    --out VGGT-360-fisheye/outputs/sweep_f$F 2>&1 | tee VGGT-360-fisheye/outputs/sweep_f${F}.log
done
```

## F. Backend control — is it VGGT-Omega or the whole VGGT family?

```bash
for F in 2 60 120; do
  python VGGT-360-fisheye/checks/center_view_sweep.py \
    --adt-root "$ADT" --backend vggt1b \
    --frame $F --fovs 40 60 80 100 120 --modes tangent raw_roi rectifier \
    --out VGGT-360-fisheye/outputs/sweep_vggt1b_f$F 2>&1 | tee VGGT-360-fisheye/outputs/sweep_vggt1b_f${F}.log
done
```

## G. Ablations on the best config from A

Replace `<BEST_FOV>` with whichever FoV won in A.

```bash
python VGGT-360-fisheye/main_adt.py --adt-root "$ADT" --fov <BEST_FOV> \
  --fuse mean --eval-domains z range --align-modes scale_shift \
  2>&1 | tee VGGT-360-fisheye/outputs/ablate_fuse_mean.log

python VGGT-360-fisheye/main_adt.py --adt-root "$ADT" --fov <BEST_FOV> \
  --no-sa-mask --eval-domains z range --align-modes scale_shift \
  2>&1 | tee VGGT-360-fisheye/outputs/ablate_no_sa.log

python VGGT-360-fisheye/main_adt.py --adt-root "$ADT" --fov <BEST_FOV> \
  --no-adaptive --eval-domains z range --align-modes scale_shift \
  2>&1 | tee VGGT-360-fisheye/outputs/ablate_no_adaptive.log
```

---

## Not runnable yet — needs a code change first (do not attempt)

- **`pose_enc` FoV-error probe.** `--head depth` already converts using the *known*
  render FoV secant, not `pose_enc[7:9]`, so B measures head agreement, not
  `pose_enc` error. Measuring that needs a new switch.
- **`pose_enc[7:9]` override with the known FoV.**
- **Cone constraint on `tiling_views`.** Every multi-tile config has corner patches
  reaching past the 62.33° turnover (2×2 → 80.0°, 4×4 → 74.3°), so outer patches
  carry black regions regardless of `--overlap`.
- **Per-patch valid-pixel fraction reporting.**
