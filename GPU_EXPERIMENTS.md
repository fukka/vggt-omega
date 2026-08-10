# GPU run list

Commands only. Run from the repo root on the GPU box. Every flag below was checked
against the scripts' argparse — nothing here is invented.

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
