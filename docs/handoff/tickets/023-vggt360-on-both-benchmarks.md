# Our VGGT-360-fisheye, as a row in both published tables

**Owner:** gpu
**Files I may touch:** nothing — runs only. Results to `results`.
**Blocked by:** nothing. Code is on `organized` @ `e583017`, CPU suites green.
**Does not collide with #019 or #020.** Both published paths were checked
bit-identical rather than assumed — see "What moved and what did not".

## Goal

Two numbers this repository does not have: **what the port is worth against the
four vanilla models on ADT, and against the two lens strategies on ego-synth.**
When it is done there is a `results/fovbench-023-<sha>` carrying a `vggt360` row
beside `vggt_1b`/`vggt_omega`/`dav2_large`/`da3_large` on the fisheye radial
cell, and a `results/slambench-023-<sha>` carrying a `vggt360` arm beside `raw`.

Every VGGT-360-fisheye number in this repo so far comes from `main_adt.py`, which
is its own driver with its own protocol. Nothing has ever placed the port on the
axes the two benchmarks publish. That is the whole ticket.

## What is new on `organized`

The port's forward pass now lives in `VGGT-360-fisheye/utils/pipeline.py` and
`main_adt.py` calls it, so three drivers share one implementation of the method
rather than three copies. On top of that:

* `fovbench` gains a `vggt360` **model** key, restricted to the fisheye radial
  cell. The rectified arm is not an input this model has and a 40 deg window is
  not a 55 deg cone; both are refused before any weight loads.
* `slambench` gains a `vggt360` **baseline arm**, beside `raw` and
  `rect_derect` — a lens strategy in the slot the other two occupy. ego-synth is
  FISHEYE624, so the take's own lens answers both the forward warp and the
  inverse ray field; no KB4 is fitted to stand in for it.

"The 60 deg model" is `main_adt.py`'s configuration — `--fov 60 --ring-tilt 26
--n-ring 8`, adaptive views, SA-mask attention, `--fuse attn`, `--head depth`,
bf16 — with one deliberate departure: views render at **518**, the backbone's
token grid, not that driver's 512. 512 is not a multiple of 14, so VGGT's own
preprocessing was bicubic-ing every view up by 1.0117x for nothing.
`--vggt360-persp-size 512` reproduces `main_adt.py` exactly.

## The commands

Full text, both runs, in [`GPU_EXPERIMENTS.md`](../../../GPU_EXPERIMENTS.md)
section **0b**. In short:

```bash
python -m fovbench.run --adt-root "$ADT" --n-frames 25 \
  --models vggt_1b,vggt_omega,dav2_large,da3_large,vggt360 \
  --views fisheye --protocols radial \
  --out eval_out/fovbench_vggt360 2>&1 | tee eval_out/fovbench_vggt360.log
```

```bash
python -m slambench.run --egosynth-root "$EGOSYNTH" --calib-root "$EGOSYNTH_CALIB" \
  --datasets aea,nymeria --models vggt_1b --baselines raw,vggt360 \
  --context-frames 1 --n-frames 25 --takes 8 \
  --out eval_out/slambench_vggt360 2>&1 | tee eval_out/slambench_vggt360.log
```

Plus the resolution control, which is what makes the FOV row readable:

```bash
python -m fovbench.run --adt-root "$ADT" --n-frames 25 --models vggt360 \
  --views fisheye --protocols radial --vggt360-source view \
  --out eval_out/fovbench_vggt360_matched 2>&1 | tee eval_out/fovbench_vggt360_matched.log
```

## Three refusals, so a wrong invocation fails in a second

Each guards a run that would otherwise return a full, plausible table about
something that never happened:

1. `vggt360` with `--views rect` or `--protocols window` — refused. It consumes
   a fisheye frame plus a lens.
2. `--baselines vggt360` with any `--models` other than `vggt_1b` — refused. The
   arm runs the vendored VGGT-1B itself (its fusion reads frame attention off a
   37x37 patch grid no other backbone exposes), so a `dav2_large x vggt360` row
   would contain no Depth-Anything at all.
3. `--baselines vggt360` with `--context-frames > 1` — refused. It already hands
   VGGT a nine-view reconstruction of one frame; an N-frame context is 9N views
   in one pass, not the temporal sweep the column claims.

`--baselines vggt360` also needs `--calib-root`, like `rect_derect`.

## The two caveats that must travel with the numbers

**The FOV row is not resolution-matched, and that favours it.** The nine tangent
views are cut from ADT's native 1408 frame — the resolution the port is designed
for — while the vanilla four see a 518 resize. The answer is still fused onto the
518 scoring grid, so every metric, mask and eccentricity bin is the one everyone
else is scored under; only the sharpness of what the network saw differs. A
60 deg view at 518 px is a **0.62x downsample** of the 1408 source against
**1.69x up** from the 518 view. That is why the `--vggt360-source view` control
above is not optional: `native` says what the method is worth, `view` says what
it is worth on the same pixels, and the gap between them is the resolution term
rather than the lens term.

**The SLAM arm is not resolution-matched either, in the same direction.** Nine
518 views out of the 896 frame is a 1.04x resample — essentially all of those
2.4 M pixels carry real detail — against `raw`'s single 518 resize. In angular
terms **7.83 px/deg against 4.37**, a 1.79x sampling advantage on a 9x pixel
budget. Exploiting resolution by tiling *is* the method, so this is not a flaw in
the comparison; it is a sentence that has to appear next to it.

The headline SLAM run deliberately omits `rect_derect`: arms are scored on the
points every arm could answer for, and a 110 deg pinhole has no answer at the
rim, so including it truncates the whole comparison at ~55 deg — which is exactly
the field this method exists to cover. The three-arm run is in 0b and is read
only against itself.

## Read the log, not only the table

Both runs print things the tables cannot carry:

* **`lens cone ... | layout reaches ... | ring stops N deg short of it`**, per
  take (slambench). The 60/tilt-26 layout is sized against Aria's *nominal*
  54.83 deg cone; ego-synth calibrations are per take. A ring that stops short
  leaves the rim to the centre view's corners alone, which reads as the method
  degrading at eccentricity when it is the layout not arriving. If this says
  "short" on most takes, re-run with `--vggt360-ring-tilt $(cone - 30)` before
  quoting anything about the rim.
* **`N% of the imaged cone ... filled with a constant`** (fovbench only). Expect
  well under 1%.

## What moved and what did not

**Nothing published moved, and this was measured rather than argued.** A no-data
digest of the `fovbench` radial *and* window scoring path (analytic model,
synthetic fisheye scene, both views) and of `slambench`'s `raw`/`rect_derect`
arms returns the identical hash in this tree and in a clean checkout of `ce3d001`:

    fovbench   5866025bec134fb1     slambench  44643a387e51e593

So **#019 and #020 are unaffected** and neither needs re-running on account of
this. `DEFAULT_BASELINES` is still `(raw, rect_derect)` and `DEFAULT_MODELS` is
still the four vanilla models, so every command already in `GPU_EXPERIMENTS.md`
measures what it measured before.

Shared code did change, additively: `fisheye_ray_lut` is memoised and returns
read-only shared arrays (every driver was rebuilding an identical KB4 inversion
per frame); `fisheye_to_persp` split out a cacheable `persp_maps`;
`Fisheye624` gained vectorised `project_bulk`/`unproject_bulk`, which are a
calling convention rather than a second camera model — the per-point reference
loop cannot serve a 21-million-ray warp, and `test_camera.py` already pins the
vectorised path to the reference at 1e-9 px.

## Done when

- [ ] `python -m pytest fovbench/tests slambench/tests tests -q` passes on the box
- [ ] the four vanilla `fovbench` cells in `fovbench_vggt360` match the
      corresponding cells of the #019 headline run — same digest, so they must
- [ ] `results/fovbench-023-<sha>` and `results/slambench-023-<sha>` pushed
      (`results.json`, `results.csv`, `report.txt`, logs — not figures)
- [ ] both `--vggt360-source` arms present, or the FOV row is labelled
      "native only, resolution not matched"
- [ ] issue commented with the sha and the two headline numbers

## Cost

Untimed, deliberately. Nine to thirteen 518 px views in one VGGT-1B pass per
frame, against one 518 px view for a vanilla row — an order more compute per
frame, and `fovbench`'s README notes the harness is otherwise CPU-bound, so this
row will be the first that is not. Smoke it at `--n-frames 3` before the grid.
