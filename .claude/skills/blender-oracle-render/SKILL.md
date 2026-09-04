---
name: blender-oracle-render
description: Render paired ground-truth ("oracle") and masked ADT frames with Blender — a full-frame Aria fisheye, a pinhole frame pixel-aligned to the rectified fisheye, and the masked twin of each, with RGB and depth for all four. Use when an experiment needs the true content a real fisheye could not capture: oracle fills for rectification wedges, black-vs-truth arms that differ in nothing else, or fisheye-vs-perspective comparisons with matched content. Launches a dedicated rendering agent.
---

# Blender oracle renders for ADT

Produce, for the same ADT camera pose, **four frames — RGB and depth for each** —
forming a 2×2 of projection × validity:

| Output | What it is | Invalid pixels |
|---|---|---|
| `fisheye_full` | Aria KB4 fisheye at ADT's real intrinsics, KB4 extended monotonically past its turnover so the square frame's corners carry true content | **0%** |
| `fisheye_masked` | The same frame with the real imaged-disc mask applied — the corners the sensor genuinely cannot see, blacked | ~14.7% |
| `persp_full` | Pinhole frame **pixel-aligned to the rectified fisheye**, i.e. that rectification with its wedges filled by ground truth | **0%** |
| `persp_masked` | The same frame with the analytic rectification mask applied — the true wedges, blacked | ~32.3% |

The point is the oracle. A real Aria frame cannot supply the filled pixels — the
lens never imaged them — so the only honest source is a renderer that owns the
scene. And because each masked frame is a *mask of a frame whose true content we
also hold*, the black and oracle arms are exactly paired: same scene, same pose,
same pixels, differing only in whether the invalid region carries truth or zeros.
That pairing is what makes the comparison clean, and it is not obtainable from
real footage at all.

## How to run this skill

Launch **one dedicated agent** with the brief below (`Agent` tool,
`subagent_type: "general-purpose"`, or a fork if the caller's context already
holds the experiment details). Give it the whole "Agent brief" section verbatim,
plus whatever run-specific parameters the user asked for (sequence, frame range,
resolution). Do not do the work inline — Blender renders are long, and the agent
should stay resident to iterate on verification failures.

If the user has not said which sequence or how many frames, default to a **single
frame from one sequence** first and get the verification numbers green before
rendering a set. A wrong camera model rendered across 500 frames is a wasted day.

---

## Agent brief

You are setting up and validating a Blender render path that produces oracle
frames for a fisheye-vs-perspective depth experiment. Read this whole brief
before touching anything.

### The codebase

`adt_egocentric` renders photorealistic egocentric frames from ADT ground-truth
poses. Read its `README.md` first — it is accurate and will save you an hour.

- **Mac**: `/Users/f.zhang2/Desktop/projects/adt_egocentric`, Blender at `/Applications/Blender.app`
- **lambda_63**: `/user/f.zhang2/projects/adt_egocentric`, Blender at `~/blender/blender`,
  ADT at `~/Documents/projectaria_tools_adt_data_clean`

Render on **lambda_63** — that is where Blender, the ADT data and the GPUs are.
Several scripts have `_lambda_` twins (`render_from_poses_blender_lambda_.py`);
check which one carries the correct paths before editing either.

### The pipeline shape (already established — do not rediscover)

```
ADT GT poses → Blender 3.6 Cycles → 360×180° EQUIRECTANGULAR panorama
                                        ↓  cached Newton-Raphson LUT
                                   FISHEYE624 remap → fisheye PNG
```

The intermediate is a **full equirectangular panorama**, which is the whole
reason this task is possible: an ERP holds every direction, so any camera model
at any field of view can be sampled from it with no invalid pixels. Producing a
perspective output is a *second remap off the same panorama*, not a second render.

Key functions in `render_from_poses_blender.py`:

- `build_fisheye624_remap(cam_calib, out_w, out_h, eq_w, eq_h)` → `(map_x, map_y, valid_mask)`
- `remap_equirect_to_fisheye(equirect_img, map_x, map_y, valid_mask)` — `cv2.remap` over the LUT

Relevant CLI: `--output_size`, `--focal`, `--fisheye`, `--frame_step`,
`--num_frames`, `--output_dir`. Path constants (`BASE`, `BLENDER_BIN`,
`BLEND_SCRIPT`) are at the top of the script.

### What to build

**1. `fisheye_full` — full-frame fisheye.**

Keep ADT's real Aria 214-1 intrinsics and focal, so that everything inside the
imaged disc is bit-identical to the real Aria framing. The KB4 polynomial turns
over at θ_max ≈ 62.33° and is undefined beyond, but the frame corners sit at
~75°. **Extend θ_d(θ) linearly past the turnover** (continue with the slope at
the turnover) so corner rays get real directions.

This extension is a modelling choice, not the real lens. Say so in the code
comment and in any writeup. It affects only the corners; the disc is untouched.

**2. `persp_full` — pinhole, pixel-aligned to the rectified fisheye.**

This is the load-bearing requirement. The frame must land on **exactly** the
pixel grid that `finetune/data/rectify.py::FisheyeRectifier` produces, so that
`persp_full` *is* the rectified fisheye with its wedges filled by ground truth.

Read `FisheyeRectifier._intrinsics()` in the `vggt-omega-finetune` repo
(`/user/f.zhang2/projects/vggt-omega-organized`, branch `fisheye-2x2`) and use
the identical output intrinsics `Knew`:

```
focal_out = focal_out_norm * max(H, W)
Knew = [[focal_out, 0, W/2], [0, focal_out, H/2], [0, 0, 1]]
```

Default `focal_out_norm = 0.262` (circumscribed; hFoV 124.7°, and the framing
whose invalid region is largest — 32.3% of pixels, 12.2% of solid angle). Expose
it as a CLI parameter; 0.371 (inscribed, 106.8°) and 0.55 (the repo's historical
default, 84.5°) are the other framings in use.

Mind the **rotation convention**: raw ADT frames need a 270° CCW rotation and the
principal point is off-centre, which `FisheyeRectifier` already accounts for via
`finetune/aria_calibration.py`. Match whatever the renderer does — a
90°-increment mismatch will still "look fine" and be completely wrong.

**3. The masked counterparts — required.** Emit `fisheye_masked` and
`persp_masked` alongside the full frames: the same rendered pixels with the
analytic invalid mask applied. Two rules:

- The mask must be **analytic** — for the fisheye, the real imaged disc
  (`r ≤ f·θ_d(θ_max)`, the *unextended* polynomial, plus a source-frame bounds
  check); for the pinhole, `FisheyeRectifier.valid_mask()` at the same
  `focal_out_norm`. Never derive it from pixel darkness.
- The masked frame must be **byte-identical to its full twin inside the mask**.
  Assert this; it is a one-line check and it is what guarantees the two arms
  differ in nothing but the invalid region.

**4. Passes: RGB and depth, for all four outputs.** Depth is not optional —
without it these frames can only be looked at, not scored. Blender's panoramic Z
pass needs converting to perpendicular Z for the target camera; the repo already
does this for fisheye (see the note near the remap builder), so do the equivalent
for the pinhole target. Depth in the masked variants must be zeroed on exactly
the same mask as the RGB, so a downstream evaluator's `depth > 0` validity test
agrees with the analytic mask. Instance segmentation only if the caller asks.

### Acceptance test — this is the deliverable, not the pictures

**Alignment check.** Take `fisheye_full`, rectify it with
`FisheyeRectifier(preset='aria-214-1', focal_out_norm=<the same value>)`, and
compare against `persp_full` **inside the rectifier's analytic valid mask**.
They are two routes to the same rays and must agree to resampling error.

Report actual numbers — PSNR and median |Δ| in 8-bit levels. Treat **PSNR ≥ 30 dB
and median |Δ| ≤ 2/255** as the bar. If you miss it, do not tune the threshold:
find the cause. Ranked by likelihood — a rotation/flip convention mismatch, a
principal-point offset, an off-by-one in the ERP sampling grid, or a stale LUT.

**Report these too:**
- invalid-pixel fraction of each output, measured not assumed: **0.000%** for the
  two full frames, ≈14.7% for `fisheye_masked`, ≈32.3% for `persp_masked` at
  `focal_out_norm=0.262`
- that each masked frame equals its full twin inside the mask (assert, don't eyeball)
- that RGB and depth carry the *same* mask in the masked variants
- the geometry of the aligned frame (hFoV, diagonal FoV) against the expected
  124.7° / 139.3° for `focal_out_norm=0.262`
- a sanity render of the *unextended* fisheye alongside, so the corner content
  the extension bought is visible as a difference rather than asserted

### Traps that have already cost time here

- **The LUT cache is keyed on output size only** (`_remap_{size}.npz`). Change the
  camera model or the focal and it will silently reuse the stale LUT and render
  something plausible and wrong. Key the cache on the model parameters, or delete
  it between configurations.
- **Never test validity with `rgb.sum() > 0`.** It misfires on dark scene content
  and on vignetting, and past the KB4 turnover a remap folds back and samples
  real-but-wrong pixels rather than leaving the frame. Compute validity
  analytically from the ray angle *and* a source-frame bounds check.
- **A silently no-op config is the failure mode to fear.** In this project an
  experiment once produced five identical metrics to four decimals because one
  kwarg never reached the dataset; the run completed and the numbers read as a
  clean negative result. Fingerprint your inputs (hash the array you actually
  write) and assert that configurations which should differ *do* differ.
- **The sensor frame clips the disc.** The Aria image circle has radius
  ≈271 px at 512 while the frame half-width is 256, so the disc is cut by the
  square edges along the axes. Directions between those radii are inside the cone
  yet never imaged. This is why circumscribed rectification of a *real* frame
  recovers only 94.1% of the cone — and why the oracle render is worth doing.
- **Blender renders are slow.** Launch on lambda_63 inside a **named tmux
  session**, `tee` to a logfile, and grep the log remotely. Never stream a render
  log into context. Check GPU memory first — the box is shared and both GPUs are
  often largely occupied.

### Deliverables

1. The render script (new, or a clearly-named variant) committed to
   `adt_egocentric`, with the KB4 extension and the pinhole remap documented in
   comments — including that the extension is a modelling choice.
2. One frame rendered through all four outputs (RGB + depth each), with the
   invalid-pixel fractions measured.
3. The alignment verification numbers, reported plainly — including if they fail.
4. A short note on what the extension changes, i.e. the angular extent the
   corners actually reach.

Report the numbers you measured. If something does not verify, say so and say
why; a wrong oracle is worse than no oracle, because everything downstream
inherits it silently.
