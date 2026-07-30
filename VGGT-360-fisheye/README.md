# VGGT-360-fisheye — VGGT-360 on ADT / Aria KB4 fisheye

A port of [VGGT-360](https://github.com/Yuanjiayii/VGGT-360) (*Geometry-Consistent
Zero-Shot Panoramic Depth Estimation*, Yuan et al., CVPR 2026) from full 360°
equirectangular panoramas to the **single Aria RGB fisheye frame** of ADT
(Aria Digital Twin), evaluated against ADT's dense GT depth.

The paper's core idea transfers unchanged: split the input into overlapping
perspective views sharing one optical center, let a VGGT-like 3D foundation
model reconstruct **one scale-consistent 3D model** from all views at once
(its cross-view attention resolves the per-view monocular scale ambiguity),
then re-project onto the original pixel grid with attention-derived
confidence weights. Only the two sphere↔perspective projection steps are
ERP-specific — this port replaces them with the Kannala-Brandt (KB4) camera
model; the attention machinery is untouched.

## What changed vs upstream

| | upstream (ERP panorama) | this port (Aria fisheye) |
|---|---|---|
| input domain | 360°×180° ERP, every pixel imaged | KB4 fisheye, imaged cone θ ≤ **62.33°** (the KB4 polynomial turnover) |
| view layout | 6 yaw ring @ FOV 110° + 2 poles | **1 center + 8-direction ring** (azimuth step 45°, tilt 32°, FOV 60°; design rule `tilt + FOV/2 ≈ θ_max`) |
| pixel↔ray | lon/lat linear mapping | KB4 forward (render) / inverse LUT (fusion) — `utils/fisheye_cam.py` |
| view split | `ERP2Persp` | `fisheye_to_persp` (`utils/fisheye_views.py`) |
| invalid pixels | none (poles cropped in eval) | **analytic cone mask** threaded through SA masks → VGGT `rgb_mask` attention bias → fusion weights |
| adaptive views (module 1) | yaw/pitch neighbors, pole special-case | (azimuth, tilt) neighbors, one cone clamp — no poles on a cone |
| SA attention (module 2) | unchanged | unchanged (only the valid mask is analytic) |
| fusion (module 3) | ERP lon/lat ray grid | fisheye KB4 ray LUT (`utils/fisheye_fusion.py`); attention-metric weights (`build_selfview_confidence`) vendored **verbatim** |
| output / eval | ERP depth, median align, crop 68 pole rows | fisheye depth vs ADT GT, scale-shift (affine-invariant) align, cone∧GT validity mask; metrics shared with the baselines (`finetune/eval/metrics.py`) |
| depth convention | ERP depth **is** range — no conversion needed | GT is planar **z**, prediction is **range** → one side always converted (`--eval-domains`) |
| dataset | `Datasets.Stanford2D3D` (not shipped upstream) | `datasets/adt.py` (raw fisheye, 270° CCW rotation, mm→m) |

Bug fixes applied to the upstream copy (documented inline):
`vggt_visfeat/layers/attention.py` imported a module upstream never shipped
(`attention_utils`, symbol unused — import removed) and hardcoded `.cuda()`
on the attention-save path (now device-safe). Upstream `main.py` is kept as
`main_erp_upstream.py` for reference.

## Layout geometry (why 1 + 8 views)

- ring tilt 32° + FOV 60° → each ring view covers incidence 2°…62° along its
  azimuth; the outer edge lands exactly on the imaged cone.
- adjacent ring-view centers are `2·asin(sin32°·sin22.5°) ≈ 23.4°` apart ≪ 60°
  FOV → large azimuthal overlap; ring↔center separation 32° < 60° → radial
  overlap. Verified: zero coverage holes, ≥2-view overlap on 88% of the cone.
- ring-view corners poke past θ_max; they carry no image and are zeroed by the
  analytic valid mask (which also gates attention and fusion).

## Run

```bash
# 1) geometry self-check (numpy/cv2 only, no GPU/torch needed)
python VGGT-360-fisheye/checks/check_fisheye2persp.py \
    [--adt-root <ADT_ROOT> --rgb-subdir videos_rgb]     # optional real-frame visuals

# 1b) is a model's depth actually following its input? (one model x one view)
python VGGT-360-fisheye/checks/depth_probe.py --backend vggt1b \
    --images room.png --adt-root <ADT_ROOT> --view tangent
#    swap --backend {vggt1b,vggt_omega,official} on the SAME inputs to tell a
#    checkpoint flaw from a VGGT-family trait; --view {tangent,raw_roi,rectifier}
#    tells the view construction from the model.

# 2) full eval on the GPU box
python VGGT-360-fisheye/main_adt.py \
    --adt-root <ADT_ROOT> --max-frames 100 \
    --qual-dir outputs/vggt360_fisheye_qual
```

Ablations: `--fuse mean` (uniform vs attention-weighted fusion),
`--no-adaptive` (module 1), `--no-sa-mask` (module 2),
`--n-ring 6 --ring-tilt 32` (7-view budget layout),
`--eval-domains z` (score only planar z instead of both conventions).

## Correctness checks (`checks/check_fisheye2persp.py`)

All geometry is proven before any GPU run; current status **PASS**:

- **A. ray round-trip** — render direction (KB4 forward) and fusion direction
  (KB4 inverse) are mutual inverses: max error **0.0002°**.
- **B. synthetic sphere pattern** — `fisheye_to_persp` output matches a direct
  analytic render of every view (catches any flip/rotation/scale bug).
- **C. fusion round-trip + coverage** — a synthetic range field pushed through
  per-view grids and fused back matches the ground-truth field to **0.000%**;
  the layout has **0 coverage holes** inside the cone (asserted over the full
  cone up to ~1 view-pixel of rim quantisation).
- **D. real ADT frame** — view montage, footprint overlay (θ_max circle lands
  on the physical image circle), coverage map, and an RGB re-fusion of the 9
  views reproducing the original photo at **46.6 dB PSNR**.

## Debugging bumpy / seamy fused depth

Run with `--debug-dir <dir>` — per frame it dumps: per-view inputs,
per-view predicted range (view space), **each view re-projected alone onto
the fisheye grid** (seams in the fused map are disagreements visible directly
between these tiles), attention weights, a gradient "seam map" of the fused
depth, the **pairwise overlap median-ratio matrix** and VGGT's per-view
translation norms (all views share one optical center, so ‖t‖ should be ≈0).

Decision tree:
- pairwise ratios ≫ 1 (>5–10% spread) → VGGT gave the views inconsistent
  monocular scales (pure-rotation input is triangulation-degenerate).
  Mitigate with `--harmonize-scales` (least-squares per-view log-scale
  correction on the overlap graph; validated by checks test E).
- ratios ≈ 1 but fused map bumpy → fusion side: compare `--fuse mean` vs
  `attn` (patch-level weight blockiness), raise `--erode-valid-px` toward 7
  (half a ViT patch — DPT decodes garbage near cone-clipped black corners),
  and compare `--fisheye-size 512` vs native 1408 (518-res views upsampled
  ~2.7× make small artifacts visible).
- bumps within single views (visible already in the per-view range montage,
  while `*_views_rgb.png` inputs are clean) → the problem is inside the VGGT
  forward, not projection/fusion.  Our forward differs from vanilla VGGT
  inference in exactly four ways — ablate them in this order:
  1. `--no-sa-mask` — the module-2 log-bias is injected into ALL 24 frame-
     attention layers; a Sobel-shaped bias applied 24× can imprint image
     texture into the depth (check whether bumps follow texture/edges!).
     With this flag the attention path is bit-identical to vanilla VGGT.
  2. `--head` — CONFIRMED on ADT: the point head (`||world_points||`,
     upstream's choice) is much bumpier than the depth head; `depth`
     (z × secant) is now the default.  `--head point` reproduces upstream.
  3. `--dtype fp32` — rules out bf16/fp16 autocast noise in the aggregator
     (tested on ADT: no visible effect, precision is not a factor).
  4. `--fov 45 --ring-tilt 25` — a corner-free layout (max corner incidence
     ~55° < θ_max): every view has zero invalid pixels, isolating black-
     corner poisoning of the DPT head (costs rim coverage 55–62°, fine for
     an ablation).
  The model loader also self-checks checkpoint↔model key matching at start
  ("weight check OK") to rule out silently-unloaded weights.

## Synthetic-vs-real (blur) A/B

To test whether distorted depth is caused by real-sensor motion blur, run the
same frames through the sharp rendered stream and the blurred real stream.
`--require-streams` restricts both runs to the sequences AND (per-sequence)
frames common to both streams, so the only difference is the pixels:

```bash
# sharp (synthetic) — scores exactly the frames real also has
python VGGT-360-fisheye/main_adt.py --adt-root <ROOT> \
    --rgb-subdir videos_synthetic --require-streams videos_rgb --max-frames 50
# blurred (real) — same frame set
python VGGT-360-fisheye/main_adt.py --adt-root <ROOT> \
    --rgb-subdir videos_rgb --require-streams videos_synthetic --max-frames 50
```

Sequence discovery is transparent: `find_adt_sequences` logs every sequence it
skips and why (missing/empty stream or depth dir), so a stream that is absent
for some sequences is never dropped silently — the reason a `videos_rgb` run
and a `videos_synthetic` run could otherwise cover different sequences.

## Notes / knobs

- **Rim coverage** is erosion-, not layout-, driven: the view-frustum union
  already covers 100.000% of imaged pixels, but eroding the per-view valid
  masks (which protects fusion from boundary-bleed) retires the same
  `θ_max`-rim band in *every* view at once — enlarging or re-tilting the ring
  views does not help (measured 0.36–0.46% miss across tilt 32–36 / FOV 60–66 /
  8–12 views, while per-view wasted pixels grow 5.5%→17%). The fix is the
  **two-tier rim rescue** in `fuse_views_to_fisheye`: eroded weights
  everywhere, un-eroded fallback only where the eroded tier is empty →
  0.000% miss at 512², 0.005% at native 1408².
- The fused quantity is euclidean **range**; **ADT GT is planar z**.  They
  differ by `cos(theta)` — 1.0 on axis, up to **2.15x** at the 62.3° rim — so
  one side must always be converted.  `--eval-domains` (default: both) scores
  in each convention on the fisheye grid: `z` scales the *prediction* by
  `cos`, `range` scales the *GT* by `1/cos` (Depth-Any-Camera's protocol; its
  fisheye loader calls the z→euclid conversion "critical").  Both share one
  validity mask defined on the z-domain GT, matching `benchmark_adt.py`, so
  the two rows isolate the domain and not a different pixel set.
  > Earlier versions defaulted to `--pred-domain range`, which compared range
  > against z-GT with *neither* side converted.  The error is radial, so
  > affine alignment cannot absorb it: a **perfect** reconstruction scored
  > AbsRel 0.146 / δ1 0.79 under that default.  Any number produced before
  > this change should be regenerated.
  `check_gt_depth_domain.py` now settles the convention numerically (RANSAC
  plane fit over the full cone, peak at `a=1.00` ⇒ planar z) and exits
  non-zero if it ever flips.  The same script also shows that GROUND-TRUTH
  range montages have curved iso-depth bands on flat walls — "curvy" per-view
  depth is geometry, not a bug (use `*_views_z.png` to judge smoothness).
- Output is up-to-scale (VGGT); metrics and alignment come from
  `finetune/eval/metrics.py` — the *same* module the DAC / UniK3D / DAv2 rows
  use — so results table directly next to them.  `--depth-max-m` (GT validity
  cap, 10 m) and `--metric-max-depth` (prediction exclusion, 100 m) are
  deliberately separate; the latter matches the baselines and should not be
  set to the former.
- KB4 constants are vendored from `finetune/eval/baselines/aria_fisheye.py`
  (single source of truth for the calibration; keep in sync).

## Citation

```
@InProceedings{Yuan_2026_CVPR,
  author    = {Yuan, Jiayi and Jiang, Haobo and Soh, De Wen and Zhao, Na},
  title     = {VGGT-360: Geometry-Consistent Zero-Shot Panoramic Depth Estimation},
  booktitle = {CVPR},
  year      = {2026},
}
```

Upstream code: [Yuanjiayii/VGGT-360](https://github.com/Yuanjiayii/VGGT-360);
`vggt_visfeat/` is their modified copy of Meta's
[VGGT](https://github.com/facebookresearch/vggt) (license headers preserved).
