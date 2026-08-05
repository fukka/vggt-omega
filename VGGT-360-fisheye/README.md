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
| input domain | 360°×180° ERP, every pixel imaged | KB4 fisheye, imaged cone θ ≤ **54.83°** (the lens' usable FOV — *not* the 62.33° KB4 polynomial turnover, see below) |
| view layout | 6 yaw ring @ FOV 110° + 2 poles | **1 center + 8-direction ring** (azimuth step 45°, tilt 26°, FOV 60°; design rule `tilt + FOV/2 ≳ θ_max`) |
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

## Two different angular limits (θ_max ≠ the KB4 turnover)

Conflating these is a bug, and `utils/fisheye_cam.py` keeps them apart:

- **fold-back turnover**, `kb4_max_incidence(k)` = **62.33°**. The KB4 polynomial
  is only monotonic up to here; past it the projection is non-injective and
  sampling those rays aliases onto wrong in-cone pixels. A property of the *fit*,
  and the right guard for warps and for truncating the inverse LUT.
- **usable FOV**, `aria_valid_theta_max()` = **54.83°**. What the lens actually
  images. Two independent measurements agree: (a) the image circle is inscribed
  in the 1408² frame — the smallest principal-point-to-border margin is 690.3 px,
  which unprojects to 54.83°, while the circle at the turnover would need
  745.8 px; (b) photometrically, on a real ADT frame the p90 brightness per
  incidence band falls 139 → 92 → 52 → 12 across 48° → 55° → 57° → 61°, and the
  60–64° band is 98.6% black.

`FisheyeCam.theta_max()` returns the min of the two, so the ray LUT, view
rendering, layout cone clamp, fusion and the eval mask all inherit the usable
limit. Earlier versions used the turnover as the usable FOV, which fed ~5–12% dead
vignette pixels per ring view into VGGT flagged as valid and put **10.9%** dead
pixels into the eval mask — scored against GT with a black input. Any number
produced before this change should be regenerated. The same authority is used by
the baselines (`aria_fisheye.usable_max_incidence`) so the rows stay comparable.

## Layout geometry (why 1 + 8 views)

- ring tilt 26° + FOV 60° → each ring view covers incidence 4°…56° along its
  azimuth; the outer edge just clears the imaged cone.
- adjacent ring-view centers are `2·asin(sin26°·sin22.5°) ≈ 19.3°` apart ≪ 60°
  FOV → large azimuthal overlap; ring↔center separation 26° < 60° → radial
  overlap. Verified: zero coverage holes, ≥2-view overlap on 92.5% of the cone,
  and 94.5% of each ring view's area carries image.
- ring-view corners still reach 60°, i.e. ~5° past θ_max; they carry no image and
  are zeroed by the analytic valid mask (which also gates attention and fusion).
- 26° is the knee of the trade-off at FOV 60 / 8 views — tilt 28 wastes 9.4% of
  each view, tilt 24 opens 0.46% holes, and the old tilt 32 (aimed at the
  turnover) wasted **16.7%** per view on pixels the sensor never images:

  | ring tilt | view valid % | holes | ≥2-view |
  |---|---|---|---|
  | 32 (old) | 83.3% | 0.0000% | 96.6% |
  | 28 | 90.6% | 0.0000% | 94.1% |
  | **26** | **94.5%** | **0.0000%** | **92.5%** |
  | 24 | 98.0% | 0.4613% | 90.6% |
- module 1's adaptive augmentation now adds **1–2** views, not 4: at this ring
  density an azimuth-twist candidate lands 8.7° from an existing view and
  `min_sep_deg` correctly rejects it. Measure `--no-adaptive` before assuming the
  extra views earn their compute.

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
- **G. usable-FOV photometry** — bins a real frame's p90 brightness by incidence
  angle and asserts θ_max sits on the vignette shoulder, so the cone can never
  drift back onto the KB4 turnover unnoticed. Runs on the checked-in
  `outputs/sweep_omega/raw_fisheye.png` when `--adt-root` is absent (skipped, not
  failed, if neither is available). Verified to accept 54.8–58° and reject 52°,
  60° and the old 62.33°.
- **D. real ADT frame** — view montage, footprint overlay (the θ_max circle now
  lands *on* the visible image circle; with the old turnover-based θ_max it sat
  outside it, in the vignette), coverage map, and an RGB re-fusion of the views
  reproducing the original photo.

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
