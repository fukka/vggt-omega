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
| output / eval | ERP depth, median align, crop 68 pole rows | fisheye depth vs ADT GT, scale-shift (affine-invariant) align, cone∧GT validity mask (`utils/metrics_adt.py`) |
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

# 2) full eval on the GPU box
python VGGT-360-fisheye/main_adt.py \
    --adt-root <ADT_ROOT> --max-frames 100 \
    --qual-dir outputs/vggt360_fisheye_qual
```

Ablations: `--fuse mean` (uniform vs attention-weighted fusion),
`--no-adaptive` (module 1), `--no-sa-mask` (module 2),
`--n-ring 6 --ring-tilt 32` (7-view budget layout),
`--pred-domain range` (score euclidean range instead of planar z).

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
  2. `--head depth` — the point head (`||world_points||`, upstream's choice)
     is empirically noisier than the depth head; this switches to per-view
     z × secant.
  3. `--dtype fp32` — rules out bf16/fp16 autocast noise in the aggregator.
  4. `--fov 45 --ring-tilt 25` — a corner-free layout (max corner incidence
     ~55° < θ_max): every view has zero invalid pixels, isolating black-
     corner poisoning of the DPT head (costs rim coverage 55–62°, fine for
     an ablation).
  The model loader also self-checks checkpoint↔model key matching at start
  ("weight check OK") to rule out silently-unloaded weights.

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
- The fused quantity is euclidean **range** ‖world_points‖; `--pred-domain z`
  (default) converts to planar z via cos θ — a >2× factor at the FOV edge, so
  match this to your GT's convention deliberately.
- Output is up-to-scale (VGGT); the default `scale_shift` alignment matches
  the repo's ADT baseline protocol (`finetune/eval/baselines/benchmark_adt.py`),
  so results can be tabled next to DAC / UniK3D.
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
