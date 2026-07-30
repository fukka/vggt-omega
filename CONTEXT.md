# Domain glossary

Shared vocabulary for this repo. The terms below are the ones that have
actually caused bugs when left fuzzy — each entry says what the term means
here, and where the authority for it lives.

## Depth conventions

**Planar z** — depth measured along the camera's optical axis (a z-buffer
value). Constant across a plane parallel to the image plane.

**Euclidean range** — distance from the optical centre along each pixel's own
ray. Related to planar z by `range = z / cos(theta)`, where `theta` is the
incidence angle of that pixel's ray.

These are *not* interchangeable on a wide-FOV camera. On the Aria 214-1
fisheye the factor `1/cos(theta)` runs from 1.0 on axis to **2.15x** at the
62.3° rim. The difference is radial, so an affine (scale+shift) alignment
cannot absorb it — mixing the two makes a perfect prediction score AbsRel
0.146 / δ1 0.79.

Which is which, in this repo:

| Quantity | Convention | Authority |
|---|---|---|
| ADT `depth_npy` GT | **planar z** | measured by `VGGT-360-fisheye/checks/check_gt_depth_domain.py` (RANSAC plane fit, peak at `a=1.00`) |
| VGGT-360 fused output (`\|\|world_points\|\|`) | euclidean range | `VGGT-360-fisheye/utils/fisheye_fusion.py` |
| VGGT depth head output | planar z *per view* | converted to range via the tangent secant in `main_adt.py` |
| ERP / panorama depth | euclidean range by definition | upstream VGGT-360 needs no conversion — every ERP pixel is a ray |
| Depth-Any-Camera fisheye GT | converted z → range on load | `third_party/depth_any_camera/dac/dataloders/scannetpp.py` |

**Scoring domain** — the convention a comparison is carried out in. Because
prediction and GT can disagree, exactly one side is always converted. See
`main_adt.py --eval-domains`, which reports both:

- `z` — scale the *prediction* by `cos(theta)`; compare against GT as stored.
- `range` — scale the *GT* by `1/cos(theta)`; compare against the fused range.
  This is Depth-Any-Camera's protocol.

The ERP-era assumption that "depth is range" is true for panoramas and false
for fisheye. That inheritance is what made the fisheye port score range
against z-GT.

## Camera and image geometry

**Imaged cone** — the set of rays a lens physically images. For KB4 the
forward polynomial `theta_d(theta)` is monotonic only up to a *turnover*
angle (~62.33° for Aria 214-1); beyond it the projection folds back and is
non-injective. Rays past the turnover are never imaged and must be masked
everywhere, or they alias onto wrong in-cone pixels ("fold-back ghosting").
Authority: `VGGT-360-fisheye/utils/fisheye_cam.py:kb4_max_incidence`.

**Tangent view** — a gnomonic (perspective) crop rendered from a fisheye or
ERP frame about some view direction. Straight 3D lines stay straight in a
tangent view; this is the input format VGGT expects.

**View parameters** — a tangent view is `(azimuth, tilt, fov)` in the fisheye
port: `tilt` is the angle from the optical axis, `azimuth` the direction of
that tilt in the image plane. Replaces upstream's `(yaw, pitch)`, which
carried pole special-cases a cone does not have.

**Analytic validity** — a per-pixel mask derived from *geometry* (ray inside
the imaged cone AND source pixel inside the frame), not from pixel content.
Upstream detected invalid ERP pixels with `rgb.sum() > 0`, which is unreliable
on fisheye (dark scenes, vignetting).

## Diagnostics

**View source** — where a probe's input image comes from, independent of which
model consumes it: a file on disk, a `tangent` crop, the wide `rectifier`
pinhole, or a `raw_roi` crop (same angular coverage, no undistortion). Swapping
view source with the model held fixed separates "the view construction is the
problem" from "the model is the problem". Authority:
`VGGT-360-fisheye/checks/depth_probe.py`.

**Backend** — a loaded model that turns an image path into planar-z depth plus
its own inferred FoV: `vggt1b` (this port's vendored `vggt_visfeat`),
`vggt_omega` (local `.pt`), `official` (the pip `vggt` package, as an
independent control on the port). Swapping backend with the view held fixed
separates a checkpoint flaw from a VGGT-family trait.

**Edge alignment (`align%`)** — fraction of the input's strong Sobel edges that
have a depth edge within ~2 px, measured on **planar z** so a flat surface's
radial range-bowl doesn't register. High means the depth's structure follows the
image's. It is a *relative* diagnostic: compare runs, don't read an absolute
value as a grade. One definition, one percentile (96), one erosion of the cone
mask — otherwise numbers from different runs are not comparable. Calibration
figures are in the `edge_alignment` docstring; note it saturates on
high-frequency noise.

**Inferred FoV** — `pose_enc[7:9]`, the camera field of view the model estimates
for itself. VGGT couples depth to this estimate, so a large gap against a view's
*known* render FoV is the suspected mechanism for depth that bends away from its
input. Only a view this repo rendered has a known true FoV; for a view loaded
from a file it is genuinely unknown and is reported as such.

## Evaluation

**Alignment** — how an up-to-scale prediction is fitted to metric GT before
scoring. Modes live in `finetune/eval/metrics.py:align_depth`:
`scale_only` (median of per-pixel ratios, `median(gt/pred)`), `scale_shift`
(least-squares affine in depth space), `disparity_scale_shift` (affine in
1/depth — the correct protocol for MiDaS/Depth-Anything-style models), `none`.

**GT validity cap vs metric cap** — two different limits, deliberately
separate:
- `--depth-max-m` (10 m) bounds which GT pixels count as valid.
- `--metric-max-depth` (100 m) excludes out-of-range *predictions* from the
  metrics, as the official DAv2 eval does.
Conflating them silently changes which pixels are scored. `finetune/eval/metrics.py`
is the single protocol; there is no second copy.
