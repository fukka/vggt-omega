# ScanNet++ DSLR — camera facts, processing, and the traps

**Read this before touching ScanNet++ in this repo.** Every number below was
measured from the calibration ScanNet++ ships, not transcribed from a paper.
Where a published claim disagrees, the disagreement is called out.

Written 2026-08-19, from an audit prompted by the question "ScanNet++ uses Aria
glasses, so why is the FOV different?" — it does not use Aria, and that
misconception is worth killing on sight. Chinese-language companion covering
the depth path specifically: [scannetpp-depth-audit.zh-CN.md](scannetpp-depth-audit.zh-CN.md).

---

## 1. What ScanNet++ actually is

Three capture devices, **none of them head-mounted**:

| Device | Role |
|---|---|
| **Faro Focus Premium laser scanner** | sub-mm point clouds (0.9 mm avg spacing) → Poisson mesh. The geometry ground truth. |
| **Sony Alpha 7 IV DSLR + wide fisheye lens** | 33 MP stills, fixed white balance, 1/100 s. The `dslr/` tree. |
| **iPhone** | RGB-D stream, automatic mode. The `iphone/` tree. |

460 scenes, ~280 k DSLR images, ~3.7 M iPhone RGB-D frames (v1; later versions
are larger). Verified against the official
[documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation) and
[changelog](https://scannetpp.mlsg.cit.tum.de/scannetpp/changelog): **no version
of ScanNet++ has ever used Project Aria or any egocentric/head-mounted device.**

> If you are comparing against ADT/Aria, you are comparing two physically
> different lenses. See §6.

---

## 2. What the dataset ships, per scene

```
<scene>/dslr/nerfstudio/transforms.json      OPENCV_FISHEYE intrinsics + poses
<scene>/dslr/resized_images/<file_path>      RGB, fisheye, 1752x1168
<scene>/dslr/resized_undistorted_images/     RGB, PINHOLE, same resolution
<scene>/dslr/resized_anon_masks/<stem>.png   anonymisation mask (NOT a lens mask)
<scene>/dslr/resized_undistorted_masks/      anonymisation mask for the pinhole set
<scene>/dslr/render_depth/<stem>.png         uint16 mm, PLANAR Z, mesh-rendered
<scene>/scans/mesh_aligned_0.05.ply          the laser mesh
```

Full-resolution originals also exist. `render_depth/` is produced by the
official toolbox from the laser mesh via [`renderpy`](https://github.com/liu115/renderpy)
— **it is not shipped by default and was absent on our download**; it has to be
rendered.

**Calibration is per-scene.** Every scene has its own `fl_x, fl_y, cx, cy,
k1..k4`. Never hard-code one scene's numbers.

---

## 3. The camera model, and the real field of view

`camera_model = OPENCV_FISHEYE` = **Kannala–Brandt 4 (KB4)**, the same family as
Aria and KITTI-360:

```
r(θ) = θ · (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
```

where `r` is the **normalised image radius** (pixel distance from the principal
point ÷ focal length).

Measured on the two scenes whose calibration we hold:

| | `3f15a9266d` | DAC sample scene |
|---|---|---|
| `fl_x, fl_y` | 616.721, 617.354 | 789.908, 791.557 |
| `cx, cy` | 878.593, 589.767 | 879.204, 584.789 |
| `k1..k4` | 0.06109, 0.003350, 0.002988, −0.001002 | −0.02947, −0.005770, −0.002148, 0.0001484 |
| **fisheye diagonal FOV** | **169.68°** | **174.20°** |
| fisheye horizontal | 146.46° | — |
| fisheye vertical | 103.89° | — |
| max incidence θ (frame corner) | **84.84°** | 87.10° |
| undistorted (pinhole) diagonal | 132.34° | 118.62° |
| undistorted horizontal | 124.10° | 109.04° |

### Ω is the whole rectangle — the opposite of Aria

ScanNet++'s DSLR is a **full-frame** fisheye: the image circle covers the sensor,
so all four corners carry real content and the valid region is the entire
rectangle. Corner 40×40 patches read **grey ≈ 80 ± 11**; a vignetted circular
fisheye would read ≈ 0 ± 0.

`cameras._default_theta_max` therefore bounds θ by the **corner** radius, not the
nearest-edge radius. Using the inscribed circle instead would silently drop
**47% of a 504×336 frame** from Ω — and Ω is what every loss (Eq. 8, 10) and
every metric (Eq. 16–18) sums over.

**Aria is the reverse case** and needs the inscribed circle. Do not copy one
lens's rule to the other. See [CONTEXT.md](../../CONTEXT.md) § "Imaged cone".

---

## 4. The published "115°" is wrong, and here is the likely origin

The RayTun3R paper ([PAPER.md](../../raytun3r/PAPER.md) §6) lists ScanNet++ as
115°. Nothing about the fisheye images is 115°.

| family | measured, two scenes | contains 115°? |
|---|---|---|
| **fisheye** (what we feed the model) | diagonal 169.7° / 174.2° | no, by a wide margin |
| **undistorted / pinhole** | diagonal 132.3° / 118.6°, horizontal 124.1° / 109.0° | **yes** |

So the most probable reading is that the paper quotes the FOV of ScanNet++'s
**undistorted** image set while the method runs on the **fisheye** set. This is a
supported hypothesis, not a settled fact — confirming it means tabulating the
`undistorted` intrinsics across many scenes, which is cheap and has not been done.

### How the number was established, and what that does *not* prove

Established (pure arithmetic on the shipped calibration, three independent
inversions agreeing to <1e-6°: dense scan, bisection, and `cv2.fisheye.projectPoints`
run backwards):

- the frame corner sits at `r = 1.7153`, which inverts to **θ = 84.84°**;
- the KB4 polynomial's fold-back turnover is at **124.2°**, so 84.84° is deep
  inside the monotone, invertible range;
- the corners carry real texture (§3);
- a second scene and two third-party repos land in the same 170–180° family.

**Not** established: that the polynomial is *physically valid* at `r = 1.7153`.
COLMAP's fit is only constrained where it had observations, and that support
region is unknown.

> ⚠️ An earlier version of this repo argued that `project ∘ unproject`
> round-tripping to 1.5e-5 px at the corner proved the corner was "inside the
> lens model, not extrapolation". **That argument is invalid** — a round trip
> only shows the forward map and its numerical inverse agree, which is true of
> *any* polynomial including one that has left physical reality. The real support
> for "not extrapolation" is corner texture plus cross-scene/cross-repo agreement.

---

## 5. Traps — every one of these has cost time here

**① `render_depth` is planar z. Predictions are usually euclidean range.**
They differ by a per-pixel `1/cos θ`, which reaches **10.9×** at this lens's
84.8° rim and averages **1.93×** over the frame. No global scale absorbs a
radially-varying factor. Before this was fixed, a *perfect* range predictor
scored AbsRel 0.426 / δ₁.₂₅ 0.412 — worse than the paper's worst reported
method, and 2× the entire effect Tab. 3 exists to show.
`depth_any_camera` flags the same thing independently (`depth = depth / fisheye_grid_z`).

**② Convert the depth convention at the *rendered* resolution, then resample.**
Resampling first pairs each z with a neighbour's θ, and `1/cos` amplifies the
mismatch exactly where cos is smallest. Worth 0.0031 → 0.0002 AbsRel.

**③ `cv2.fisheye.undistortPoints` silently saturates past ~78°.**
On `3f15a9266d` it returns 80.09° for the corner where the true answer is 84.84°
— a 9.5° error in total FOV. Its own `cv2.fisheye.projectPoints` contradicts it.
It is exact below 78°, so the failure is invisible on narrow lenses and appears
only in the outer ~3.5% of pixels. **Use a bisection-safeguarded inverse**
(`cameras.KannalaBrandt.theta_of_r`), never OpenCV's, for this dataset.

**④ The shipped masks are anonymisation masks, not lens masks.**
Both `resized_anon_masks` and `resized_undistorted_masks` remove ~0.2–0.4% of
pixels — blacked-out faces and screens — with **0.000%** off in the corners and
no radial structure. `has_mask: true` in `transforms.json` means only that these
exist. **Ω comes from `camera.valid_mask` / `theta_max`, never from a mask file.**

**⑤ `mask_path` is a bare filename, and masks live in a sibling directory.**
`transforms.json` gives `"mask_path": "DSC07484.png"` with no directory, and the
files are in `dslr/resized_anon_masks/` (there is no `dslr/masks/`). Every
plausible-looking wrong path reads exactly like "this dataset ships no masks".

**⑥ 16% of frames are flagged `is_bad` and must be dropped.**
143 of 896 on `3f15a9266d`, in 8 contiguous runs, the longest 132 frames.
Dropping them splices the sequence: consecutive-pair rotation goes from max 4.46°
to max 64.4° while the median barely moves (0.943°). **Use medians, not means,
on any per-pair statistic.**

**⑦ `frames` is not in filename order; `test_frames` is not an eval split.**
`data.py` sorts by `file_path`, so "stride 1" really is temporally consecutive.
`test_frames` is 10 frames with wild inter-frame rotation (median 45°) — not a
sequence.

**⑧ Poses are nerfstudio/OpenGL convention and metric.**
Converted to OpenCV camera-from-world on load; validated against SIFT+MAGSAC++ to
0.17°. Camera bbox 1.12 × 1.94 × 0.29 m, path 11.2 m — metres, not normalised units.

---

## 6. ScanNet++ DSLR vs Aria/ADT — the domain gap, quantified

Measured on real frames from both datasets, at a common 504-long-side working size.

| | ScanNet++ DSLR | Aria 214-1 RGB (ADT) |
|---|---|---|
| device | Sony A7 IV + fisheye lens | Aria glasses |
| frame | 1752 × 1168 (3:2) | 1408 × 1408 (square) |
| **image circle** | **covers the frame — corners have content** | **inscribed — corners are black** |
| corner grey level | 80 ± 11 | **3.0 ± 1.7** |
| valid-pixel fraction | 100% | 75.5% |
| **imaged cone θ_max** | **84.84°** (frame corner) | **54.83°** (inscribed circle) |
| horizontal FOV | 146.5° | 109.7° (circle diameter) |
| `k1` | 0.0611 | **0.3852 — 6.3×** |
| stretch vs equidistant @ 50° | +4.9% | +17.3% |
| max `1/cos θ` | 10.9× | 1.74× |
| depth GT | laser-mesh render | synthetic digital twin |

Three consequences:

1. **41% of ScanNet++ pixels sit at incidence angles Aria never images**
   (54.83°–84.84°). Anything learned there has no counterpart on ADT.
2. **The distortion curves are different shapes, not scaled versions.** Aria
   pushes pixels outward hard and early then rolls over; ScanNet++ stays close
   to equidistant throughout.
3. **Radius-indexed adapters do not transfer.** RayTun3R's learned tables are
   indexed by normalised radius `ρ ∈ [0,1]` (`N_r = 20` bins). On ScanNet++
   `ρ = 1` means 84.8°; on Aria everything past **`ρ = 0.68` is black** — 32% of
   the table is dead — and at matched `ρ` the two lenses differ by 4° / 8° / 30°
   at `ρ` = 0.4 / 0.7 / 1.0. A table fitted on one indexes wrongly on the other.

Non-geometric gaps also matter: static DSLR sweeps of empty rooms (0.94° between
consecutive frames) versus egocentric walking with hands and people in the near
field; laser-scanned versus synthetic depth.

> **For RayTun3R specifically this is not a defect.** It is a per-sequence online
> adaptation method; the paper's own Sec. 6 limitation (1) states the correction
> is camera-specific and a different lens needs a new fit. "Train on ScanNet++,
> test on ADT" is contrary to its design.
>
> **For the wider project**, the recommendation is: use ScanNet++ as a
> *verification* set, not a training set for ADT work. Its laser-grade dense
> depth and clean geometry make it excellent for catching camera-model,
> depth-convention and Ω bugs — the failure modes in §5 that are otherwise
> silent. For training toward ADT, in-distribution ADT data is worth more.

---

## 7. Reproducing every number here

All of it comes from `transforms.json` plus a handful of frames — no GPU, no
weights, seconds to run:

```bash
python -m raytun3r.experiments.data_audit --path <scene> --json audit.json
```

That prints the FOV derivation, the ignored `transforms.json` keys, pose
sanity, and the per-stride rotation regime. The archived output for
`3f15a9266d` is on the `results` branch:

```bash
git show results:results/rt3r/s10-snpp-3f15a9266d-da3s/data_audit.json
```

To check a claim independently of this repo's code, invert the KB4 polynomial by
bisection and confirm against `cv2.fisheye.projectPoints` (**not**
`undistortPoints` — trap ③).

A staged few-MB sample of `3f15a9266d` with the full `transforms.json` can be
produced with `raytun3r.experiments.make_local_sample`. **It must not be
committed** — this repo is public and ScanNet++ is licensed per recipient.

---

## Sources

- [ScanNet++ paper (arXiv:2308.11417)](https://arxiv.org/pdf/2308.11417) · [overview](https://www.alphaxiv.org/overview/2308.11417v1)
- [Official documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation) · [changelog](https://scannetpp.mlsg.cit.tum.de/scannetpp/changelog) · [NVS benchmark](https://scannetpp.mlsg.cit.tum.de/scannetpp/benchmark/nvs)
- [Official toolbox `dslr/undistort.py`](https://github.com/scannetpp/scannetpp/blob/main/dslr/undistort.py) — `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(..., balance=0.0)`, outputs `PINHOLE` at unchanged resolution
- [`liu115/renderpy`](https://github.com/liu115/renderpy) — the renderer behind `render_depth`
- [Depth Any Camera (arXiv:2501.02464)](https://arxiv.org/html/2501.02464v1) — treats ScanNet++ DSLR as `crop_wFoV: 180`, converts z-buffer → euclidean
- [Calibration Tokens (arXiv:2508.04928)](https://arxiv.org/html/2508.04928v1) — RayTun3R's CalTok baseline, also on ScanNet++ fisheye
