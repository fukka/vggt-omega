# Dataset scope, 2026-08: two splits and one lens question

**Status: design, nothing run.** No training, no rendering, no GPU job was
launched to produce this. Every number below came out of a command on
`lambda_63` or a CPU-only computation on the Mac, and all of them are
reproducible with [`tools/dataset_scope_2026_08.py`](../../tools/dataset_scope_2026_08.py):

```bash
# on lambda_63 (needs the data)
python3 tools/dataset_scope_2026_08.py --adt ~/Documents/projectaria_tools_adt_data_clean
python3 tools/dataset_scope_2026_08.py --emit /netapp/datasets/f.zhang2/scannetpp/data | python3 -
cat third_party/depth_any_camera/splits/scannetpp/*.txt | cut -d/ -f2 | sort -u > dac_scenes.txt
cp /netapp/datasets/f.zhang2/scannetpp/metadata/scene_types.json .
# anywhere (CPU only, no data, ~20 s)
python tools/dataset_scope_2026_08.py --scannetpp /tmp/spp_scan.csv /tmp/spp_intr.csv
python tools/dataset_scope_2026_08.py --budget /tmp/spp_scan.csv scene_types.json dac_scenes.txt
python tools/dataset_scope_2026_08.py --coverage --grid 1408          # one scene
python tools/dataset_scope_2026_08.py --coverage --intr /tmp/spp_intr.csv   # all 1006
```

`--coverage` evaluates on a `--grid`-px Aria grid; the cone is scale-free, so
the fractions move only in the last reported digit between 352 and 1408. The
single-scene numbers quoted in §3.2 are at 1408, the 1006-scene sweep at 352.

The motivation is the external review that correctly downgraded the ADT result
from *cross-scene* to *cross-sequence*: six ADT sequences are one apartment on
one device. Nothing here fixes that on ADT — ADT has only the one apartment.
The fix has to come from ScanNet++, and §2 and §3 are about what that costs.

---

## 1. ADT

### 1.1 What is on the box

Nineteen `Apartment_*` sequences, all M1292 (`ls` on
`~/Documents/projectaria_tools_adt_data_clean`). Every one of them holds the
complete raw sources — `video.vrs`, `synthetic_video.vrs`, and
`depth/depth_images.vrs` — but only six have been **extracted** to frames:

| sequence | `videos_rgb` | `videos_synthetic` | `depth_npy` | synthetic frame-id span | synthetic stride | `syn ∩ depth` |
|---|---:|---:|---:|---|---|---:|
| `Apartment_release_clean_seq131_M1292` | 2879 | **2878** | 2939 | 313–3190 | **1** (×2877) | **2878** |
| `Apartment_release_clean_seq133_M1292` | 2844 | 400 | 2904 | 275–3116 | 7 (×351), 8 (×48) | 400 |
| `Apartment_release_clean_seq134_M1292` | 2815 | 400 | 2876 | 255–3068 | 7 (×379), 8 (×20) | 400 |
| `Apartment_release_clean_seq135_M1292` | 2880 | 400 | 2940 | 478–3355 | 7 (×315), 8 (×84) | 400 |
| `Apartment_release_clean_seq136_M1292` | 2879 | 400 | 2939 | 506–3384 | 7 (×314), 8 (×85) | 400 |
| `Apartment_release_decoration_seq132_M1292` | 2730 | 400 | 2791 | 878–3607 | 6 (×64), 7 (×335) | 400 |
| `seq137,138,140–150` (13 seqs) | 2801–2990 | **0** | **0** | — | — | 0 |

### 1.2 The synthetic-stream finding — this is the load-bearing one

**The 400 is a stride, not a ceiling, and not a container limit.** Three
independent facts settle it:

1. The 400 frames are **spread over the whole take** (seq132: ids 878–3607,
   inter-frame gaps of 6 and 7 only), not the first 400. So this is a strided
   extraction of a complete source, not a truncated one.
2. `seq131` was extracted from the *same container type* by the *same pipeline*
   and holds **2878 contiguous frames at stride 1**. Full-rate extraction
   demonstrably works.
3. Every sequence — including the thirteen with nothing extracted — carries its
   own `synthetic_video.vrs` (794–971 MB) and the identical `ADT_*_synthetic.zip`,
   which `unzip -l` shows contains exactly that one file. The source is complete
   everywhere.

The extractor is `processing/prepare_syn_real_dataset.py --stride N` in the
sibling `adt_egocentric` repo (`indices = [i for i in range(0, n_real, stride)
if first <= i <= last]`). Somebody ran it at `--stride 7` for five sequences and
`--stride 1` for one.

**Cost of re-extracting at stride 1**, measured from file sizes and mtimes of
the extractions already on disk:

| | seq131 `videos_synthetic` (stride 1) | seq132 `videos_synthetic` (stride 7) |
|---|---:|---:|
| frames | 2878 | 400 |
| MB / frame | 1.956 | 1.340 |
| total | 5.63 GB | 0.54 GB |
| wall-clock span of the write | 187 s | 161 s |
| s / frame | 0.065 | 0.404 |

Note the second column: a *strided* extraction took almost as long in wall-clock
as a full one, because the VRS is decoded frame by frame either way. Re-running
the five strided sequences at stride 1 therefore costs roughly **3 minutes and
5.6 GB each — about 15 minutes and 28 GB in total.**

`df` on `lambda_63`: `/` (where ADT lives) is **94 % full, 214 GB free**. 28 GB
fits. Extending to the thirteen unextracted sequences would not: those need
synthetic (5.6 GB) *and* `depth_npy` (11.6 GB at 3.965 MB/frame) per sequence,
≈17.2 GB × 13 ≈ **224 GB, which does not fit in 214 GB.**

### 1.3 The bedroom segment — the loud part

`Apartment_release_decoration_seq132_M1292/videos_rgb/room_annotations.csv`
has 4168 rows: 502 `bedroom` (ids **1745–2246**) and 3666 `unknown`.
`unknown` means *unannotated*, not *not-bedroom*.

| stream | bedroom frames present |
|---|---:|
| `videos_rgb` | **502 / 502** |
| `depth_npy` | **502 / 502** |
| `videos_synthetic` | **74 / 502** |
| `videos_synthetic ∩ depth_npy` | **74 / 502** |

**As the data stands today, a synthetic-RGB bedroom test set is 74 frames.**
That is the direct consequence of the stride-7 extraction: 502 / 7 ≈ 74. It is
not a container limit and it goes away with §1.2's 3-minute re-extraction — but
until that is run, the split below has a 74-frame test arm, and 74 frames of one
room in one apartment supports very little.

There is a second labelled segment, not previously written down:
`Apartment_release_clean_seq131_M1292/videos_rgb/room_annotations.json` is
interval-coded and labels ids **0–782 `kitchen`** (783 frames). Only 313–782 were
ever extracted, so it yields **470 frames with RGB, synthetic and depth all
present** — because seq131 is the stride-1 sequence. seq131 is a *training*
sequence under the current convention.

### 1.4 The split

Held out, following the existing `autoresearch` convention:
`Apartment_release_clean_seq136_M1292` and
`Apartment_release_decoration_seq132_M1292`.

Stream: **`videos_synthetic`**, per the caller's instruction that real ADT RGB
carries hands with no ground-truth depth in the twin, which would poison the
metric. *(That rationale is asserted upstream and is not verified anywhere in
this repo — see §4.)* Ground truth: `depth_npy`, **planar z** (ticket 016).
Frame key: the ADT frame id, intersected across `videos_synthetic` and
`depth_npy`.

**Split A — runnable today, no re-extraction:**

| | sequences | frames |
|---|---|---:|
| train | seq131 (2878) + seq133 (400) + seq134 (400) + seq135 (400) | **4078** |
| test — sequence arm | seq136, all synthetic∩depth frames | **400** |
| test — room arm | seq132 ids 1745–2246 ∩ synthetic ∩ depth | **74** |

Split A is badly unbalanced: 71 % of the training frames come from seq131 alone,
because seq131 is the only sequence extracted at stride 1. Any per-sequence
effect will be dominated by that one take.

**Split B — after the 15-minute stride-1 re-extraction of seq132–136 (recommended):**

| | sequences | frames |
|---|---|---:|
| train | seq131 + seq133 + seq134 + seq135 | **≈ 11 400** (projected) |
| test — sequence arm | seq136 | **≈ 2879** (projected) |
| test — room arm | seq132 ids 1745–2246 | **502** (measured — rgb and depth are both already 502/502) |

The 502 is measured. The other two rows are **projections**, not measurements:
they assume a stride-1 extraction fills the same VRS index window the stride-7
one spans (seq133 275–3116, seq134 255–3068, seq135 478–3355, seq136 506–3384),
which is what the matching `videos_rgb` counts (2844 / 2815 / 2880 / 2879) imply
but which nothing has yet produced. Re-run `--adt` after the extraction and
replace them.

### 1.5 What this split does and does not buy

* It is **cross-sequence**, exactly as the reviewer said. One apartment, one
  device (M1292), for both arms. Nothing here changes that.
* The room arm is **not** a held-out room. seq131/133/134/135 are unannotated
  and the wearer walks the whole apartment in each of them, so the bedroom is
  almost certainly in the training frames — just not labelled. Calling the
  bedroom arm "unseen room" would be false. It is a *labelled-room* arm, useful
  for reporting error broken down by room, not for claiming room-level
  generalisation. Annotating seq131/133/134/135 with the same
  `annotate_rooms.py` and excluding bedroom frames from training is the ticket
  that would make the claim true; it has not been written.
* seq132 is `decoration`, a different object layout from the four `clean`
  training sequences. That is a real, if small, distribution shift, and it is
  a property of the sequence, not of the room label.

---

## 2. ScanNet++

### 2.1 What is actually on the box — far more than the 5 DAC scenes

`/netapp/datasets/f.zhang2/scannetpp/data` holds **1018 scene directories**.

| | scenes |
|---|---:|
| on disk | 1018 |
| with `dslr/nerfstudio/transforms.json` | 1006 |
| with `scans/mesh_aligned_0.05.ply` | 956 |
| with `dslr/render_depth/` | **1** (`3f15a9266d`, 906 frames) |
| **complete** = images + transforms + mesh | **956** |

**956 complete scenes, 1 032 862 DSLR frames** (min 121, median 964, max 5448
per scene). All 956 are `OPENCV_FISHEYE` at 1752×1168. `metadata/scene_types.json`
labels them across ~34 room types — 238 apartment, 201 office, 161 bedroom/hotel,
79 bathroom, 70 conference room, 56 kitchen, 48 classroom, … This is the actual
answer to the external-validity gap: it is many buildings, not one apartment.

The vendored DAC splits (`third_party/depth_any_camera/splits/scannetpp/`, not
in the Mac checkout) name **41 scenes**: 32 train, 4 val, 5 test, and 2
"test_easy" which are a subset of test. All 41 are present; **4 of the 32 DAC
train scenes have no mesh** (`03f7a0e617`, `1c876c250f`, `4bc04e0cde`,
`5a14f9da39`), so DAC's own train split cannot be fully re-rendered here. The
5 test scenes total exactly 2725 DSLR frames, which matches
`scannetpp_tiny_test.txt`'s 2725 lines — DAC scores every frame of them.

### 2.2 The split

Hold out all **41** DAC scenes, not just the 5 test ones — using DAC's val or
train scenes for our training and its test scenes for evaluation still leaves
the DAC-protocol comparison clean, but excluding all 41 costs almost nothing
and removes the question entirely.

| | scenes | DSLR frames |
|---|---:|---:|
| available pool (956 complete − 41 DAC) | **919** | **1 015 015** |
| DAC test (evaluation, unchanged protocol) | 5 | 2 725 |
| DAC val | 4 | 2 585 |

The pool covers 34 scene types. `tools/dataset_scope_2026_08.py` selects from it
round-robin over scene type, so a subsample stays diverse rather than becoming
221 apartments.

### 2.3 The render cost, quantified

Dense GT depth is not shipped; it is rendered from the laser mesh by the
official toolbox. **The path is complete and already proven on this box:**

* `renderpy.cpython-311-x86_64-linux-gnu.so` at `/user/f.zhang2/opt/renderpy/`,
  imports in the py3.11 `raytun3r` env with only `PYTHONPATH` set;
* toolbox at `/user/f.zhang2/projects/scannetpp/`, with `common/render.py` and a
  working config `common/configs/render_3f15.yml` (`data_root`, `render_dslr:
  True`, `scene_ids`, `near: 0.05`, `far: 20.0`, `output_dir`);
* it produced `3f15a9266d/dslr/render_depth` — 906 files, `uint16`, 1168×1752,
  **planar z in millimetres** (`depth * 1000` clipped to 16 bit in `render.py`),
  3.5 % zero pixels (mesh holes / beyond `far`).

Measured from that run:

| | measured |
|---|---:|
| wall clock | 906 frames in 184.5 s → **0.204 s / frame** |
| `render_depth` on disk | 417 MB → 0.476 MB / frame |
| `render_rgb` on disk (written too) | 156 MB → 0.172 MB / frame |
| **total** | **0.632 MB / frame** |

Extrapolating (single process, one GPU):

| option | scenes | frames | render time | disk |
|---|---:|---:|---:|---:|
| whole pool, every frame | 919 | 1 015 015 | **57.5 h** | **642 GB** |
| 300 scenes, every frame | 300 | 321 012 | 18.2 h | 203 GB |
| 300 scenes, stride 4 | 300 | 80 369 | **4.6 h** | 51 GB |
| 150 scenes, stride 4 | 150 | 40 978 | 2.3 h | 26 GB |
| 120 scenes, stride 6 | 120 | 22 005 | 1.2 h | 14 GB |
| 60 scenes, stride 4 | 60 | 16 591 | 0.9 h | 10 GB |
| DAC test (5 scenes, needed for the protocol) | 5 | 2 725 | 0.15 h | 1.7 GB |

`/netapp` has **6.4 TB free**, so disk is not the constraint; wall clock is.
**Recommended: 300 scenes at stride 4 → 80 369 training frames for 4.6 h of
render**, plus the 5 DAC test scenes (0.15 h). Stride 4 matters — consecutive
DSLR stills within a scene are near-duplicates in the same way consecutive ADT
frames are, and `fovbench/split.py` already argues that case. If 80 k turns out
to be too few, the same command extended to stride 1 on those 300 scenes adds
240 k more for another 13.6 h without changing the scene list.

---

## 3. ScanNet++ → Aria: is there public code to re-map the lens?

### 3.1 The two camera models, measured

| | ScanNet++ DSLR | Aria RGB (ADT) |
|---|---|---|
| model | `OPENCV_FISHEYE` = Kannala–Brandt 4 | **FISHEYE624** (KB + 2 tangential + 4 thin-prism, 16 params) |
| calibration | **per scene**, in `dslr/nerfstudio/transforms.json` | per device; repo's calibration of record in `finetune/aria_calibration.py`, exposed by `finetune/eval/baselines/aria_fisheye.py` |
| frame | 1752×1168, all 956 complete scenes | square (1408×1408 native, 512/896 binned) |
| image circle | **covers the sensor** — Ω is the whole rectangle, corners carry real content | **inscribed** — Ω is a disc, the corners are dark |

Field of view over all **956** complete scenes (KB4 inverted numerically,
`--scannetpp`):

| | min | p5 | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| diagonal | 165.40° | 169.06° | **169.83°** | 174.59° | 175.85° |
| horizontal | 133.90° | 134.80° | **146.80°** | 147.26° | 147.84° |
| **vertical** | **85.12°** | 86.29° | **103.39°** | 104.19° | 105.16° |

This confirms `docs/research/scannetpp-camera-reference.md` at scale, on 956
scenes rather than 2, and confirms that the RayTun3R paper's 115° is not the
fisheye set (the undistorted `PINHOLE` set measures 132.34° / 124.10° / 102.91°
diagonal / horizontal / vertical on `3f15a9266d`, 118.67° / 109.05° / 86.16° on
`1f7cbbdde1` — the 115° falls in that family's range, not the fisheye's).
No scene's corner reaches its KB4 turnover, so the polynomial is monotone over
the whole frame everywhere.

Aria, from the repo's own calibration (`--coverage`, 1408×1408):

```
fx=fy=610.94  cx=690.29  cy=715.11   k=(0.3852, -0.4442, 0.5591, -0.3254)
KB4 turnover      62.33° half-angle
usable theta_max  54.83° half-angle  (109.65° cone)
pixels inside the cone: 1 496 970 / 1 982 464  (75.5 %)
```

### 3.2 FOV coverage — which direction loses pixels, and where the void is

Both directions lose something, and they lose it differently.

**ScanNet++ → Aria loses source pixels.** Aria's 54.83° cone retains
**58.90 %** of a ScanNet++ frame's pixels (`3f15a9266d`, `src kept` column); the
outer 41 %, out to 84.8° incidence, has nowhere to go.

**Aria → ScanNet++ leaves a void, in every single scene.** Fraction of Aria's
imaged cone that lands inside a ScanNet++ frame, over all 1006 scenes with
intrinsics:

```
covered:  min 86.92 %   p5 87.64 %   median 98.01 %   p95 98.14 %   max 98.27 %
void:                                median  1.99 %                max 13.08 %
scenes with zero void: 0 / 1006
```

**Where the void sits is the problem.** For the reference scene the void is
**29 779 of 1 496 970** cone pixels (1.99 %), all of them at incidence
**50.98°–54.83°** — the outermost annulus — and all at azimuth ±60°…±120°, i.e.
the **top and bottom** of Aria's disc. The cause is arithmetic: ScanNet++'s sensor is 3:2 landscape, so
its vertical half-FOV is 51.7° (median) against Aria's 54.83° requirement. As a
share of the rim band (θ ≥ ⅔·θ_max, 60.6 % of the cone):

```
void as % of the rim band:  min 2.86 %  p5 3.07 %  median 3.28 %  p95 20.41 %  max 21.58 %
```

That distribution is bimodal: ~5 % of scenes have a vertical half-FOV near 43°
and lose a fifth of the rim. **The void lands precisely in the band this project
measures.** A remapped ScanNet++ frame whose void is filled — with black, with
ImageNet mean, or with `BORDER_REPLICATE` — hands the rim experiment a fabricated
signal in the only region the experiment reads.

Rotating the source frame to portrait moves the void from top/bottom to
left/right; it does not remove it. Using the shipped `resized_undistorted_images`
instead of the fisheye set does not help either: the pinhole set's vertical FOV
is 102.91° against the fisheye's 103.89° on the same scene — the void is set by
the sensor's aspect ratio, not by which image set you resample from.

The only void-free options are to shrink the target cone (a per-scene cap at the
source's vertical half-FOV: 51.69° median reaches 88.4 % of Aria's cone pixels;
the p5 scene's 43.14° reaches only 58.1 %), or to carry a validity mask
everywhere and exclude void pixels from Ω. **Carry the mask.** That is what
`slambench/baselines.py::RectDerectBaseline._remap` already does — `np.where(ok,
u, -1)` plus an explicit `in_cone` array, `cv2.BORDER_CONSTANT` with the comment
"a replicated pixel is a fabricated observation" — and what
`test_the_run_settings_leave_no_void_for_the_sampler_to_average_over` exists to
enforce. That test passes today only because a 110° pinhole is *fully* backed by
the 896 Aria fisheye. It would fail on this remap, and it should.

### 3.3 What code exists

**(a) In-repo, and it already does exactly this operation.**
`RectDerectBaseline` (`slambench/baselines.py:204–320`) is a complete
camera-to-camera resampler through ray space: target grid → unit rays
(`Pinhole.rays`) → source pixels (`Fisheye624.project`) → `cv2.remap` with an
explicit validity mask, plus `derectify` for the return trip. Only the target
model is hard-wired to a pinhole; swapping in another `project`/`unproject` pair
is the whole change. The pieces for the other pair are already there:
`slambench/camera.py::Fisheye624` (exact 16-parameter model, with
`project_bulk` / `unproject_bulk` / `theta_of` / `max_imaged_theta`),
`slambench/vggt360.py::Fisheye624Lens` (`ray_lut()`, `cos_theta()`, `maps_for`,
`theta_max`), and — most directly — `raytun3r/cameras.py`, which already
carries **both** constructors: `from_scannetpp(intrinsics, w, h)` reads the
`transforms.json` KB4 entry, and `from_aria(h, w)` builds the Aria side, both
into the same `KannalaBrandt` class with a shared `_default_theta_max` that
knows the full-frame-vs-inscribed distinction. `raytun3r/cameras.py::convert_depth`
does planar-z ↔ euclidean-range per pixel.

**(b) The vendored DAC (`third_party/depth_any_camera`, CVPR '25).** It is
complete on the box (`dac/` package present, `dac/utils/erp_geometry.py`) and its
ERP path is genuinely a general cross-camera converter:

* `cam_to_erp_patch_fast(...)` — source camera → ERP patch by explicit gnomonic
  projection. Supports `OPENCV_FISHEYE` (with an explicitly FOV≥180-safe branch),
  `MEI`, and pinhole. **It has no FISHEYE624 forward model**, so it cannot render
  *into* Aria's exact lens.
* `erp_patch_to_cam_fast(..., fisheye_grid2ray=...)` — ERP → target camera, and
  for `kitti360`/`scannetpp`/`zipnerf` it takes the target's rays from a
  **per-pixel LUT with a NaN validity channel** rather than a closed-form model.
  That LUT is the general hook: a FISHEYE624 grid built from
  `Fisheye624.unproject_bulk` drops straight in, and DAC will then resample into
  Aria's exact lens without a line of DAC being changed.
* Both hops use `F.grid_sample(..., padding_mode='border')` and then multiply by
  `mask_active`, so fabricated border pixels are zeroed rather than kept — but
  the mask is a separate return value and it is on the caller to carry it.
  DAC's own `padding_rgb=[123.675, 116.28, 103.53]` (ImageNet mean) fill is
  present but commented out in `erp_geometry.py`; `scannetpp_erp.py` then calls
  `resize_for_input(..., padding_rgb=[0,0,0], mask=erp_mask)`.
* **Depth convention: DAC works in euclidean range, we work in planar z.**
  `dac/dataloders/scannetpp_erp.py` line 143: *"convert depth from zbuffer to
  euclid (critical for fisheye dataset to use euclid depth)"* — it does
  `depth = depth / fisheye_grid_z` on load and never converts back.

**(c) Other public options.** `projectaria_tools.core.calibration.distort_by_calibration`
maps between two `CameraCalibration` objects and does support FISHEYE624 — this
repo already constructs one that way in `_library_rectify`. Whether it accepts a
KB4 *source* was not tested here. OpenCV's `cv2.fisheye.initUndistortRectifyMap`
only goes fisheye→pinhole, so it covers half the trip. There is no maintained
public "ScanNet++-to-Aria" tool; every route is an assembly of a projection and
an unprojection.

### 3.4 Recommendation

**Do a single one-hop remap in ray space, in-repo. Do not route through ERP.**

```
Aria pixel (u,v)  --unproject_bulk-->  unit ray  --project-->  ScanNet++ pixel
   (FISHEYE624, exact)                              (KB4, per scene)
```

then one `cv2.remap` per frame with `BORDER_CONSTANT` and the `ok` mask, exactly
the shape of `RectDerectBaseline._remap`. Cost: one LUT per (scene, output size),
built once and cached — per-scene calibration means 956 of them, ~1–2 MB each at
a 352–518 px target. Reasons to prefer it over DAC's ERP path: one resampling
instead of two (ERP costs a second interpolation and a second round of blur),
the exact FISHEYE624 rather than a KB4 fit of it, no euclid/planar round trip,
and code this repo already has tests for.

Use DAC's ERP path **only** if the goal is to reproduce DAC's protocol itself. It
works — via the `fisheye_grid2ray` hook — but it buys nothing here.

**Failure modes, in order of how quietly they bite:**

1. **The void.** §3.2. Every scene leaves one, median 3.3 % / worst 21.6 % of the
   rim band, at the top and bottom of Aria's disc, at 51–55° incidence. Carry the
   mask into Ω. Never fill. Add a `--coverage`-style assertion to whatever builds
   the split, so a scene whose rim void exceeds a threshold is dropped rather
   than trained on.
2. **Depth convention.** ScanNet++ `render_depth` is **planar z** in mm
   (`render.py` writes the z-buffer × 1000). This project is planar z (ticket
   016). DAC is euclidean range. Whether the remapped depth needs converting
   depends on one thing only: *does the target camera share the source's optical
   axis?*
   * **Pure lens re-parameterisation, no rotation** (the recommended route): the
     world point's ray direction is unchanged, so θ is unchanged, so planar z is
     **invariant** — resample the values with `INTER_NEAREST` and change nothing.
     Assert this rather than assume it: remap an all-ones z map and check it comes
     back all-ones inside the mask.
   * **Any rotation** — a tilt or azimuth (as in `vggt360`'s ring), a roll, or
     DAC's `use_pitch` φ taken from the pose — changes θ per pixel and the values
     **must** be converted: `range = z / cosθ_src`, resample, `z' = range · cosθ_dst`.
     `raytun3r.cameras.convert_depth(..., src="z", dst="range")` is that function.
     Getting this wrong is a smooth radial error that scale alignment cannot
     absorb (a single scalar cannot undo a per-pixel `1/cosθ`), so it reads as
     "the model is worse at the rim" — which is the exact conclusion this project
     is trying to measure. This is the classic silent bug and it is one
     line away in either direction.
   * If you do go through DAC, its ERP tensors are **euclidean range**; convert
     back before anything in this repo touches them.
3. **Interpolation.** RGB bilinear, depth **nearest**. Bilinear depth across a
   mesh-hole boundary or a depth discontinuity invents surfaces. DAC already
   does nearest for depth and bilinear for RGB; match it.
4. **Per-scene calibration.** `fl_x, fl_y, cx, cy, k1..k4` differ per scene
   (median diagonal 169.8° but range 165.4°–175.9°). One cached LUT per scene,
   never a repo-wide constant. `scannetpp-camera-reference.md` says the same.
5. **Ω is the whole rectangle on ScanNet++ and a disc on Aria.** Copying either
   rule to the other side is a 47 %-of-frame error in one direction and a
   fabricated-corner error in the other. `_default_theta_max` already encodes the
   distinction; do not re-derive it.
6. **Resolution.** Source 1752×1168, target square. The remap resamples a 3:2
   frame into a disc; effective sampling density is not uniform and the rim of
   the target reads a *sparser* part of the source than the centre does. That is
   a real, unquantified confound for a rim-vs-centre experiment, and it is
   separate from the void.

**The alternative worth naming:** do not warp the pixels at all. Feed the model
the source frame with its own ray grid — which is what `raytun3r` is built to do
— and the void, the resampling blur and the density confound all disappear,
because nothing is resampled. That does not answer "how does a model trained on
Aria behave on ScanNet++", but it does answer "how does FOV affect depth", which
is the actual research question.

---

## 4. What is **not** established

* **That real ADT RGB lacks GT depth on hands.** This is the stated reason for
  the synthetic-only convention and it is repeated here as given. Nothing in this
  repo verifies it; `fovbench/split.py` deliberately scores *both* streams and its
  docstring gives a different reason for doing so ("quoting only the synthetic
  stream would overstate every model"). Someone should check a hand-containing
  frame against `depth_npy` before this is written down as fact.
* **Split B's frame counts.** 502 bedroom frames is measured. The ≈11 400 train /
  ≈2879 test numbers are projections from the VRS index windows the stride-7
  extractions span. Re-run `--adt` after re-extracting and replace them.
* **That the bedroom arm is an unseen room.** It is not — see §1.5. No sequence
  other than seq131 and seq132 has room annotations at all, so nothing is known
  about what rooms the training frames contain.
* **That ScanNet++ training will transfer to Aria.** Nothing here trains
  anything. §3 establishes that a remap is geometrically possible and what it
  costs; it establishes nothing about whether a model trained on remapped
  ScanNet++ helps on ADT.
* **The 0.204 s/frame render rate at scale.** It is one scene, 906 frames, on
  one GPU on 2026-08-07, measured from mtimes. Both `lambda_63` GPUs were 55–69 %
  full at the time of writing and the box is shared; a 4.6 h render competing for
  a GPU is not a 4.6 h render.
* **Whether `projectaria_tools.distort_by_calibration` accepts a KB4 source.**
  Not tested. §3.3(c).
* **Whether DAC's `fisheye_grid2ray` hook actually accepts a FISHEYE624 LUT.**
  The code path takes an (H, W, 4) array of rays plus a NaN flag and never
  inspects the model that produced it, so it should — but it has not been run.
* **The 41 %/58.9 % source-retention figure** is one scene (`3f15a9266d`). The
  void figures are all 1006 scenes; that one is not.

---

## 5. Decisions needed

1. **Re-extract ADT synthetic at stride 1 for seq132–136?** ~15 min, ~28 GB, on
   a filesystem at 94 %. Without it the bedroom test arm is 74 frames and the
   train set is 71 % one sequence. This is a `gpu` ticket (it reads ADT).
2. **ScanNet++ render budget:** 300 scenes × stride 4 (4.6 h, 51 GB, 80 k
   frames) is the recommendation. Bigger, smaller, or not yet?
3. **Does the ScanNet++ arm remap to Aria at all, or does it train in its own
   lens with its own ray grid?** §3.4's last paragraph. The remap costs a void in
   the rim band; the no-remap route costs the ability to say "the same model on
   the same lens".
4. **Annotate rooms in seq131/133/134/135** so the bedroom arm can be a genuine
   held-out room? `annotate_rooms.py` exists. This would be a new ticket.
