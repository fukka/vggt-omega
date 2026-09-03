# Aria datasets with depth, in a scene that is not our apartment

Written 2026-09-02, prompted by the overfitting check: every H14/H15 number is
trained and tested inside one apartment on one device, and `seq136` was measured
to behave almost exactly like a training sequence (labelled arm −86.6% in
sample, −82.0% there, against a 2.6× collapse on the redecoration). We need a
different scene, on an Aria camera, with depth.

## 1. The correction: ADT has a second environment, and we have been treating it as junk

`research-state.yaml` carries, from the #40 replan:

> ADT has NO non-Apartment scene (184 Apartment + 52 Lite object-recog)

The count is right and the conclusion is probably wrong. The official ADT
documentation says the dataset is **284 apartment + 52 office** sequences —
"the office scene, a single room with minimal office furniture" — and the 52
non-`Apartment_*` entries in our own `ADT_download_urls.json` are exactly 52.
They are also on a **different device**:

| | sequences | device | `depth` | `segmentation` | `synthetic` | `main_groundtruth` |
|---|---|---|---|---|---|---|
| `Apartment_release_*` | 284 | `M1292` | 3.92 GB | 299 MB | 962 MB | 20.1 MB |
| `Lite_release_recognition_*` | 52 | **`61283`** | **2.68 GB** | 111 MB | 708 MB | 1.88 MB |

So they carry the **full ADT ground-truth stack, dense depth included**, and the
much smaller `main_groundtruth` is consistent with "a single room with minimal
furniture" rather than a four-room apartment.

**VERIFIED, 2026-09-03.** With a fresh CDN file the metadata pulls in seconds:

```json
{ "scene": "LiteOffice", "serial": "1WM09380061283", "dataset_name": "ADT_2023" }
```

and its 27 object instances are `LiteOffice_Table`, `LiteOffice_Desk_A`,
`LiteOffice_Sofa_Pleather`, `LiteOffice_Door`, `LiteOffice_Window_Shade_A`,
`LiteOffice_Wall_A`, … — an office, not an apartment. **`research-state.yaml`'s
"ADT has NO non-Apartment scene" is wrong**, and #40 was pushed down to a
within-sequence bedroom segment for no reason.

What LiteOffice is and is not:

* ✅ **a different room**, with ADT's full dense-depth ground truth;
* ✅ **a different device** — serial `…061283` against the Apartment's `…011944`;
* ⚠️ **almost no camera motion**: the wearer's trajectory over 2783 frames spans
  1.24 m × 0.19 m × 0.61 m. These are object-recognition clips, someone looking
  at one object. That is fine for "does the method transfer to another room" and
  bad for anything needing baseline — H9's parallax anchors will starve here.
* ⚠️ **27 instances against the Apartment's hundreds**, so it is a sparser scene
  as well as a different one.

### The calibration is genuinely different, and it matters

Both devices are Aria Gen 1 RGB, but the units are not interchangeable:

| | focal @504 | k1 | k2 | k3 | k4 |
|---|---|---|---|---|---|
| Apartment `M1292` | 218.689 | 0.4147 | −0.6293 | 0.9011 | −0.5185 |
| LiteOffice `61283` | 212.834 | 0.4243 | −0.6693 | 0.9713 | −0.5575 |

(both KB4-fitted from each device's own FISHEYE624 by `tools/adt_camera.py`)

**Focal differs by −2.68%.** Scoring LiteOffice frames through the Apartment
camera would carry a **6.4 px radial error at the rim** at 504 px — 0.56 px of
it from the lens shape and the rest from the focal. In an experiment whose
entire subject is radial behaviour, that is not a rounding error; it is the
smooth radial error that reads as "the model is bad at the rim". So the camera
must come from the sequence, which is what `camera.json` and
`tools/adt_camera.py` are for.

### And a check that came out clean

Refitting the **Apartment's** own FISHEYE624 gives k = (0.4147, −0.6293,
0.9011, −0.5185) against the repo's constant (0.3852, −0.4442, 0.5591,
−0.3254). The coefficients look different, so the fits were compared where it
matters — in pixels, against the true lens, over the whole cone at 504 px:

| | max error | rms | at 54.8° |
|---|---|---|---|
| repo `_ARIA_KB4` | **0.226 px** | 0.102 px | +0.172 px |
| refit here | 0.118 px | 0.018 px | −0.118 px |

The repo's constant is a quarter-pixel approximation of the real lens across the
entire imaged cone. **No existing result needs revisiting.** The two coefficient
sets are different parameterisations of nearly the same curve, and the
cross-device gap above is a real difference while this one is not.

The extractor was also validated the only way that settles it: it re-derives
frames from the raw VRS and compares them byte-for-byte against the existing
extraction — 8 frames, 0 mismatched — and the calibration it reads from the
provider (`f=610.941, cx=715.115, cy=716.715` at 1408) matches the repo's
hard-coded constants to three decimals.

## 2. The strongest answer: Aria Synthetic Environments (ASE)

**100,000 procedurally generated multi-room interiors**, each ~2 minutes,
rendered through the simulated Aria RGB camera. Per sequence: RGB (JPEG),
**depth (16-bit PNG, millimetres)**, instance segmentation, 6DoF trajectory,
semi-dense points, and a 3D floor plan.

| | |
|---|---|
| scenes | 100,000 — the cross-scene axis this project has never had |
| camera | simulated Aria RGB, **fisheye** |
| depth | 16-bit PNG, mm, **along the pixel's ray** |
| rate | 10 FPS |
| total | ~23 TB, but the downloader takes `--scene-ids 0-9`, so ~2.3 GB buys ten scenes |
| access | sign-up + CDN file, same shape as ADT |

> ### ⚠️ The convention trap, stated before anyone writes a loader
> ASE's depth is **"the depth along the pixel's ray direction"** — that is
> **euclidean range**. ADT's `depth_npy` is **planar z**, measured by
> `VGGT-360-fisheye/checks/check_gt_depth_domain.py`. On the Aria rim the two
> differ by up to **2.15×**, radially, which no affine alignment can absorb.
> This is the exact bug class that invalidated #38 v1 and cost a four-row
> re-run. Any ASE loader declares `range` at its boundary, once, per
> `CONTEXT.md`.

Two more things to check before trusting a cross-lens claim: whether ASE ships a
per-sequence calibration or renders every scene with one nominal Aria lens
(if the latter, ASE varies the SCENE and not the lens, which is exactly the axis
we want here but is not an H15-style cross-lens test); and what the render
resolution is.

## 3. Already on `lambda_63`, and already used

`ego-synth 5B` gives four real egocentric datasets — `aea`, `nymeria`,
`egoexo4d`, `oxford` — with **semi-dense MPS SLAM points** projected into every
frame, on both raw fisheye and a rectified 110° pinhole. Many different scenes,
real Aria footage, no download needed. What it is not: dense depth. slambench
already scores on it, and `#22`/`#23` are built on it.

Use it for the cross-scene question when sparse is enough (it was enough for the
oracle-null work); it cannot replace dense GT for the zone tables.

## 4. Checked and rejected

| dataset | why not |
|---|---|
| **Aria Gen 2 Pilot (A2PD)** | its depth images are produced by **Foundation Stereo** — estimated, not ground truth. Scoring a depth model against another depth model's output is not an evaluation. Ground-truth annotations are listed as future work. |
| **HOT3D** | Aria + Quest3, 3.7M images, but the 3D ground truth is hand and object **poses** with object meshes. Scene depth is not shipped; only object-level depth could be rendered. `data_hot3d_5b` in ego-synth has 0 bytes of sparse depth. |
| **Nymeria / AEA / Ego-Exo4D directly** | already covered through ego-synth 5B, and their native releases carry MPS semi-dense points, not dense depth. |
| **Digital Twin Catalog** | object scans, not scenes. (`DTC_objects_ADT_download_urls.json` is already on the box.) |

## 5. Recommendation

1. **Re-download the ADT CDN file and pull two `Lite_release_recognition_*`
   sequences.** Cheapest possible cross-room + cross-device test, zero new code,
   and it also settles whether our own standing note is wrong.
2. **ASE `--scene-ids 0-9`.** ~2.3 GB, ten genuinely different interiors with
   dense depth on an Aria fisheye. This is the cross-scene axis the project has
   never had, and it is the direct answer to "is H14 just learning this
   apartment".
3. Keep ego-synth for anything sparse-sufficient.

Both (1) and (2) need a human to accept a licence in a browser before anything
can be fetched.

## Sources

- ADT: <https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset>, <https://www.projectaria.com/datasets/adt/>
- ASE: <https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset>, its `ase_data_format` and `ase_download_dataset` pages, and <https://huggingface.co/datasets/projectaria/aria-synthetic-environments>
- Aria Gen 2 Pilot: <https://arxiv.org/abs/2510.16134>, <https://www.projectaria.com/datasets/gen2pilot/>
- HOT3D: <https://www.projectaria.com/datasets/hot3D/>
- Local: `ADT_download_urls.json` on lambda_63, `docs/data/ego-synth-5b-sparse-depth.md`
