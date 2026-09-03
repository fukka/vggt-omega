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

**Not verified, and here is why.** I tried to pull one sequence's trajectory and
ground truth to confirm the room, and every URL returned **403**: the signed
links in `ADT_download_urls.json` have expired (`oe=` in the query string is an
expiry stamp). Re-download the CDN file from
<https://www.projectaria.com/datasets/adt/> — it needs a browser and a licence
click, so it is a human step — and then one 12 MB fetch settles it.

If it holds, this is a **cross-room + cross-device** test with our existing
loader, our existing depth convention, and no new code. That is a stronger
external-validity probe than the within-sequence bedroom segment #40 fell back
to, and it was available the whole time.

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
