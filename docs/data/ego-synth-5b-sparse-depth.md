# ego-synth 5B — sparse SLAM depth GT

Semi-dense MPS SLAM points, projected into every frame of every clip, for four
egocentric Aria datasets. This is **new ground truth**: metric, occlusion-aware,
and available on *both* the raw fisheye and a rectified 110° pinhole of the same
frame. That last property is what makes it interesting here — it is the first GT
in this repo that can score `rect` and `fisheye` on the same rays without a
re-render.

It was produced for Wan-5B video training, not for depth evaluation, and
**everything has been resized to 896×896**. That single fact is behind most of
the gotchas below.

## Where it is

| | path |
|---|---|
| source of truth | space-container `/group-volume/ttaa/egolabel/ego-synth/data_<ds>_5b/` |
| storage service | `file://groups/SR-TORAIC-IVU/ttaa/egolabel/ego-synth` (`space storage download file …`) |
| lambda_63 | `/data/f.zhang2/ego-synth-5b/<ds>/<take>/` |

`<ds>` ∈ `aea`, `nymeria`, `egoexo4d`, `oxford`. There is a `data_combined_5b`
that symlinks all of them — ignore it. `data_hot3d_5b` exists but has **no**
sparse depth at all (0 bytes), so it is not part of this.

Sizes, `sparse_depth` only, measured 2026-08-11:

| dataset | takes with depth | clips | `sparse_depth` |
|---|---|---|---|
| `aea` | 143 | 9 278 | 23 GiB |
| `nymeria` | 254 | 8 113 | 57 GiB |
| `egoexo4d` | ~1 090 of 2 380 take dirs | 9 277 | 122 GiB |
| `oxford` | 124 | 10 475 | 18 GiB |
| | | **37 143** | **220 GiB** |

Adding the `rectified/` and `fisheye/` mp4s roughly triples that (~4 MB per clip
each). `latents/` and `text_embeds/` are Wan-only, ~840 MB per take, and are not
copied.

## A take

```
<take>/
  meta.json                    clip table, rectification, source geometry
  camera_poses.json            T_world_camera per frame, per clip
  rectified_valid_mask.png     896×896, the imaged cone of the pinhole render
  fisheye/<clip>.mp4           raw Aria RGB, upright, 896²
  rectified/<clip>.mp4         110° pinhole render, 896²
  sparse_depth/<clip>.npz      ← the GT
  gaze/, captions-*.json, latents/, text_embeds/    (not needed here)
```

A **clip** is 121 consecutive source frames; `<clip>` is its start index in the
source recording. `meta.json → clips[]` gives `source_frame_indices` and
`source_timestamps_ns` for each.

## The npz

**Every file carries its own `meta` key, and that key is the authority — read it
before trusting this document.**

Nine arrays. Four for each variant, plus `meta`:

| array | dtype | shape | |
|---|---|---|---|
| `<v>_frame` | uint16 | (N,) | frame index **within the clip**, 0…120 |
| `<v>_uvd` | float16 | (N, 3) | `u`, `v` in the 896² image; `d` in metres |
| `<v>_dist_std` | float16 | (N,) | MPS metric range 1σ, **metres** |
| `<v>_inv_dist_std` | float16 | (N,) | MPS inverse-distance 1σ, **1/metres** |

with `<v>` ∈ `fisheye`, `rectified`. One row is *one SLAM point seen in one
frame*. Typical file: ~600 k rows per variant, ~6 MB.

* `d` is **metric camera-frame Z in metres** — planar z, not range. Same
  convention the repo already audited for ScanNet++ (`3f15a92`, `3f8ded5`).
* Points are **occlusion-aware**: only points the SLAM system actually observed,
  taken from the nearest observation keyframe within 60 ms.
* Capped at 10 000 points per frame by a hybrid sampler (half by 32×32 image-tile
  round-robin for even coverage, half uniform from the remainder). The cap only
  binds on dense frames.
* **Unfiltered by design.** Both uncertainty columns are kept so you choose the
  cut. `inv_dist_std` is scale-invariant triangulation quality; the meta is
  explicit that it is *not* `1/dist_std` and not derivable from `dist_std` and
  depth.

## Gotchas

These are the ones that cost a run if missed.

1. **896, not 1408.** The source Aria frames are 1408². Any Aria calibration you
   bring in must be scaled by `896/1408 = 0.636364`. The `u`,`v` here are already
   in the 896 frame.

2. **`u`,`v` are float16, so pixel coordinates are quantised.** Above 512 the
   step is 0.5 px; between 256 and 512 it is 0.25 px. Round when you index, and
   do not expect sub-quarter-pixel agreement with anything you reproject
   yourself.

3. **`d` is float16 too** — ~0.05 % relative, so ≈5 mm at 10 m and ≈6 cm at the
   120 m cap. Cast to float32 before any metric. Fine for AbsRel/δ1; do not quote
   RMSE to millimetres.

4. **Planar Z, not range.** Do not compare it against a model emitting distance
   without converting.

5. **Frames are upright already.** `meta.args.pose_orientation = "upright"`, and
   the poses are rolled to match by `R_ROLL_CW90`. Do **not** apply the 270°-CCW
   rotation that `datasets.adt.ADTFisheyeFrames` applies to ADT.

6. **The rectified stream is a known pinhole and the fisheye one is not.**
   `meta.rectification` gives `fov_deg = 110.0`, `focal_px = 313.69297712`,
   render 896², principal point at the centre (448, 448). So for `rectified_*`:

   ```
   theta = atan(hypot(u - 448, v - 448) / 313.69297712)
   ```

   — 55° at the middle of an edge, 63.6° at a corner. Mask with
   `rectified_valid_mask.png`. **There is no fisheye camera model anywhere in
   this data.** Incidence angle on the `fisheye_*` points cannot be computed from
   these files alone; it needs the Aria calibration from the source VRS. Radius
   binning works on both; θ binning works only on `rectified` until that
   calibration is brought in. That distinction is exactly the one
   [`fovbench/README.md`](../../fovbench/README.md) already makes.

7. **The two variants are not row-aligned.** Their row counts differ (602 247 vs
   619 329 in the take inspected). Use each as its own point set; do not join
   them by index.

8. **20 fps content labelled at 24.** `meta.fps_note` — motion reads ~20 % fast
   if you assume 24. Irrelevant for per-frame depth, relevant for anything
   temporal.

9. **`egoexo4d` is less than half populated** — about 1 090 of 2 380 take
   directories have a non-empty `sparse_depth/`. Filter on that, not on the
   directory listing.

## Poses

`camera_poses.json` → `clips.<clip>.frames[i].T_world_camera`, 4×4 row-major,
camera→world, metres, in that recording's MPS gravity-aligned frame (per-take
origin — **not** comparable across takes). Quaternions elsewhere in the file are
xyzw. Each frame's pose is interpolated at its exact VRS capture timestamp.

## Reading it

```python
import json
import numpy as np

z = np.load(f"{take}/sparse_depth/{clip}.npz", allow_pickle=True)
print(json.loads(str(z["meta"][0])))          # the authority, per file

v = "rectified"                                # or "fisheye"
fr  = z[f"{v}_frame"].astype(np.int32)
uvd = z[f"{v}_uvd"].astype(np.float32)         # float16 -> float32 first
sig = z[f"{v}_inv_dist_std"].astype(np.float32)

m = (fr == 0) & (sig < 0.01)                   # frame 0, well-triangulated
u, v_, d = uvd[m].T
```

Scoring a prediction is then a gather, not a resize: the model's depth map is
896² and `u`,`v` index straight into it.

```python
pred_at_gt = pred[np.rint(v_).astype(int), np.rint(u).astype(int)]
```
