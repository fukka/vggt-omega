# Any-camera depth baselines: UniK3D & Depth-Any-Camera

Evaluate two "any-camera" metric monocular depth models on the **raw,
non-rectified Aria fisheye** stream — the setting they are designed for (no
fisheye→pinhole rectification):

| Model | Paper | How it handles fisheye | Weights |
|-------|-------|------------------------|---------|
| **UniK3D** | Piccinelli et al., CVPR 2025 | Native `Fisheye624` camera; predicts a metric 3D point map | `lpiccinelli/unik3d-{vits,vitb,vitl}` (auto-download) |
| **Depth-Any-Camera (DAC)** | Guo et al., CVPR 2025 | Warps fisheye→equirectangular (ERP) using the intrinsics, predicts there | `yuliangguo/depth-any-camera` (`download_weights.py`) |

* **ADT** (dense GT) → **quantitative** depth metrics.
* **EgoExo4D** → **qualitative** depth visualisations (the ego renders have no metric GT).
* **official** → each repo's *own* demo asset run end-to-end, as a sanity check that
  the model + weights + environment are correct, independent of the Aria pipeline.

Every saved prediction writes the full triple — **input RGB + metric depth
(`.npy` + colorized `.png`) + coloured point cloud (`.ply`)**.

Data loading mirrors [`finetune/eval/run_eval.py`](../run_eval.py) /
[`adt_depth.py`](../adt_depth.py): the ADT path reuses `ADTWindowDataset` with
`seq_len=1` and **`rectify=False`** — real `videos_rgb` Aria fisheye + `depth_npy`
GT, 270° CCW rotation, uint16 mm → metres.

The single shared fact that makes both work: the **Aria 214-1 KB4** coefficients
`[0.3852, -0.4442, 0.5591, -0.3254]` (from [`finetune/data/rectify.py`](../../data/rectify.py))
are the *same* radial polynomial as UniK3D's `Fisheye624` (k1..k4, rest 0) and
DAC's `OPENCV_FISHEYE` cam_params — see [`aria_fisheye.py`](aria_fisheye.py).

## Setup (GPU box)

These two baselines are **separate upstream repos**, not vendored here. The setup
script clones them into `third_party/` **inside this repo** (gitignored) and
installs them — `run_baselines` then finds them automatically, no flags or env
vars needed.

UniK3D needs **python ≥3.11, torch ≥2.4** (+ xformers/triton — fine on CUDA); DAC
needs **numpy<2** + an older torch. These conflict, so install each in its **own
env** by running the script once per env (it clones once, installs the one you
name):

```bash
# from your vggt-omega checkout, inside your UniK3D env:
bash finetune/eval/baselines/setup_baselines.sh unik3d   # clone both + pip install -e UniK3D

# inside your DAC env:
bash finetune/eval/baselines/setup_baselines.sh dac      # + pip install DAC reqs + download weights
```

Other targets: `clone` (just clone both, no pip), `both` (install both into the
current env — only if it happens to satisfy both). The script is re-runnable
(existing clones are pulled). To install elsewhere instead, pass
`--unik3d-root` / `--dac-root` per run, or set `$UNIK3D_ROOT` / `$DAC_ROOT`.

## Run

```bash
# ADT — quantitative (both models), non-rectified fisheye
python -m finetune.eval.baselines.run_baselines --mode adt \
    --adt-root /group-volume/Fengjia/data/projectaria_tools_adt_data_clean \
    --dac-config checkpoints/dac_swinl_indoor.json \
    --dac-weights checkpoints/dac_swinl_indoor.pt \
    --adt-max-frames 100 --out eval_out/baselines

# EgoExo4D — qualitative
python -m finetune.eval.baselines.run_baselines --mode egoexo \
    --egoexo-root EgoX/example/egoexo4D \
    --dac-config checkpoints/dac_swinl_indoor.json \
    --dac-weights checkpoints/dac_swinl_indoor.pt

# official-example sanity check (each repo's own demo asset)
python -m finetune.eval.baselines.run_baselines --mode official \
    --dac-config checkpoints/dac_swinl_indoor.json --dac-weights checkpoints/dac_swinl_indoor.pt
#   UniK3D -> assets/demo/scannet.jpg (+ scannet.json camera)
#   DAC    -> demo/input/scannetpp_sample.json
#   --mode all  runs official + adt + egoexo

# one model at a time (recommended given the env split)
python -m finetune.eval.baselines.run_baselines --mode all --models unik3d ...
python -m finetune.eval.baselines.run_baselines --mode all --models dac ...
```

Useful flags: `--unik3d-backbone {vits,vitb,vitl}`, `--no-unik3d-camera`
(withhold intrinsics → UniK3D predicts rays itself), `--image-resolution`,
`--n-frames` (EgoExo frames/clip), `--n-qual` (ADT panels), `--device`,
`--adt-rgb-subdir`.

**ADT fisheye RGB source** (`--adt-rgb-subdir`, both non-rectified):
* `videos_synthetic` *(default)* — rendered fisheye, pixel-aligned with the
  `depth_npy` GT → cleanest quantitative numbers (= `run_eval --blender-norectify`).
* `videos_rgb` — the real Aria sensor stream; slightly mis-registered to the GT
  (= `run_eval --real-norectify`). Use it for the true real-image domain.

## Output

`--out` is **auto-generated** from the active models and their key settings if not
given explicitly, e.g.:

```
eval_out/unik3d-vitl_dac-swinl_indoor/   # both models, default variant
eval_out/unik3d-vitl/                     # unik3d only
eval_out/dac-swinl_indoor/               # dac only
eval_out/unik3d-vitl-nocam_dac-swinl_indoor/  # unik3d without intrinsics
```

Contents:

```
adt_results.json          {fisheye_planar, erp_range} → variant → mode → metrics
adt_summary.txt           the two comparison tables
qual/{unik3d,dac}/        ADT samples — <i>_input.png, _depth.npy, _depth.png, _pcd.ply
egoexo/{unik3d,dac}/      EgoExo samples — <clip>_f####_{input.png,depth.npy,depth.png,pcd.ply}
official/unik3d/          sanity-check on kitti360 (MEI fisheye, 933×933):
  kitti360_input.png          the demo image the repo ships
  kitti360_depth.{npy,png}    our prediction (planar z)
  kitti360_pcd.ply            our point cloud
  (UniK3D only ships scannet.npy as reference; no npy for kitti360)
official/dac/             sanity-check on scannetpp (OPENCV_FISHEYE — same family as Aria KB4):
  scannetpp_input.png           the demo image
  scannetpp_depth.{npy,png}     our prediction (ERP, euclidean range)
  scannetpp_pcd.ply             our point cloud
  scannetpp_gt_depth.{npy,png}  GT depth (repo's scannetpp_depth.png, mm→m)
  scannetpp_official_output.jpg repo's own demo output visualization
```

**Why these demo assets?** — camera-type audit of all shipped demo assets:

| Asset | Camera model | Type |
|-------|-------------|------|
| UniK3D `kitti360` | MEI | **fisheye** ← default |
| UniK3D `equirectangular` | Spherical | 360° |
| UniK3D `dl3dv` | OPENCV | perspective |
| UniK3D `scannet` | Fisheye624, k≈0 | *effectively perspective* (ScanNet uses pinhole-class lenses; near-zero distortion coefficients) |
| DAC `scannetpp` | OPENCV_FISHEYE | **fisheye** ← default (same distortion family as Aria KB4) |
| DAC `kitti360` | MEI | fisheye (GT depth + official output available; pass `--dac-official-sample demo/input/kitti360_sample.json`) |
| DAC `matterport3d` | ERP | 360° |
| DAC `nyu`, `kitti` | PINHOLE | perspective |

UniK3D outputs are in the **fisheye image frame** (depth = planar z, point cloud
from its 3D point map); DAC outputs are in **ERP** (depth = euclidean range, point
cloud from `reconstruct_pcd_erp`). Depth `.npy` is the model's raw metric output
(unaligned). PLYs are binary little-endian, viewable in MeshLab/CloudCompare.

## Two ADT metric domains (and why)

The models predict in different native domains, so a single image-plane number
would unfairly penalise one. We report both:

1. **Fisheye-frame, planar depth** — UniK3D's native output vs GT directly. Same
   convention as the VGGT/DAv2 ADT eval, so these numbers are comparable to your
   existing results.
2. **ERP, euclidean range (head-to-head)** — both models scored on the identical
   ERP grid DAC predicts on (UniK3D's depth is converted to range and warped there
   via DAC's own `cam_to_erp_patch_fast`). This is the fair common-domain
   comparison.

Both use alignment modes `none` (metric, absolute scale) and `scale_shift`.

**Cone mask (ERP domain).** `crop_wFoV=180` makes the ERP patch span ±90°
longitude, but the Aria lens only images ±62° (the KB4 forward polynomial turns
over there). `cam_to_erp_patch_fast` applies the raw polynomial, which past the
turnover *folds back* and samples wrong, in-cone source pixels — ghosting that the
`active`/`valid` masks miss (~40% of the patch for Aria). `fisheye_to_erp_fwd`
therefore zeroes `active` beyond the per-camera turnover (`kb4_max_incidence`), so
GT, UniK3D and DAC are all scored only inside the physically imaged cone. It
auto-adapts per camera (Aria → 62°; ScanNet++ → ~101°, effectively no-op).

**GT convention.** ADT `depth_npy` is treated as **planar z** (matching the
existing VGGT eval) and converted to euclidean range for the ERP domain via the
Aria KB4 ray grid. If your ADT depth is actually along-ray distance, pass
`--gt-euclidean` (the conversion is then inverted).

## What is validated vs. GPU-only

Validated locally (CPU, no weights): the Aria intrinsics + 270° rotation, the
KB4 ray grid and z↔range round-trip, and DAC's fisheye→ERP warp + `resize_for_input`
on real EgoExo fisheye frames (geometrically correct ERP unwrapping). The model
**forward passes** (UniK3D needs py3.11/torch2.4; DAC needs its weights) run on
your GPU box — everything feeding them is exercised.
