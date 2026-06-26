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

Clone both repos somewhere visible (default location: `<ADT>/third_party/`):

```bash
cd <ADT>/third_party
git clone https://github.com/lpiccinelli-eth/UniK3D.git
git clone https://github.com/yuliangguo/depth_any_camera.git
```

UniK3D needs **python ≥3.11, torch ≥2.4** (+ xformers/triton — fine on CUDA);
DAC runs on an older stack (`numpy<2`). They have conflicting requirements, so
use **two environments**, or run the two models in separate passes
(`--models unik3d` and `--models dac`).

```bash
# env A — UniK3D
pip install -e third_party/UniK3D            # weights auto-download on first run

# env B — DAC
pip install -r third_party/depth_any_camera/requirements.txt
python -m finetune.eval.baselines.download_weights --variant dac_swinl_indoor
#   -> checkpoints/dac_swinl_indoor.{json,pt}
```

Point the scripts at the repos via `--unik3d-root` / `--dac-root`, or
`$UNIK3D_ROOT` / `$DAC_ROOT` (default: `<ADT>/third_party/{UniK3D,depth_any_camera}`).

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

## Output (`--out`, default `eval_out/baselines/`)

```
adt_results.json          {fisheye_planar, erp_range} → variant → mode → metrics
adt_summary.txt           the two comparison tables
qual/{unik3d,dac}/        ADT samples — <i>_input.png, _depth.npy, _depth.png, _pcd.ply
egoexo/{unik3d,dac}/      EgoExo samples — <clip>_f####_{input.png,depth.npy,depth.png,pcd.ply}
official/{unik3d,dac}/    sanity-check outputs — same input/depth/pcd triple
```

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
