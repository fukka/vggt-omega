# Fisheye3R (reproduction)

Re-implementation of **"Fisheye3R: Adapting Unified 3D Feed-Forward Foundation
Models to Fisheye Lenses"** (Duan et al., arXiv:2603.28896v2, Yale + Google)
on the **VGGT-Omega** backbone of this repo.

The official repo (`github.com/android-xr/fisheye3r`) is an empty placeholder
(README with unfilled TODO sections, no code) as of 2026-07, so everything
here is reconstructed from the paper + supplementary material.

## Paper → code map

| Paper | Where |
|---|---|
| KB distortion synthesis T, undistortion T⁻¹ (supp Eq. 16–18, randomized params) | `distortion.py` |
| Calibration tokens in E/F/G, per-layer insert-then-drop (Sec. 3.2, Eq. 6–8) | `model.py::Fisheye3R._frame_level/_global_level` |
| Frozen backbone, tokens init N(0, 1e-6), K=8, L0=12 (supp Sec. 6) | `model.py` |
| Camera-type classifier ψ on the L0 class token (Eq. 14) | `model.py::predict_camera_type`, `train.py fit-classifier` |
| Masked attention M_F / M_G for mixed sequences (Eq. 15, Fig. 3) | `model.py` (batch-split for F/E, two-pass key-selection for G — exactly equivalent to the block masks, see module docstring) |
| SSL / SL / SL+ objectives (Eq. 11–13) | `losses.py::scheme_loss`, `train.py` |
| Supervision in the perspective domain after T⁻¹, poses untransformed | `losses.py::undistort_predictions` |
| AdamW, lr 1e-5→1e-7, 40k iters, ≤24 frames/GPU (supp Sec. 6) | `train.py` |
| Metrics: RRA/RTA/AUC@30, ATE, AbsRel/RMSE/δ1 (scale+shift per sequence), FoV hErr/vErr (Sec. 4.1) | `eval.py` |

Module correspondence on VGGT-Omega: **E** = `aggregator.patch_embed`
(DINOv3 ViT, 24 blocks; tokens in the last 12), **F** = `frame_blocks` (24),
**G** = `inter_frame_blocks` (24, incl. the register-only ones).

## Usage

```bash
# camera-type classifier (logistic regression; paper: ~1 min, 99.9% acc)
python -m fisheye3r.train fit-classifier \
    --weights checkpoints/vggt_omega.pt --data-root /data/perspective --out runs/f3r

# SSL: unlabeled perspective RGB only (Eq. 11)
python -m fisheye3r.train train --scheme ssl \
    --weights checkpoints/vggt_omega.pt --data-root /data/perspective --out runs/f3r

# SL: perspective GT (Eq. 12) — needs depth/ + cameras.npz per scene
python -m fisheye3r.train train --scheme sl ...

# SL+: real fisheye GT (Eq. 13) — point --data-root at the fisheye dataset
python -m fisheye3r.train train --scheme slplus ...

# evaluate on real fisheye (ADT-style: fisheye GT depth in-domain)
python -m fisheye3r.eval --weights checkpoints/vggt_omega.pt \
    --tokens runs/f3r/tokens_ssl_final.pt --data-root /data/adt_test

# evaluate with synthesized distortion (ScanNet++-fisheye-style protocol)
python -m fisheye3r.eval ... --synthesize-fisheye
# unadapted baseline numbers
python -m fisheye3r.eval ... --baseline
```

Dataset layout: `root/scene/rgb/*.jpg`, optional `depth/*.npy` (meters) and
`cameras.npz` (`intrinsics (N,3,3)`, `extrinsics (N,4,4)` cam-from-world).

`python fisheye3r/smoke_test.py` runs a CPU end-to-end check on a tiny
random backbone (distortion round-trip, init-equivalence with the frozen
model, hybrid masked attention, one SSL step, metrics, save/load).

## Known deviations from the paper

1. **Backbone**: VGGT-Omega only (paper: VGGT, π³, MapAnything). VGGT-Omega
   exposes depth + camera heads, so **point-map Acc/Comp/CD are not
   evaluated** and the FoV metric comes from the 9D pose encoding rather than
   a dense FoV map.
2. **Token position**: tokens are *prepended* (paper appends). Equivalent
   under attention permutation-invariance; required by VGGT-Omega's RoPE
   convention (patch tokens must be the trailing H×W tokens).
3. **Masked attention**: implemented as batch-split (frame level) and a
   two-pass key-selection (global level) instead of materializing the
   (SN+K)² mask — mathematically identical to Eq. 15, avoids the giant mask.
4. **Frame sampling**: random window + stride instead of the precomputed
   ≥25%-covisibility matrix; augmentations lack random resize/crop.
5. **VGGT loss details**: the paper defers to each model's original loss; we
   use conf-weighted L1 depth + log-depth gradient matching + smooth-L1 on
   the 9D pose encoding, and gate SSL pseudo-labels by a teacher-confidence
   quantile (pseudo-label filtering is unspecified in the paper).
6. **Batching**: one sequence per step (`--accum` for effective batch);
   the paper packs 1–12 sequences / ≤24 frames per GPU across 4 GPUs.
