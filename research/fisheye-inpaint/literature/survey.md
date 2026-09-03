# Literature survey — rectified-fisheye border completion for 3D foundation models

Status: bootstrap sweep, 5 search agents. VERIFIED = seen in search result/fetch.
UNCERTAIN = surfaced only indirectly; do not cite without checking.

## A. Mainstream answer: don't rectify at all
- **Fisheye3R** — Duan et al., arXiv:2603.28896 (2026). VERIFIED. Says VGGT/π³/MapAnything
  "trained on large-scale datasets of perspective images; when tested on wide field-of-view
  images, e.g., from a fisheye camera, their performance degrades." Rectification "either
  sacrifices scene coverage or introduces extreme resampling artifacts and peripheral
  stretching"; undistortion "sever[s] crucial cross-view covisibility... destabilizes the
  geometric constraints required for pose estimation, leading to structural drift."
  Improves camera pose, depth, pointmap AND FoV estimation on fisheye. Reproduced in this repo (fisheye3r/).
- **Wid3R** — Jung et al., arXiv:2602.05321 (2026). VERIFIED. Existing feed-forward recon
  "assume rectified or pinhole inputs"; ray-based camera-token conditioning. COMPETING WORK.
- **Calibration Tokens** — Gangopadhyay et al., arXiv:2508.04928. VERIFIED (fetched).
  Frozen FMDEs (MiDaS/DepthAnything/UniDepth) systematically wrong on fisheye due to input
  distribution mismatch; fixed by latent token, no backbone finetune. "Entirely lossless for
  input images". LogL1 beats L1 "especially in border regions".
- **Depth Any Camera (DAC)** — Guo et al., CVPR 2025, arXiv:2501.02464. VERIFIED. Canonical ERP
  space + pitch-aware augmentation; up to 50% delta1 gain on fisheye/360.
- **Metric3D v2** — arXiv:2404.15506. VERIFIED. Canonical camera space via intrinsics, not rectification.
- **PanoVGGT** — arXiv:2603.17571, CVPR 2026. VERIFIED (abstract). VGGT-style on panoramas,
  spherical positional embeddings. Padding handling UNKNOWN — needs full-text read.
- **DarSwin / DarSwin-Unet** — arXiv:2304.09691 (ICCV'23) / arXiv:2407.17328 (WACV'25). VERIFIED.
  Radial-azimuthal patches keyed to the lens distortion profile; never rectifies.
- **360MonoDepth** (arXiv:2111.15669), **OmniFusion** (arXiv:2203.00838). VERIFIED. Tangent-patch
  gnomonic projection — each patch fully valid by construction.
- **WoodScape** (arXiv:1905.01489), **OmniDet** (arXiv:2102.07448). VERIFIED. Train directly on
  unrectified fisheye, 6 tasks.
- **Surround-View Fisheye 3D Detection** — arXiv:2511.18695. VERIFIED. Objects in fisheye occupy
  ~15% of pinhole pixel area; loss "irreversible and cannot be recovered through rectification";
  pinhole-trained detectors lose >12 pts on rectified fisheye.

## B. Precedent for completing fisheye invalid regions
- **FisheyeEX** — Liao et al., arXiv:2206.05844. VERIFIED. Polar outpainting of fisheye invalid
  border, spiral distortion-aware perception module so generated content matches the FISHEYE
  distortion model (not a pinhole prior). ~27% more valid content. NOTE: operates BEFORE
  rectification, in fisheye domain. Evaluates its own outpainting, not a downstream model delta.
- **FishDreamer** — CVPRW 2023 (OmniCV), arXiv:2303.13842. VERIFIED. Joint outpainting +
  segmentation beyond sensor FoV, Polar-aware Cross Attention. Cityscapes-BF/KITTI360-BF.
  Code: github.com/MasterHow/FishDreamer. Closest "completion feeds a recognition task".

## C. The risk: generated content misleads geometry
- **Enhancing Diffusion Models with 3D Perspective Geometry Constraints** — Upadhyay et al.,
  SIGGRAPH Asia 2023 / ACM TOG 42(6), arXiv:2312.00944. VERIFIED. *** KEY RISK CITATION ***
  Perspective-geometry loss during diffusion training -> consistent vanishing points/FoV;
  outputs improve DOWNSTREAM model performance. i.e. unconstrained diffusion output is
  perspective-inconsistent and measurably hurts downstream tasks.
- **HAD: Hallucination-Aware Diffusion Priors for 3D Reconstruction** — CVPR 2026, arXiv:2605.16873.
  VERIFIED. Diffusion-augmented views "inevitably introduce hallucinated content"; pixel-wise
  hallucination score maps mask unreliable pixels during reconstruction.
- **GaMO** — arXiv:2512.25073. VERIFIED. Avoids hallucination structurally: outpaint from
  EXISTING poses (not new viewpoints). Plücker-ray conditioning + focal lengths scaled to the
  enlarged FoV. Replica/ScanNet++/Mip-NeRF360.
- **Missingness Bias in Model Debugging** — Jain et al., ICLR 2022, arXiv:2204.08945. VERIFIED.
  Blacking out pixels is NOT a neutral no-information signal; introduces measurable bias.
  Transformers can drop tokens instead — "more natural" missingness.

## D. Tools for perspective-consistent generation (if we build the fix)
- **PreciseCam** — arXiv:2501.12910. VERIFIED. ControlNet-style explicit control of vFoV,
  roll/pitch, AND lens distortion in T2I. Mechanism to force a generator to a specified 125 deg FoV.
- **ControlVP** — arXiv:2512.07504. VERIFIED. Vanishing-point consistency loss on depth-conditioned LDM.
- **CameraCtrl** — He et al., arXiv:2404.02101. VERIFIED. Plücker camera-pose encoder for video diffusion.
- **DiffPano** — NeurIPS 2024, arXiv:2410.24203. VERIFIED. Spherical epipolar-aware multi-view diffusion.
- **PanoDiffusion / PanFusion** — UNCERTAIN. Surfaced only inside other papers' comparisons.

## E. Veridical alternative: warp from neighbouring frames
- **Deep Flow-Guided Video Inpainting** — Xu et al., CVPR 2019. VERIFIED.
- **E2FGVI** — flow-completion -> warp -> synthesize. VERIFIED.
- **Flow-Guided Diffusion for Video Inpainting** — arXiv:2311.15368. VERIFIED.
- Aria/ADT reconstruction context: **Photoreal Scene Reconstruction from an Egocentric Device**
  (SIGGRAPH 2025, arXiv:2506.04444), **EgoLifter** (arXiv:2403.18118). VERIFIED they exist;
  neither frames the border-fill operation.

## F. Why black regions could break a camera head
- **Kayhan & van Gemert** — CVPR 2020, arXiv:2003.07064. VERIFIED. CNNs exploit image boundary
  effects from zero-padding to learn absolute spatial location.
- **Islam, Jia, Bruce** — ICLR 2020, arXiv:2001.08248. VERIFIED. CNNs encode substantial
  absolute position information.
- **RSPC** — Guo, Stutz, Schiele, CVPR 2023. VERIFIED. Occluding ~10% of patches with noise
  causes severe accuracy drops and destabilizes intermediate self-attention. (Noise patches,
  not black — flag the distinction.)
- **MAE** — He et al., CVPR 2022, arXiv:2111.06377. VERIFIED. Encoder DROPS masked tokens rather
  than zeroing them; only the decoder uses a learned mask token. Design lesson, and a
  counterexample to "ViTs tolerate holes".
- **Perspective Fields** — CVPR 2023 Highlight, arXiv:2212.03239. VERIFIED. Dense per-pixel
  up-vector + latitude, "invariant or equivariant to cropping, warping, rotation". Implies
  GLOBAL FoV heads (VGGT's) lack this invariance by construction.
- **Tame a Wild Camera** — NeurIPS 2023, arXiv:2306.10988. VERIFIED. Incidence field, explicitly
  crop/resize invariant.
- **CTRL-C** (ICCV 2021, arXiv:2109.02259), **DeepCalib** (CVMP 2018). VERIFIED they exist;
  no border/padding sensitivity analysis found.

## THE GAPS (candidate novelty)
1. No paper completes fisheye/rectified invalid regions as PREPROCESSING for a frozen
   downstream geometric model and reports a controlled accuracy delta.
2. No paper proves black padding biases a learned FoV/camera regressor toward a wrong FoV.
   This is our hypothesis; supported only by analogy (Kayhan/van Gemert + Missingness Bias).
3. No paper combines (a) rectified-fisheye border inpainting + (b) explicit FoV/perspective
   conditioning of the generator + (c) measured effect on a downstream camera-FoV estimator.
4. Nobody uses Aria/ADT 6DoF pose metadata to warp temporally-adjacent frames into a rectified
   fisheye border before foundation-model inference.
5. No numeric characterization of the FoV/focal distribution of VGGT's training corpora.

## G. Round 2 — VGGT family, FoV limits, camera↔depth coupling
- **VGGT** — CVPR 2025 best paper, arXiv:2503.11651. VERIFIED. Camera head predicts g=[q,t,f]
  (quaternion, translation, 2D FoV). *** Limitations section states explicitly: "the current
  model does not support fisheye or panoramic images." *** Author-acknowledged.
- **RayTun3R: Online Camera Adaptation in 3D Foundation Models** — Sinitsyn, Araslanov, Cremers
  (TUM), arXiv:2607.02711 (Jul 2026). VERIFIED. *** STRONGEST COMPETING EXPLANATION ***
  DUSt3R/MASt3R/VGGT/π³/DepthAnything3 "degrade sharply under fisheye camera geometry";
  attributes it to PINHOLE BIAS BAKED INTO POSITIONAL ENCODINGS. Fix = 10,752-param adapter to
  absolute/rotary position encodings, no rectification. 2–12x rotation-error reduction across
  FoV 110°–200°; only "competitive" on depth.
- **CAM3R: Camera-Agnostic Model for 3D Reconstruction** — arXiv:2603.22631. VERIFIED.
  Perspective-only models "suffer significant geometric degradation... via non-rectilinear
  optics." Couples per-pixel ray estimation with radial-distance estimation architecturally.
- **MapAnything** — arXiv:2509.13414. VERIFIED. *** KEY COUNTER-EVIDENCE ***
  Table 2 (two-view): images-only rel=0.20 τ=43.9; +known intrinsics rel=0.19 τ=45.3 (SMALL);
  +intrinsics+poses rel=0.10 τ=63.2 (LARGE). Knowing intrinsics alone buys little.
- **VGGT-360** — Yuan, Jiang, Soh, Zhao, arXiv:2603.18943. VERIFIED. The paper ported in
  VGGT-360-fisheye/. Frames it as DOMAIN GAP, training-free reprojection. Does NOT frame it as
  camera-parameter-estimation failure, does NOT compare to DAv2 => our commit 04b6d4f diagnosis
  is novel relative to it.
- **On Geometric Understanding and Learned Priors in Feed-forward 3D Reconstruction Models** —
  arXiv:2512.11508, Bratulić, Mittal, Brox, Rupprecht. VERIFIED exists; probes DUSt3R/VGGT/DA3
  internals. Whether it covers intrinsics-depth coupling UNCONFIRMED — needs full read.
- **FS-Depth** arXiv:2307.14624 (focal-and-scale coupling). UNCERTAIN, title only.
- UNCERTAIN, not fetched: DriveVGGT arXiv:2511.22264; PAGE-4D (OpenReview);
  Depth Any Panoramas arXiv:2512.16913; DepthMaster arXiv:2606.12368.
