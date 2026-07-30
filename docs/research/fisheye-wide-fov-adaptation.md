# Adapting frozen depth / 3D foundation models to fisheye and wide-FOV input

> **New location.** This repo previously had no cross-subproject notes convention — only
> per-subproject READMEs (`VGGT-360-fisheye/`, `fisheye3r/README.md`, `finetune/`) and the
> root `CONTEXT.md` glossary. `docs/research/` is introduced here for literature notes that
> span subprojects. Nothing else references it yet.
>
> Survey date: 2026-07-29. Every paper below was checked against its arXiv abstract page,
> official project page, or official repo; §5 lists what could not be confirmed.
> "**Verified**" = I fetched the primary source and read it. "**Claims**" = the paper's own
> assertion, which I did not independently reproduce.

---

## 1. Scope and constraint

Goal: run **Depth Anything V2** ([arXiv:2406.09414](https://arxiv.org/abs/2406.09414), NeurIPS 2024)
and **VGGT / VGGT-Ω** ([arXiv:2503.11651](https://arxiv.org/abs/2503.11651);
[arXiv:2605.15195](https://arxiv.org/abs/2605.15195), CVPR 2026 Oral) on **fisheye / wide-FOV**
imagery — here Aria 214-1 KB4 fisheye — with the distortion handled *without* retraining the
backbone. Admissible budget: nothing, a reprojection wrapper, a handful of learned embeddings,
a LoRA/adapter, or a small residual on positional encodings. Full retraining is out of scope.

The concrete failure this survey is aimed at: **VGGT couples its depth output to its own
self-estimated FoV.** This is structural, not incidental — `pose_enc` is
`[T(3), quat(4), fov_h, fov_w]`, and the depth→points conversion divides by
`f = (H/2)/tan(fov/2)` derived from exactly those two channels
(`vggt_omega/utils/pose_enc.py:16,24-26,32-43` in this repo). On narrow tangent crops cut from a
fisheye frame, `pose_enc[7:9]` is wrong, so the geometry bends relative to the input. DAv2 has no
such channel and stays aligned on the same crops. Therefore the literature that matters most is:
(i) can the camera estimate be **overridden** instead of inferred, and (ii) is the deeper cause
the **positional encoding's pinhole prior** rather than the FoV head.

Two papers answer those two questions almost exactly:
**OmniVGGT** (override, §3.2) and **RayTun3R** (PE prior, §3.3).

---

## 2. The three families of approach

| | **(a) Reproject input → frozen pinhole model** | **(b) Make the model camera-aware / camera-conditioned** | **(c) Small distortion/camera embeddings or adapters on a frozen backbone** |
|---|---|---|---|
| **Exemplars** | VGGT-360, 360MonoDepth, OmniFusion, PatchFusion, Depth Pro (multi-scale patches), Depth Anywhere (cubemap pseudo-labels), PaGeR (cubemap + DA3) | DAC, UniK3D, UniDepth/V2, Metric3D/v2, Pow3R, MapAnything, X-Lens, Prompt Depth Anything | **Fisheye3R**, **RayTun3R**, **OmniVGGT** (GeoAdapter), RePer-360, FishRoPE, LoRA3D |
| **How distortion is handled** | Removed before the model sees it: gnomonic/tangent or cubemap patches are locally rectilinear. Distortion never enters the network; it re-enters only in the fuse-back step. | Encoded as an explicit camera representation the network consumes: ERP canonicalization (DAC), spherical-harmonic pencil of rays (UniK3D), dense ray map (Pow3R, MapAnything), canonical focal rescale (Metric3D), generic unprojection map (X-Lens). | Left in the image; the network's *interface* is corrected. Learned tokens recalibrate features (Fisheye3R), or positional encodings get a radial/angular residual so token geometry matches the lens (RayTun3R, FishRoPE), or a zero-init side-branch injects known intrinsics (OmniVGGT). |
| **Training cost** | Zero for the backbone. VGGT-360 is fully training-free. Some members train only a fusion net (OmniFusion, PatchFusion). | Highest. All of DAC / UniK3D / UniDepth / Metric3D / Pow3R / MapAnything / X-Lens are trained from scratch or fully finetuned on large mixed corpora (MapAnything: 64×H200 for 10 days; Pow3R: 8×A100 for 5 days). | Lowest non-zero. RayTun3R: **10,752** params, ~30 three-frame windows, 2–3 h per scene. Fisheye3R: **~344 k** params, 40 k iters / ~20 h on 4 GPUs. OmniVGGT's GeoAdapter is 26.8 M params but was trained on 32×A100 for 10 days. |
| **Backbone frozen?** | Yes, entirely. | No. | Yes for Fisheye3R (claims), RayTun3R (verified: "all backbone weights remain frozen"), RePer-360, FishRoPE (frozen DINOv2 + LoRA). **No for OmniVGGT** — it initializes from VGGT weights and finetunes. |
| **Known failure modes** | Seams and per-view scale disagreement in the fuse-back step; loses genuinely wide-FOV context inside each crop; and — the case measured in this repo — **a narrow crop can itself trip the backbone's camera self-estimation**, so "locally rectilinear" is not sufficient for VGGT. Cost grows with view count. | Needs calibration at inference (DAC: yes, verified). Requires the very retraining this project excludes. UniK3D notes wide-FOV outputs *contract* without the angular loss. | Fisheye3R: no public code (verified placeholder, 2026-07); needs a camera-type classifier and masked attention to avoid regressing perspective input. RayTun3R: assumes **known calibration**, and adapts **per sequence** — not a single global adapter. FishRoPE: no depth results. |

Cross-cutting observation from the field's own survey — *Panoramic Scene Understanding: A Survey
from Distortion-Aware Engineering to Sphere-Native Modeling*, Zhu & Fan
([arXiv:2606.27745](https://arxiv.org/abs/2606.27745)) — is that sphere-native operators
"cannot reuse perspective-pretrained backbones and thus have not scaled," and the field has
converged on **geometry-aware tokenization**: adapt the input interface, keep the pretrained
weights. That is precisely family (c), and it is the family this project's constraint selects.

---

## 3. Paper notes

### 3.1 Training-free / test-time projection strategies

**VGGT-360: Geometry-Consistent Zero-Shot Panoramic Depth Estimation** — Jiayi Yuan, Haobo Jiang,
De Wen Soh, Na Zhao. CVPR 2026. [arXiv:2603.18943](https://arxiv.org/abs/2603.18943) ·
code [github.com/Yuanjiayii/VGGT-360](https://github.com/Yuanjiayii/VGGT-360)
(**verified: real code present** — `main.py`, `utils/`, `vggt_visfeat/`, `assets/`; no LICENSE
file and no checkpoints in the README).
Three training-free modules, all verified from the method section. *Uncertainty-guided adaptive
projection*: start from `N_B ≥ 6` base views giving full coverage with controlled overlap, score
each by a Sobel-gradient-derived per-pixel uncertainty `U(p) = σ(−Z(p))` with
`Z(p) = (G(p) − median)/τ`, then give the top-K most uncertain base views two extra neighbours at
predefined yaw/pitch offsets. *Structure-saliency enhanced attention*: an **additive log-confidence
bias on the attention logits** in VGGT's intra-frame attention,
`softmax(QKᵀ/√d + log(M_s))` — a bias before softmax, not a reweighting of values, and no weights
change. *Correlation-weighted 3D correction*: from final-layer attention maps it computes
sharpness (normalized Shannon entropy), locality (Gaussian spatial compactness) and symmetry
(Bhattacharyya coefficient), adds and normalizes them into a per-point weight `C_vk`, and fuses
with `D_erp(r) = Σ C_vk D_vk / Σ C_vk`. **Claims** 27–36% Abs Rel improvement over prior SOTA on
Stanford2D3D and Replica360-2K.
*Why it matters:* this is the anchor being ported. Note what the paper does **not** contain
(verified): no discussion of whether VGGT's *predicted* intrinsics or the *known* render intrinsics
of each view are used, and no FoV/focal error analysis. The FoV-coupling effect measured in this
repo is therefore a genuine gap in the upstream method, not something it addressed and you missed.

**360MonoDepth: High-Resolution 360° Monocular Depth Estimation** — Rey-Area, Yuan, Richardt.
CVPR 2022. [arXiv:2111.15669](https://arxiv.org/abs/2111.15669) ·
code [manurare.github.io/360monodepth](https://manurare.github.io/360monodepth/).
Projects the ERP onto a set of tangent planes, runs an off-the-shelf **perspective** depth
estimator per tangent view, then recombines with **deformable multi-scale alignment followed by
gradient-domain blending**. The perspective model is untouched.
*Why it matters:* the cleanest prior art for the fuse-back half of family (a), and its blending
stage is a stronger seam handler than confidence-weighted averaging. If the ADT fisheye fusion
shows seams, this is the recipe to borrow.

**OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion** — Li, Guo et al.
CVPR 2022 Oral. [arXiv:2203.00838](https://arxiv.org/abs/2203.00838) ·
code [github.com/yuliangguo/OmniFusion](https://github.com/yuliangguo/OmniFusion).
Tangent-image patches → per-patch CNN → fusion that concatenates **3D geometric features with 2D
image features** to absorb patch-wise disagreement, plus a self-attention transformer for global
aggregation across patches and an iterative depth refinement loop.
*Why it matters:* it identifies patch-wise scale/offset disagreement as the core problem of
tangent-patch methods and fixes it with explicit geometric features — the same disease VGGT-360
treats with cross-view attention. Same first author as DAC, which is a useful lineage to know.

**PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth
Estimation** — Zhenyu Li, Shariq Farooq Bhat, Peter Wonka.
[arXiv:2312.02284](https://arxiv.org/abs/2312.02284) · project
[zhyever.github.io/patchfusion](https://zhyever.github.io/patchfusion/).
Coarse global prediction + finer inconsistent tiles, fused by a patch-wise network under
high-level feature guidance; a Global-to-Local module supplies context so no patch-selection
heuristic is needed; Consistency-Aware Training/Inference explicitly penalizes disagreement in
tile **overlap** regions. Built on ZoeDepth.
*Why it matters:* CAT/CAI is a directly transplantable idea — an overlap-consistency objective or
test-time consistency check across your tangent views, orthogonal to VGGT-360's attention-derived
confidence.

**Depth Pro: Sharp Monocular Metric Depth in Less Than a Second** — Bochkovskii, Delaunoy,
Germain, Santos, Zhou, Richter, Koltun. ICLR 2025.
[arXiv:2410.02073](https://arxiv.org/abs/2410.02073) ·
code [github.com/apple/ml-depth-pro](https://github.com/apple/ml-depth-pro).
Multi-scale ViT for dense prediction; metric output **without camera metadata**, with a dedicated
state-of-the-art **focal-length estimation head** from the single image.
*Why it matters:* the counter-example that a separate FoV head can be accurate. Its focal head is
a candidate external estimator to feed VGGT rather than trusting `pose_enc[7:9]` — but see §5, I
did not verify its behaviour on fisheye-derived crops.

**Depth Anywhere: Enhancing 360 Monocular Depth Estimation via Perspective Distillation and
Unlabeled Data Augmentation** — Ning-Hsu Wang, Yu-Lun Liu. NeurIPS 2024.
[arXiv:2406.12849](https://arxiv.org/abs/2406.12849).
Uses a perspective depth model as **teacher** via **six-face cube projection** to pseudo-label
360 images, plus offline invalid-region masking and online semi-supervised joint training.
*Why it matters:* if you ever want a student that natively eats fisheye, this is the cheapest
supervision route — the teacher runs on rectilinear cube faces where it is already correct. It is
distillation, so it trains a student; the *teacher* stays frozen.

**Unified Panoramic Geometry Estimation via Multi-View Foundation Models (PaGeR)** — Bozic,
Slavkovic, Narnhofer, Metzger, Rozumny, Schindler, Kalischek.
[arXiv:2605.26368](https://arxiv.org/abs/2605.26368) ·
code [github.com/prs-eth/PaGeR](https://github.com/prs-eth/PaGeR) · project
[pager360.github.io](https://pager360.github.io/).
Lifts **Depth Anything 3** (da3-giant) to panoramas through a **fixed 6×504×504 cubemap**, so VRAM
and runtime are constant in input resolution; one forward pass returns scale-invariant depth,
metric depth, world-frame normals and a sky mask. Ships two new datasets (ZüriPano real eval,
PanoInfinigen synthetic train).
*Why it matters:* the closest published relative of VGGT-360 but on DA3 and *with* training — it
shows the cubemap-as-multi-view trick works on a modern feed-forward FM, and its fixed-budget
cubemap is a cheaper view schedule than VGGT-360's adaptive one. Useful as a baseline; not
training-free.

### 3.2 Camera-conditioned / camera-prompted geometry — how each *injects* the camera

This is the section that answers "can the camera estimate be overridden rather than inferred". The
short answer, verified across four papers: **the injection format that keeps winning is a dense
per-pixel ray map, patchified and added to the image tokens.**

**OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer** — CVPR 2026 Highlight.
[arXiv:2511.10560](https://arxiv.org/abs/2511.10560) ·
code [github.com/Livioni/OmniVGGT-official](https://github.com/Livioni/OmniVGGT-official)
(MIT; HF weights `Livioni/OmniVGGT`) · project
[livioni.github.io/OmniVGGT-official](https://livioni.github.io/OmniVGGT-official/).
**The single most directly relevant camera-conditioning paper for a VGGT backbone.** Verified
mechanism: a **GeoAdapter** with two branches (camera, depth) injects into VGGT's
**Alternating-Attention blocks**. The camera branch is `L+1 = 25` independent encoders, each a
**single linear layer**, one per AA block. Intrinsics are parameterized not as `K` but as
**`f ∈ ℝ²`, the field of view** — the *same two numbers as VGGT's `pose_enc[7:9]`* — bundled with
quaternion `q ∈ ℝ⁴` and translation `t ∈ ℝ³` into `g = {q, t, f}`. Poses are first normalized
relative to camera 1 (`Gⱼ′ = Gⱼ G₁⁻¹`) and scaled by the mean camera-to-origin distance. Injection
is **zero-initialized convolution** gated by an availability mask:
`e′_{c,i,l} = e_{c,i,l} + 𝒵𝒞_l( m_i · e^aux + (1 − m_i) · placeholder )`.
Depth uses a **single** encoder: one conv, kernel 14 (matching the patch stride). Training uses
**stochastic multimodal fusion** — sample `Q ∈ [0,S]` images to receive camera params and
`O ∈ [0,S]` independently for depth, plus 10% RGB-only batches — which is what makes arbitrary
inference-time subsets work. Cost: GeoAdapter is **26.8 M** params; trained on **32×A100 for
10 days**, initialized from VGGT weights (so the backbone is finetuned, not frozen). Table 4
**claims** `w/ K+RT` beats RGB-only on AUC@30°.
*Why it matters:* it proves the exact override you want is architecturally sound — VGGT's camera
information can be *supplied* through a zero-init gated side-channel at every AA block, in the
very `(q, t, fov_h, fov_w)` parameterization your `pose_enc` already uses, without destroying the
representation. The catch: as published this is a 10-day retrain. The transplantable part is the
**interface**, and the fact that zero-init gating means an untrained adapter is exactly the
identity — so a *tiny* version of this can be trained on ADT alone. See idea I2.

**Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors** — Jang,
Weinzaepfel, Leroy, Agapito, Revaud. CVPR 2025.
[arXiv:2503.17316](https://arxiv.org/abs/2503.17316) ·
code [github.com/naver/pow3r](https://github.com/naver/pow3r).
Verified mechanism: intrinsics become a **dense ray map** — the ray at pixel `(i,j)` is
`K⁻¹[i, j, 1]` — which is then **patchified and embedded exactly like RGB** and injected into the
encoder through **per-block MLPs**. Depth priors are `[D/‖D‖, M] ∈ ℝ^{W×H×2}` (normalized depth
stacked with a sparsity mask), same patchify-and-inject path. Relative pose `P₁₂`, being non
pixel-aligned, is instead **added to the global CLS token of both decoders** after self- and
cross-attention. Trained with **random modality dropout** — pick a random count `m`, then pick `m`
modalities — over 8.5 M pairs. Added parameters are small (**+0.1%** for the head, **+4%** for
auxiliary injection in the inject-1 variant) but the DUSt3R backbone is **finetuned**, 3 days at
224 px + 2 days at 512 px on 8×A100.
*Why it matters:* the `K⁻¹[i,j,1]` ray map is the cleanest, most portable camera-injection format
in the literature — and crucially it is **not tied to a pinhole `K`**. Replace `K⁻¹` with the KB4
unprojection `kb4_unproject(u,v)` from `VGGT-360-fisheye/utils/fisheye_cam.py` and the *same*
tensor shape now describes an Aria fisheye. Verified caveat: **Pow3R never tests non-pinhole
intrinsics** — the mechanism generalizes, the evidence does not.

**MapAnything: Universal Feed-Forward Metric 3D Reconstruction** —
[arXiv:2509.13414](https://arxiv.org/abs/2509.13414) ·
code [github.com/facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) ·
project [map-anything.github.io](https://map-anything.github.io/).
Same ray-map idea, industrialized. Intrinsics enter as **ray directions** `R^i ∈ ℝ^{3×H×W}` through
a shallow conv encoder with a single **pixel-unshuffle of size 14**, projected to the DINOv2 latent
width (1024 × H/14 × W/14) and **summed with image patch features**. Outputs are deliberately
**factored**: ray directions, up-to-scale depth along ray, pose (quaternion + up-to-scale
translation relative to frame 1), and one **global metric scale** `m`, with
`X_i^metric = m · X̃_i`. Modality dropout: overall geometric-input probability 0.9, and ray
directions / ray depth / pose each 0.5 independently; when depth is chosen it is dense or 90%
sparsified with equal probability. DINOv2 ViT-L encoder (fine-tuned at lr 5e-6), randomly
initialized 24-layer alternating-attention transformer. 64×H200, 6 + 4 days.
*Why it matters:* two things. (1) It is one of the three backbones Fisheye3R adapts, so its
interface and Fisheye3R's tokens are known-compatible. (2) The paper **claims** generic *central*
camera calibrations are representable as ray directions and that the method can extend to fisheye
"with appropriate training" — i.e. the representation is FoV-agnostic even though the released
model was not trained that way. The **factored** output is also worth copying: separating "which
ray" from "how far along it" is exactly the decoupling that would stop a wrong FoV from bending
depth.

**Depth Any Camera (DAC): Zero-Shot Metric Depth Estimation from Any Camera** — Yuliang Guo,
Sparsh Garg, S. Mahdi H. Miangoleh, Xinyu Huang, Liu Ren. CVPR 2025.
[arXiv:2501.02464](https://arxiv.org/abs/2501.02464) ·
code [github.com/yuliangguo/depth_any_camera](https://github.com/yuliangguo/depth_any_camera)
(MIT) · project [yuliangguo.github.io/depth-any-camera](https://yuliangguo.github.io/depth-any-camera/).
Already a baseline here, so only the mechanism, verified: **ERP is the canonical space** — every
pixel is a `(latitude λ, longitude φ)`, patches are a fixed 500×700. Perspective training images
are converted into ERP patches via gnomonic geometry using **the given camera parameters'
distortion and projection functions**, with the **pitch-aware** part being noise added to `λ_c` so
patches take varying shapes and land in high-distortion latitudes. At inference an **FoV alignment**
scales each input so its FoV matches the crop-area FoV. Backbone is **iDisc**, chosen for its
self/cross-attention; SILog loss; a virtual focal length is derived from image dimensions for
metric scaling. **Intrinsics are required at inference** (verified).
*Why it matters:* positions the repo's existing baseline correctly — DAC is family (b) and needs
calibration, which you have for Aria. Its ITA conversion is also reusable as *data augmentation*
for any minimal-finetuning scheme: it is a verified recipe for synthesizing high-distortion
training samples from perspective images, i.e. the same trick Fisheye3R's SSL/SL schemes need.

**UniK3D: Universal Camera Monocular 3D Estimation** — Piccinelli, Sakaridis, Segu, Yang, Li,
Abbeloos, Van Gool. CVPR 2025. [arXiv:2503.16591](https://arxiv.org/abs/2503.16591) ·
code [github.com/lpiccinelli-eth/UniK3D](https://github.com/lpiccinelli-eth/UniK3D) · project
[lpiccinelli-eth.github.io/pub/unik3d](https://lpiccinelli-eth.github.io/pub/unik3d/).
Represents the **pencil of rays** as a learned superposition of **spherical harmonics** —
**up to degree 3, constant component excluded, 15 harmonic tensors** — which is model-independent,
so no pinhole/rectified assumption. Output lives in a **spherical** 3D space that disentangles
camera from scene, and an **angular loss** together with the camera module is what
"prevents the contraction of the 3D outputs for wide-view cameras."
*Why it matters:* the wide-FOV **contraction** failure it names is a close cousin of your bending
artifact — both are "the model got the ray field wrong, so the geometry deforms radially." The
15-coefficient spherical-harmonic ray field is also the most compact camera parameterization in
the literature: a plausible drop-in for a *learned* few-parameter camera prompt (§4, I5).

**UniDepth / UniDepthV2** — Piccinelli, Yang, Sakaridis, Segu, Li, Van Gool, Yu. CVPR 2024;
V2 [arXiv:2502.20110](https://arxiv.org/abs/2502.20110) ·
code [github.com/lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth) · project
[lpiccinelli-eth.github.io/pub/unidepth](https://lpiccinelli-eth.github.io/pub/unidepth/).
A **self-promptable camera module** predicts a **dense** camera representation that **conditions
the depth features**, in a **pseudo-spherical** output space that disentangles camera from depth;
a **geometric invariance loss** enforces that camera-prompted depth features stay invariant. V2
adds an edge-guided loss, a simplified architecture and an uncertainty output.
*Why it matters:* the word "self-**promptable**" is the point — the camera representation is a
*dense conditioning signal that a prompt can replace*. This is the conceptual template for
overriding VGGT's FoV: don't fight the estimator, feed it. The geometric-invariance loss is also
the right objective shape for a minimal adapter — see I3.

**Metric3D / Metric3Dv2** — Yin, Zhang, Chen, Cai, Yu, Wang, Chen, Shen. ICCV 2023 /
TPAMI. [arXiv:2307.10984](https://arxiv.org/abs/2307.10984) ·
[arXiv:2404.15506](https://arxiv.org/abs/2404.15506) · project
[jugghm.github.io/Metric3Dv2](https://jugghm.github.io/Metric3Dv2/).
The **canonical camera space transformation**, explicitly designed to be "effortlessly plugged
into existing monocular models," in two interchangeable forms: **(1)** rescale the GT depth by
`f_canonical / f_original` during training; **(2)** **resize the input image** to emulate the
canonical camera, again by the focal ratio. At inference a de-canonical transform maps the
predicted metric depth back.
*Why it matters:* the cheapest possible camera conditioning — it is a **resize**, zero parameters,
zero training. Form (2) is a genuinely test-time-only operation. On a tangent crop whose true FoV
you *rendered and therefore know*, resizing the crop so its effective focal matches whatever focal
the frozen model behaves best at is a one-line experiment. See I1.

**Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation** — Lin et al.
CVPR 2025. [arXiv:2412.14015](https://arxiv.org/abs/2412.14015) · project
[promptda.github.io](https://promptda.github.io/).
First to bring **prompting** to depth foundation models: a low-cost **LiDAR** depth map acts as
the prompt, fused **at multiple scales inside the depth decoder** (not the encoder), with a data
pipeline of synthetic LiDAR simulation + real pseudo-GT.
*Why it matters:* proves a **decoder-side** prompt is enough to re-anchor a DAv2-family model's
metric behaviour. The analogue here: prompt the decoder with a sparse *known-geometry* signal —
e.g. the analytic per-pixel incidence angle `θ(u,v)` from `fisheye_cam.py`, or a handful of ADT GT
depths — rather than modifying the encoder.

**X-Lens: Real-Time Metric Depth Estimation with Heterogeneous Cameras** — Heng Zhou, Shuhong Liu,
Yonghao He, Bohao Zhang, Fa Fu, Chenhui Hou, Xianbao Hou, Lijun Han, Wei Sui (D-Robotics /
U. Tokyo / Soochow). 2026-07-14. [arXiv:2607.12993](https://arxiv.org/abs/2607.12993) ·
code [github.com/zhouhengamerica/XLens](https://github.com/zhouhengamerica/XLens).
The most recent synthesis of families (b) and (c). Replaces the intrinsic matrix with a **generic
unprojection map** covering pinhole, fisheye and 360 without architecture changes; adds
(i) **layer- and camera-type-specific calibration tokens** (Fisheye3R-style), (ii) a **Jacobian
distortion bias** injected into **cross-attention** so distorted fisheye regions match undistorted
pinhole observations, (iii) **FishRoPE** for ray-space positional encoding. Outputs a factored
`(normalized depth, confidence, single global metric scale)` and **deliberately does not predict
pose or intrinsics** because deployment assumes a calibrated rig. DINOv2 + DPT, alternating
within-/cross-view layers, 0.04 B params, up to 41 FPS. Three-stage training: pinhole pretrain →
fisheye token adaptation → heterogeneous joint. **Claims** −25.4% AbsRel vs the strongest baseline.
*Why it matters:* two design decisions to steal directly. (1) **Dropping the camera head entirely**
when calibration is known — that is the cleanest possible answer to FoV coupling. (2) The
**Jacobian distortion bias in cross-attention**: since the local fisheye Jacobian is analytic for
KB4, this is a *zero-parameter* attention modification. Also independent confirmation that
calibration tokens + spherical RoPE compose.

**AnyCalib: On-Manifold Learning for Model-Agnostic Single-View Camera Calibration** —
Javier Tirado-Garín, Javier Civera. ICCV 2025.
[arXiv:2503.12701](https://arxiv.org/abs/2503.12701) ·
code [github.com/javrtg/AnyCalib](https://github.com/javrtg/AnyCalib) (Apache-2.0).
Regresses the **per-pixel ray field**, from which intrinsics follow in **closed form** for pinhole,
Brown-Conrady **and Kannala-Brandt**; handles cropped/stretched images. **Claims** it outperforms
3D foundation models at calibration despite far less training data.
*Why it matters:* it is the sanity-check instrument for the FoV-coupling diagnosis. Run AnyCalib on
the same tangent crops and compare its KB/pinhole estimate against VGGT's `pose_enc[7:9]` and the
known render FoV — three numbers, one of which is ground truth. It also supplies the calibration
that RayTun3R needs when true calibration is unavailable (RayTun3R's Appendix B does exactly this).

### 3.3 Parameter-efficient adaptation of geometry backbones

**RayTun3R: Online Camera Adaptation in 3D Foundation Models** — Daniil Sinitsyn, Nikita Araslanov,
Daniel Cremers (TUM / MCML, per [mcml.ai](https://mcml.ai/research/groups/cremers/)). 2026-07-02.
[arXiv:2607.02711](https://arxiv.org/abs/2607.02711). **No code URL** — the paper says
"our code will be made publicly available."

**Read this one first.** It is the closest published match to this project's constraint *and* it
diagnoses the mechanism behind the observed bending.

*Diagnosis (verified).* The failure is attributed to a **pinhole bias in the positional encodings**
of pretrained 3D FMs, established by a **Jacobian** argument: for a pinhole camera the
backprojection Jacobian `J_κ⁻¹ = ∂κ⁻¹/∂(u,v)` is constant over the image, because a one-pixel step
changes the viewing direction by the same `1/f_{x,y}` everywhere. For fisheye it is strongly
position-dependent. They measure the largest singular value `σ₁` and the local area element
`det(J_PE^⊤ J_PE)` of the *positional-embedding* Jacobian against normalized radius, and find
pretrained embeddings are **nearly flat** (pinhole-like) while adapted ones become
**radius-dependent**.

*Method (verified).* Backbone entirely frozen (**Depth Anything 3, VGGT, π³**), all residual
adapter parameters zero-initialized. Absolute PE gets a polar-binned residual around the
calibrated principal point:
`P′(u,v) = P_A(u,v) + t_r(ρ_{u,v}) + ρ_{u,v} · δ_θ(θ_{u,v})`
with `t_r`, `δ_θ` learnable lookup tables over **20 radial** and **8 angular** bins. RoPE gets
`ω′(u,v) = ω(u,v) + Δ_r(ρ_{u,v})`, a **radial lookup table shared across RoPE frequencies**, one
parameter per bin, **20 bins**. For DA3-Small (`C = 384`): `(20 + 8) × 384` PE parameters + 20 RoPE
+ overhead ≈ **10,752 trainable parameters**. Two parameter-free pieces: patches outside the valid
lens circle are replaced by the **mean valid token**, and each patch is **resampled using the local
linearization of the fisheye→pinhole map at its centre**; and for DPT-style heads the regular
**prediction-grid coordinates** are replaced by camera-aware ones obtained by undistorting each
grid location through the calibrated map.

*Training (verified).* Three-frame windows, `L_reproj` (reprojection error after transforming
predicted 3D points from `I_i` to `I_j`), `L_pose` (angular difference against MAGSAC++ poses,
arc-cos rotation + translation direction), edge-aware smoothness, plus L2 and total-variation
regularizers on the positional corrections. Adaptation set: **30 three-frame windows** per
sequence; ~2–3 h per ETH3D scene end to end; **no added inference cost** (~100 ms/frame, same as
vanilla DA3).

*Results (claims, verified as reported).* Datasets: KITTI-360 (185°), TUM-VI (195°),
ScanNet++ (115°), ETH3D (110°), FIORD (200°). DA3-Small rotation error ETH3D 8.59° → 0.70°,
KITTI-360 1.69° → 0.84°, TUM-VI 10.41° → 2.41°; translation ETH3D 15.16° → 4.48°,
KITTI-360 12.81° → 2.92°. Depth AbsRel ETH3D 0.178 → 0.107, ScanNet++ 0.282 → 0.108. Improvements
also reported for **π³ and VGGT** (Table 2). Ablation: the **learned absolute-PE residual gives the
largest gain**; radial bins alone already do most of it, angular bins refine; **RoPE-only performs
poorly**; patch undistortion alone is minimal; parameter-free pieces are smaller. Appendix B shows
it still works with **AnyCalib-predicted** calibration instead of true calibration, and beats both
a LoRA baseline and a "CalTok" (calibration-token) baseline. Fisheye3R is cited as
**contemporaneous** in related work, **not compared against numerically**.

*Why it matters:* it is a 10 k-parameter, frozen-backbone, VGGT-compatible adapter that fixes
fisheye geometry, and its Jacobian diagnostic gives you a **measurable target**: if
`det(J_PE^⊤J_PE)` versus radius is flat for your vendored PEs, the pinhole prior is present and the
FoV head may be a symptom rather than the disease. Two honest limitations: it **assumes known
calibration** (fine — you have Aria KB4), and it adapts **per sequence** from a short temporal
segment, so it is online adaptation rather than one universal fisheye adapter.

**Fisheye3R: Adapting Unified 3D Feed-Forward Foundation Models to Fisheye Lenses** — Ruxiao Duan,
Erin Hong, Dongxu Zhao, Eric Turner, Alex Wong, Yunwen Zhou (Yale + Google). ECCV 2026.
[arXiv:2603.28896](https://arxiv.org/abs/2603.28896) (v1 2026-03-30, v2 2026-07-01) ·
[github.com/android-xr/fisheye3r](https://github.com/android-xr/fisheye3r) —
**verified still a placeholder** as of this survey: only `.github/`, `docs/`, `LICENSE`
(Apache-2.0), `README.md`, with "Implementation — Coming soon." No weights.
Verified mechanism, and it matches this repo's `fisheye3r/README.md` reconstruction on every
number: **K = 8 calibration tokens per layer**, inserted into every image-encoder layer and every
alternating-attention (frame-wise and global) block **except the first `L₀ = 12`**, initialized
`N(0, 1e-6)`; **insert-then-drop** — tokens are concatenated with image tokens at layer `ℓ`, take
part in attention, then discarded, which "localizes the latent calibration effect to each layer."
Mixed perspective/fisheye batches are handled by a **linear (logistic-regression) camera-type
classifier** on the `L₀`-layer class token, `M_s = 𝕀(ψ(x_{s,0}^{(L₀)}) > 0.5)`, plus binary
attention masks `M_{F,s}` / `M_G` that let calibration tokens influence only fisheye tokens.
Three schemes: **SSL** `L = ℒ(f(I^p); T⁻¹ ∘ f(T(I^p), φ))` — unlabeled perspective images only,
distortion `T` synthesizes fisheye and the *unadapted* model is the teacher; **SL** — perspective
GT through the same synthesis; **SL+** — real fisheye GT directly. Cost: **~344 k** trainable
params vs **1.23 B** frozen (Table 4), AdamW 1e-5 → 1e-7, **40 k iters, ~20 h on 4 GPUs**, ~35 GiB
peak. Adapts **VGGT, π³ and MapAnything**; **claims** consistent gains in pose, depth, point map
and **FoV** estimation.
*Why it matters:* the other anchor. The mechanism is orthogonal to RayTun3R's (tokens vs PE
residuals), which is exactly why RayTun3R's ablation beating a "CalTok" baseline is informative and
why the two could be **stacked**. Its **SSL** scheme is the standout practical property: it needs
**no fisheye data and no GT at all**, only unlabeled perspective images plus your KB4 forward model
— and this repo already has `fisheye3r/distortion.py`.

**OmniVGGT** — see §3.2. Belongs to both families: an adapter, but a 26.8 M one trained for 10 days.

**LoRA3D: Low-Rank Self-Calibration of 3D Geometric Foundation Models** — ICLR 2025.
[arXiv:2412.07746](https://arxiv.org/abs/2412.07746) · project
[520xyxyzq.github.io/lora3d](https://520xyxyzq.github.io/lora3d/) · OpenReview
[LSp4KBhAom](https://openreview.net/forum?id=LSp4KBhAom).
Specializes a pretrained DUSt3R-family model **to the target scene using its own multi-view
predictions**: robust global optimization aligns sparse-view predictions, prediction confidence is
**recalibrated** to better reflect actual point accuracy, and the recalibrated confidence gates
**pseudo-labels** for a **LoRA** finetune. No external priors, no manual labels. **Claims** up to
88% improvement over 160+ scenes (Replica, TUM, Waymo); **5 minutes on one GPU**, **18 MB** per
adapter.
*Why it matters:* the self-supervision engine that pairs with everything else here. VGGT-360's
overlapping tangent views already give you the multi-view consistency signal LoRA3D consumes — so
a per-sequence ADT adapter could be trained with **no GT depth at all**. Same "short segment, own
predictions, tiny adapter" shape as RayTun3R, but with LoRA instead of PE residuals.

**RePer-360: Releasing Perspective Priors for 360° Depth Estimation via Self-Modulation** —
Cheng Guan, Chunyu Lin, Zhijie Shen, Junsong Zhang, Jiyuan Wang. 2026-03-06.
[arXiv:2603.05999](https://arxiv.org/abs/2603.05999).
Adapts a depth **foundation model** to panoramas while preserving pretrained perspective priors: a
lightweight **geometry-aligned guidance module** derives a modulation signal from two
complementary projections (**ERP** and **CP**/cubemap), and a **Self-Conditioned AdaLN-Zero**
mechanism emits **pixel-wise scaling factors** that close the perspective→panorama feature
distribution gap. **Claims** it beats standard finetuning using **1% of the training data**, and
~20% RMSE improvement in the same in-domain setting.
*Why it matters:* AdaLN-**Zero** is the same zero-initialized-gate trick as OmniVGGT's zero-init
convs and LLaMA-Adapter's zero gating — identity at init, so it cannot damage the frozen prior.
The "1% of data" result is the strongest evidence in this survey that **feature modulation
conditioned on the projection** is a high-return, low-data intervention. Applied to a
DAv2-style model, i.e. directly transferable to the DAv2 half of this project.

**LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention** — Zhang, Han,
Liu, Gao, Zhou, Hu, Yan, Lu, Li, Qiao. ICLR 2024.
[arXiv:2303.16199](https://arxiv.org/abs/2303.16199).
Learnable adaption prompts prepended at higher transformer layers, injected through a
**zero-initialized attention with zero gating** so new cues enter gradually and pretrained
knowledge is preserved. **1.2 M** trainable params on a frozen 7 B model, <1 h on 8×A100.
*Why it matters:* the origin of the zero-init gate that OmniVGGT, RePer-360 and RayTun3R all rely
on. Worth citing when you justify why an adapter initialized to identity is safe to bolt onto
VGGT-Ω.

**Visual Prompt Tuning** — Jia, Tang et al. ECCV 2022.
[arXiv:2203.12119](https://arxiv.org/abs/2203.12119) ·
code [github.com/KMnP/vpt](https://github.com/KMnP/vpt).
<1% of parameters, trainable only in the **input space**, backbone frozen; often beats full
finetuning.
*Why it matters:* the formal ancestor of Fisheye3R's calibration tokens; the reference for
"learned tokens in input space are enough."

**PPEA-Depth** ([arXiv:2312.13066](https://arxiv.org/abs/2312.13066)) and **ER-LoRA: Effective-Rank
Guided Adaptation for Weather-Generalized Depth Estimation**
([arXiv:2509.00665](https://arxiv.org/abs/2509.00665)) — PEFT specifically for **dense depth
regression**, the former progressive over encoder *and* decoder, the latter a
Selecting–Tuning–Maintaining strategy using effective rank to pick what to adapt.
*Why it matters:* evidence that PEFT transfers to dense regression rather than just
classification, and ER-LoRA's rank criterion is a principled way to choose *which* VGGT blocks
deserve the adapter instead of guessing.

**FiT / FiTv2: Flexible Vision Transformer** —
[arXiv:2402.12376](https://arxiv.org/abs/2402.12376) /
[arXiv:2410.13925](https://arxiv.org/abs/2410.13925), and **RoPE-ViT: Rotary Position Embedding for
Vision Transformer** ([arXiv:2403.13298](https://arxiv.org/abs/2403.13298)).
FiT treats images as variable-length token sequences with masked MHSA and **2D RoPE**, and shows
that naive LLM length-extrapolation transfers poorly — hence **VisionNTK / VisionYaRN**,
interpolation schemes tailored to 2D RoPE, for unseen resolutions and aspect ratios.
*Why it matters:* if you feed VGGT tangent crops at non-native resolution or aspect, the PE
resampling scheme is a confound sitting on top of the fisheye problem. FiT is the reference for
doing it deliberately, and VisionNTK-style frequency rescaling is a **zero-parameter** knob in the
same place RayTun3R puts its learned residual.

### 3.4 Positional encoding and attention surgery for distortion

**FishRoPE: Projective Rotary Position Embeddings for Omnidirectional Visual Perception** —
Qualcomm. [arXiv:2604.10391](https://arxiv.org/abs/2604.10391). No code URL found.
Verified mechanism. Pixels are mapped to angles by **inverse Kannala–Brandt**:
`r = √((u−c_x)² + (v−c_y)²)`, `θ = r⁻¹_KB(r)` by polynomial inversion,
`φ = atan2(v−c_y, u−c_x)`. The embedding dimension is split in half — `d/2` for `θ`, `d/2` for
`φ` — each with standard RoPE rotations `R(θ·ω)`, `R(φ·ω)`, so attention logits depend on
**angular separation instead of pixel distance**, in both encoder self-attention and
BEVFormer-style cross-attention. Frozen **DINOv2 ViT-B/14** with **LoRA r=16, α=32 on query and
value projections**, ≈**3 M** trainable params on 86 M frozen; features from layers 3/6/9/12.
Degrades gracefully: as `θ → 0`, `r(θ) ≈ k₁θ` and FishRoPE becomes a scaled 2D RoPE — but the
paper is explicit this equivalence "does not extend to large incidence angles." **Not
training-free.** Evaluated on WoodScape 2D detection (**claims** 54.3 mAP) and SynWoodScapes BEV
segmentation (65.1 mIoU) — **depth is not evaluated**.
*Why it matters:* the exact KB4 → RoPE construction you would need, spelled out, using the same
lens model as Aria. It is the *analytic* counterpart to RayTun3R's *learned* radial RoPE residual —
and RayTun3R's ablation (RoPE-only performs poorly) is a warning that RoPE surgery alone may not
be enough for geometry tasks. X-Lens adopting FishRoPE for depth is the counter-evidence.

**SpheRoPE: Zero-Shot Optimization-Free 360° Panorama Generation with Spherical RoPE** —
Or Hirschorn, Aaron Olender, Eli Alshan, Ianir Ideses, Lior Fritz, Sagie Benaim. 2026-06-30.
[arXiv:2606.32033](https://arxiv.org/abs/2606.32033).
Injects spherical priors into **frozen pretrained** diffusion transformers, **training-free and
optimization-free**, by re-parameterizing RoPE's **low-frequency channels as 3D Cartesian
coordinates** on the sphere while **harmonically quantizing high-frequency channels** to enforce
exact periodicity; paired with Semantic-Distortion classifier-free guidance. Inherits Flux.1,
Flux.2, LTX-Video.
*Why it matters:* the existence proof that **spherical RoPE surgery can be entirely training-free
on a frozen transformer**. The low/high-frequency split is the transferable insight: low
frequencies carry the geometry worth re-parameterizing, high frequencies mainly need to stay
periodic. Domain is generation, not depth — so treat as mechanism, not evidence.

**PanoFormer: Panorama Transformer for Indoor 360° Depth Estimation** — Shen et al. ECCV 2022.
[arXiv:2203.09283](https://arxiv.org/abs/2203.09283) ·
code [github.com/zhijieshen-bjtu/PanoFormer](https://github.com/zhijieshen-bjtu/PanoFormer).
Divides patches **on the spherical tangent domain** so tokens are minimally distorted, and adds a
**learnable token flow** to the self-attention module (deformable-attention style) plus
panorama-specific metrics.
*Why it matters:* "tangent-domain tokens" is architecturally the same move as tangent-view
rendering, but *inside* the tokenizer instead of as a preprocessing step — a middle path between
family (a) and family (c). Closely related to RayTun3R's parameter-free **patch resampling by
local linearization**.

**EGformer: Equirectangular Geometry-biased Transformer for 360 Depth Estimation** — Yun et al.
ICCV 2023. [arXiv:2304.07803](https://arxiv.org/abs/2304.07803).
Rather than trying to remove distortion, it **uses equirectangular geometry as an explicit bias for
local attention**, extracting geometry-aware local attention with a large receptive field at low
parameter/compute cost.
*Why it matters:* the "bias the attention with the known geometry" pattern, which is also what
VGGT-360's log-confidence bias and X-Lens's Jacobian bias do. For a frozen model, an additive
geometric bias on attention logits is the cheapest intervention that exists — no parameters.

**SGFormer: Spherical Geometry Transformer for 360 Depth Estimation** — Junsong Zhang, Zisong Chen,
Chunyu Lin, Lang Nie, Zhijie Shen, Junda Huang, Yao Zhao.
[arXiv:2404.14979](https://arxiv.org/abs/2404.14979).
Integrates spherical priors into ViTs and retargets the decoder into a **spherical prior decoder
(SPDecoder)** using **bipolar re-projection**, **circular rotation** and **curve local embedding**
to preserve equidistortion, continuity and surface distance respectively.
*Why it matters:* decoder-side geometric priors — relevant because VGGT's DPT head also carries a
2D prediction grid, the exact object RayTun3R corrects.

**Sector Patch Embedding (SPE): An Embedding Module Conforming to The Distortion Pattern of Fisheye
Image** — Dianyi Yang, Jiadong Tang, Yu Gao, Yi Yang, Mengyin Fu (BIT). 2023-03-26.
[arXiv:2303.14645](https://arxiv.org/abs/2303.14645).
Samples **circular, sector-shaped patches aligned to the fisheye distortion pattern** and encodes
position via **learnable polar coordinates**. **Claims** +0.75% / +2.8% top-1 for ViT / PVT.
*Why it matters:* replaces the square-grid patchify assumption with a polar one. Note the strong
family resemblance to RayTun3R's polar-binned PE residual — polar coordinates around the principal
point are the natural chart for a fisheye, whether you use them for sampling or for indexing PEs.

**SphereNet: Learning Spherical Representations for Detection and Classification in Omnidirectional
Images** — Coors, Condurache, Geiger. ECCV 2018
([open access](https://openaccess.thecvf.com/content_ECCV_2018/html/Benjamin_Coors_SphereNet_Learning_Spherical_ECCV_2018_paper.html)).
Adapts the **sampling locations** of convolutional filters to reverse distortion and wraps filters
around the sphere; because it builds on regular convolutions, it **enables transfer of existing
perspective CNN models** to the omnidirectional case.
*Why it matters:* the earliest clean statement of this project's thesis — change *where* you
sample, keep the pretrained weights. Also **Gauge Equivariant Convolutional Networks and the
Icosahedral CNN**, Cohen et al. ([arXiv:1902.04615](https://arxiv.org/abs/1902.04615)), for the
principled equivariant treatment; both are CNN-era and cannot reuse ViT weights, which is the
survey's stated reason sphere-native methods did not scale
([arXiv:2606.27745](https://arxiv.org/abs/2606.27745)).

### 3.5 Fisheye-specific depth estimation

**FisheyeDistanceNet** — Ravi Kumar et al. ICRA 2020.
[arXiv:1910.04076](https://arxiv.org/abs/1910.04076) ·
[github.com/rvarun7777/FisheyeDistanceNet](https://github.com/rvarun7777/FisheyeDistanceNet).
Self-supervised, **scale-aware Euclidean distance** and ego-motion from **raw** monocular fisheye
video, **without rectification**.
*Why it matters:* the origin of the "predict Euclidean **range**, not planar z, on a fisheye"
convention that `CONTEXT.md` already treats as load-bearing here. Independent support for the
repo's scoring-domain discipline.

**SynDistNet** — Ravi Kumar, Klingner et al. WACV 2021.
[arXiv:2008.04017](https://arxiv.org/abs/2008.04017).
Multi-task: jointly learns semantic segmentation and uses its prediction to **guide** self-supervised
distance estimation (semantic masking of dynamic objects), for both fisheye and pinhole.

**SVDistNet: Self-Supervised Near-Field Distance Estimation on Surround View Fisheye Cameras** —
Ravi Kumar, Klingner, Yogamani, Bach, Milz, Fingscheidt, Mäder. IEEE T-ITS.
[arXiv:2104.04420](https://arxiv.org/abs/2104.04420).
**The mechanism to note:** "camera-geometry adaptive multi-scale convolutions which utilize the
**camera parameters as a conditional input**," letting one model generalize across different
fisheye cameras **without retraining per variant**, plus pairwise/patchwise vector self-attention.
*Why it matters:* the earliest instance in this survey of *conditioning on intrinsics to absorb
lens variation* — the pre-foundation-model ancestor of Pow3R/MapAnything ray maps and OmniVGGT's
GeoAdapter. Good evidence that camera conditioning solves *cross-lens* generalization specifically.

**OmniDet: Surround View Cameras based Multi-task Visual Perception Network for Autonomous Driving**
— Ravi Kumar, Yogamani et al. IEEE RA-L 2021.
[arXiv:2102.07448](https://arxiv.org/abs/2102.07448) ·
code [github.com/valeoai/WoodScape/tree/master/omnidet](https://github.com/valeoai/WoodScape/tree/master/omnidet).
Six tasks on **unrectified** fisheye with a shared encoder and synergized decoders, and "a novel
camera geometry based adaptation mechanism [that] encodes the fisheye distortion model **both at
training and inference**."
*Why it matters:* the WoodScape/OmniDet line is where fisheye-native dense prediction was worked
out; it is also the evaluation ground FishRoPE and X-Lens use, so it is the bridge between the old
fisheye literature and the 2026 adapter papers.

### 3.6 Feed-forward 3D foundation models, and which are camera-parameter-free

The property that matters here: whether the model has an internal camera estimate that its depth
can be *wrong through*. All entries verified for existence, authors and code.

| Model | arXiv | Code | Camera-parameter-free by design? |
|---|---|---|---|
| **DUSt3R** | [2312.14132](https://arxiv.org/abs/2312.14132) | [naver/dust3r](https://github.com/naver/dust3r) | **Yes.** Regresses pointmaps "without prior information about camera calibration nor viewpoint poses," explicitly "relaxing the hard constraints of usual projective camera models." |
| **MASt3R** | [2406.09756](https://arxiv.org/abs/2406.09756) | [naver/mast3r](https://github.com/naver/mast3r) | Yes — DUSt3R + a dense local-feature head and matching loss. |
| **Spann3R** | [2408.16061](https://arxiv.org/abs/2408.16061) | [HengyiWang/spann3r](https://github.com/HengyiWang/spann3r) | Yes — "without any prior knowledge of the scene or camera parameters"; external spatial memory puts pointmaps in a global frame. 3DV 2025. |
| **Fast3R** | [2501.13928](https://arxiv.org/abs/2501.13928) | project [opencv.org/fast3r](https://opencv.org/fast3r/) | Yes — N images in one forward pass, no global alignment. CVPR 2025. |
| **CUT3R** | [2501.12387](https://arxiv.org/abs/2501.12387) | [CUT3R/CUT3R](https://github.com/CUT3R/CUT3R) | Yes — recurrent persistent state, online metric-scale pointmaps in a common frame. |
| **MoGe** | [2410.19115](https://arxiv.org/abs/2410.19115) | project (CVPR 2025) | **Yes, and pointedly so.** Predicts an **affine-invariant** pointmap, "agnostic to true global scale and shift"; camera shift, **focal length** and depth are *derived from* the pointmap afterwards — the inverse of VGGT's dependency direction. |
| **MoGe-2** | [2507.02546](https://arxiv.org/abs/2507.02546) | NeurIPS 2025, [OpenReview](https://openreview.net/forum?id=16mDq7m2OK) | Extends MoGe to metric scale without losing the affine-invariant relative geometry; real-data refinement using sharp synthetic labels. |
| **π³** | [2507.13347](https://arxiv.org/abs/2507.13347) | [yyfz/Pi3](https://github.com/yyfz/Pi3) | **Yes.** Fully **permutation-equivariant**, no reference view; predicts affine-invariant poses and scale-invariant **local** pointmaps. ICLR 2026. |
| **VGGT** | [2503.11651](https://arxiv.org/abs/2503.11651) | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | **No.** Appends **camera tokens**; a **camera head** predicts extrinsics *and intrinsics*; a DPT head gives dense output. This is the coupling. |
| **VGGT-Ω** | [2605.15195](https://arxiv.org/abs/2605.15195) | this repo | **No** — same 9-D pose encoding (`vggt_omega/utils/pose_enc.py`). CVPR 2026 Oral; ~70% less GPU memory, **single dense prediction head with multi-task supervision**, high-res layers removed. |
| **MapAnything** | [2509.13414](https://arxiv.org/abs/2509.13414) | [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | **Optionally conditioned** — predicts rays but also *accepts* them. See §3.2. |
| **Depth Anything 3** | [2511.10647](https://arxiv.org/abs/2511.10647) | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Works "with or without known camera poses"; a **plain DINO transformer** suffices as backbone and a **single depth-ray target** replaces multi-task learning. **Claims** +44.3% pose / +25.1% geometry over VGGT. |

*Why this table matters:* MoGe and π³ are the natural **controls** for the FoV-coupling
experiment. If the same tangent crop that bends under VGGT stays straight under π³ or MoGe — both
of which derive camera from geometry rather than the reverse — the diagnosis is confirmed from the
architecture side, complementing the DAv2 control you already have. Note that RayTun3R found π³ and
VGGT still degrade on fisheye even though they are camera-parameter-free, which is precisely its
argument that **the positional encoding, not the camera head, is the deeper cause**.

---

## 4. Ideas worth trying here

Ranked by effort × likelihood. Every idea names the paper whose mechanism it borrows.

### Tier 1 — hours of work, high information value

**I1. Metric3D canonical resize on tangent crops (zero parameters, zero training).**
Mechanism: Metric3D's canonical-camera transform **method 2** — resize the input to emulate a
canonical focal, then de-canonicalize the output
([arXiv:2307.10984](https://arxiv.org/abs/2307.10984),
[arXiv:2404.15506](https://arxiv.org/abs/2404.15506)).
Because you *render* your tangent views, their true focal is known. Sweep the resize factor and
watch both `pose_enc[7:9]` and `align%`. If a resize that brings the crop's effective focal to
VGGT's training-typical focal makes the inferred FoV converge on the true one and the bending stop,
you have a free fix and a clean explanation. This is the cheapest experiment in the document and it
directly attacks the measured coupling.

**I2. Override `pose_enc[7:9]` — literally.**
Mechanism: OmniVGGT injects `f ∈ ℝ²` = **field of view** as auxiliary camera tokens through
zero-init convolutions at each AA block, i.e. it treats the same two numbers as an *input*
([arXiv:2511.10560](https://arxiv.org/abs/2511.10560)). Before building any adapter, test the
*downstream* half for free: run VGGT normally, then convert depth → points using the **known render
FoV** instead of the predicted one (your `pose_enc.py:42-43` already isolates this — swap `fov_h`,
`fov_w`). If the bending disappears, the coupling is purely in the depth→points conversion and no
training is needed at all. If it persists, the FoV estimate has contaminated the **features**, and
you need I3/I4. This single experiment cleanly partitions the problem, and it is worth doing before
anything else in Tier 2.

**I3. Reproduce RayTun3R's Jacobian diagnostic on the vendored PEs.**
Mechanism: measure `σ₁` and `det(J_PE^⊤ J_PE)` of the positional-embedding Jacobian against
normalized radius; flat ⇒ pinhole prior present
([arXiv:2607.02711](https://arxiv.org/abs/2607.02711)).
This is a read-only measurement on `vggt_visfeat/layers/vision_transformer.py` and
`layers/attention.py`. It tells you whether to spend effort on PE residuals (RayTun3R) or
calibration tokens (Fisheye3R) — RayTun3R's own ablation says the **absolute-PE residual carries
the largest gain** and RoPE-only underperforms, so this measurement decides where the 10 kB of
parameters should go. It also slots naturally into `checks/depth_probe.py` alongside the existing
backend/view-source swap matrix.

**I4. Control experiment with camera-parameter-free backbones.**
Mechanism: MoGe derives focal *from* the pointmap rather than the reverse
([arXiv:2410.19115](https://arxiv.org/abs/2410.19115)); π³ is fully permutation-equivariant with no
reference view ([arXiv:2507.13347](https://arxiv.org/abs/2507.13347), code
[yyfz/Pi3](https://github.com/yyfz/Pi3)). Add them as `depth_probe.py` **backends** next to
`vggt1b` / `vggt_omega` / `official`. Expected outcome per RayTun3R: they degrade too, but
*differently* — pinhole PE bias without FoV coupling. That separates the two mechanisms
experimentally, which no single-model experiment can.

### Tier 2 — days of work, strong expected payoff

**I5. KB4 ray map as the camera prompt (Pow3R format, VGGT injection point).**
Mechanism: Pow3R patchifies the dense ray map `K⁻¹[i,j,1]` exactly like RGB and injects it via
per-block MLPs ([arXiv:2503.17316](https://arxiv.org/abs/2503.17316), code
[naver/pow3r](https://github.com/naver/pow3r)); MapAnything does the same with a pixel-unshuffle-14
conv encoder summed into DINOv2 patch features
([arXiv:2509.13414](https://arxiv.org/abs/2509.13414)).
Substitute your **KB4 unprojection** (`VGGT-360-fisheye/utils/fisheye_cam.py`) for `K⁻¹` and the
ray map now describes an Aria fisheye — the tensor shape is unchanged, so a shallow conv (stride 14
to match VGGT's patch grid) plus a **zero-initialized** output conv is a drop-in adapter that is the
identity at init (LLaMA-Adapter zero gating,
[arXiv:2303.16199](https://arxiv.org/abs/2303.16199); OmniVGGT's `𝒵𝒞_l`).
This is the highest-ceiling idea here: it lets the model see the *actual* lens rather than a
rectified approximation, and it removes the need for tangent crops entirely. Verified caveat:
Pow3R never tested non-pinhole intrinsics, and MapAnything only **claims** fisheye is
representable "with appropriate training" — so this is mechanism-grounded but not
evidence-grounded, and needs its own training run.

**I6. Fisheye3R SSL on VGGT-Ω with no fisheye data.**
Mechanism: `L_SSL = ℒ(f(I^p); T⁻¹ ∘ f(T(I^p), φ))` — synthesize fisheye with `T`, supervise the
adapted model against the **unadapted** model's perspective prediction pulled back through `T⁻¹`
([arXiv:2603.28896](https://arxiv.org/abs/2603.28896)). ~344 k params, 40 k iters, ~20 h on 4 GPUs.
This repo already has `fisheye3r/train.py --scheme ssl` and `fisheye3r/distortion.py`, so the
remaining work is a corpus of unlabeled perspective images and GPU time. It needs **no ADT GT and
no fisheye data**, which makes it the lowest-risk *training* option. Note the official code is
still a placeholder (verified), so your reproduction is currently the only implementation — worth
saying out loud in any writeup.

**I7. RayTun3R-style polar PE residual, and stack it with I6.**
Mechanism: `P′(u,v) = P_A(u,v) + t_r(ρ) + ρ·δ_θ(θ)` with 20 radial + 8 angular bins, plus
`ω′ = ω + Δ_r(ρ)` with 20 shared radial RoPE bins ≈ **10,752** params for `C = 384`; zero-init;
losses are reprojection + pose-vs-MAGSAC++ + edge-aware smoothness + TV/L2 on the corrections
([arXiv:2607.02711](https://arxiv.org/abs/2607.02711)). Adaptation set: **30 three-frame windows**.
Two reasons this is attractive for ADT specifically: ADT sequences are exactly the "short temporal
segment" it needs, and the mechanism is **orthogonal to Fisheye3R's tokens** — RayTun3R beats a
calibration-token baseline, so stacking PE residuals *and* tokens is unexplored and cheap. Also
adopt its two free pieces immediately: **mean-valid-token substitution outside the lens circle**
(you already have `analytic validity` masks) and **patch resampling by local linearization** of the
KB4 map at each patch centre.

**I8. Zero-parameter geometric attention bias.**
Mechanism: three papers add a bias to attention logits rather than learning anything — VGGT-360's
`softmax(QKᵀ/√d + log M_s)` ([arXiv:2603.18943](https://arxiv.org/abs/2603.18943)), EGformer's
equirectangular geometry bias ([arXiv:2304.07803](https://arxiv.org/abs/2304.07803)), and X-Lens's
**Jacobian distortion bias** in cross-attention
([arXiv:2607.12993](https://arxiv.org/abs/2607.12993)).
The KB4 local Jacobian is analytic, so a bias of the form `−λ · d_angular(i,j)` or a
Jacobian-derived term costs **zero parameters** and no training. Given you already modify
`layers/attention.py` for VGGT-360's saliency bias, this is a small extension to code you own.
Cheapest of the Tier-2 items; ceiling unknown for depth (FishRoPE reports no depth numbers,
X-Lens's bias is entangled with two other changes).

**I9. LoRA3D-style self-calibration using your own overlapping tangent views.**
Mechanism: recalibrate prediction confidence to reflect true point accuracy, gate pseudo-labels
with it, finetune a **LoRA** — 5 minutes on one GPU, 18 MB per adapter, no external priors or
labels ([arXiv:2412.07746](https://arxiv.org/abs/2412.07746)).
VGGT-360's overlapping views plus your `correlation-weighted` fusion already produce exactly the
multi-view consistency signal and confidence that LoRA3D consumes. This yields a **per-sequence ADT
adapter with no GT depth**, and it composes with I7 (LoRA3D supplies the labels, RayTun3R supplies
the parameterization).

### Tier 3 — worth knowing, higher effort or lower certainty

**I10. AdaLN-Zero self-modulation for the DAv2 half.** RePer-360 derives a modulation signal from
two complementary projections (ERP + cubemap) and emits pixel-wise scaling factors through a
**Self-Conditioned AdaLN-Zero**, **claiming** it beats full finetuning with **1% of the data**
([arXiv:2603.05999](https://arxiv.org/abs/2603.05999)). For fisheye, substitute your
`rectifier`/`tangent`/`raw_roi` view sources as the "complementary projections" — the probe
machinery is already there. Targets DAv2, which is the model in this project that *works* but is
only relatively-scaled; this is the route to making it panorama/fisheye-native cheaply.

**I11. Analytic FishRoPE for the vendored RoPE.** Replace pixel-grid RoPE with
`θ = r⁻¹_KB(r)`, `φ = atan2(...)`, splitting `d/2` + `d/2`
([arXiv:2604.10391](https://arxiv.org/abs/2604.10391)); and consider SpheRoPE's split —
re-parameterize **low-frequency** channels geometrically, **harmonically quantize** high-frequency
ones — which is fully **training-free on a frozen transformer**
([arXiv:2606.32033](https://arxiv.org/abs/2606.32033)). Tempered by RayTun3R's finding that
RoPE-only correction underperforms; treat as a component of I7, not a standalone fix.

**I12. Drop the camera head when calibration is known.** X-Lens **deliberately does not predict
pose or intrinsics** because deployment assumes a calibrated rig, and outputs a factored
`(normalized depth, confidence, global scale)` ([arXiv:2607.12993](https://arxiv.org/abs/2607.12993)).
MapAnything's factored `(rays, depth-along-ray, pose, scale)` is the same idea
([arXiv:2509.13414](https://arxiv.org/abs/2509.13414)). If I2 shows the FoV estimate has
contaminated the features, the principled fix is to stop asking VGGT for a camera at all and use
the factored form with your KB4 rays fixed. Architecturally invasive but the correct end state for
a calibrated Aria rig.

**I13. Fuse-back upgrades for the existing port.** 360MonoDepth's **deformable multi-scale alignment
+ gradient-domain blending** ([arXiv:2111.15669](https://arxiv.org/abs/2111.15669)) and
PatchFusion's **Consistency-Aware Training/Inference** on overlap regions
([arXiv:2312.02284](https://arxiv.org/abs/2312.02284)) are both stronger than confidence-weighted
averaging. Independent of the FoV problem — do this only if seams, not bending, dominate your error.

**I14. Instrument, don't assume, the camera estimate.** Run **AnyCalib**
([arXiv:2503.12701](https://arxiv.org/abs/2503.12701), code
[javrtg/AnyCalib](https://github.com/javrtg/AnyCalib), Apache-2.0) on the same crops: it regresses
a per-pixel ray field and recovers **Kannala–Brandt** intrinsics in closed form. That gives an
independent third number next to VGGT's `pose_enc[7:9]` and the known render FoV, and it is what
RayTun3R uses when true calibration is unavailable. Low effort; belongs in `checks/`.

**I15. ER-LoRA's effective-rank criterion for choosing where to adapt.** If you go the
LoRA/adapter route, ER-LoRA's Selecting–Tuning–Maintaining strategy uses effective rank to decide
which layers are task-relevant ([arXiv:2509.00665](https://arxiv.org/abs/2509.00665)) — a
principled alternative to Fisheye3R's hand-set `L₀ = 12` cutoff, which is worth revisiting since
VGGT-Ω's block structure differs from VGGT's.

### Cross-cutting note on the FoV-coupling problem

Three distinct answers exist in the literature, and they are not mutually exclusive:

1. **Fix the number** — override `pose_enc[7:9]` with the known FoV (OmniVGGT's injection format;
   I2), or canonicalize the input so the estimate becomes correct (Metric3D; I1).
2. **Remove the dependency** — factored outputs with rays supplied, no camera head (X-Lens,
   MapAnything; I12); or use a backbone that derives camera from geometry (MoGe, π³; I4).
3. **Fix the actual cause** — if RayTun3R is right, the FoV error is downstream of a **pinhole prior
   in the positional encodings**, and correcting the PEs fixes both
   ([arXiv:2607.02711](https://arxiv.org/abs/2607.02711); I3, I7).

I3 and I2 together decide which of the three you are in, and both are cheap. Do them first.

---

## 5. Unverified / could not confirm

Statements I could **not** trace to a primary source. Treat as open questions, not facts.

- **RayTun3R code.** The paper says "our code will be made publicly available" but gives no URL, and
  I found no repository. Reimplementation is currently the only path.
- **FishRoPE code.** No repository URL in the paper text I read
  ([arXiv:2604.10391](https://arxiv.org/abs/2604.10391)); the Qualcomm affiliation came from the
  paper HTML, and I did not confirm the full author list.
- **UniK3D with ground-truth camera at inference.** I could not confirm whether the spherical-harmonic
  camera module can be *prompted* with a known ray field instead of its own prediction. This matters
  for idea I5 and should be checked directly in
  [lpiccinelli-eth/UniK3D](https://github.com/lpiccinelli-eth/UniK3D). Its **backbone** is likewise
  unconfirmed — the abstract page does not name it.
- **Depth Anything 3 intrinsics input.** Verified it works "with or without known camera **poses**";
  I did **not** confirm it accepts known **intrinsics**. Also did not verify what the
  "depth-ray prediction target" is concretely.
- **Depth Pro's focal head on fisheye-derived crops.** Its focal estimation is **claimed**
  state-of-the-art on ordinary images; nothing verified about narrow crops cut from a fisheye
  frame, which is the case that matters for I1/I14.
- **Whether RayTun3R's or Fisheye3R's numbers reproduce.** Both are reported as published; I did not
  reproduce anything, and the two papers **do not compare against each other** (verified — RayTun3R
  cites Fisheye3R only as contemporaneous work in related work).
- **VGGT-360 CVPR camera-of-record vs arXiv.** The arXiv abstract page gives no venue field, and I
  did not open the CVPR 2026 proceedings PDF (403 on direct fetch); the CVPR designation comes from
  a search-result title for
  `openaccess.thecvf.com/content/CVPR2026/papers/Yuan_VGGT-360_...pdf`, not from a fetched page.
  Same caveat for Fisheye3R's "ECCV 2026" (from the arXiv comments field and the placeholder repo
  README) and OmniVGGT's "CVPR 2026 Highlight" (from its GitHub repo title).
- **DAC's exact ERP-conversion source files.** The repo listing (`dac/`, `configs/`, `scripts/`)
  did not surface the specific module implementing image→ERP / FoV alignment. The architecture is
  named `IDiscERP`, "our modified version of the IDisc model, incorporating isolated image and
  positional encoding features" — the actual file needs a local clone to pin down. Note this repo
  already vendors it at `third_party/depth_any_camera/`.
- **PaGeR's training budget and whether any part is frozen.** Confirmed backbone (DA3 da3-giant),
  cubemap resolution (6×504×504) and outputs; not the training cost or which weights move.
- **"Focusable Monocular Depth Estimation"** ([arXiv:2605.11756](https://arxiv.org/abs/2605.11756))
  surfaced in a search for focal-length conditioning, but on inspection it is **region/prompt**
  conditioning (box/text prompts, SAM3 features aligned to Depth Anything), **not** focal-length
  conditioning. Listed here so nobody re-chases it.
- **Depth Any Panoramas** ([arXiv:2512.16913](https://arxiv.org/abs/2512.16913), Insta360, DINOv3-L)
  and **DA360** ([arXiv:2512.22819](https://arxiv.org/abs/2512.22819), code
  [Insta360-Research-Team/DA360](https://github.com/Insta360-Research-Team/DA360)) are verified to
  exist and are panorama-focused, but both are **trained** panoramic models rather than
  minimal-adaptation methods, so I did not read their method sections. DA360's stated mechanism —
  learn a **shift** parameter off the ViT backbone to turn scale-and-shift-invariant output into
  scale-invariant, plus **circular padding** in the DPT decoder — is a small-parameter idea that may
  deserve a second look for the DAv2 half.
- **DepthMaster** ([arXiv:2606.12368](https://arxiv.org/abs/2606.12368), "Unified Monocular Depth
  Estimation for Perspective and Panoramic Images") appeared in search results with a plausible
  title but I did not fetch it. Unverified.
- **`pose_enc` layout** is the one claim here verified from *local* source rather than a paper:
  `vggt_omega/utils/pose_enc.py` lines 16, 24-26, 32-43 give
  `[translation(3), quaternion(4), fov_h, fov_w]` with
  `fy = (H/2)/tan(fov_h/2)`, `fx = (W/2)/tan(fov_w/2)`. The official VGGT README does **not**
  document this layout (verified — it only exposes `pose_encoding_to_extri_intri`).
