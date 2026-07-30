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

## VGGT-Omega input contract and the scale/FoV question

Written to explain the center-view sweep on one ADT Aria frame (tangent crops peak in edge
alignment near FoV 100 and fall off on both sides; `raw_roi` is worse than tangent everywhere;
the ~84.5° rectifier view is best). Everything below is from the VGGT-Ω paper
([arXiv:2605.15195](https://arxiv.org/abs/2605.15195), CVPR 2026 Oral), the official repo
[facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega), or the vendored
source in this repo — which is a byte-identical clone of upstream for the files quoted here
(verified: `vggt_omega/utils/load_fn.py` at `39a0cb8 Initial commit`, never touched since, and
matches
[raw.githubusercontent.com/.../vggt_omega/utils/load_fn.py](https://raw.githubusercontent.com/facebookresearch/vggt-omega/main/vggt_omega/utils/load_fn.py)).

### The documented input contract

There are exactly **three** places the contract is written down. The Hugging Face model card
([huggingface.co/facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega)) is gated and
carries **no** technical content — no resolution, no preprocessing, no training-data or camera
statement (verified by fetching it). The project page
([vggt-omega.github.io](https://vggt-omega.github.io/)) likewise documents no input constraints
(verified). So:

**(1) The checkpoint table** (`README.md` lines 30-33) names resolution as a *property of the
weights*, not a runtime flag: `VGGT-Omega-1B-512` → Resolution **512**; the text-alignment
checkpoint is a separate **256** model. The README's own benchmark (lines 119-123) states that
`mode="balanced"` with `image_resolution=512` on 3:2 landscape images "produces 624x416 inputs" —
i.e. **the model is not fed 512×512; it is fed whatever shape carries the 512² token budget.**

**(2) `vggt_omega/utils/load_fn.py`** — the actual contract. Signature and docstring, lines 15-21:

```python
def load_and_preprocess_images(image_path_list, mode="balanced", image_resolution=512, patch_size=16):
    """Load images for VGGT-Omega inference.

    `balanced` keeps the total token count close to image_resolution**2.
```

The docstring is loose; the code is exact. Line 86 defines the budget as a **token count**, not a
pixel count:

```python
token_number = (image_resolution // patch_size) ** 2     # = (512//16)**2 = 1024
w_patches = np.sqrt(token_number / aspect_ratio)
h_patches = token_number / w_patches
```

1024 tokens × 16² px = **262,144 px = 512² of image area**, distributed to match the aspect ratio
and snapped to whole patches. Two hard constraints are enforced (lines 30-31):
`image_resolution % patch_size != 0` raises, and extreme aspect ratios are **center-cropped** into
`[0.5, 2.0]` before anything else (line 68, `_crop_to_supported_aspect_ratio`). The resize itself is
`Image.Resampling.BICUBIC` (line 47).

**(3) Paper §4.1, Implementation Details** — the reason the loader is shaped that way:

> "We augment images by randomly varying the aspect ratio within [0.33,1.33], keeping the image area
> approximately 512 × 512 pixels, and applying color jittering, grayscale conversion, and random
> patch masking."

So `mode="balanced"` is **not a heuristic — it reproduces the training-time area budget exactly**
(262,144 px, both at train and at test). This is the single most important sentence in the paper for
our purposes: the invariant VGGT-Ω was trained under is **constant image *area* with varying aspect
ratio**, not constant side length.

**Patch size and backbone.** §4.1: variants are 200M/500M/1B/10B with 12/12/24/16
alternating-attention blocks and hidden sizes 384/768/1024/4096; "The vision transformer is
initialized from DINOv3 [147] and is not frozen during training." Patch size 16 is confirmed in the
efficiency section: "512 × 336 for VGGT-Ω with 16-pixel patches", against "518 × 336 for VGGT and
DA3 with 14-pixel patches". So the local defaults (`image_resolution=512`, `patch_size=16`,
`mode="balanced"`) are **correct and match upstream**; nothing to fix there.

**Minimum/maximum resolution: not documented, and structurally unnecessary.** There is no stated min
or max and no resolution ablation anywhere in the paper (§4.3 ablates model size, data size, register
attention, multi-task losses, and self-supervision — *not* resolution, and *not* out-of-distribution
cameras). Structurally there is no learned absolute position embedding to break: the backbone uses
**RoPE on normalized coordinates**, `vggt_omega/models/layers/rope_position_encoding.py` lines
70-73 and 85:

```python
if self.normalize_coords == "max":
    max_HW = max(H, W)
    coords_h = torch.arange(0.5, H, **dd) / max_HW
    coords_w = torch.arange(0.5, W, **dd) / max_HW
...
coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]
```

and `vggt_omega/models/aggregator.py` line 40 / lines 226-228 both set `normalize_coords="max"`.
The model even guards this: `vggt_omega/models/vggt_omega.py` lines 80-89 warn
`"the released VGGT-Omega checkpoint was trained with 'max'"` if it is anything else. This is
DINOv3's scheme, described in its §"Updated Model Architecture"
([arXiv:2508.10104](https://arxiv.org/abs/2508.10104)):

> "our base implementation assigns coordinates in a normalized [−1, 1] box to each patch... In order
> to improve the robustness of the model to resolutions, scales and aspect ratios, we employ
> RoPE-box jittering. The coordinate box [−1, 1] is randomly scaled to [−s, s], where s ∈ [0.5, 2]."

and later, explicitly: "our model can seamlessly process images at varying resolutions without
requiring adaptation, thanks to the adoption of Rotary Positional Embeddings (RoPE)."

**A consequence worth internalising:** because coordinates are normalised to [−1,1] *regardless of
token-grid size*, and because the backbone was additionally jittered over a 4× coordinate-scale
range, the positional encoding carries **no information about angular scale**. VGGT-Ω cannot tell a
40° crop from a 120° crop positionally — it must read FoV out of image *content*. That is why the
FoV estimate is a learned prior over appearance, and why it should be expected to regress toward the
training distribution whenever the content is unfamiliar.

**Camera estimation, and whether intrinsics can be supplied.** §3.1:

> "gi = (qi, ti, fi) ∈ R⁹ is the concatenation of the rotation quaternion qi ∈ R⁴, the translation
> vector ti ∈ R³, and the field of view fi ∈ R² ... As is commonly done, we **assume that the
> principal point is at the center of the image**."

Confirmed in `vggt_omega/utils/pose_enc.py` lines 41-49: `fy = (H/2)/tan(fov_h/2)`,
`fx = (W/2)/tan(fov_w/2)`, `cx = W/2`, `cy = H/2`, and **zero distortion parameters**. The head is
"a lightweight transformer to the camera tokens and registers... followed by an MLP on each updated
camera token... predicts camera parameters in a single pass, without iterative refinement" (§3.1.3),
with "a ReLU activation for the focal length" (Appendix A.2).

**There is no supported way to supply known intrinsics.** `vggt_omega/models/vggt_omega.py` line 36
is `def forward(self, images: torch.Tensor)` — images and nothing else. And the depth head does not
consume the camera prediction: `dense_head(aggregated_tokens_list, ...)` and
`camera_head(aggregated_tokens_list, ...)` are **parallel** readouts of the same aggregator. This is
a sharper constraint than it first looks, and it differs from Depth Pro. Depth Pro predicts a
*canonical* inverse depth C and recovers metric depth by an explicit rescale
(§3.2, [arXiv:2410.02073](https://arxiv.org/abs/2410.02073)): "To obtain a dense metric depth map
Dm, we scale by the horizontal field of view, represented by the focal length fpx and the width w:
Dm = fpx/(w·C)", with the FoV head existing only "to handle images that may have inaccurate or
missing EXIF metadata" (§3.3). For Depth Pro, substituting a known focal length **corrects the
depth**. For VGGT-Ω, substituting a known FoV corrects only the *unprojection* — the predicted depth
map is already committed and cannot be repaired that way.

The paper does, however, address this directly, in §5 "Further Insights" under **Auxiliary Inputs**
— this is the most actionable paragraph in the paper for us:

> "Theoretically, incorporating auxiliary inputs, such as temporal order, **camera parameters**,
> depth maps, or scale factors, can further enhance performance. However, we empirically observe
> that introducing these priors during pretraining, even when applied randomly or masked across
> training iterations, is often **detrimental**. Conversely, our preliminary experiments indicate
> that providing conditional auxiliary inputs **exclusively during the fine-tuning phase is highly
> effective**, improving task-specific performance without compromising the integrity of the learned
> representations."

i.e. the authors tried camera conditioning, and their guidance is: inject it at **fine-tune** time,
not pretrain time. That is exactly the regime we are in, and it is a first-party endorsement of the
"condition on known Aria intrinsics during finetuning" plan rather than a test-time override.

### Does feeding 518×518 violate the contract? No — and the model never sees 518

This turns out to be a non-issue, for a concrete reason: **`load_and_preprocess_images` silently
resizes our crops to 512 before the model sees them.** Trace it for a square input:
`aspect_ratio = 518/518 = 1.0` → `token_number = 1024` → `w_patches = sqrt(1024/1.0) = 32`,
`h_patches = 32` → target `(32·16, 32·16) = (512, 512)` → `image.resize((512,512), BICUBIC)`
(line 47). Our tester does go through this path: `VGGT-360-fisheye/checks/depth_probe.py` lines
315/333-334 call the official `vggt_omega.utils.load_fn.load_and_preprocess_images(...,
image_resolution=image_resolution)`, and `center_view_sweep.py` line 127 defaults
`--image-resolution 512`. So the 518 in `--persp-size` is a *rendering* size only; the network is
fed a 512×512, 1024-token image in every row of the sweep table.

What that costs is a **518→512 bicubic downsample, a ratio of 1.0117** — a ~1.2% non-integer
resample. It is a mild low-pass, applied *identically to every row of the sweep*, so it cannot
produce the non-monotonicity we observed. It is not free (a non-integer bicubic resample does
attenuate the top of the spectrum, which is where an edge-alignment metric lives), and the tidy fix
is to render crops at 512 directly and skip the round trip — but it is a second-order effect, not
the explanation.

Two related things that are **not** problems, worth ruling out explicitly:

- **No positional-embedding interpolation happens.** The classic hazard is the ViT one
  ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929), §3.2): "The Vision Transformer can handle
  arbitrary sequence lengths (up to memory constraints), however, the pre-trained position
  embeddings may no longer be meaningful. We therefore perform 2D interpolation of the pre-trained
  position embeddings... Note that this resolution adjustment and patch extraction are the only
  points at which an inductive bias about the 2D structure of the images is manually injected."
  VGGT-Ω has no such embedding to interpolate — see the RoPE code above. (Incidentally the number
  518 traces to that same paper: it is ViT-H/**14**'s fine-tuning resolution, 518/14 = 37 tokens
  across, which is why VGGT-1B uses it. It has no meaning for a patch-16 model — 518/16 = 32.375.)
- **Off-native inference is not automatically lossy, but native-size patching does measurably win on
  boundaries.** Depth Pro's Table 9 is the cleanest primary comparison, because it holds training
  identical and varies only the architecture: a plain **ViT-L DINOv2 run at 1536×1536 with bicubic
  interpolated position embeddings** versus **Depth Pro's 35 patches at the ViT's native 384×384**.
  Metric depth is a wash (NYUv2 δ1 96.5 vs 96.1; iBims δ1 90.3 vs **91.3**), but boundaries are not:
  iBims F1 0.161 → **0.177** (+9.9%) and DIS R 0.065 → **0.080** (+23%), *and* Depth Pro is faster
  (392 → 341 ms). Their text: "our architecture improves the boundary recall by relative 23% over
  DINOv2." Since our `align%` is a boundary metric, this is the most relevant number in this whole
  document: **the native-resolution-patch design buys roughly 10-23% on exactly the quantity we are
  measuring, while barely moving global depth accuracy.**

### What is known about VGGT-Ω's training FoV distribution

The **distribution** is not published. The **bounds on half of the data** are, and they are sharp.

Training data comes in two halves (§3.5). The public half (§3.5.1) lists ~3M sequences from Aria
series, Bedlam, BEHAVIOR-1K, Co3Dv2, uCo3D, DL3DV, Dynamic Replica, EDEN, EFM3D, HOT3D, Habitat,
Hypersim, Mapfree, Mapillary Metropolis, MPSD, Megadepth, Megasynth, Mid-Air, Mvssynth,
ParallelDomain-4D, Replica, SAIL-VOS, ScanNet series, TartanAirV2, TartanGround, Taskonomy,
UnrealStereo4K, Virtual KITTI, Waymo, WildRGBD, plus unspecified internal datasets. **No FoV or
camera-model statement is made about this half.** Note that Aria data *is* in there (Aria series,
HOT3D, EFM3D) — but in whatever projection those releases ship, which for the standard distributions
is rectified/pinhole, not raw fisheye.

The annotated half (§3.5.2, ~40M internet videos → ~800K kept sequences) has explicit filters, and
they are the load-bearing evidence. Under "Reconstruction and filtering":

> "For successful reconstructions, we discard sequences that fail heuristic checks, e.g., an image
> registration ratio < 99.5%, **a field of view outside [30°, 120°]**, or a **distortion ratio >
> 0.1**. These criteria aggressively remove cases with degenerate motion or extreme zoom."

And upstream of that, the VLM pre-filter prompt (Appendix A.3) makes it a **Step 1 hard reject**,
alongside cartoons and corrupted footage:

> "5. Non-Pinhole Projections: Is the footage 360° equirectangular or heavily distorted fisheye
> without calibration?"

Finally, Appendix C, Limitations, states the failure mode in the authors' own words:

> "reconstruction quality often degrades if the **field of view changes abruptly** (e.g., shifting
> from 10° to 160° in a few seconds) **or the camera is highly distorted**... These limitations are
> **primarily attributable to the distribution of our training data**."

**Verdict on the brief's key question.** A 120° input is at the exact upper filter bound — the last
FoV that survived annotation, and therefore the thinnest part of the prior. A 40° input is inside
[30°,120°] but near the lower bound and additionally in "extreme zoom" territory, which the same
sentence says was aggressively removed. Neither is nominally out of distribution; both are at the
edges of it. **Raw, un-undistorted fisheye is unambiguously out of distribution** — rejected twice
over (VLM hard reject; distortion ratio > 0.1) — and, independently of any training statistics, is
**unrepresentable** by the R⁹ pinhole-with-centered-principal-point parameterization in
`pose_enc.py`. That is a complete and sufficient explanation for `raw_roi` being worse than `tangent`
at every FoV, and it needs no appeal to scale at all.

This also predicts the FoV read-outs we saw. The model is biased toward the interior of [30°,120°],
so it should under-report at the top (tangent 120 → 105.8) and under-report worst when distortion is
present (raw_roi 120 → 91.5, a −24% error, versus tangent's −12%). The 84.5° rectifier view sits
mid-distribution and is recovered to 84.0 — a 0.6% error, the best in the table — while also scoring
the best `align%`. Accuracy of the FoV read-out and quality of the depth track together, which is
what you expect if both are governed by proximity to the training prior.

### The angular-resolution hypothesis: the evidence runs against it

The hypothesis is that a 518 px crop at FoV 40 (~13 px/deg, above the sensor's ~11.35 px/deg)
supplies interpolated, information-free detail that hurts dense prediction. Two independent lines of
primary evidence say the sign of that effect is **the opposite**, and the arithmetic of our own
sweep says it is not the driver.

**(a) Upsampling with zero new information measurably *helps* dense depth.** BoostingMonocularDepth
([arXiv:2105.14021](https://arxiv.org/abs/2105.14021)) ran precisely this control, §3 with Fig. 5:

> "We use an original input image of 192 × 192 pixels and simply upsample it to generate higher
> resolution results. This way, **the amount of high-frequency information remains the same in the
> input but we still see an increase in the high-resolution details in the result**, demonstrating a
> limit in the network capacity."

Their conclusion in §7: "upscaling low-resolution images does help in generating more high-frequency
details. Hence, our estimation resolution depends mainly on the **image content and not on the
original input resolution**." The bottleneck they identify is *how much detail the network can emit
per forward pass*, not how much detail the input contains. Interpolated pixels are not inert — they
buy the decoder more output capacity for the same scene.

**(b) The same paper shows the *real* mechanism is context density, and it is non-monotonic.** §3:

> "when these cues in the image gets further apart than the receptive field, the network is not able
> to generate a coherent depth estimation around pixels that do not receive enough information"

and, in the other direction, §5:

> "**Resolutions below the receptive field size do not improve the structure and in fact reduce the
> performance as the full capacity of the network is not utilized.**"

Too little context density under-uses the model; too much overwhelms it. They operationalise this
with an edge map as a proxy for contextual cues, dilate it by a receptive-field-sized kernel, and
define **R₀** as "the maximum resolution where every pixel will receive context information in a
forward pass", with **R₂₀** the resolution leaving 20% of pixels without nearby edges. The quality
curve peaks near R₀-R₂₀ and degrades past it: "beyond (c), the estimates become unstable in terms of
the overall structure" (Fig. 6). **That is the same shape as our FoV sweep** — a hump with a peak
away from the obvious choice — and it is driven by edge/context density, which is exactly what
changes when you vary FoV at a fixed token budget.

Their Table 1 also carries a warning we should heed. Adaptive resolution (Single-est R₀) versus the
original model, MiDaS: on high-resolution **Middlebury2014** everything improves (ORD 0.3840→0.3554,
D³R 0.3343→0.2504, RMSE 0.1708→0.1481). But on **Ibims-1**, whose originals are only 640×480, the
boundary metric improves while the global metrics get *worse* (D³R 0.3698→**0.3269** better; ORD
0.4002→0.4504, RMSE 0.1596→0.1687, δ1.25 0.6345→0.6633 all worse). **Pushing resolution/detail up
can improve a boundary metric while degrading global geometry.** `align%` is a boundary metric. We
should not conclude from `align%` alone that a configuration is geometrically better.

**(c) The classic non-monotonic train/test scale result.** "Fixing the train-test resolution
discrepancy" ([arXiv:1906.06423](https://arxiv.org/abs/1906.06423)) §3.3 shows a ResNet-50 trained at
224 peaking at test resolution **288**, not 224:

| K_test | 64 | 128 | 224 | 256 | **288** | 320 | 384 | 448 |
|---|---|---|---|---|---|---|---|---|
| top-1 | 29.4 | 65.4 | 77.0 | 78.0 | **78.4** | 78.3 | 77.7 | 76.6 |

The mechanism (§3.1) is **apparent object size**: train-time RandomResizedCrop zooms in, so the
optimum at test is whatever setting reproduces the *apparent scale statistics of training* — not
whatever matches the nominal training resolution. Read across: for VGGT-Ω at a fixed 1024-token
budget, the apparent-scale knob **is** FoV. A peak at some FoV other than "the one that matches the
sensor" is the expected result, not an anomaly.

**(d) Our own numbers do not fit the angular-resolution story.** The source is 1408 px over ~124°
≈ 11.35 px/deg. A 512 px input (post-loader) matches that at **FoV ≈ 45°**. If matching native
sensor detail drove `align%`, the peak would sit near 45-60°. It sits at **100°**, where the crop is
5.12 px/deg — a **2.2× downsample** below sensor detail — and 40° (the *only* upsampling row) is the
worst tangent score in the table. The observed ordering is inconsistent with an angular-resolution
explanation and consistent with the training-prior/context-density explanation given under *What is
known about VGGT-Ω's training FoV distribution* and in (b) above: 100° is comfortably interior to
[30°,120°], 120° is at the filter bound, 40° is near the lower bound and
in the "extreme zoom" regime the pipeline removed.

**One confound to control before trusting the shape of the curve.** `align%` is normalised by the
count of input Sobel edges, and both the numerator and denominator change with FoV: a narrow crop
upsampled from the sensor has fewer, softer edges (a weaker, noisier denominator), while a 120° crop
has many edges compressed near the periphery. Boosting's design makes exactly this control explicit
— they select patches by "comparing the density of the edges in the patch to the density of the
edges in the whole image". Before drawing conclusions from the hump, re-run the sweep with the edge
threshold set to hold **edge count** (not percentile) roughly constant across FoV, or report
`align%` against a matched-edge-count baseline. A meaningful part of the 17→28 spread could be the
metric.

### Patch-based inference: what the field actually does, and the few-large vs many-small numbers

Three primary systems, and they agree on the design rule that matters most to us.

**The shared rule: never change the pixel size the backbone sees.** PatchFusion
([arXiv:2312.02284](https://arxiv.org/abs/2312.02284)), §3.1: "We use a **fixed patch size that is
equal to or similar to the native resolution of the base depth model**." Depth Pro, §3.1: the canvas
is fixed at 1536×1536 "chosen as a multiple of the ViT's 384×384", and "the input image is split into
patches of **384 × 384 at each scale**" — patch size is constant across the pyramid, only the
*downsampling of the canvas* changes, and "the patch encoder shares weights across all scales, [so]
it may intuitively learn a scale-invariant representation." Boosting fixes its tile size to the
network's receptive field. **All three vary which part of the world lands in the window; none vary
the window's pixel size.** Our sweep does the opposite — it holds pixels fixed at 512 and varies the
world content, i.e. it varies angular resolution, which is the one knob these systems deliberately
freeze.

**Geometry of each scheme.**

| system | patch size | how chosen | overlap | fusion |
|---|---|---|---|---|
| Depth Pro | 384² (= ViT native) | fixed; canvas 1536² downsampled to 3 scales → 25 + 9 + 1 = **35 patches** | **25%** intersection (two finest scales only, "to avoid seams") | merged to feature maps by a **Voronoi partition** of the target area, then upsampled and fused by a DPT decoder; a separate image encoder on the whole image at 384² "anchors the patch predictions in a global context" |
| PatchFusion | fixed at base model's native res (e.g. 540×960 on UnrealStereo4K) | P=16 non-overlapping grid; +33 shifted → P=49; +N random → R=N | shifted/random placement rather than a fixed fraction | end-to-end, no post-hoc optimisation: a coarse global branch + fine patch branch with **consistency-aware training/inference** (L2 loss on overlapping regions of intermediate *features* and depth). Explicitly "freedom from heuristic patch selection and post-processing" |
| Boosting | = receptive field size, **grown per-tile** | tile at base res with **1/3 overlap**; discard tiles whose edge density is *below* the whole image's; **grow** any tile whose edge density is *above* the image's until it matches | 1/3 | trained Pix2Pix/10-layer-U-Net merging network at 672², merging patch estimates onto a double-estimation base at R₂₀ |

Boosting's selection rule is the one to steal: **choose the crop size that equalises edge density
against the full image**, "This makes sure that each patch estimate has a stable structure." It is an
edge-density criterion, and our alignment metric is edge-based, so the two are directly compatible.

**The few-large vs many-small numbers.** PatchFusion's Table 3 (supplementary; trained on
UnrealStereo4K, zero-shot to Middlebury 2014) is the only clean patch-count sweep I found:

| config | RMS ↓ | SEE ↓ |
|---|---|---|
| ZoeDepth COARSE (no tiling) | 1.0777 | 0.8326 |
| + PatchFusion P=16 | 1.0743 | 0.8284 |
| + PatchFusion P=40 | 1.0678 | 0.8219 |
| + PatchFusion R=128 | 1.0620 | 0.8195 |
| + PatchFusion R=256 | 1.0580 | 0.8194 |
| + PatchFusion R=1024 | 1.0536 | 0.8178 |

Monotone in patch count, and **brutally diminishing**: 16 → 1024 patches is 64× the compute for
**1.9% relative RMS** and **1.3% relative SEE**; 128 → 1024 is 8× the compute for 0.8%. Most of the
value is in the first few dozen patches; there is no cliff and no reversal.

Their in-domain Table 1 adds a caveat that matters for us. On UnrealStereo4K, P=16 → P=49 → R=128
improves every *global* metric monotonically (δ1 98.419→98.450→98.469, REL 0.0399→0.0392→0.0388, RMS
1.0878→1.0747→1.0655) while the **boundary metric moves the wrong way**: SEE
0.8382→0.8462→**0.8488**. On MVS-Synth SEE is non-monotonic (1.0759→1.0700→1.0833). So **more, more
overlapped patches reliably helps global geometry but does not reliably help the boundary metric** —
the two can and do dissociate, in both directions (compare with the Ibims-1 result above). Since `align%` is a boundary
metric, "add more crops" is not guaranteed to move it.

Depth Pro's Table 9, quoted earlier, is the complementary result and points the other way on
boundaries: **35 native-384 patches beat one interpolated 1536 ViT by +9.9% F1 and +23% DIS R** at
equal metric accuracy and lower latency. The distinction between the two results is worth holding
onto: Depth Pro's gain comes from **patching at native resolution instead of stretching the
backbone**; PatchFusion's SEE wobble comes from **adding ever more redundant patches at an already
correct scale**. The first is a large, reliable effect; the second is small and sign-unstable.

**Reading this back onto the sweep.** The literature's consistent recommendation would be: stop
sweeping FoV at fixed pixels. Instead fix the window at VGGT-Ω's native budget (1024 tokens,
262,144 px area, aspect ratio inside the training band), and vary **how many such windows tile the
fisheye cone and how much they overlap** — i.e. move along the raw_roi/tangent/rectifier axis and the
tile-count axis, not the angular-resolution axis. The rectifier row already being the best in the
table (30.1) is consistent with this: it is the one configuration whose FoV (84.5°) sits
mid-distribution *and* whose projection the R⁹ pinhole camera model can actually represent. Depth
Pro's global "image encoder" anchor also has an obvious analogue here — one whole-cone view for
global consistency plus N native-budget windows for detail — and VGGT-Ω is natively multi-frame, so
the windows can be passed as frames of one sequence and let register/global attention do the fusion
rather than a hand-built merge.

### Unverified / could not confirm (this section)

- **VGGT-Ω's actual training FoV *distribution*.** Genuinely not documented. Only a hard bound
  (`[30°, 120°]` + distortion ratio ≤ 0.1) on the ~800K annotated internet-video sequences; the ~3M
  public-dataset sequences carry **no** stated FoV or camera-model constraint. No histogram, no mean,
  no per-dataset breakdown. Treat "peak near 100° reflects the training prior" as a plausible
  inference, not a documented fact.
- **The convention of `[30°, 120°]`** — horizontal, vertical, or diagonal FoV — is not stated in the
  paper. This matters: 120° diagonal on a 4:3 frame is only ~98° horizontal, which would move where
  our 120° tangent crop sits relative to the bound.
- **The convention of the training aspect-ratio band `[0.33, 1.33]`** (§4.1) is likewise unstated
  (h/w or w/h). Either way it does **not** coincide with the loader's `[0.5, 2.0]` crop band
  (`load_fn.py` line 68) — the loader admits shapes the paper does not claim to have trained on.
  Irrelevant to the current square-crop sweep (aspect 1.0 is inside both), but a live hazard the
  moment we feed non-square crops.
- **Whether the released 1B checkpoint's DINOv3 ViT-L init inherits the full high-resolution
  adaptation.** DINOv3 §5.1 describes high-res adaptation (global crops from {512, 768}) applied to
  the **7B**, and §5.2 distills the 7B into ViT-S/B/L. I did not confirm that the distilled ViT-L
  retains the 512-768 robustness, nor which exact DINOv3 checkpoint VGGT-Ω initialises from — the
  paper cites [147] generically.
- **The exact cost of the 518→512 bicubic resample** on `align%`. I established the resample happens
  and is uniform across the sweep; I did **not** measure it. Re-rendering at 512 and re-scoring is a
  cheap control that would settle it.
- **No resolution ablation and no OOD-camera ablation exist in the VGGT-Ω paper.** §4.3 ablates model
  size, data size, register attention, multi-task losses, self-supervision, and annotation quality
  only. So there is no first-party number for "how much does VGGT-Ω degrade off 512" or "how much on
  a wide-FoV camera". Anything we say there is our own measurement.
- **Depth Pro's Table 9 ViT-L DINOv2 baseline** is trained under Depth Pro's Stage 1 protocol, not
  the released Depth Pro recipe, and it is a *monocular metric depth* model, not a multi-view
  feed-forward reconstructor. The +23% boundary-recall transfer to VGGT-Ω is an analogy, not a
  measured result.
- **Whether register/global attention actually fuses multi-window inputs the way a hand-built merge
  would.** The suggestion at the end of *Patch-based inference* is mine; no paper I read tests
  feeding overlapping crops of one image to a multi-frame reconstructor as if they were separate
  views. VGGT-Ω §3.5.2 filters
  training sequences for "Insufficient Parallax", which is a reason to expect trouble: overlapping
  crops of a *single* frame have exactly zero baseline.

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
