# Survey: distortion handling, cross-frame attention, efficiency (2026-08-19)

Requested by the human while GPU tickets #35-#38 are pending. All entries
verified via the arXiv API (id + date + abstract read); no memory citations.

## A. Distortion handling / wide-FOV geometry

### Wid3R (arXiv:2602.05321, 2026-02) — NEAREST NEIGHBOR, must cite
Feed-forward multi-view reconstruction supporting wide-FOV cameras directly:
ray-based representation with **spherical harmonics** + a **camera model
token** for distortion-aware reconstruction; first feed-forward method
claiming 360° support. Conditioning on diverse camera types improves
generalization (+33.67 AUC@30 on Zip-NeRF fisheye).
- Difference from us: Wid3R **trains a new model** on diverse-camera data;
  we **adapt frozen backbones** with tiny parameter budgets, driven by a
  measured failure field. They need the training corpus; we need 30 min of
  finetuning. Complementary; also a candidate baseline if weights released.

### UniDAC (arXiv:2603.27105, 2026-03) — DAC's successor, must cite
Single model for any camera by **decoupling relative depth from a spatially
varying scale map** + RoPE-φ latitude-aware positional embedding.
- Their core premise — metric failure on new cameras is a *spatially varying
  scale field* — is exactly what our run_009 grid measured on frozen
  backbones (radially-modulated range compression, dispersion 2-10%). They
  solve it by training a scale-estimation module at dataset scale; we show
  the field is low-dimensional enough that 48 parameters recover most of it
  on a frozen model. Cite as convergent evidence + stronger-supervision
  contrast. Candidate #37-style baseline row (successor of DAC which we
  already run).

### DAPETR (arXiv:2606.08680, 2026-06)
BEV detection with mixed pinhole+fisheye; learned distortion-aware positional
embedding beats explicit polar reparameterization, and **combining learned
adaptation with explicit geometric reparameterization can conflict**.
- Useful citation for our discussion: analytic remap vs learned adaptation is
  not additive (matches RayTun3R's "naive PE remap improves rotation but…").

### PFDepth (arXiv:2509.26008, 2025-09), OmniDS (arXiv:2607.03038, 2026-07)
Multi-fisheye rig depth (volumetric fusion / ERP cost volumes with deformable
cross-attention + distortion bias). Different setting (calibrated rigs,
trained end-to-end), cite in related work's "fisheye-specific architectures"
paragraph to contrast with backbone adaptation.

### Spherical PE line: SpheRoPE (arXiv:2606.32033), RoPE-φ (in UniDAC)
Growing consensus that **global positional encoding is the right lever for
distortion**, consistent with our within-patch distortion measurement
(≤0.21px on Aria KB4 ⇒ patch-content adaptivity is pointless) and with
RayTun3R's PE-table design. Our angle: we quantified *why* PE-level is
enough (patch-level distortion is sub-pixel at 110°) — worth stating as a
measured justification the PE papers assume implicitly.

## B. Cross-frame attention / video depth

### ViGeo (arXiv:2605.30060, 2026-05)
Feed-forward video geometry with **dynamic chunking attention** (bidirectional
+ causal contexts, adapt at test time). Full-model training on public data.
- Contrast: our H6 module is a bolt-on for a frozen backbone, rim-tokens-only
  (0.48× FLOPs of full-token cross-attention), zero-init so identity at start.

### PPD/PPVD (arXiv:2601.05246, 2026-01)
Pixel-space diffusion for depth; video via reference-guided token propagation.
Generative lane, orthogonal, cite for completeness in video-depth related work.

### Older anchor: context-aware temporal attention for dynamic scenes
(arXiv:2305.07397). Cross-frame attention for monocular depth is established;
what is new in H6 is *where* (peripheral tokens only, geometry-selected) and
*why* (measured: rim carries pose/alignment signal but weak monocular depth
evidence — H1.2, 024B).

## C. Token efficiency for 3D transformers

### Spark3R (arXiv:2605.06270, 2026-05) — closest efficiency neighbor
Training-free asymmetric token reduction for feed-forward 3D (VGGT, π³, DA3,
VGGT-Ω!): query tokens sensitive, KV tokens compressible; up to 28× on 1000
frames.
- Ours selects tokens by **imaging geometry (θ)** not saliency, and *adds*
  cross-frame capacity at the rim instead of removing tokens. Orthogonal and
  stackable: Spark3R compresses KV context; H6 chooses which queries deserve
  cross-frame KV at all. Cite; possible "combine" future-work line.

### Lite3R (arXiv:2605.11354), TAP (arXiv:2608.10989)
Sparse linear attention + FP8 QAT; task registers for pruning. Orthogonal
efficiency lanes; one-line citations.

## D. Finetuning recipes

### GIFT (arXiv:2608.02068, 2026-08)
PEFT for depth FMs **without depth labels** via geometric invariance across
appearance changes (non-Lambertian). Same genus as our recipe — identify a
structured failure, build a targeted low-cost post-training objective — in a
different domain. Good "recipe framing" citation.

## What the survey changes

1. **Nobody does diagnosis-driven adaptation**: measure the frozen backbone's
   conditional error field first, then aim parameters/losses at it. Wid3R and
   UniDAC train against distortion generically; RayTun3R adapts PE per scene
   unsupervised. Our loop (probe → 48-param field → compression-weighted
   LoRA → rim-only attention) stays unclaimed.
2. **Nobody treats the periphery asymmetrically** (pose signal vs depth
   compression division of labor). Center-PH throws the rim away; everyone
   else treats all angles alike.
3. UniDAC + RayTun3R must enter related work as the two closest lanes
   (trained scale-field model / unsupervised PE TTA), with our supervised
   cross-scene tiny-budget adaptation as the third lane.
4. Efficiency claims need a Spark3R citation and a stackability sentence.

## Addendum: Spark3R method details (HTML read, 2026-08-19 evening)

Confirmed from the paper body (arXiv:2605.06270v2): query tokens get
INTRA-GROUP MERGING (light), KV tokens get PRUNING with a LAYER-ADAPTIVE
schedule (heavier where empirically insensitive); training-free; prior
uniform-reduction methods cap at ~10x before quality drops, the asymmetry
is what unlocks 28x.

### What this gives OUR efficient model (H6) — brainstorm with the human
1. **Their asymmetry validates our query side and exposes our KV waste.**
   Our rim-query design already treats queries as precious (geometry-
   selected, uncompressed). But our KV = ALL 1296 prev-frame grid tokens,
   including ~323 dead-corner vignette tokens (pure noise) — Spark3R says
   KV is exactly where pruning is safe. H6.1 probe locked+running: KV in
   {full, cone, rim, center} on the delivered #36 checkpoint.
2. **H6.2 temporal KV pyramid** (design): rolling multi-frame peripheral
   memory where KV from frame t-k is pruned/merged with factor growing in
   k. Constant-cost video model; the "efficient fisheye video" novelty the
   human asked for, now with two independent justifications (their
   role-asymmetry + our angular-redundancy measurement: rim tokens carry
   1/1.73 the solid angle of center tokens, so ring-merging rim KV is the
   geometry-licensed compression).
3. **Composition claim for the paper**: Spark3R compresses WITHIN existing
   global attention (many-frame offline models); ours ADDS a gated
   cross-frame path to single-frame-style frozen models. Orthogonal axes
   (their role asymmetry x our geometric asymmetry) — cite + one-line
   stackability note, plus the H6.3 ablation (geometry-selected vs
   saliency-selected queries at equal budget) if time allows on the box.
