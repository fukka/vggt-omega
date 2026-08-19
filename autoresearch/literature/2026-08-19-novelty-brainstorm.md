# Novelty brainstorm under the CVPR bar (2026-08-19)

Constraint from the human: novelty must live at the **model** or **training**
level and promise strong wins in **efficiency or performance**. Constraints
from our own measurements (these kill or license ideas):

- K1: within-patch distortion ≤0.21px on Aria KB4 → any patch-content
  adaptivity (deformable conv on patches, per-patch unwarping) is dead.
- K2: the depth failure is a smooth radially-modulated range compression,
  dispersion 2-10% → low-dim conditional fixes suffice; capacity is not the
  bottleneck.
- K3: rim tokens carry pose/alignment signal a frozen FM already uses
  (H1.2); context helps center not rim (024B) → cross-frame is the only
  known source of *new* rim depth evidence.
- K4: hands/dynamics behave as plain occlusion (H4) → no dedicated dynamics
  module is warranted on ADT.

## Ranked candidates

### C1. θ-conditioned adapter modulation (training/arch, performance) — STRONG
LoRA whose update is gated by incidence angle: ΔW(x) = B·diag(g(θ))·A where
g is a tiny MLP on the token's θ. Rationale: uniform LoRA spends capacity
uniformly; K2 says the needed correction is a smooth function of θ. Expected:
match H5-full quality at r=4 or beat it at r=8 with zero extra FLOPs at
inference (g(θ) precomputable per token position). Nobody in the survey does
input-geometry-conditioned PEFT. Cheap to implement on top of h5 lora.py
(one extra file); CPU pilot on seq131 feasible this week. **Do first.**

### C2. Foveated tokenization / equal-solid-angle patching (arch, efficiency) — STRONG but heavier
MEASURED 2026-08-19 (scratchpad solid_angle.py, spherical-excess per patch
on the true Aria KB4 at 504²/14px): **a center patch covers 1.73× the solid
angle of a rim patch** (0.00407 vs 0.00235 sr, θ>45° band) — the *rim* is
the angularly oversampled side, opposite to the initial guess. Equal-area
tokenization at center-patch density therefore **merges rim patches**: 651
tokens instead of 973 in the cone = **33.1% fewer**, with zero change to
center resolution. Two consistent readings: (a) efficiency — rim tokens are
angularly redundant, merge them; (b) science — rim depth failure is NOT a
sampling-starvation story (rim has *more* pixels per steradian), which
independently corroborates K3: what the rim lacks is evidence (cross-frame),
not resolution. Risk: breaks the pretrained PE distribution — mitigation is
RayTun3R-style table refit. Remaining probe: remap-only forward (no
training) — does vanilla DA3 on the equal-area remap move the bias field?
**Do the remap probe; commit to training only if the field moves.**

### C3. Peripheral memory across many frames (arch, performance) — H6 extension
H6 is pairwise (t-1 only). Extend to a rolling KV cache of rim tokens from
the last N frames (cheap: rim tokens only, 627/1296). The story upgrades
from "cross-frame attention" to "peripheral memory": periphery = where new
scene content enters the FOV first on egocentric motion. Ablation axis: N.
Defer until #36 confirms the pairwise direction on dense windows.

### C4. Diagnosis-driven adaptation as a *recipe* (training, generality) — the umbrella claim
Formalize the pipeline: (1) probe the frozen FM on a small labeled set,
(2) fit the conditional error field E(θ, d̂), (3) spend parameters/losses
only where E says (weighted loss, θ-gated LoRA, rim-only modules).
GIFT (2608.02068) does invariance-driven; nobody does error-field-driven.
This is the framing that unifies rungs 0-3 into one method instead of a
ladder of tricks. Costs nothing new; it is writing + one transfer test
(does the ADT-measured field transfer to ScanNet++ 115°? → reuse #37 data).

### C5. KILLED ideas (keep for the paper's justification narrative)
- Patch-level deformable/undistortion modules — K1.
- Output-space recalibration beyond rung 0 — many-to-one compression (H3).
- Dynamics-specific module — K4.
- Naive PE remap — DAPETR + RayTun3R Tab. 4a both find it conflicts with
  learned adaptation.

## Recommended CVPR story (v2, post-pivot)

"**Peripheral vision for 3D foundation models**": frozen FMs treat every
viewing angle alike; egocentric fisheye breaks that symmetry — the rim is a
pose asset (H1.2, and RayTun3R's Center-PH gap confirms externally) and a
depth liability (run_009 field). We (a) measure the asymmetry, (b) exploit
it with θ-conditioned tiny adaptation (C1 ⊃ H5) and rim-only cross-frame
attention at 0.48× FLOPs (H6/C3), (c) show the recipe transfers (C4:
ADT → ScanNet++), against RayTun3R's baseline family with an honest
adaptation-data column. Efficiency headline from H6 + optionally C2;
performance headline from H5/C1 near-rim numbers.

## Immediate actions
1. C1 pilot code (θ-gated LoRA) — extends h5/code/lora.py, pilot on seq131.
2. C2 probe — solid-angle ratio computation + remap-only bias-field probe.
3. Center-PH baseline (from comparison-protocol.md) — CPU, DA3-S.
These three are CPU-sized and do not touch the GPU queue.
