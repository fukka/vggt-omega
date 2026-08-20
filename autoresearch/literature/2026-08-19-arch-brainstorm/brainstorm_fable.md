# Brainstorm (Fable agent) — decoupled pose/depth architectures for fisheye egocentric video

Scope respected: no experiments run; all justifications cite F1–F8, plus two in-repo
refutations the F-list implies but doesn't spell out: **H7** (θ-gated LoRA — the learned
gate goes flat; the PE already conditions adapters spatially) and **H8** (equal-solid-angle
resampling degrades every zone at equal tokens). Both are used below as kill evidence.

Design invariants derived from the facts, which every proposal obeys:

- **I1 (from F1, H8):** never resample the image or the token lattice the encoder sees.
  All geometry enters *behind* the encoder or as attention structure.
- **I2 (from F2, F4):** the depth failure is a smooth, low-dimensional radial log-range
  compression field; the fix needs evidence or a structured field, not capacity.
- **I3 (from F6 + findings act 2):** the frozen pose path lives on rim features; depth
  adapters must be zero-init / readout-side so pose is untouched by construction.
- **I4 (from F5):** cross-frame evidence belongs to rim queries; center KV is poison;
  temporal value saturates by t−4.
- **I5 (from F3, F8):** any learned map from appearance to correction is a one-apartment
  overfit risk until proven otherwise; prefer geometry-bottlenecked or self-supervised paths.

---

## Ranked proposals

### P1. EpiRim — pose-conditioned spherical epipolar attention (a cost volume disguised as 16 KV tokens)

**Mechanism.** Upgrade F5's rim cross-frame module from *retrieval* to *geometry*. Given
relative pose (from P2 below, or MPS/IMU, or GT during ablation), each rim query token's
unit ray (exact KB4 lift) is sampled at D≈16 inverse-range hypotheses; each 3D hypothesis
is projected into frames t−1…t−4 via pose + KB4, and the KV set for that query becomes the
D bilinearly-gathered tokens along its epipolar curve on the fisheye sphere — plus a learned
inverse-range positional encoding on each hypothesis. Attention over these D candidates is
then literally a matching distribution over range; a zero-init gated head decodes the
softmax's expected inverse-range into a **log-range residual** added to the backbone's depth.
Center tokens get no module at all (I4). The backbone stays frozen; the module is the only
video-specific part of the network.

**Why it should work.** F5 already proves rim-query/rim-KV cross-frame attention improves
rim depth and that geometry-blind restriction of KV costs nothing (0.684 vs 0.685) — i.e.
the attention is *finding correspondences on its own*; giving it the epipolar answer should
only sharpen it. F6 says rim parallax is large and real (biggest apparent motion under head
rotation), so the inverse-range hypotheses are well-separated exactly where depth is worst.
F4 says the failure is a compression field — a *bias* — and triangulated parallax is the one
evidence source that is unbiased in range, so the residual directly nulls the field rather
than re-fitting it (fixes the one-affine problem at its cause). Findings act 3 ("multi-frame
context buys the center, not the field") is exactly what geometry-free temporal attention
does; EpiRim is the missing mechanism that routes context *to the rim*.

**Expected gain.** Near-rim scale_shift AbsRel 0.684 (F5 module) → **0.45–0.55** guess
(approaching the 0.567 LoRA pilot, F2, without touching a backbone weight), and — the more
important row — a large cut in the *frozen-affine* near-rim number, since metric parallax
attacks the compression field itself (F4/F7). Cost: KV per query drops from ~600 rim tokens
(F5) to D×4 ≈ 64 across four frames → attention FLOPs ≈ **0.01–0.03× all-token** cross-frame
attention (F5's rim-only was already 0.23×); the gather is memory-bound but tiny. Module
params ≈ 50–100k.

**Cheapest falsifier (<1 day, 1×A100).** Training-free probe on the delivered #36 F5
checkpoint: with GT pose, mask KV to within ε of each query's epipolar curve (no retraining,
the same trick as the F5 rim-KV probe) and read near-rim zone AbsRel vs 0.684. Then one
20-epoch retrain (≈ the H6 budget, 236 pairs/epoch) with the D=16 hypothesis sampler.
Kill condition: epipolar-restricted ≤ unrestricted at matched KV budget means attention
already saturates the correspondence problem and geometry adds nothing.

**Novelty positioning.** Plane-sweep cost volumes exist (MVS, ManyDepth-style video depth,
PFDepth/OmniDS ERP volumes) but always in end-to-end-trained networks on calibrated rigs.
VGGT/DUSt3R fuse frames by undirected global attention with no camera model; MASt3R regresses
pointmaps and matches with a *trained* head; MoGe/UniK3D/DAC are single-image. Spark3R prunes
KV by saliency inside existing attention — EpiRim *constructs* KV by calibrated geometry in an
added zero-init path (orthogonal, stackable, and our per-query KV count is below anything
pruning can reach). "A KB4-exact epipolar attention bolted onto a frozen FM, restricted to the
lens periphery because that's where the measured give/receive asymmetry lives" is unclaimed
(2026-08-27 novelty check: StableDPT is all-token and geometry-free; FDT foveated selection
points compute at the *center*, we invert it with a measured reason).

**Main risk / how it dies.** Pose error. The epipolar curve for a near-field rim point (the
0.26–0.94 m hand/body band, F8/H4) moves a lot per degree of pose error; if P2's pose (or MPS)
is off by >1–2°, the gathered KV misses the true correspondence and the module degenerates to
noise — worse than F5's blind retrieval, which at least learned robustness. Secondary risk:
bilinear token gather at 14-px patch granularity may quantize away near-field disparity.
Mitigation for both: keep the F5 unrestricted-retrieval path as a parallel head and let the
zero-init gates arbitrate; the falsifier's GT-pose arm vs estimated-pose arm separates the
two failure modes in one day.

---

### P2. RimPose — detector-free ray-space pose from frozen features (the decoupling itself)

**Mechanism.** Pose gets its own stream and never touches predicted depth (direction A).
Take the frozen encoder's rim tokens (θ ≥ ~35°, ~40% of the cone) from frames t and t−k;
build the 4D correlation volume between the two rim sets (≈600×600 at 14-px granularity —
trivially cheap); mutual-nearest matching + cycle check gives coarse matches; refine to
sub-patch precision by correlating the high-res DPT decoder features in a local window
around each coarse match. Lift both endpoints to unit rays by exact KB4 inversion (no
undistortion image ever exists — I1), then solve rotation + translation direction with a
ray-space essential/eigen solver under GNC robust weighting. Optionally a ~10k-param match
reweighting MLP (confidence from correlation shape + θ) trained on ADT GT pose. Total new
parameters: 0–10k; no extra backbone pass (features are already computed for depth).

**Why it should work.** F6: masking rim correspondences degrades two-view pose on 17/17
pairs — the rim is where pose signal lives; H1.1 says span-under-noise is the mechanism, and
a detector-free dense matcher on ViT features is precisely a high-count, full-span match
source. Findings act 2: the frozen FM's own pose already relies on rim features, so those
features demonstrably encode match-able rim structure. Decoupling is also *robustness by
construction*: F4's compression field corrupts any pose path that consumes predicted depth
(pointmap-style), while epipolar geometry on rays is immune to it.

**Expected gain.** Two-view median rotation at or below the classical SIFT+KB4 harness
(which needed hand-tuned span admission, H1.1) and RRA/RTA@15 competitive with VGGT-Ω's
learned pose head at essentially zero trained parameters — plus it's the enabling input to
P1. Latency: ~1–2 ms/pair of matching on top of the existing forward pass.

**Cheapest falsifier (<1 day).** The H1 harness already exists with ADT GT pose. Run
frozen-DA3-feature mutual-NN rim matching + ray solver on the 17 pairs (extend to ~100),
report rotation / translation-direction error vs (a) SIFT+KB4, (b) VGGT-Ω native pose,
(c) full-image variant of the same matcher (tests whether rim-only loses anything — F6
predicts full-image ≈ rim-only, which would also be an efficiency claim). CPU-feasible for
the matcher; A100 only for backbone features.

**Novelty positioning.** MASt3R's matching is a trained descriptor head on a fine-tuned
backbone with pinhole assumptions; DUSt3R/VGGT derive pose from regressed geometry (entangled
with the depth failure); RoMa-class dense matchers train the whole matcher. "Frozen depth-FM
features + exact fisheye ray lifting + classical solver, restricted to the periphery, are
already a pose method" is a training-free result nobody has stated — and it's the measured
counterpoint to Center-PH, which wins depth by discarding exactly these pixels.

**Main risk / how it dies.** 14-px patch quantization: if sub-patch refinement on DPT
features can't get matches below ~2–3 px reprojection error, rotation error lands at 5–10°
and the method is a toy. It also dies if frozen features are apartment-specific texture
detectors (F3's worry) — the ScanNet++ 170° replication in the falsifier settles that
cheaply. A partial death is acceptable: even a mediocre-accuracy RimPose can still gate P1
via its GT-pose-vs-estimated-pose arm.

---

### P3. Field-Token head — per-frame low-rank nulling of the compression field, provably center-neutral

**Mechanism.** The single-image depth fix, built to make **one affine serve the whole image**
(F4's stated criterion). Predict log-range as ℓ̂(u) = ℓ_backbone(u) + Σ_{k=1..K} c_k · B_k(θ(u)),
with B_k a *fixed* radial spline basis that is identically zero for θ < 15° (center neutrality
is a property of the basis, not a hope — answers Q2), and c ∈ R^K (K≈8) regressed per frame
by a two-layer MLP from the backbone's global token + pooled rim tokens (~20k params). Train
with a one-affine objective: fit a single scale-shift per frame inside the loss and add a
penalty on the variance of zone-wise optimal affines, so the network is explicitly optimized
for the F7 alignment-robustness rows.

**Why it should work.** F4: the failure is a smooth radial log-range field and 82% of the
apparent penalty is affine placement — a K-dim radial field is the minimal object that can
null it. F3 is the crucial design constraint: the *static* (θ, d̂) map recovers <half the
gain and damages the center, while image features double the gain — so the coefficients must
come from features; but F3/I5 also warn that free-form feature heads may memorize the
apartment — so we bottleneck the features to K numbers per frame that can only express a
smooth radial gain. It sits exactly between the failed 48-param table (H2.1) and the 25k
free readout head (F2): image-adaptive like the head, structurally incapable of appearance
overfit like the table.

**Expected gain.** Frozen-affine near-rim AbsRel cut ≥50% (it is optimized for that row);
scale_shift near-rim between the table's −18–25% transfer and the head's −51–67% (F2, H2.2),
guess −35–55% on held-out sequences; near-center change bounded at 0 by the basis. ~20k
params, ~0 FLOPs (K coefficients + a per-pixel basis lookup).

**Cheapest falsifier (<1 day).** Train on the 4 sequences (minutes, mostly CPU-scale), eval
the two held-out ADT sequences and ScanNet++ 3f15 with the full F7 table including frozen-
affine and zone-fit rows, plus the mandatory aux-only (θ, d̂) baseline row. Kill conditions:
(a) the regressed c collapses to a per-dataset constant → it *is* the H2.1 table and inherits
its failure; (b) frozen-affine rows don't move → the field isn't being nulled, only re-dressed.

**Novelty positioning.** UniDAC trains a spatially-varying scale map at dataset scale inside
a new model; DAC/UniK3D bake camera awareness into full training; RayTun3R fits a PE-table
per scene by TTA without an output-space model. A *diagnosis-derived, feature-conditioned,
provably-center-neutral K-dim field head on a frozen backbone* — with the one-affine training
objective as its defining loss — is a different, and much cheaper, point in that space. It is
admittedly the least "architectural" proposal here; it earns its place as the single-frame leg
that P4 and the benchmark need, not as the headline.

**Main risk / how it dies.** The field may not be expressible as g(θ) alone — run_009 showed
the bias is (θ × depth)-structured (near content pushed far, far content pulled near), and a
pure θ-field can only fix one depth regime per angle. If the falsifier shows the θ-only basis
saturating well below the readout head, the honest conclusion is that P1's evidence-based
route is *necessary*, which is itself a paper claim (echoes H2.1's "recalibration cannot
invert a compression").

---

### P4. Periscope-TTA — the rim supervises itself (pose → triangulation → field distillation, no depth GT)

**Mechanism.** Close the loop between P2 and P3 to get **GT-free adaptation**: on a new
scene/camera, run P2's matcher over dense windows (t±4, F5's saturation horizon), triangulate
the rim matches in ray space (midpoint method) with P2's pose, and use the resulting sparse
metric-up-to-scale ranges as pseudo-labels to fit P3's field coefficients (or, richer, to
finetune the F2 readout head) — robustly, with GNC weights from reprojection residuals.
Scale ambiguity from monocular pose is *harmless by protocol*: the eval of record is per-frame
scale-shift aligned (F7), and the object being corrected is the shape of the compression
field, not absolute scale.

**Why it should work.** Every link is measured: rim matches carry real geometry (F6);
rim parallax is the largest in the frame under head motion (F6), so triangulation is
best-conditioned exactly where depth is worst (F4); the correction target is low-dimensional
(F4/I2), so sparse noisy pseudo-labels suffice; and the whole thing sidesteps I5/F8 — the
one-apartment generalization risk — because adaptation needs only video, no depth sensor.
This converts the project's biggest weakness (F8: one room, generalization untested) into
its headline capability.

**Expected gain.** Recover ≥60% of the supervised readout-head gain (F2) on a held-out
scene with zero depth labels; on ScanNet++ (different camera, 170°) any statistically clear
rim improvement at all is a result no adapter baseline can match (RayTun3R's TTA is the only
competitor in the lane, and it doesn't exploit pose). Cost: adaptation = minutes of matching
+ a least-squares fit; zero inference-time overhead beyond P3.

**Cheapest falsifier (<1 day).** Oracle-first: on one held-out ADT sequence use **GT pose**
(remove P2 risk from the loop), triangulate frozen-feature rim matches, fit the field, eval
F7 zones. If the oracle can't beat the frozen baseline, match density/accuracy at the rim is
insufficient and the whole loop is dead — no amount of learning fixes it. If the oracle
works, swap in P2 pose the same day.

**Novelty positioning.** Self-supervised monocular depth (Monodepth lineage) trains whole
networks with photometric losses on pinhole video; RayTun3R's TTA is entropy-style PE
adjustment with no explicit geometry; MASt3R-SfM does global SfM but never distills back
into a monocular head. "The periphery is a pose asset and a depth liability — so let the
pose asset *pay for* the depth liability" is the N3 asymmetry turned into a closed
architecture, which no surveyed work states or exploits.

**Main risk / how it dies.** Compounding: pseudo-label quality = matcher accuracy × pose
accuracy × triangulation conditioning, and near-field rim points (the hand band, F8/H4)
have big parallax but also the most occlusion and motion. If the oracle arm passes but the
estimated-pose arm fails, P4 degrades to "needs MPS trajectories" — still useful on Aria
(MPS is standard) but a weaker paper claim.

---

### P5. Angular KV Pyramid — constant-cost peripheral memory for video (efficiency headline)

**Mechanism.** Extend F5 (t−1) to a rolling multi-frame rim memory with *geometry-licensed*
compression: KV from frame t−k is ring-merged along φ within iso-θ rings with merge factor
2^k (measured license: a rim patch covers only 1/1.73 the solid angle of a center patch —
the rim is the angularly *oversampled* band, so ring-merging rim KV discards redundancy,
not information). Per-age zero-init gates. Memory footprint per frame: 600 + 300 + 150 + 75
KV tokens for t−1…t−4 ≈ 1125, vs 4×1296 = 5184 for naive all-token history → a video model
whose temporal cost is **constant and ~0.2× one frame of all-token cross-attention**, with
the horizon chosen by measurement (F5: saturates by t−4).

**Why it should work.** F5 gives every ingredient: rim-KV loses nothing vs full-KV; t−2 adds
+1.9% *training-free* (so the gates generalize across ages); saturation at t−4 bounds the
pyramid. Merging is the only new bet, and the solid-angle measurement plus Spark3R's finding
that KV (not queries) is the compressible side both point the same way.

**Expected gain.** +2–4% near-rim over t−1-only (capturing most of the measured multi-frame
headroom) at ≤1.9× the t−1 module's KV, i.e. still ≈0.4× of even the single-frame all-token
module. This is the deployment/AR story: fixed small memory, no growth with video length.

**Cheapest falsifier (<1 day).** Entirely training-free on the #36 checkpoint (F5 showed
training-free extension works): compare near-rim AbsRel for {t−1} vs {t−1,t−2} unmerged
(should reproduce +1.9%) vs {t−1..t−4} pyramid-merged. Kill condition: merging destroys the
multi-frame gain (merged ≈ t−1-only), meaning rim KV redundancy is angular in theory but
appearance-critical in practice.

**Novelty positioning.** Spark3R: saliency-driven, training-free pruning *inside* existing
global attention of many-frame offline models; ours adds a gated temporal path to
single-frame-style frozen models and compresses by imaging geometry — orthogonal axes
(their role asymmetry × our angular asymmetry), explicitly stackable. StreamVGGT-style
causal caches keep all tokens. FDT foveates toward the center; we invert with a measured
reason. Note H8 does NOT kill this: H8 resampled the *encoder input*; P5 merges *KV of an
added module* — I1 is respected.

**Main risk / how it dies.** It's a mid-size efficiency delta on a module that is already
cheap (0.23×), so reviewers may read it as an ablation rather than a contribution unless it
ships inside P1/P6's larger story. Dies quietly if the training-free probe shows merged
multi-frame ≤ t−1-only; costs one GPU-day to find out.

---

### P6. PERISCOPE — the composed system (the actual CVPR submission)

**Mechanism.** One frozen encoder, two decoupled streams sharing its features:
**pose stream** = P2 (rim ray matching, classical solver, ~0 params); **depth stream** =
backbone + P3 field head (single-frame leg) + P1 epipolar rim memory with P5's pyramid
(video leg), all zero-init/readout-side so the pose stream's features are untouched (I3).
P4 is the deployment mode: on a new device/scene, the pose stream generates the depth
stream's adaptation labels. Everything trainable together ≈ 0.6M params on a frozen
backbone.

**Why it should work.** It is the minimal architecture in which every measured fact has a
home: F1→ no input surgery anywhere; F2→ all learning behind the encoder; F4→ the field
head and metric parallax both target the compression field; F5→ the video leg's query/KV
budget; F6→ pose from the rim, and pose safety by decoupling; F3/F8→ P4's self-supervision
answers the generalization objection structurally rather than with more data.

**Expected gain (system-level guesses).** Near-rim scale_shift AbsRel ~1.4 → **≤0.5** with
no center collateral (vs 0.567 LoRA pilot that edits the backbone); frozen-affine rows
improved by design; pose ≥ VGGT-Ω at 0 pose params; temporal cost 0.2× single-frame
all-token cross-attention; adaptation to a new scene without a depth sensor.

**Falsifier.** None of its own — it lives or dies by P1–P5's falsifiers, which is the point:
each component is independently killable in <1 GPU-day, in dependency order P2 → P1 → P3 →
P4 → P5.

**Novelty positioning.** VGGT/DUSt3R entangle pose and depth in one trunk and are camera-
naive; RayTun3R adapts PE but holds no pose story; UniDAC/Wid3R retrain at dataset scale;
Center-PH wins depth by amputating the pose-critical band. PERISCOPE's thesis — *the fisheye
periphery is a pose asset and a depth liability, and the right architecture routes the asset
to pay the liability* — holds both sides of the RayTun3R-vs-Center-PH trade nobody currently
holds (the measured N1 lane).

**Main risk.** Systems papers die by diffusion: if P1 and P4 both fall to their falsifiers,
what remains (P3+P5) is an adapter paper with good diagnostics — real, but not the flagship.
The falsifier ordering exists to learn this within a week, not at the deadline.

---

## Killed ideas (kept for the paper's justification narrative)

### K-A. Center/Rim dual-expert mixture with learned fusion — KILLED
The seductive one: it matches direction B's first bullet, "MoE for fisheye" writes itself,
and reviewers recognize the shape. It dies on three measurements. (1) **H7**: the learned
θ-gate on LoRA converged to flat — the ViT's positional embedding already conditions
adapters spatially, so a learned spatial router is redundant machinery. (2) **F2**: a single
0.49M LoRA (and even a 25k head) already improves the rim *without* hurting the center —
there is no measured interference between regimes for a second expert to resolve; the
mixture solves a problem that does not exist in the data. (3) **F4/I2**: the failure is a
smooth low-dimensional field; capacity partitioning is the wrong axis — the bottleneck is
evidence (P1) or structure (P3), not parameters. A dual-expert would add params, a fusion
seam through the θ≈38° band (exactly where hands live, F8/H4), and zero measured mechanism.

### K-B. Latent-space equal-area resampling feeding the experts — KILLED
The "process the fisheye into distortion-corrected latent features and let experts merge
them" variant of direction B. F1 kills the input-space version outright (+16–31% everywhere);
moving the resample after the encoder does not rescue it, because the thing H8 broke — the
pretrained token/PE statistics the frozen decoder and any readout consume — is broken
identically by a latent remap. And the within-patch distortion measurement (≤0.21 px on Aria
KB4) removes the motivation: there is no patch-level distortion to correct. Deformable-
attention fisheye works (DarSwin lineage) are prior art for the idea *and* our measurements
are the explanation for why that whole family underperforms frozen-FM tolerance at 110°.

---

## Top pick — 3-line summary

**P1+P2 (EpiRim fed by RimPose), wrapped as PERISCOPE.** It is CVPR-grade because it changes
*what computes what*: pose is extracted classically from frozen rim features (no trained pose
head), and that pose constructs a 16-token spherical epipolar KV set that turns cross-frame
attention into a calibrated cost volume on a frozen backbone — a new mechanism with a measured
justification chain (F4 bias needs unbiased evidence; F5 shows the attention path works and
prices it; F6 shows the rim funds the pose), not a reweighting of an existing adapter; and
every component is falsifiable on one A100 in under a day using the F7 zone protocol.
