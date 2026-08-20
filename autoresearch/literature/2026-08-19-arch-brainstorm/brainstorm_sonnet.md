# Brainstorm (Sonnet agent) — Efficient architecture for fisheye depth+pose

Emphasis: token/KV economy, tiny experts, latent fusion. All proposals respect F1
(no pixel-space geometric surgery on the frozen encoder's input) and build capacity
*behind* the encoder or *between* frames, per the measured "where to intervene"
gradient (F1–F4). Working numbers: ViT patch 14, ADT training res 504 → N≈1296
tokens/frame; rim (θ≥~38°) ≈ 40% of tokens (research-state.yaml, H6 motivation);
F5's rim-restricted temporal attention already measured at 0.23× all-token cost.
All FLOPs/param figures below are quantified guesses, explicitly flagged as such.

---

## 1. Sliding-Window Rim-KV Cache for Video Depth (RANK 1)

**Mechanism.** Formalize F5/H6 into a standing architecture rather than a one-off
probe: at inference, every frame's rim tokens (θ≥38°, ~40% of N) get their K/V
projected once and pushed into a fixed-length ring buffer (window T=3–4, per F5's
observed saturation by t−4). Each new frame's rim queries cross-attend to the
cached rim K/V from t−1…t−T through a zero-init gated residual (exactly F5's
verified module); center tokens never enter this path and are never cached.
No pixel resampling, no change to the frozen spatial attention — this is a purely
additive temporal module bolted onto the existing per-frame forward pass.

**Why it should work.** F5 already measured the two load-bearing facts: rim-only
KV matches full-KV near-rim accuracy (0.684 vs 0.685) at half the KV and 0.23×
the attention FLOPs of all-token cross-frame attention; center KV actively hurts
(+18%), so excluding it is a correctness gain, not just an efficiency one. F6
independently confirms the rim is where real cross-frame pose/parallax signal
lives, so restricting temporal fusion there is targeting the actual evidence, not
an arbitrary sparsity trick.

**Expected gain (guess).** Near-rim AbsRel: matches or slightly beats the current
single-frame F2 head (already −51 to −67%), plus a further few % from the t−2…t−4
evidence F5 flagged as still additive at t−2 (+1.9%) before saturating — call it
near-rim −55 to −65% vs single-frame baseline, roughly flat vs F2 alone (the win
here is efficiency, not accuracy). FLOPs/frame for temporal fusion: ~0.23× naive
all-token cross-frame attention (measured), and caching removes K/V *projection*
recompute for T−1 of the T cached frames each step — at T=4 that's ~75% of the
K/V-projection matmuls avoided relative to recomputing the window from scratch
every frame. Params: the existing F5 module (~sub-1M, same order as the LoRA in
F2). Net: this is the cheapest large FLOPs win on the table because it consumes
work already measured, not new work.

**Cheapest falsifying experiment.** Single A100, <1 day: take the existing H6
module (already trained, #36), wrap it in a ring-buffer cache over a 20–30 frame
dense-sampled clip, and report (a) zone AbsRel with the cache vs recompute-every-
frame (should be numerically identical — it's an inference-time cache, not a new
training run) and (b) measured wall-clock/FLOPs per frame at T=1,2,4,8 to confirm
the saturation point F5 already flagged. This validates the *engineering* claim
(cache ≡ recompute, cost drops) without touching training.

**Novelty positioning.** VGGT/DUSt3R/MASt3R/MoGe/UniK3D/DAC do all-pairs or
all-token attention with no geometry-aware KV pruning. Spark3R does saliency-
driven token reduction on the *same backbones* we target — the explicit
difference (already logged in findings.md) is that Spark3R's reduction is
content/saliency-driven and ours is geometry-driven (θ known at zero training
cost) and *additive* (adds rim capacity) rather than *subtractive* (prunes
tokens); the two are stackable, not competing. Vs generic KV-cache literature
(streaming LLMs, sliding-window attention): novel in that the eviction policy is
angular geometry, not recency or attention-score heuristics, and the geometry
argument is independently justified by a *pose* experiment (F6), not just a
speed argument.

**Main risk / how it dies.** If a paper reviewer asks "why not also cache center
KV for a bigger window at the same cost" — F5's answer (center KV hurts) is
strong but was measured on a *training-scene* probe (H6.1 explicitly flagged
exploratory); it needs a held-out-scene confirmation or the efficiency headline
just becomes "we deleted half the KV and lost nothing," which is a fine result
but a smaller claim than "we deleted the harmful half." Also dies if T-frame
saturation doesn't replicate under dense real egomotion (head rotation) rather
than the near-static clips used so far — F6 says rim motion is largest under
rotation, so a fast-turning clip could break the "saturates by t−4" assumption.

---

## 2. Theta-Gated Annular Token Merging (Latent ToMe, post-encoder only)

**Mechanism.** After the frozen encoder's full forward pass (so the pretrained
weights see the untouched full token grid, honoring F1/F3), apply a bipartite
token-merging step *only inside the trainable head/expert stage*: tokens are
grouped into angular annuli by θ, and within each annulus merge ratio is a
monotonic function of θ — aggressive merging in the center (θ≤11°, where F2/F3
show the frozen backbone's own signal already suffices and a tiny head recovers
it) and no merging at the rim (θ≥38°, full resolution preserved for the
compression-field fix). Merging is a learned or similarity-weighted average
(ToMe-style) over token *features*, never over pixels.

**Why it should work.** F2 established that center zones need very little added
capacity (25k-param head already helps them; the center isn't the hard part).
That implies the *decoder/head-side* compute spent on dense center tokens is
mostly wasted resolution — merging there should be nearly free in accuracy while
cutting the O(N²) cost of any attention happening downstream of the merge point
(e.g., a rim-expert transformer per Idea 3, or a global refinement layer). F1
already ruled out pixel-level equal-area remapping (+16–31% worse) — this is the
latent-level analogue done right: it never touches the frozen encoder's own
attention, only the *added* modules that consume its output.

**Expected gain (guess).** If center ≈60% of N is merged 4:1 and rim ≈40% is
untouched: post-merge token count ≈0.15N + 0.40N = 0.55N (~45% token reduction
entering any added downstream attention stage), which is ~70% FLOPs reduction in
that stage (self-attention scales ~N²: 0.55²≈0.30). Zone AbsRel: expect near-flat
center/far zones (±2–5%, within the noise band F7 already treats as such) and
unchanged near-rim (untouched resolution) — i.e., an efficiency-only win for any
head built on top, not a new accuracy claim by itself.

**Cheapest falsifying experiment.** Single A100, <1 day: implement the merge as
a post-hoc pooling step feeding the existing F2 readout head (params unchanged),
sweep center merge ratio {1:1 (control), 2:1, 4:1, 8:1}, report full (θ×depth)
joint-table zone AbsRel per F7's protocol plus measured FLOPs of the head's
forward pass. Falsifies immediately if near-rim or far-zone AbsRel degrades at
any merge ratio (would mean the "center is disposable resolution" claim from F2
doesn't hold once merging couples neighboring tokens' gradients).

**Novelty positioning.** ToMe and successors merge tokens *inside* a trained
transformer, uniformly or by attention-score similarity; the novelty here is
merge ratio parameterized by a known, training-free geometric quantity (θ) and
restricted to sit strictly downstream of a frozen backbone whose own tokens must
stay unperturbed (a constraint ToMe doesn't have, and which F1 makes load-bearing
for us specifically). Vs UniDAC's spatially-varying scale map / DarSwin's radial
tokenization: those act at input or early-layer resolution; this acts purely on
already-extracted frozen features, so it's a strictly cheaper, F1-compliant
sibling. Vs Spark3R: complementary layering point (post-encoder vs their
in-encoder saliency pruning).

**Main risk / how it dies.** Merging is a form of information mixing across
tokens that are geometrically close but semantically different (e.g., a hand at
θ=25° next to background) — could quietly reintroduce exactly the "input
resampling hurts" failure mode (F1) one level later, just averaged over features
instead of pixels. If the falsifying experiment shows any zone degrading, this
idea folds into "not worth the added engineering for a downstream-only speedup."

---

## 3. Polar Latent Re-Tokenization for a Rim Expert (isotropic receptive field)

**Mechanism.** Take the frozen encoder's rim-band tokens (θ≥38°, image-grid
ordered) and re-index them — purely as a sequence reordering/lookup, no
resampling, no interpolation — into a polar (angle, radius) 2D layout. A small
added transformer or depthwise-conv expert (same param budget class as F2's
head, ~25–100k params) then operates on this re-indexed grid, where neighboring
positions are metrically closer to isotropic than in the original fisheye-warped
image grid. Output residual is added back to the frozen backbone's rim
predictions via a zero-init gate (same safety pattern as F5).

**Why it should work.** F1 shows the *frozen* ViT must never see resampled
pixels — but that constraint says nothing about how an *added, trained-from-
scratch* small module should be laid out internally. F2/F3 already prove the
active ingredient is frozen image features (not geometry alone: an MLP on
(θ, depth) recovers less than half the gain and damages the center) — so the
expert must consume real features, which this does; it merely changes their
*addressing*, giving the small conv/attention kernels in the expert a receptive
field that is locally isotropic in physical angle rather than warped by KB4
projection, which a fixed-size square kernel on the raw grid cannot get.

**Expected gain (guess).** Speculative accuracy edge over F2's flat head:
near-rim AbsRel modestly better (guess −5 to −15% relative to F2's already-strong
number) because local structure (edges, hand boundaries) that is anisotropically
stretched in the native grid becomes isotropic in polar layout, letting a *small*
kernel do more work per parameter — i.e., same param budget, better use of it.
Params: comparable to F2 (tens of k). FLOPs: the re-indexing itself is a gather,
effectively free; the expert only ever runs over the rim's ~40% of tokens.

**Cheapest falsifying experiment.** Single A100, <1 day: train two heads at
matched parameter count — (a) F2's existing flat readout head, (b) the same
architecture but with rim tokens re-indexed to polar order before the head's
conv/attention layers — on the same rim-band data, report zone AbsRel per F7.
Falsifies if (b) does not beat (a) outside noise; that would show the frozen
features already carry whatever local structure was needed and re-addressing
buys nothing, consistent with (but not identical to) H7's finding that θ-gating
was redundant with the PE.

**Novelty positioning.** DarSwin does radial/distortion-aware tokenization at
the *input* stage of the backbone (ruled out for us by F1's empirical result on
this exact pretrained ViT). RayTun3R's polar PE residual conditions a fixed-grid
readout by angle but does not re-address tokens into a new topology. This
proposal is the annular re-tokenization the human's direction C asked for,
executed strictly as a *post-hoc, added-module* operation — the paper-level
claim is "polar re-tokenization is only safe/useful behind a frozen encoder,"
which is itself a finding, win or lose.

**Main risk / how it dies.** This is the most speculative idea in the set: H7
already showed that once positional encoding is in play, extra geometric
conditioning (θ-gating) was *redundant* with what uniform LoRA could already
learn. The same could hold here — the flat head may already implicitly learn
whatever the polar re-indexing would hand it "for free," making this an
engineering complication with no measured payoff. Treat as the third priority to
run, not the first.

---

## 4. Rim-Only Pose Head (near-free pose branch)

**Mechanism.** Direction A, briefly: build the camera/pose head to read *only*
from the rim-token KV cache already materialized by Idea 1 (or, if run
standalone, only from a rim-token forward pass) — never touch center tokens.
This is not new compute so much as *deleted* compute: today's pose head presumably
attends over the full token set; this restricts it to the ~40% rim subset it
already provably relies on.

**Why it should work.** F6 (masking rim correspondences degrades two-view pose,
median −2.15°, 17/17 pairs) and the H1.2/H1.3 line (center-masked pose ≈
vanilla; rim-masked pose collapses) both say the pose signal is already almost
entirely rim-resident in the frozen backbone. Restricting the pose head's input
is therefore closer to "stop paying for center tokens the pose head was already
ignoring" than a capability cut.

**Expected gain (guess).** Pose (RRA/RTA@15 or rotation-gain α): expect
near-parity with full-token pose head (±1–2° on rotation, consistent with F6's
scale of effect), since the ablation evidence already shows center contributes
little. FLOPs: if implemented as reading only from Idea 1's cached rim K/V, the
pose head's attention cost drops to roughly the rim fraction squared relative to
a full-token pose head (~0.4²≈0.16× if it's self-attention over pose queries and
rim K/V only, better if it's cross-attention with few pose queries). Params:
near-zero marginal — mostly a masking/routing change to an existing head.

**Cheapest falsifying experiment.** Single A100, <1 day: take the existing pose
head, run it three ways on the same held-out pairs — (a) full tokens, (b) rim
tokens only, (c) center tokens only — report RRA/RTA (or the repo's rotation-gain
metric) per pair, expect (b)≈(a)≫(c), directly extending H1.2/H1.3's masking
protocol to whatever pose head the current architecture uses.

**Novelty positioning.** No baseline (VGGT, DUSt3R/MASt3R, UniK3D, DAC,
RayTun3R, CAM3R) restricts its pose head's *input token set* by incidence angle;
they all pool or attend globally. The novelty is small but concrete: an
architectural consequence of F6, not just an eval-time ablation — i.e., ship the
restriction into the model, not just the analysis.

**Main risk / how it dies.** If the pose head is already cheap relative to the
backbone forward pass (likely — pose heads are typically small compared to a ViT
encoder), the FLOPs win is real but marginal to the *end-to-end* budget, making
this a "nice, cheap, low-risk" addition rather than a headline result on its
own — best presented as a component of Idea 1's system rather than standalone.

---

## 5. Compression-Field Residual Expert (attacking F4's actual mechanism)

**Mechanism.** F4 states the causal fix is removing a *smooth radial compression
field* from the prediction so a single global affine serves the whole image —
not patching per-pixel bias post hoc (F2's head) and not a 48-param output table
(H2.1, refuted-with-mechanism). Build a tiny expert (matched param class to F2,
~25–50k) that predicts a smooth, low-frequency correction field g(θ, d̂; frozen
features) — parameterized as a small number of radial basis functions or a
low-degree polynomial in θ modulated by frozen-feature evidence — applied
multiplicatively/additively to predicted depth *before* the per-frame affine is
fit, rather than as an arbitrary unconstrained per-pixel residual.

**Why it should work.** F4 is explicit that ~82% of the apparent near-rim
penalty is the affine's placement, and that the survivor is a genuine smooth
compression field. F2's head already works, but nothing in its design *enforces*
smoothness or a single-affine-suffices property — it's an unconstrained MLP that
happens to help. Building the smoothness constraint into the architecture (few
degrees of freedom in θ, shared across the whole image) directly targets F4's
named mechanism, which should (a) make the fix more data-efficient (fewer params
than F2's already-tiny head), (b) make it more likely to transfer across scenes
(a smooth global field is a much smaller hypothesis class than F2.3's per-scene
head, which the decoration-scene cross-fold already showed straining under
genre shift), and (c) directly produce the eval-protocol payoff F4 asks for:
one frozen affine sufficing for the whole image, checkable via F7's
alignment-robustness rows.

**Expected gain (guess).** Near-rim AbsRel: comparable to F2 (−50 to −65%) but
with *less* per-scene variance (tighter cross-scene transfer than H2.3's
−18.7%…−78% spread, because the hypothesis class is smaller) and a measurable
win on the alignment-robustness metric specifically (frozen-affine zone AbsRel
should move much closer to scale-shift zone AbsRel than F2's head does). Params:
likely *fewer* than F2's 25k (a handful of radial basis coefficients modulated
by a small feature-gating network). FLOPs: negligible, same order as F2.

**Cheapest falsifying experiment.** Single A100, <1 day: train this constrained-
field expert alongside F2's existing unconstrained head on identical data/split,
report both under F7's alignment-robustness protocol (frozen / scale_only /
zone-restricted affine) — the constrained version should show a *smaller* gap
between "after scale_shift" and "after frozen affine" zone AbsRel than F2's head,
which is the direct, falsifiable prediction of F4's mechanism claim.

**Novelty positioning.** This is the one proposal that is explicitly a
mechanism-matched redesign of F2, not a new axis — its novelty claim is
methodological: most adaptation papers (UniDAC, Wid3R, RayTun3R, Fisheye3R) fit
unconstrained per-pixel or per-token corrections; this constrains the correction
to the diagnosed functional form (F4's compression field), which is closer to
DAC's "spatially-varying scale map" lane but *derived from a measured
diagnosis* rather than fit from scale at dataset scale — the diagnosis-driven
framing findings.md already flags as unclaimed territory.

**Main risk / how it dies.** If the true compression field isn't well
approximated by a low-order radial function (e.g., it depends on scene content,
not just θ and predicted depth) then constraining the expert's capacity will
underperform F2's flexible head — this is directly testable and could revert to
"F2's head was already doing the right thing implicitly, don't bother
constraining it."

---

## 6. Full-Pipeline Distillation into a Deployment Student

**Mechanism.** Once a full "best" pipeline exists (frozen backbone + F2/Idea-5
head + Idea-1 rim-KV video module), distill it into a single small student
network — either a smaller ViT backbone or the frozen backbone plus only the
lightweight modules, trained end-to-end this time (no longer frozen) against
the teacher's outputs, using F2's compression-weighted loss plus a rim-teacher
soft-target term so the student inherits the rim-behavior without needing the
teacher's temporal cache at deployment.

**Why it should work.** This is standard distillation logic applied to a
specific, already-diagnosed target (the compression field, F4) rather than
generic whole-model compression — the teacher's rim behavior is a well-defined,
low-dimensional signal (F4/F5) that should distill more reliably than
distilling "depth in general."

**Expected gain (guess).** Student at, say, half the backbone width/depth:
expect a real but partial retention of near-rim gains (guess: retains 50–70% of
the teacher's rim improvement) at roughly 2–4× lower encoder FLOPs — genuinely
useful for edge/AR deployment framing, but this is the least certain number in
the set since no distillation run exists yet in this project.

**Cheapest falsifying experiment.** Not really <1 day on a single A100 for a
full run — the honest cheapest falsifier is a *partial* one: distill only the
rim-expert module (Idea 5) into a smaller version of itself (e.g., halve its
already-tiny hidden width) using teacher soft targets vs training the small
version from scratch on hard labels; compare near-rim AbsRel. If distillation
gives no edge over from-scratch training at this tiny scale, full-pipeline
distillation is unlikely to be worth the engineering before other ideas are
locked in.

**Novelty positioning.** Distillation of foundation depth models exists broadly,
but "distill only the diagnosed correction pathway, keep the frozen backbone as
is (or also distill it) with a compression-field-aware loss" is not the generic
recipe — most baselines here (UniK3D, DAC, VGGT) are not built around a
diagnosed, localized failure mode to distill against.

**Main risk / how it dies.** F3's own finding — that the readout head "may
encode one apartment's appearance" — is a live threat to distillation too: if
the teacher's rim fix is partly scene-specific, the student inherits that
overfit rather than a general mechanism, and the efficiency win becomes a
generalization loss. This should not be started before the cross-scene/
cross-dataset generalization question (open in F3, ticket #29 lineage) is
better resolved.

---

## KILLED: Discrete Radial Mixture-of-Experts over Tokens (hard routing by θ-bin)

**The seductive pitch.** Route each token to one of K small expert MLPs by its
θ-bin (center/mid/near-rim/far-rim), each expert a LoRA-scale module — a natural
generalization of F2's single head into a proper MoE, and it sounds like the
textbook way to give "tiny experts" their own capacity per the human's
direction B.

**Why it dies.** H7 already ran almost exactly this idea's mechanism — a
θ-conditioned gate on LoRA capacity, dW(x) = B·diag(g(θ))·A — and it was
REFUTED cleanly: gated LoRA matched uniform (ungated) LoRA at both rank 8 and
rank 4 (0.572/0.567, 0.754/0.753), and the gate curve stayed flat
(|g−1|≈0.06). The diagnosed mechanism is that the positional encoding already
puts spatially-varying correction inside a *single* uniform adapter's
hypothesis class — there is no unused capacity sitting idle in the center that a
separate expert would unlock, because the shared adapter already conditions on
position through the PE. A hard-routed discrete MoE is a strictly more
complex, less differentiable version of the same idea (adds a router,
adds train-time load-balancing loss, adds discrete-routing inference branching)
for a mechanism H7 already showed buys nothing at equal rank. This is exactly
the seductive-but-weak case: "give each zone its own expert" sounds right until
you notice the shared model was never capacity-constrained by geometry in the
first place. Do not spend GPU time on this without first re-deriving why H7's
finding wouldn't also kill it (e.g., if hard routing frees experts to use
*nonlinear* capacity differently, not just gated linear updates — untested, but
the burden of proof is now on the idea, not for it).

---

## Summary

**Top pick: Idea 1, Sliding-Window Rim-KV Cache.** It is the only proposal that
consumes zero new hypothesis risk — F5 already measured both halves of its
claim (rim-KV ≈ full-KV, center-KV hurts) — and turns them into a standing
video architecture with a genuinely new systems contribution: geometry-driven,
zero-router KV eviction that is provably *not* just a speed/accuracy tradeoff
but a correctness improvement, positioned explicitly against Spark3R's
saliency-driven pruning as the geometry-driven, additive-not-subtractive
sibling. That distinguishes it from an adapter tweak: it is an architectural
policy for how *any* frozen fisheye depth backbone should do video KV, backed by
a pose experiment (F6) as well as a depth one (F5), which is the kind of
cross-task justification a single accuracy number can't give.
