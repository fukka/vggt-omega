# Critique (Sonnet agent, efficiency angle)

## 1. Strongest technical objections

**Fable's EpiRim (P1/P6 PERISCOPE).** The FLOPs accounting (0.01–0.03× all-token)
is the wrong currency. Every rim query gets its *own* epipolar curve (function of
its ray + the frame's pose), so the D=16×4-frame KV set differs per query — this
is not a cache in the reuse sense, it's a per-query gather of bilinearly-
interpolated features from HBM, ~40k irregular gather locations per frame
(627 queries × 16 hyp × 4 frames). GPUs do not vectorize scattered gathers the
way they vectorize a dense K/V matmul; realized cost is memory-bandwidth bound,
not FLOP bound. Opus's own P3 (their EpiRim variant) admits this outright
("memory-bandwidth bound... realized speedup may be 3–5×, not 19×") — Fable's
P1 never flags it, and reports the smaller number as if it were wall-clock.
Also unaccounted: computing each query's epipolar curve requires a KB4
*forward* projection per hypothesis (iterative/polynomial, not free) — this
per-query geometry cost has no line item in P1's budget. Net: P1's efficiency
headline needs a wall-clock measurement, not a FLOPs ratio, before it can be
trusted at all.

**Opus's RayCal (P1).** The "0 inference FLOPs" claim is real and is the best
efficiency number in all three documents — but it is conditioned on triangulation
being well-posed, and Opus's own risk section names the likely killer:
rotation-dominant egocentric motion collapses translation baseline, and this is
not a tail case for AR headset video, it is close to the *typical* case (people
look around far more than they translate). If RayCal needs DeSweep/P2's 32-shell
cost-volume machinery to be reliable across real head motion (as Opus's own
composition implies — P1 depends on P2 for the hard cases), the "zero FLOPs"
framing describes only the easy-motion regime, and the true deployed cost is
P1+P2 together, which reintroduces exactly the memory-bound-gather profile of
Fable's EpiRim (32-shell warp-and-correlate per rim token). The efficiency win
should be reported conditional on parallax-sufficiency, with the fallback path's
cost included, not as a flat "40 numbers, 0 FLOPs" headline.

## 2. Where their proposals beat mine

**Fable's P5 (Angular KV Pyramid) strictly supersedes my Idea 1.** My ring
buffer caches T=4 frames of full-resolution rim KV (~518 tokens/frame, ~2072
total). P5 adds age-dependent ring-merging (φ-merge within iso-θ rings, factor
2^k) justified by a real geometric fact I didn't use — rim patches are
angularly *oversampled* relative to center patches (solid-angle ratio 1.73), so
older-frame rim KV can be halved/quartered without discarding information, only
redundancy. Their total (≈1125 tokens for a 4-frame history) beats my naive
sum by roughly 2×, uses the same F5 justification I used, and is genuinely a
better version of the same architecture. I'd fold P5's ring-merge into Idea 1
rather than defend a separate design — this is a straightforward concession.

**The H6.1/train-scene caveat does temper my top pick, but doesn't sink it.**
research-state.yaml is explicit that H6.1's rim-KV≈full-KV finding is
"exploratory, train scene" — I flagged this myself as Idea 1's main risk, and
it applies identically to Fable's P1/P5/P6 and to Opus's P3, since all of them
build directly on the same F5 module and its same probe. It's not a
differential critique of my idea vs theirs — it's a shared, unresolved
precondition for the entire "rim-only temporal KV" family across all three
documents. The fix is the same for everyone: the falsifier must run on a
held-out scene, not re-confirm cache≡recompute on the training scene as I
originally scoped it.

## 3. Merged top-3 (one paper, <1 GPU-week total, robust to F8)

1. **RayCal (Opus P1)** — the depth headline. It is the only proposal in all
   three sets whose evidence source is a *(backbone, camera)* property, not a
   *(backbone, scene)* property — it needs unlabeled video, not per-apartment
   depth GT, so it is structurally the most robust answer to F8's one-room
   limitation of the whole trio. Falsifier: oracle-pose triangulation vs
   run_009's GT field, <1 day.

2. **RimPose** (Fable P2 and Opus P4 converged on nearly the same mechanism
   independently — frozen rim features → classical/differentiable solver,
   pose decoupled from depth by construction). Two agents reaching the same
   design from different starting emphases is itself evidence; it's also the
   only proposal across all documents that turns the standing "adapters must
   not perturb rim features" constraint (findings.md act 2) into an
   architectural guarantee rather than a training-time hope. Falsifier:
   CPU-feasible, existing H1 harness, <1 day.

3. **Rim-KV pyramid** (my Idea 1 + Fable's P5 ring-merge, folded together) —
   the video/efficiency leg, needed so the paper has an architecture story for
   video depth, not just two single-frame/pose mechanisms. Falsifier:
   training-free on the existing #36 checkpoint, <1 day, provided it is run on
   held-out scenes per the H6.1 caveat above.

Together: RimPose supplies pose almost free, RayCal spends that pose on the
depth field almost free, the KV pyramid supplies the video leg almost free —
one coherent "the rim funds pose, pose funds depth" thesis (Fable's framing,
Opus's cheapest mechanism) at a combined falsification cost of ~3 GPU-days,
leaving slack under the 1-week budget for the composed-system check.

## 4. One thing all three of us missed

Every proposal in all three documents treats this as a **monocular** problem —
parallax has to be earned from egomotion (RayCal, DeSweep, EpiRim all live or
die on translation being observable). Aria is not a monocular rig: it carries
multiple rigidly-mounted, factory-calibrated cameras (SLAM/grayscale pairs
alongside RGB — the same calibration pipeline that gave us the 38.44° device→RGB
rotation number). If ADT exposes synchronized multi-camera frames, a *static*
stereo baseline exists at every single timestamp, independent of head motion —
which would sidestep the #1 named killer of RayCal and DeSweep (rotation-
dominant motion → ill-conditioned triangulation) entirely, at zero motion
dependency. Worth a cheap check before committing to a motion-triangulation
architecture: does the local ADT sample include a second synchronized camera
stream with known extrinsics to the RGB sensor? If yes, it's a strictly better,
motion-independent evidence source for exactly the field RayCal wants to
identify. Secondary miss, smaller: all three documents benchmark efficiency in
A100 FLOPs/wall-clock: the project's stated first-class goal is edge/AR
deployment, and nobody discusses quantization robustness or on-device memory
bandwidth for the KV caches/gathers being proposed — an A100-cheap module can
still be the wrong shape for a mobile NPU.
