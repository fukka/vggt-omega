# H2.0 analysis — the depth baseline

## run_008 (2026-08-19, superseded) and run_008b (protocol of record)

run_008 used `scale_only` alignment as locked; its absolute level disagreed 2×
with the GPU lane's ticket-024 table and its depth rows flipped direction —
diagnosed as an alignment-protocol mismatch, since fovbench's `align_mode` of
record is **`scale_shift`** (`fovbench/models.py:321`). run_008b re-runs the
identical measurement through `finetune.eval.metrics.align_depth(mode=
"scale_shift")` — the single protocol authority. Deviation from the locked
protocol documented here; both JSONs kept.

## run_008b — DA3-Small, seq131, 28 frames, range domain, scale_shift/frame

Uncontrolled AbsRel by θ: 0.28 (3°) → 0.57 (51°), spread **2.04** — the
folklore curve, reproduced.

Joint table (AbsRel; rows = GT range bands, cols = θ bin midpoints):

| depth \ θ | 3.4° | 10.3° | 17.1° | 24.0° | 30.8° | 37.7° | 44.5° | 51.4° | spread |
|---|---|---|---|---|---|---|---|---|---|
| 0–1 m | 0.350 | 0.466 | 0.878 | 0.818 | 0.992 | 1.261 | 1.611 | 1.982 | **5.66** |
| 1–2 m | 0.386 | 0.442 | 0.450 | 0.465 | 0.469 | 0.507 | 0.667 | 0.817 | 2.12 |
| 2–3 m | 0.209 | 0.202 | 0.193 | 0.206 | 0.223 | 0.251 | 0.262 | 0.331 | 1.72 |
| 3–5 m | 0.262 | 0.286 | 0.271 | 0.239 | 0.203 | 0.168 | 0.150 | 0.177 | 1.91 (**falling**) |
| 5–10 m | 0.280 | 0.318 | 0.343 | 0.332 | 0.287 | 0.233 | 0.173 | 0.155 | 2.22 (**falling**) |

**Reads:**

1. **A real radial degradation survives the distance control at near/mid range**
   (0–3 m rows rise monotonically with θ), so the fisheye rim penalty is not
   purely "the furniture" — consistent with the GPU lane's raw-fisheye
   `survives` = 0.57–0.85 across models.
2. **The far rows invert**: 3–10 m content is predicted BETTER at the rim than
   at the center. So the adapter target is not "the rim", it is the
   **near-field rim** — which in egocentric video is precisely the hands /
   manipulation zone (ties N1 to N4). A purely radial correction (RayTun3R's
   binning) may be the wrong parameterization; the failure is radial×depth.
3. **Caveats, in order of concern:** (i) one frozen affine per frame can
   redistribute a structured bias across the (θ, depth) plane — the repo's
   U-shape warning in 2D; the inversion could partly be the affine's seesaw.
   Robustness arm queued: alignment-free per-cell scale-ratio read
   (`fovbench/geometry.py` has `raw_scale_ratio` for exactly this reason).
   (ii) AbsRel >1 in the 0–1 m row means the affine essentially fails there;
   treat magnitudes as ordinal, not metric. (iii) One scene; the GPU table's
   six-sequence pooled controlled penalty (1.25 for this model) is consistent
   with these mixed-sign rows pooling to a modest number, but the row-level
   structure is only measured here so far. (iv) Absolute level (~0.3–0.4
   pooled) is ~2× the GPU lane's — plausibly seq131's heavy near-field content;
   unresolved, flagged.

## run_009 (2026-08-19, CONFIRMATORY — H2.0b alignment-free maps)

Bias (per-cell median log residual vs frame level) and dispersion (MAD around
the cell's own median), no alignment anywhere:

- **Dispersion is small EVERYWHERE** — 0.02–0.10 in log depth (2–10% depth
  noise), including the near rim. The model *sees* the near rim fine.
- **Bias is large and structured**: 0–1 m content is placed e^0.53–e^1.20 =
  1.7–3.3× too far (worsening toward the rim: −0.53 at 3° → −1.20 at 51°);
  5–10 m content 1.4–1.8× too near (shrinking toward the rim: +0.49 → +0.34).

**Verdict: the run_008b structure is real and it is (mis)calibration, not
noise.** The model applies a **depth-range compression** around the frame
level, radially modulated. The far-rim "inversion" of run_008b is the far
bias shrinking at the rim, not better perception. This matches two prior
signatures: the RayTun3R repro's depth gain 0.406 ("right ordering, range
compressed") and UniK3D's wide-FOV "contraction" the angular loss exists to
prevent. Prediction 1 of H2.0b: bias half confirmed, dispersion half refuted
(near-rim dispersion is NOT elevated) — i.e. per the protocol's refutation
clause, claims of "the model can't see the near rim" are withdrawn in favor
of "**the model mis-scales the near rim, systematically and precisely**",
which is exactly what a recalibration table (H2.1) can fix if it transfers.

## run_010 (2026-08-19, CONFIRMATORY — H2.1 recalibration table, both splits + fixed-affine variant)

48-param table (16/48 cells had ≥2000 train px; |c|max 0.24), fit on train
frames, applied to held-out frames, protocol-of-record eval:

| zone (held-out) | even/odd | halves | even/odd, fixed affine |
|---|---|---|---|
| near rim (≤2 m, ≥38°) | 1.023→0.765 (**−25%**) | 0.639→0.502 (**−21%**) | −18% |
| center (≤11°, pooled) | −1.8% | −15.4% | +6.5% |
| far (≥3 m) | −12.4% | −5.5% | −11.3% |

But the row detail shows **near-CENTER cells worsen consistently** (0–1 m at
3.4°: 0.124→0.477; 1–2 m rows at low θ likewise), in every variant — including
with the eval affine frozen from the uncorrected prediction. So the damage is
not the re-alignment coupling; it is the table. **Diagnosis:** the correction
is indexed by *predicted* depth, and the model's compression makes prediction
many-to-one in GT depth — a 0.5 m object predicted at 1.4 m lands in the same
bin as a true 1.4 m object, and the cell's median fix (fit on the majority)
is wrong for the minority. **A post-hoc, output-indexed recalibration cannot
invert a compression; disambiguation needs input evidence** (appearance,
context, or multi-frame parallax).

**H2.1 verdict: partially supported, with the failure mode being the finding.**
The transferable-miscalibration component is real (−18…−25% near-rim on
held-out frames, far zone also improves), so any learned adapter must beat
this table. But the near-center collateral establishes, on measurement, why
the next rung must condition on the image, not the output — and the
ticket-024B multi-frame result says parallax evidence exists and is currently
spent on the center. The learned-adapter protocol (H2.2) should therefore be
input-conditioned and report the full joint table, not zone aggregates
(zone pooling hid this failure until the rows were read).

## run_011 (2026-08-20, CONFIRMATORY — H2.2 feature head, both splits)

~25k-param head on frozen DA3-Small final patch tokens (+sinθ, cosθ, log
pred-depth), zero-init output, trained on train-split patch residuals (300
epochs, L1 0.35→0.13), held-out eval under protocol of record:

| zone (held-out) | even/odd | halves | H2.1 table (even/odd) |
|---|---|---|---|
| near rim (≤2 m, ≥38°) | 1.023→0.333 (**−67%**) | 0.639→0.314 (**−51%**) | −25% |
| near center (≤2 m, ≤11°) | 0.412→0.354 (**−14%**) | 0.283→0.259 (−9%) | **damaged** |
| center pooled (≤11°) | −38% | −28% | −2% |
| far (≥3 m) | −38% | −19% | −12% |

**Both success criteria met**: the head beats the table 2–3× at the near rim
and removes the near-center collateral — frozen features DO carry the
disambiguating signal the output lacks. Residual weak spot: the single most
central 0–1 m cell in even/odd regresses (0.124→0.398; that cell's "before"
was unusually good and its pixel count is the smallest — flagged, not
explained). Pose is untouched by construction (readout-side only).

**What this establishes:** on this scene, the entire practical fix for the
radially-modulated compression is available from the frozen model's own
features via a head that fits in minutes on CPU — the RayTun3R online-
adaptation setting (hours, input-side surgery) is not needed for the *depth*
half on Aria. The open question is transfer: per-scene fit on other sequences
(ticket filed), then cross-scene generalization, then VGGT-Ω (highest
headroom).

## Cross-lane synthesis (ticket 024 A+B × H1 family)

- Part A: raw-fisheye rim depth penalty survives the depth control (0.57–0.85);
  rectified input's penalty is mostly composition ("the furniture").
- Part B: adding context frames (3/5/10) improves overall AbsRel — but the
  rim/center penalty **stays flat or worsens**: "context buys the centre, not
  the field."
- H1 family (this workspace): the rim is what the model's cross-frame
  alignment runs on.

**The asymmetry that now defines the project: the periphery supplies the
alignment that multi-frame fusion depends on, yet the fusion's depth gains are
spent almost entirely in the center. The rim gives and does not receive.**
The H2 adapter's sharpest formulation: route multi-frame evidence back into
near-field rim depth without perturbing the rim features that alignment uses.

## Alignment-robustness rows (2026-08-19, review change #4, RECORD harness)

Same vanilla DA3-S predictions, seq131 odd, 504px, three alignments
(results/alignment_robustness_seq131odd.json):

| mode | near_rim | near_center | center | far |
|---|---|---|---|---|
| scale_shift all-valid (record) | 1.032 | 0.401 | 0.322 | 0.224 |
| scale_only all-valid | 1.126 | 0.688 | 0.362 | 0.239 |
| scale_shift near-fit (<=2m) | 0.257 | 0.167 | 0.484 | 0.643 |

Confirms and sharpens the external review: the record-alignment rim penalty
is mostly the global affine serving the mid/far pixel mass — fitting near
drops near_rim -75% while far degrades 2.9x. NO single affine serves both,
which is the range-compression field seen from the alignment side (run_009
alignment-free measurement remains the causal home). Paper phrasing: zone
AbsRel numbers are rim-under-this-alignment; the method's before/after
deltas stay internally fair (both arms share the record alignment), and the
proper "how broken is the rim really" statement cites run_009's bias field
+ this three-row table together.
