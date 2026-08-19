# Method section pre-draft (numbers pending #35/#36/#37)

Drafted from the locked protocols so the section is ready to receive numbers.
Style: measurement-dictated design; each component cites the finding that
forces it.

## 4. Method: repairing the near-field rim without touching what pose needs

### 4.1 Design constraints from the measurements

Three measured facts constrain any repair. (i) The failure is a precise,
radially modulated range compression (§3): the model's evidence is intact,
its calibration is not — so capacity should go to *re-reading* existing
features, not re-learning them. (ii) The compression is many-to-one in true
depth: a 0.5 m object predicted at 1.4 m shares an output bin with a true
1.4 m object, so any correction indexed by the output alone provably
conflates populations (we measure this as the near-center collateral of a
48-parameter lookup baseline). Input evidence is necessary. (iii) The
model's cross-frame alignment lives in the rim features (§3): a repair that
rewrites them gambles with pose. We therefore keep the backbone bit-frozen
where pose reads it, and prove — not measure — pose safety where possible.

### 4.2 The ladder

**Rung 0 (baseline): the 48-parameter table.** d' = d·exp c[θ-bin, log-d̂ bin],
fitted as per-cell median residuals. Transfers at the rim, corrupts the near
center (Table X) — the measured case for input conditioning, and the bar any
learned method must clear.

**Rung 1: the readout head (single image).** A per-patch MLP
(C+3 → 64 → 1, ~25k params, zero-init output) on the frozen backbone's final
patch tokens + (sinθ, cosθ, log d̂), emitting a per-patch multiplicative
log-depth correction. Trained in minutes on CPU. The backbone forward is
untouched ⇒ pose invariant by construction. [Six-scene, cross-scene numbers:
§5, already in hand.]

**Rung 2: rim-targeted finetuning (single image, improves the backbone).**
LoRA (r=8) on the last-4 block MLPs only — attention untouched (constraint
iii). Loss = compression-weighted depth (weights from the measured bias map)
+ rim-feature distillation to the frozen teacher (the pose-safety mechanism
once weights move) + multi-frame rim consistency at train time (routes the
parallax evidence that fusion currently spends on the center — §3's
give/receive asymmetry — into rim depth; inference stays single-image).
Control: same-budget plain-L1 LoRA. [Numbers: #35.]

**Rung 3: peripheral cross-frame attention (video).** One zero-init
cross-frame block; queries = rim tokens only (~40% of the cone), KV = the
previous frame's tokens; the update is written into a depth-head-only copy
of the final feature level, so the camera path reads the originals and pose
is again invariant by construction. Efficiency is principled: the center
provably gains nothing from temporal context. Control: same-parameter
all-token queries. [Numbers: #36.]

### 4.3 What we deliberately do not do

Patch-content undistortion: on 110° KB4 at 14-px patches the within-patch
deviation of the true lens map from its linearization is ≤0.21 px (measured)
— there is nothing to undistort inside a patch; the distortion lives between
patches. Dynamic-region machinery: hands sit exactly in the repair zone but
act as plain occlusion (masking ≈ area-matched random control), so the only
dynamics measure is loss hygiene (dynamic cells excluded from fitting).

## Open slots

- Table X: rung comparison on held-out scenes (before / rung0 / rung1 /
  rung2 / rung2-control / rung3 / rung3-control) × (near-rim, near-center,
  center, far, RRA@15, RTA@15). Fill from #35/#36 evals + run_010/011.
- Fig: loss-component ablation of rung 2 (full vs plain control curves).
- Efficiency table for rung 3: FLOPs/latency rim-query vs all-token.
