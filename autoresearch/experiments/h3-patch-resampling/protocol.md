# H3 — Zero-parameter local patch undistortion: does it help the rim, per-band?

Locked before running. The last untouched bootstrap hypothesis (axis N2).

**Hypothesis.** Resampling each ViT patch by the local gnomonic (tangent-plane)
linearization of the KB4 map at its center — so every 14×14 patch's *content*
is locally pinhole-like while the patch grid stays fixed — improves DA3-Small's
rim depth at ZERO parameters, because patch content statistics move toward the
training distribution. RayTun3R's ablation reports patch undistortion alone is
"minimal" as a whole-image number; the per-(θ×depth)-cell effect is unmeasured,
and our diagnosis (precise, radially-modulated miscalibration) predicts any
gain should concentrate in the high-θ columns.

**Prediction.** In the protocol-of-record joint table on seq131 (28 frames,
same as run_008b): high-θ columns improve, center columns unchanged; overall
effect may be small (consistent with RayTun3R). Secondary read: does it stack
with the H2.2 head (resampled input → new features/preds → head refit,
even/odd split)?

**Method.**
- Input transform: per patch (14×14, 36×36 grid at 504²), build a tangent-plane
  grid at the patch-center ray (orthonormal basis e1,e2 ⊥ d0), with the local
  pixel scale matched at the patch center (Jacobian-matched so the center is
  identity); sample the fisheye image through the exact KB4 projection with
  bilinear grid_sample. Patches outside the cone: left as-is. Seams between
  patches are inherent (documented upstream); no blending.
- Sanity check before trusting: resampled image visually inspected (one frame,
  sent to dashboard assets) + center patch must be pixel-identical to vanilla
  (identity at center by construction).
- Arms: (a) vanilla (= run_008b cached), (b) resampled input, full forward.
  Eval identical to run_008b. Then (c) EXPLORATORY: head refit on resampled
  features, even/odd, vs run_011.
- CPU, ~28 forwards.

**Refutation.** High-θ columns unimproved ⇒ patch-content statistics are not
the binding constraint (the compression lives elsewhere — consistent with
RayTun3R's "PE residual carries the gain, resampling minimal") — N2's
remaining candidates then need training (DarSwin-style token layout), which is
out of the limited-finetuning budget's cheap tier; record and close N2's
zero-parameter branch.
