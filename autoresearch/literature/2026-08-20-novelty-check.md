# Novelty check for the H2.2 direction — 2026-08-20

Two searches on "lightweight head on frozen depth-FM features for fisheye/radial
recalibration". Status: abstracts only, not verified to the survey's standard.

## Nearest neighbors found

- **DepthFisheye: Efficient Fine-Tuning of Depth Estimation Models for Fisheye**
  (ICCVM 2025, [pdf](https://iccvm.org/2025/papers/lncs/56.pdf)) — closest in
  goal. Distortion-Aware Adapter + LoRA on attention/linear layers, backbone
  otherwise frozen. **Touches the backbone's computation** (LoRA in blocks), so
  it cannot make our pose-safety-by-construction claim; no radial×depth
  diagnosis; no pose axis. TODO: verify eval protocol and whether any radial
  breakdown is reported.
- **Calibration Tokens** ([arXiv:2508.04928](https://arxiv.org/abs/2508.04928))
  — input-side tokens, frozen backbone; changes feature computation for all
  outputs (depth AND pose paths). Already in the bootstrap notes.
- **DrivingDepth: Sparse-Prompted Pixel-wise Scale Correction**
  ([arXiv:2606.31488](https://arxiv.org/abs/2606.31488)) — pixel-wise scale
  correction for driving depth via sparse prompts; output-side correction
  exists as a pattern, but prompt-driven (needs sparse depth at test), not
  feature-driven, and pinhole driving, not fisheye.
- **Calibration-aware linear probing** (ICML 2026 workshop-ish,
  [icml.cc](https://icml.cc/virtual/2026/68328)) — relearn only the head under
  a calibration objective on frozen features; classification-confidence
  domain, but the "head-only recalibration as diagnostic" framing is kin.
- **WideDepth** ([arXiv:2605.24074](https://arxiv.org/abs/2605.24074)) —
  explicitly motivates fisheye for **near-field** robotics and has mm GT;
  strongest candidate external benchmark for the near-field-rim claims.

## What remains ours (as far as these searches show)

1. The (θ × depth) **compression diagnosis** with the distance control and
   alignment-free bias/dispersion split (precision vs calibration).
2. The **give/receive asymmetry**: rim powers frozen-FM pose (masking
   evidence) while multi-frame fusion's depth gains land in the center.
3. **Pose-safety by construction**: readout-only correction that provably
   cannot move the pose path — plus the three-axis (center, rim, pose) eval.
4. The **near-field-rim collision** with egocentric hands (0.26–0.94 m, 80%+
   beyond 41°).
5. The measured **failure of output-indexed recalibration** (compression is
   many-to-one) as the justification for feature conditioning.

Action: cite DepthFisheye and DrivingDepth as the nearest adapter/correction
patterns; position against "touches the backbone computation" vs "readout-only".
