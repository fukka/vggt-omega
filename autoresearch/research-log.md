# Research Log

## 2026-08-18 — Bootstrap

- Invoked via /autoresearch with four novelty priorities (N1 Pareto, N2 efficiency,
  N3 periphery-for-alignment, N4 dynamics) targeting Aria ~110° fisheye, limited
  finetuning of depth backbones (VGGT-360 / RayTun3R / CAM3R named as anchors).
- Continuity: cron job `75190b64`, every 20 min, session-only.
- Workspace created at `autoresearch/` (subdirectory, not repo root, because two
  Claude sessions share this working tree and the root is crowded).
- Literature: the repo already carries a verified 2026-07-29 survey
  (`docs/research/fisheye-wide-fov-adaptation.md`) and full reproductions of all
  three anchors (`VGGT-360-fisheye/`, `raytun3r/`, `cam3r/`). Ran four fresh
  searches on the novelty axes; new finds recorded in
  `literature/2026-08-18-bootstrap-search.md` (DarSwin, Calibration Tokens
  2508.04928, WideDepth 2605.24074, POMATO, ReViV, LF-VISLAM).
- Hypotheses H1–H4 formed (one per axis; see research-state.yaml). Priority order:
  H1 first because it is CPU-only, cheap, and its failure would kill axis N3 before
  anything is built on it.
- Evaluation locked at bootstrap: radially-binned metrics under the existing repo
  protocol (one frozen affine per frame, spread not endpoint ratio, distance control
  from slambench/fov.py where depth is involved; count-matched controls for any
  keypoint-band comparison; alpha/gain for rotation, never bare R_deg ratios).

## 2026-08-18 — Inner loop, runs 001–004 (H1 family, all CPU, one afternoon)

- run_001/002: H1 (per-correspondence rim advantage) REFUTED — flat synthetic
  control, coin-flip paired differences.
- run_003: H1.1 (span at fixed count) SUPPORTED 17/17 pairs; the ~20x gap between
  real (−2.15°) and ideal-noise (−0.11°) arms locates the mechanism in robustness
  to real feature noise.
- run_004: H1.2 (model ignores rim) REFUTED in the best way — DA3-Small's pose is
  rim-driven (center-masked == vanilla; rim-masked 2.5x worse on every pair).
- Net: N3's premise is now measured from three angles and the adapter design
  constraint is concrete (don't touch rim features the pose path uses; correct
  depth at the readout; report pose stability as a third Pareto axis).
- Next queued: (a) area-matched random-mask control for run_004; (b) Aria-FOV
  (θ≤55°) replication of runs 003–004; (c) outer-loop reflection + to_human
  report with figures; then H2 design.
