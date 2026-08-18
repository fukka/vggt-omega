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
