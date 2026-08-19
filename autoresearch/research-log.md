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

## 2026-08-19 — H1.3 closed; tickets filed; Chinese dashboard

- run_006/007: both day-1 findings transfer to real Aria (seq131, hand-eye
  bootstrap gate-verified at 0.77-0.96 deg, angle(C)=40.6 vs box's ~38).
  Quantitative softening: span value saturates ~45 deg; center not disposable
  on the narrower cone, but rim still ~2.3x an area-matched random deletion.
- Human feedback (recorded in memory reporting-in-plain-chinese): reports in
  plain Chinese; single refreshable dashboard.html; GPU tickets English but
  explained to the human in Chinese. Dashboard created and shipped.
- GPU tickets FILED with human's permission: #27 (calibration JSON), #28
  (hand-pixel statistics on skeleton sequences). Ticket files 026/027 in
  docs/handoff/tickets/, bodies mirrored, pushed to origin.
- H1 family now fully closed. Next: H2 (center-safe adapter) design under the
  new constraint; H3 zero-parameter tokenization probe; depth-side radial curve
  with the distance control.

## 2026-08-19 (tick 2) — H2.0 depth baseline; the project's defining asymmetry

- run_008 (scale_only) disagreed with the GPU lane 2x -> diagnosed as alignment
  protocol mismatch; run_008b reruns under the protocol of record (scale_shift
  via finetune/eval/metrics.py). Lesson re-learned: check the eval-of-record
  BEFORE locking a protocol detail it contradicts.
- run_008b: the radial depth penalty survives the distance control at 0-3m
  (near-field rim up to 5.7x row spread) but INVERTS at 3-10m. The adapter
  target is the near-field rim (= the egocentric manipulation zone; ties N4).
- Cross-lane synthesis with tickets 024 A+B: rim depth penalty is real on raw
  fisheye across 5 models; context frames buy the CENTER, not the field. With
  H1: the rim powers alignment but does not receive the fusion gains. This
  asymmetry is now the core thesis candidate.
- H2 restated accordingly (see research-state.yaml).

## 2026-08-19 (tick 3) — the compression story: H2.0b + H2.1

- run_009 (alignment-free): the near-rim "disaster" is PRECISE miscalibration
  — dispersion 2-10% everywhere, bias up to e^1.2 too far at near-rim, e^0.5
  too near at far — a depth-range compression, radially modulated. Matches the
  raytun3r depth-gain-0.406 signature and UniK3D's contraction.
- run_010 (48-param table, 2 splits + fixed-affine variant): near-rim fix
  transfers (-18..-25% held-out); near-center collateral in every variant.
  Mechanism: compression makes predicted depth many-to-one in GT, so an
  output-indexed table pushes the majority's fix onto minorities. Post-hoc
  recalibration cannot invert a compression — the measured case for an
  input-conditioned adapter (H2.2), and the table becomes the baseline any
  adapter must beat. Also: zone aggregates hid the collateral; report full
  joint tables from now on.

## 2026-08-20 — H2.2: frozen features are enough (on this scene); #27 closed

- Official calibration arrived (GPU #27): rotation 38.44 deg vs hand-eye 40.55
  — 2.33 deg apart; bootstrap validated; #27 closed with cross-check comment.
- run_011: the 25k-param frozen-feature head, both splits — near-rim -51..-67%
  held-out, near-center IMPROVED (the table had damaged it), far improved,
  center improved. Beats the mandatory table 2-3x everywhere it matters. One
  flagged cell (most-central 0-1m, even/odd split) regressed.
- Filed ticket 028 (issue #29): the same script on the six-sequence split,
  per-scene, both splits — one scene is not a result.
- Direction: deepen H2 (transfer), then VGGT-Omega variant, then multi-frame
  routing. Paper skeleton is forming: diagnosis (compression) -> asymmetry
  (rim gives, doesn't receive) -> minimal fix (feature head) -> transfer.
