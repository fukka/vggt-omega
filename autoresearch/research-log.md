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

## 2026-08-20 (tick 5) — the four axes collide at the near-field rim

- run_011 flagged cell de-flagged: 392 px of 2.45M (0.016%) — noise.
- Ticket #28 delivered (GPU downloaded 3 skeleton sequences; none existed on
  the box before — the "clean" release never contains humans). Dynamic
  (person) pixels: 0.8-4% of cone pixels; 80%+ beyond theta=41deg in
  single-skeleton sequences; median depth 0.26-0.94 m. **The hand/body zone
  IS the near-field rim** — the same cells where compression is worst
  (run_008b/009) and inside the band pose relies on (runs 004-007). All four
  of the user's novelty axes now point at one measured object.
- Filed ticket 029 (H4.1): does masking GT-dynamic pixels improve pose beyond
  an area-matched random mask; does removing hands from the input improve
  static depth. Uses the #27 calibration (no hand-eye needed on the box).
- Outer loop cycle 2 (informal): direction stays DEEPEN — transfer of the
  feature head (#29 pending) + H4.1; the paper narrative is now fixed:
  diagnosis -> asymmetry -> collision at the near-field rim -> minimal safe
  fix -> validation.

## 2026-08-20 (tick 6) — novelty check + cross-scene code ready; GPU quiet

- #29 / #31 still pending on the box (normal cadence).
- Novelty search: nearest neighbors are DepthFisheye (LoRA in blocks — touches
  backbone computation), Calibration Tokens (input-side), DrivingDepth
  (prompt-driven output scale correction). Nobody has the compression
  diagnosis, the give/receive asymmetry, pose-safety-by-construction, or the
  hands-at-the-near-rim collision. Recorded in
  literature/2026-08-20-novelty-check.md.
- H2.3 (cross-scene, leave-one-scene-out) protocol locked and cross_scene.py
  written + smoke-tested locally (self-fit only, no result claimed). Ready to
  ticket the moment #29 confirms the per-scene picture.

## 2026-08-21 — outer loop cycle 2: both GPU tickets land; the method holds on six scenes

- #29: run_011 holds on ALL SIX sequences (-21..-75% near-rim, both splits;
  near-center worst case +5.8%, noise-order). The paper's main table exists.
- #31: H4.1 prediction NOT supported — GT-dynamic masking ~ area-matched
  random (0.35-1.0 deg on 10-25 deg baselines); depth arm inconsistent. Hands
  are plain occlusion. H4 resolves to the placement measurement + loss
  hygiene. One anomaly (meal_seq131 random-beats-vanilla) recorded, open.
- Filed #32 (H2.3 cross-scene, leave-one-scene-out) — decides "one head or
  six". Closed #29, #31.
- Remaining before drafting: H2.3 result; VGGT-Omega variant (next CPU task:
  find the feature hook; its checkpoint lives on the box, so the run itself
  is a ticket); pose-stability statement (by construction, document it).

## 2026-08-21 (tick 8) — H2.4 prepared: VGGT-Omega head code validated structurally

- No VGGT-Omega weights on the Mac (empty HF cache) — the run is box-only.
- omega_head.py written against the local vggt_omega source: tokens =
  aggregated_tokens_list[-1][patch_token_start:], depth = dense_head planar z
  converted once to range via the KB4 camera; patch 16, input 512.
- Random-init smoke first produced all-NaN (untrained 24-block model
  explodes) — added a sanitize mode (random-init only, loudly labeled) and
  the full fit/eval path now runs end to end with finite dummies.
- Ticket 031 filed for the six-sequence weighted run. #32 (cross-scene)
  still pending on the box.

## 2026-08-21 (tick 9) — provenance audit ahead of drafting

- #32 / #33 still pending on the box.
- Every headline number re-derived programmatically from its source JSON —
  all match. Chain recorded in autoresearch/paper/numbers.md (with the
  caveats that must travel with each number). The paper can now be drafted
  without a single unverified quote.

## 2026-08-21 (tick 10) — human flags a GT-provenance hole in the hand numbers

- The human pointed out two things we had not verified: (1) ADT GT depth is
  rendered in two variants (with/without the person) — if #28/#31 used the
  without variant, the "hands at 0.26-0.94 m" depths are the BACKGROUND
  behind the hand, not the hand; (2) the synthetic RGB stream is a natural
  hands-free counterfactual, cleaner than mean-color masking.
- Filed ticket 032 (issue #34): verify which variants #28/#31 used, measure
  the with-vs-without depth difference at dynamic pixels, and re-run the #31
  depth comparison with real-vs-synthetic input. H4 reopened as
  under-verification; numbers.md rows marked. Theta-placement (from
  segmentation) unaffected; the six-sequence H2 results unaffected (no
  humans in those sequences).

## 2026-08-22 — H3 closed with a number; a bug caught by its own protocol

- run_014 invalid: arbitrary tangent basis rotated every patch by its azimuth
  (pinwheel shattering) — caught by the visual sanity check the protocol
  mandated. Fixed with true Jacobian linearization (identity to first order,
  correct orientation/anisotropy).
- run_014b: effect is null to the 3rd decimal — and the warp's own magnitude
  explains why: within-patch distortion on Aria KB4 at patch 14 is <=0.21 px
  even at the rim. The distortion lives BETWEEN patches, not inside them.
  N2's patch-content branch (incl. DarSwin-style content resampling at this
  granularity) is closed with a measured reason; also explains RayTun3R's
  "patch undistortion minimal" ablation row quantitatively.
- All four axes now have measured dispositions: N1+N2(inter-patch)+N3 unified
  in the compression/head story; N2(patch-content) closed; N4 under
  verification (#34). Still pending: #32, #33, #34 on the box.

## 2026-08-22 (tick 12) — findings brought current; paper skeleton drafted

- #32/#33/#34 still pending on the box.
- findings.md Key Results now carries the six-seq confirmation, H4.1, and H3.
- paper/outline.md drafted: 7 sections, contribution list, figure inventory,
  and the three claims that wait on the pending tickets. Drafting proper
  starts when they land (ml-paper-writing skill at that point).

## 2026-08-22/23 — outer loop cycle 3: #32 and #33 land; only #34 left

- #32 cross-scene: ONE HEAD, NOT SIX — five clean folds -74.5..-78.0%
  (matching or exceeding within-scene); decoration fold -18.7% with real
  center collateral (+66.5% near-center) — genre shift is the honest boundary.
  48-param cross-scene control not run (GPU time budget; flagged, optional).
- #33 VGGT-Omega: improves on all 12 runs (-19..-41%) — backbone-agnostic
  directionally — but weaker than DA3 and with halves-split center collateral
  up to +50%. Final-block tokens carry less disambiguating signal; probing
  earlier blocks is future work. The three-axis eval caught exactly what it
  was designed to catch: readout-only is pose-safe by construction, but
  center-depth safety is empirical and backbone-dependent.
- Direction: CONCLUDE once #34 answers the hand-GT question. Paper skeleton
  ready (paper/outline.md); all other numbers audited.
