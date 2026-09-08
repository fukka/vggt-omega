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

## 2026-08-23 — #34 lands: provenance confirmed; outer loop -> CONCLUDE

- #34: (i) #28/#31 used consistent with-skeleton seg+depth; (ii) with-skeleton
  depth contains the person (100% of dyn px differ >5cm from the without
  variant); (iii) #28's hand depths CONFIRMED on a fresh sample; (iv)
  synthetic stream is hands-free but (v) unusable as the hands counterfactual
  — DA3's synthetic-domain gap (+7..34% AbsRel) swamps the effect; mean-fill
  masking remains the right tool. Meal anomaly does not reproduce under
  synthetic; isolated, open.
- The human's provenance challenge is resolved in the best way: the numbers
  were right, and now they are PROVEN right — the verification goes in the
  paper's supplementary.
- All hypotheses dispositioned. Outer loop: CONCLUDE. Drafting begins next
  tick via the ml-paper-writing skill, from paper/outline.md +
  paper/numbers.md + findings.md.

## 2026-08-23 (tick 14) — paper drafting begins

- ml-paper-writing skill invoked; NeurIPS preprint template staged at
  autoresearch/paper/draft/ with the five figures.
- Full first draft of main.tex written: abstract (5-sentence form), intro
  with 4 contribution bullets, related work positioned on "readout-only vs
  touches-the-computation", the two measurement sections, the two-rung
  method, six-seq + cross-scene + backbone-transfer experiments, honest
  limitations. Every number traced to paper/numbers.md.
- Citations: assembled from the repo's verified survey, NOT yet fetched
  programmatically — 3 entries marked PLACEHOLDER; full fetch pass required
  (paper/CITATIONS-TODO.md). No LaTeX on this Mac; compile via
  tectonic/Overleaf next.

## 2026-08-23 (tick 15) — HUMAN PIVOT: CVPR bar requires method novelty + benchmarks

- The human reviewed the draft direction and ruled: diagnosis + readout head
  is not enough for CVPR or products. Required: (1) architecture novelty
  (efficient fisheye single-image/video), (2) finetuning that improves the
  backbone, (3) public-dataset evaluations with published baselines on BOTH
  pose and depth.
- Accepted. Paper drafting PAUSED; draft becomes the motivation/analysis
  skeleton. New method phase: H5 (rim-targeted pose-preserving finetuning),
  H6 (peripheral cross-frame attention for video), BENCH (the baseline
  matrix — RayTun3R/Fisheye3R/CAM3R reproductions in-repo are the moat).
- The diagnosis assets are not wasted: every H5/H6 component is dictated by
  a measured finding, which is exactly the story CVPR method papers want.

## 2026-08-23 (tick 16) — H5 protocol + losses; a shared-library bug found and fixed

- H5 protocol locked (three losses, scene-level holdout, plain-LoRA control,
  success gates); losses.py implemented as pure tensor functions with a
  5-part CPU test suite.
- The identity-warp test caught a REAL bug in raytun3r's KB4 inversion:
  plain Newton from theta=r overshoots past the turnover for Aria's k1>0,
  landing ~1 px wrong in the outermost ~5 deg of the cone (float64 too).
  Impact review of prior results: theta-binned tables (6.9 deg bins) and
  gate-checked pose runs are insensitive at this magnitude — no conclusion
  changes — but differentiable warping cannot tolerate it. Fixed with
  bisection-safeguarded Newton (max round-trip error 0.97 px -> 5e-5 px),
  regression test added, full raytun3r suite (78) + smoke pass.
- Next: LoRA injection + 2-step training smoke on real frames, then the GPU
  training ticket.

## 2026-08-23 (tick 17) — H5 mechanics verified end to end on CPU

- lora.py (dependency-free LoRA with exact-teacher toggle) + train_smoke.py.
- Discovery: DA3's DPT head is conv-only, so the protocol's "head LoRA"
  share starts empty (documented deviation; conv-LoRA is the fallback).
  Injection = last-4 ViT block MLPs, 8 linears, 122.9k trainable vs 34.3M
  frozen; cam_enc/cam_dec (pose modules) unmatched by construction.
- Smoke PASSED, five checks: intended-layers-only training; LoRA-disabled
  path bit-identical to pristine (the teacher costs no memory); all three
  losses backprop on real ADT frames with the official calibration; total
  and depth losses decrease over 5 steps; base weights bit-identical after
  training. First gate (2-step decrease) was too strict for a composite
  objective whose feat term starts at exactly zero — relaxed to 5 steps,
  a scaffolding change, not a result.
- Next: train.py (4-scene loader, pairs, epochs, LoRA checkpoints, eval
  hooks), then the GPU training ticket.

## 2026-08-23 (tick 18) — H5 trainer shipped; training ticket #35 filed

- train.py: multi-sequence loader, GT-pose pairs via the official
  calibration, three-loss objective with --depth-alpha 0 as half of the
  plain-LoRA control, LoRA-only checkpoints (~500 KB). CPU smoke: 2 tiny
  epochs on seq131, loss decreasing, checkpoint + log written.
- Ticket 033 / issue #35: two training runs on the box (full method vs
  plain-LoRA control), 4 clean sequences train, seq136 + decoration held
  out. Evaluation deliberately stays CPU-side — the checkpoints are tiny
  and come back via results.
- Next: eval_lora.py (joint-table depth + RRA/RTA pose on held-out scenes,
  loading the LoRA checkpoints) so it is ready when #35 lands.

## 2026-08-24 (tick 19) — eval_lora ready; #35 amended with the eval half

- eval_lora.py: one model, LoRA toggled — before/after share every other bit;
  depth joint table + zones + pose (median rot, RRA@15, tdir, RTA@15) against
  official-calibration GT. Smoke-tested with the trainer's smoke checkpoint.
- Logistics reality: held-out scenes live only on the box, so evaluation goes
  back into #35 as an addendum comment (script committed, still runs-only).
  Pose note recorded: adjacent pairs at 100-frame spacing are ~60deg
  rotations — the box must pair closer frames for a meaningful RRA.
- While #35 runs: next up is the H6 (peripheral cross-frame attention)
  protocol and the BENCH protocol skeletons.

## 2026-08-24 (tick 20) — H6 protocol + BENCH matrix locked; #35 still training

- H6 (peripheral cross-frame attention) protocol locked: rim-query
  cross-frame block, zero-init, ~0.6M params; pose-safety premise (DA3's
  camera path is a separate trunk) recorded as TO-VERIFY; the all-token
  same-params control separates the routing story from the efficiency story.
- BENCH matrix locked as a skeleton: datasets (ADT primary, ScanNet++
  cross-lens, KITTI-360/TUM-VI stretch, WideDepth if public), 10 methods
  (3 in-repo reproductions are the moat), metrics incl. our radial breakdown
  as a benchmark contribution; execution order ADT-first.
- #35 (H5 training + eval) still running on the box.
- H6 premise check (same tick): DA3's pose head DOES read backbone feats —
  premise false as stated; resolved better than the fallback: the module
  feeds a depth-head-only copy of the tokens (heads are parallel readouts),
  keeping pose bit-identical by construction. Protocol amended before any
  implementation.

## 2026-08-24 (tick 21) — H6 module implemented and smoke-verified

- peripheral_attn.py: rim-query cross-frame attention, zero-init gate,
  depth-head-only feats copy (camera path reads originals — the verified
  parallel-readout route). Correction to the protocol's estimate: the final
  feats level is dim 768 (not 384), so the module is 2.96M params, not 0.6M
  — still small; recorded.
- module_smoke.py PASSED on real DA3 + ADT frames: zero-init bit-identical
  through the depth head; after one gradient step depth moves while every
  camera-output tensor stays bit-identical; grads reach only the module.
  (One integration lesson: DA3's camera estimation consumes/mutates the head
  output in place — snapshot pattern documented in the smoke.)
- #35 (H5 training) still running on the box. Next: H6 trainer (reuse H5's
  loop with the module in the depth path), then its training ticket.

## 2026-08-24 (tick 22) — H6 trainer smoke-verified; ticket 034 filed

- H6 train.py: two-direction module application per pair, two losses (the
  rim-feature term is structurally unnecessary here), module-only optimizer,
  12 MB checkpoints. Name-collision with the H5 trainer fixed by explicit-
  path import. CPU smoke: loss 1.64->1.50 over 2 tiny epochs.
- Watch item recorded: mv loss hit 0.0 on one 2-pair epoch sample — flagged
  in the ticket as a health check for real training.
- Ticket 034 (H6 rim vs all-token control) filed; eval_module.py to follow
  as addendum. #35 still training.

## 2026-08-25 (tick 23) — H6 eval ready; both training tickets amended and complete on our side

- eval_module.py: video-mode eval (module needs the previous frame), with a
  like-for-like second pass so before/after pool identical frames. Smoke ok.
- #36 amended with the eval addendum (same pattern as #35). Both method
  tracks are now fully specified and runnable end-to-end on the box; the
  Mac side has no blocking work left until checkpoints or evals land.
- Idle-tick plan while training runs: BENCH cell prep (frozen UniK3D/DAC on
  ADT machinery exists in finetune/eval/baselines) and the paper's method
  section can be pre-drafted against the locked protocols.

## 2026-08-25 (tick 24) — BENCH frozen-row machinery built and smoke-tested

- eval_baseline_joint.py: any model_zoo baseline -> protocol-of-record joint
  table + whole-image AbsRel/d1. Three integration fixes found by the smoke:
  registry name, adapters snapping output shape, and the LOADER's own
  token-friendly shape (camera/theta must follow the frames, not --res).
- Smoke (da3_small, 4 local frames): table shape matches the diagnosis
  (near-rim blow-up). Ticket 035 filed for the 6-model x 2-held-out-scene
  frozen rows. In flight now: #35 (H5), #36 (H6), #37 (bench rows).

## 2026-08-25 (tick 25) — quiet GPU tick: state/dashboard synced, method section pre-drafted

- #35/#36/#37 all still in the box queue; no results files yet.
- research-state.yaml and the dashboard brought current (three tickets in
  flight, per-track verification status).
- paper/method-draft.md: the method section as a four-rung ladder
  (table -> readout head -> rim-targeted LoRA -> peripheral attention), each
  rung justified by a measured finding, with the two "deliberately not done"
  paragraphs (patch undistortion <=0.21 px; hands = occlusion). Open table/
  figure slots enumerated so numbers drop straight in when tickets land.

## 2026-08-26 (tick 26) — citation verification pass done; GPU still training

- All 13 arXiv citations fetched programmatically (titles/authors/years
  verified), including the two previously-unverified web finds (DrivingDepth
  confirmed real) and the two found by search (ADT 2306.06362, MAGSAC++
  1912.05909). refs.bib generated from fetched metadata only. Two
  non-arXiv items remain flagged (DepthFisheye/ICCVM, Lowe/CrossRef).
- #35/#36/#37 still in the box queue.

## 2026-08-26 (tick 27) — RayTun3R comparison row prepared; ticket 036 filed

- raytun3r_row.py: joint-table row for a RayTun3R-adapted backbone; vanilla
  path smoke-tested locally (ADTSequence -> backbone -> table end to end);
  adapter loading mirrors raytun3r/eval.py's own pattern.
- Ticket 036 (issue #38) filed as fourth in queue — the most GPU-hungry row
  (2-3h/scene per the paper), per-scene adaptation on each held-out scene
  (fair-to-favorable setting for the baseline).
- Main-table inventory now complete on our side: frozen rows (#37), our
  three rungs (#35/#36 + run_011 in hand), the strongest adapter baseline
  (#38). Everything else is analysis when numbers land.

## 2026-08-26 (tick 28) — H6 rim-mask cone bug caught by the efficiency pass

- Status ping posted on #35 (queue quiet for several ticks; keeping the
  blocker visible per handoff convention).
- Measuring H6's efficiency table exposed a real bug: rim_mask_for did not
  intersect the imaged cone, so the square grid's dead corners (25% of
  tokens) were attention queries. Fixed (rim = theta in (35deg, theta_max]),
  threaded through trainer/eval/smoke, regression smoke PASSED, #36 notified
  to pull before running.
- Corrected efficiency numbers (paper table): rim 627/1296 grid tokens (64%
  of cone); module FLOPs 0.48x of all-token (5.45G vs 11.28G); CPU latency
  30.2 vs 49.0 ms. Saved to h6 results/efficiency.json.

## 2026-08-26 (tick 29) — trajectory figure; GPU still quiet

- trajectory.png: the near-rim metric's march down the method ladder, every
  bar re-read from its audited JSON. Added to the dashboard.
- #35-#38 unchanged; the #35 status ping stands unanswered (the box session
  has presumably not run since).

## 2026-08-26 (tick 30) — H5 CPU pilot: mechanism confirmed on held-out frames

- Exploratory pilot (seq131 even/odd frame split, 252px, 10 epochs, ~3 min
  train on CPU): near-rim -59.7%, and — unlike the H2.1 table — every other
  zone improves too (near-center -31%, center -27%, far -21%). Loss
  components behave as designed (feat term rises then plateaus under the
  distillation pull). Pose unchanged within the saturated-pair regime
  (no-collapse signal only).
- No hyperparameter changes for #35; the pilot's job was direction + sanity
  and it delivered both. Protocol claims stay GPU-gated.

## 2026-08-27 (tick 31) — H6 pilot fails informatively; dense-window fix lands before the box runs

- H6 CPU pilot (same reduced split as H5's): near rim +2.1% — direction NOT
  confirmed. Diagnosis: 3.3s "adjacent" frames violate the module's premise,
  AND the same uniform-subsampling trap was waiting inside the trainer for
  the box run. Seq gains dense=True (contiguous block); H6 trainer/eval use
  it; #36 amended (second pre-run pull).
- Also corrected: H6's center-safety is token-level only (DPT head mixes
  spatially) — structural claims now restricted to the pose path; center
  depth is empirical for H6, same as H5.
- Contrast worth keeping: the H5 pilot (single-frame losses) worked at
  sparse spacing; the H6 pilot (multi-frame core) could not — consistent
  with the mechanism story rather than against it. Real test = #36 dense.

## 2026-08-27 (tick 32) — findings/dashboard absorb the pilots; H6 novelty contrast recorded

- findings.md gains the method-phase status block; dashboard updated.
- H6 novelty search: foveated token selection (FDT) concentrates compute at
  the CENTER — ours is the measured inversion; StableDPT is the nearest
  video-depth mechanism (all-token, no camera geometry). Recorded in
  literature/2026-08-27-h6-novelty.md.
- GPU: 0 deliveries; queue unchanged.

## 2026-08-27 (tick 33) — collator ready; the blocker is named

- collate_main_table.py: one command renders the paper's main-table draft
  from whatever JSONs exist — already fills 6 rows (frozen anchor, rung 0/1
  within-scene, 6-seq mean -63%, cross-scene -73%, omega -24%); the six
  pending rows fill themselves as #35-#38 land.
- Blocker made explicit to the human on the dashboard: the box session has
  not been started since the #32-#34 batch; all four tickets are ready and
  waiting on that single action.

## 2026-08-19 — human-directed literature + comparison + brainstorm (GPU wait)
- Survey (7 arXiv queries, 12 abstracts read): Wid3R / UniDAC / Spark3R are the
  closest lanes; diagnosis-driven adaptation and asymmetric periphery treatment
  remain unclaimed. → literature/2026-08-19-distortion-crossframe-survey.md
- RayTun3R comparison recipe extracted; gaps to compare-the-same-way: Center-PH
  baseline, ScanNet++ 3f15 row (their named sequence = external anchor),
  adaptation-data column. → paper/comparison-protocol.md
- New hypotheses H7 (theta-gated LoRA) and H8 (equal-solid-angle tokenization,
  probe-first) added to state. → literature/2026-08-19-novelty-brainstorm.md
- Next CPU work: Center-PH baseline, H7 pilot, H8 probe (no GPU queue impact).

## 2026-08-19 (tick 2) — Center-PH baseline anchor
- Protocol addendum locked first (predictions P1-P3), then run: seq131 odd,
  DA3-S, 504px, visual rectification check passed.
- P2/P3 confirmed (near_rim coverage 49.6%, cone 75.2%). P1 HALF-REFUTED:
  identical-pixel control shows center flat and near-center +62% WORSE under
  rectification — Center-PH's ScanNet++ depth win does not transfer to
  egocentric near-field ADT. Exploratory until reproduced on held-out scene.
- Next: H7 pilot, H8 remap probe.

## 2026-08-19 (tick 3-4) — ALL FOUR GPU TICKETS DELIVERED; H7 pilot read
- #35/#36 training delivered (curves healthy, mv-loss check passed); evals
  still owed, re-requested with the device-bug fixes landed (53c06c4).
- #37 frozen rows: 5 models x 2 scenes (DAC skipped, ERP-native, documented).
  KEY: held-out scenes have almost no near-field rim mass (seq136 ~50k px in
  0-1m rim cells; decoration_seq132 zero 0-1m px anywhere) — near_rim numbers
  there are small because the regime is absent, not because the failure is.
- #38 RayTun3R rows: adaptation HURTS on both held-out ADT scenes (whole
  0.172->0.229, 0.079->0.126; near-rim worse too; coverage >=0.974 so not
  matcher starvation). Strongest published competitor, negative on our data.
- Loader gap flagged: #38 vanilla vs #37 da3_small near_rim differ 3.6x on
  the same scene — reconcile before any shared table.
- H7 r=8: ties uniform on every zone; gate curve flat (|g-1|~0.06) — the
  backbone's PE already lets LoRA condition spatially. r=4 arms running.
- Fixed GPU-reported device bugs (losses.py x4, lora.py, raytun3r_row.py);
  5/5 loss tests pass. #37/#38 closed with consuming comments.

## 2026-08-19 (tick 4b) — H7 REFUTED cleanly
- gated==uniform at both ranks; gate flat; rank matters (r4 0.75 vs r8 0.57)
  but conditioning does not substitute. Mechanism recorded in analysis.md.
  Paper ablation; H7 closed without GPU spend.

## 2026-08-19 (tick 5) — #38 v1 invalidated by our own double-conversion bug
- Reconciling #37 vs #38 (3.6x near_rim gap) found raytun3r_row.py dividing
  already-range GT by cos again. Fixed (8b5c13d); #38 reopened, 4-row re-run
  requested; local seq131 vanilla cross-check running. v1 numbers quarantined.
- Lesson: eval scripts must assert GT domain against the loader's declared
  convention (the variable was even named gt_z while holding range).

## 2026-08-19 (tick 6) — H8 refuted by probe A
- Equisolid remap: +31% near-rim, +16% center at same token count. Radial
  failure is not sampling density. Probe B cancelled. Pattern recorded:
  input-space geometry surgery hurts; effective fixes are behind the encoder.

## 2026-08-19 (tick 7) — paper skeleton v2
- outline.md rewritten around the post-pivot story (asymmetry → where-to-
  intervene → ladder → radially-honest benchmark); numbers.md audited
  through today incl. quarantines. Dashboard rebuilt earlier per human
  request (method diagrams + results gallery). Awaiting 3 GPU eval batches.

## 2026-08-19 (tick 8) — H6.1 confirmed (exploratory): rim-KV == full-KV
- Spark3R method read (query merge / KV prune / layer-adaptive). Probe on
  delivered rim ckpt: rim-KV 627 == full 1296 on near_rim; center-KV +18%.
  Module cost 0.48x -> 0.23x of all-token, training-free. Held-out version
  appended to #36. H6.2 pyramid design recorded.

## 2026-08-19 (tick 9) — box alive (delivered fovbench-019, other lane); #39 posted
- ScanNet++ 3f15 pose-anchor + cross-lens probe posted as issue #39 (queue
  tail; eval batches for #35/#36/#38 remain the priority).

## 2026-08-19 (tick 10) — H6.2 probe: +1.9% free from t-2 KV, saturates at t-2
- Pyramid shape confirmed, magnitude thin; no GPU spend; revisit after #36.

## 2026-08-19 (tick 11) — external H2 review accepted in full
- Items 1-2 upheld (percentage headlines, cross-sequence naming, 82% affine
  placement), item 3 refuted our-way (features not geometry; control
  imported). Camera guard added. seq136 mass question REOPENED (frame-window
  artifact suspected: 1.359 vs 0.149). All records corrected same-tick.

## 2026-08-19 (tick 12) — alignment-robustness rows on the record harness
- near-fit affine: near_rim -75%, far +2.9x — no single affine serves both;
  compression field confirmed from the alignment side. Records updated.

## 2026-08-19 (tick 13) — external-validity plan (human-directed)
- Assessment: ideas+internal validity solid; external validity is the gap.
- Found in-repo published protocols: DAC cross-dataset splits, CAM3R ADT
  pose. Same-camera cross-scene has NO published protocol -> ours on ADT
  non-Apartment scenes. #40/#41 posted (behind eval batches).

## 2026-08-19 (tick 14) — cross_scene per-camera refactor landed
- Geometry per-sequence, cross-camera announced not asserted; smoke passed
  (in-sample seq131, near_rim -67.9% consistent). #40/#41 can now run the
  head cross-lens without code danger.

## 2026-08-19 (tick 15) — per-frame zone output in eval_lora (error-bar policy)
- Regression-identical zones; bootstrap demonstrated (r4 pilot near_rim 90%
  CI [0.65, 1.11]). #35 asked to pull once more before evals.

## 2026-08-19 (tick 16) — per-frame zones in eval_module too; both eval scripts error-bar ready

## 2026-08-19 (tick 17) — human-directed multi-agent architecture brainstorm
- Q1 answered (data sufficiency): 4 seqs enough to learn the lens, not enough
  for a generalization claim; loss still falling at ep20 -> widen data.
- 3 agents (Fable/Opus/Sonnet) x ~18 proposals + adversarial cross-critique.
  Record: literature/2026-08-19-arch-brainstorm/. Survivors -> H9 (RayCal-TTA
  label-free field removal), H10 (FrozenMatch decoupled pose), H11 (parallax-
  gated epipolar rim attention, t-8/t-16 horizon). Unanimous kills incl. the
  center/rim dual-expert MoE (H7+F2+F4 evidence chain).
- Cross-critique-only finds: two-annuli problem (pose value 35-45 vs depth
  liability 45-55), Aria SLAM-cam static baseline, hand-motion triangulation
  poisoning, rolling shutter. Pre-checks scheduled before GPU commitment.
- Dashboard gained section 二·八 (brainstorm conclusions + Q1 answer).

## 2026-08-21 (tick 18) — slamfov #23 radial half consumed; #40 replanned; plan re-ordered
- #23 (results 8e5fb5e): oracle null catches the depth confound in the act on
  real SLAM points (1.63-2.84x pooled with zero field dependence); after
  distance standardisation ALL 5 models exceed the null -> rim degradation is
  real on real footage. Accuracy vs evenness are different axes (vggt_omega
  most accurate AND steepest field) -> the paper's motivation figure.
- Protocol adoption: oracle-null row + two-way table become part of our bench
  protocol; pooled numbers no longer citable alone.
- #40 replanned by GPU recon (accepted with conditions): ADT has NO
  non-Apartment scene; seq132 bedroom segment (502 annotated frames) ->
  cross-ROOM probe; ladder cross-sequence < cross-room < cross-building.
- #41 Part B unblocked (meshes + renderpy on box). Priority posted on #40:
  #35/#36 evals -> bedroom probe -> ScanNet++ render -> #38 v2.
- Dashboard: new section 二·九 (result + revised plan), #40/#41 cards updated.

## Tick 19 — 2026-08-22

- Consumed the decisive deliveries (results aef0f04 + 1158e27): #35 rim losses
  LOSE to plain LoRA; #36 rim-KV LOSES to all-token on held-out; #38 v2 mixed
  (helps seq136, hurts dec_seq132); oracle null must be per-model.
- Added the missing error bars myself: paired bootstrap on delivered per_frame
  (data/bootstrap_h5_h6_2026-08-22.md) — both kills significant on seq136,
  H6's on both seqs.
- State updates: H5 refuted-by-control, H6 refuted-on-held-out, H11 blocked
  (precondition failed), H12 registered (GPU-authored lens-Jacobian FiLM,
  pre-registered shuffled-field kill bar), plain-LoRA = standing baseline.
- findings.md: new section "The reframe (2026-08-22)" — rim deficit is a
  global lens-prior mismatch; four region-targeted failures vs global wins.
- Dashboard: new section 二·十 (full experiment review + verdict + regear),
  header/nav/queue rows updated. Consumption comment with bootstrap CIs
  posted on #35; H12 pilot declared GPU priority 1.
- POLICY correction noted: this Mac no longer imports torch — local "verified"
  claims now only cover pure-python/numpy tests.

## Tick 20 — 2026-08-24

- Pulled: H12 pilot RAN AND LOST (results da38331) — jac worst of three arms on
  both held-out sequences; GPU stopped on the pre-registered criterion. Fifth
  controlled negative. GPU also opened H13 (teacher caching for a distillation
  ladder, organized 3afbace).
- Mac post-mortem (data/h12_gradient_and_field_sensitivity_2026-08-24.md), run
  locally — jacobian.py is pure numpy and DOES run on this Mac (10 tests/0.86s):
  (1) jac-minus-control is monotone in theta, corr +0.66 (seq132) / +0.24
  (seq136) against BOTH controls; seq132's nearest-depth column perfectly
  monotone over 8 rings, crossing zero at 38-45 deg. Real geometry helps on
  axis, hurts at the rim, and beats theta-only at the centre so it is content
  not just radiality. (2) log_aniso is 10-40x more sensitive to KB4 coefficient
  perturbation at the rim than mid-field (~31% vs ~5% relative). Reading:
  conditioning is a CENTRE tool on this lens.
- State: H12 refuted-by-preregistered-criterion; H13 registered with two
  evidence-based guardrails (teacher has the steepest rim field of the five
  benchmarked; VGGT-conf gating already measured worse than ungated on the
  DAv2 work); all refuted hypotheses' priorities set to closed. Live set is now
  BENCH, H9, H10, H13 (+ H2 family, H11 blocked).
- Strategy: every "rim as a separate sub-problem" hypothesis is closed. H9
  promoted to priority 1 — its locked bar is global by construction, the one
  property every surviving intervention shares.
- Posted to #35: the two analyses, the re-prioritisation, the H13 guardrails,
  and two cheap asks (per_frame re-emit for H12; SLAM-stream enumeration, which
  gates H9's design and cannot be checked locally — the Mac sample has
  groundtruth/ only, no VRS).
- Dashboard: new section 二·十一; 二·十's ledger and plan rows updated.

## 2026-09-08 — the roll/orientation line (H17)

Prompted by the user asking whether depth foundation models all assume a level
camera, whether egocentric data violates that, whether an IMU can supply the
roll, and whether feeding it in would help. Four sub-hypotheses were written and
locked in `experiments/h17-roll-prior/protocol.md` before H17.2 and H17.3 ran
(commit a17324f, results in 1c3e9a0 / a95c315 / 867ac5c).

**Order of events, including the two things I got wrong.**

1. Literature first: found arXiv 2608.00678 ("Breaking the Horizontal Prior"),
   which reports the same effect on four single-image depth models, fixes it
   with a training-time regulariser, uses no IMU, and whose best algorithmic
   roll estimator is off by 25.9 deg. Saved to `literature/`. It has no fisheye,
   no egocentric data, and never measures a real roll distribution — which set
   the shape of H17.1 and H17.2.
2. H17.1 (measurement): read ADT's roll from the MPS gravity vector. Median 3.7
   deg, p99 21.8, none past 40. Also confirmed UPRIGHT_K=3 from geometry.
3. **Correction to my own published claim.** Re-reading h16's three arms while
   writing this up, the report's "two thirds of the roll fall-off is the black
   wedge" is not supported: the arm that HOLDS the black region fixed is the
   steepest. Projection and boundary are not separated by the arms that exist.
   The claim was removed from the report and the honest decomposition, plus the
   fourth arm that would actually settle it, are written down instead.
4. H17.2: the locked bar passed cleanly and this is the line's best result —
   multi-view-pretrained backbones are 4-5x less roll-sensitive at 30 deg, and
   capacity does not substitute.
5. **A readout bug that looked like a finding.** The first roll probe regressed
   (sin, cos) and took atan2, giving MAE 70 deg — worse than predicting a
   constant, which reads as "roll is unreadable". The label-permutation null is
   what exposed it (a linear probe cannot legitimately lose to a constant).
   Within +-45 deg, cos spans only [0.707, 1]; direct angle regression took MAE
   from 70 to 12.75. The null now runs by default.
6. H17.3 landed between its two bars and, more usefully, split by room: 6.8 deg
   same-room vs 18.7 deg on the rearranged one.

**Direction: the line is scientifically closed and engineering-deprioritised.**
H17.4 (derotate / condition / augment) stays locked but unrun, because H17.2 +
H17.1 together leave almost nothing for a gravity input to recover on ADT. The
bottleneck is a wider-pose egocentric dataset with dense depth, not a method.

In parallel, on the user's instruction to increase ADT training frames: the data
ladder was re-run with gradient steps pinned near 48k. 48x the distinct frames
buys nothing on the honest held-out sequence and triples the near-centre damage.

## 2026-09-08 (later) — H17.5, the dose curve, and a build-process failure

**A process failure I have to record, because it made a previous report to the
user inaccurate.** The report is assembled from `body{1..4}.html` fragments in
a scratchpad. On 2026-09-08 I wrote a corrected section 02 into `body2a.html`,
merged it into `body2.html`, and then continued editing `body2a.html` — which
by then was no longer the file the build reads. Two later rebuilds therefore
silently republished the *stale* section 02, still containing the "two thirds
of the fall-off is the black wedge" claim I had already retracted, while I told
the user the correction was live. Fixed by replacing section 02 in `body2.html`
wholesale from the corrected copy, adding an assertion in the build that fails
on the known stale strings, and renaming the intermediate fragments to
`_merged_*` so they cannot be edited-then-ignored again.

**H17.5** (locked before running) put the boundary into the clean arm: mask the
rectified view to its inscribed disc. +84% at 30 deg, between the two
thresholds, so "both contribute". But the protocol's premise — that this mask
is a near-exact analogue of `fisheye_disc` — is wrong by an order of magnitude
(21.9% of the frame vs 1.7% of the cone), so the intended decomposition did not
happen. Recorded as a flawed design rather than a result.

**The dose curve** then produced the most useful thing in this block, and it
corrects a rule I wrote myself yesterday. Sweeping border width from 3.1% to
53.4% of the frame, scored 14 deg away from the border, gives +22 / +13 / +17 /
+17 / +6.6 / +32 percent — non-monotone, no dose-response, noise-scale spread.
Only the step survives. Against the +106% measured with the score taken right
against the border, the variable is **distance, not area**. The standing rule in
research-state.yaml was rewritten, and the new form explains H14's 110 deg
teacher better than the old one did.

Two guards were added along the way and both earned their keep: the dose script
refuses any border that would eat into the scored cap (it caught that an 89 deg
view leaves only 5 px outside a 44 deg cap), and the probe prints a
label-permutation null (it caught the atan2 readout bug earlier in the day).

## 2026-09-08 (later still) — H17.6, and the ladder closes

**The ladder is complete** (48k gradient steps held fixed, distinct frames
240 → 2,400 → 4,800 → 11,418): seq136 rim −67.1 → −71.3%, decoration_seq132 rim
−21.7 → −20.9% (flat), decoration_seq132 near-centre +58 → +145% (monotone).
48x the data at constant compute moves the honest held-out sequence not at all
and multiplies its near-centre damage by 2.5.

**H17.6 was locked, run, and falsified**, which is the most useful outcome this
block has produced. The prediction was that border robustness would split by
pretraining family the way roll robustness did. It does not:
vggt_omega +29%, da3:small +106%, vggt +107%, da3:large +636%. Roll and border
robustness are different axes; multi-view pretraining buys the first and not the
second.

Two things follow, and one of them is the pay-off for the whole h17 block.

* **DA3-Large's accuracy is brittle.** Best of the DA3 pair on a clean frame
  (0.0690 vs 0.1545), worst of all four with one border (0.5077). A 7.4x
  degradation from a change touching no scored pixel. Picking a backbone on
  clean-benchmark accuracy is actively misleading here.
* **The 110 deg teacher door opens, for exactly one backbone.** The protocol
  pre-committed this decision before the numbers existed: under +40% means run
  it. Only VGGT-Omega qualified — and it is also the most accurate and the most
  roll-robust model measured. So H14's next arm is a 110 deg VGGT-Omega
  teacher with 100% cone coverage, which dissolves the accuracy-vs-coverage
  tension that killed H14 rather than trading along it.

The build now asserts on the stale strings from this morning's process failure,
and it caught one unbalanced `</div>` in this session's edits before publishing.

## 2026-09-08 — H18: the roll line pays off in H14

The chain the whole h17 block turned out to be: roll sensitivity is a
pretraining-data property (H17.2) -> the black border is what actually hurts,
and by distance not area (02e dose curve) -> border tolerance is
backbone-specific, VGGT-Omega +29% vs DA3-Small +106% vs DA3-Large +636%
(H17.6) -> therefore VGGT-Omega can run the 110 deg teacher that inverted for
DA3 -> H18.

Pre-check first, as H14 always does: same view, same 22.5% black, DA3-Small
teacher +33/+39% (inverts), VGGT-Omega -65.0%, both at 100% rim coverage. H14's
shipped teacher was -14.7% at 70%. Bars were locked before training.

Students: -52.7% / -22.8% at the near rim with **no depth labels**, against a
labelled ceiling of -57.9% / -27.2%. 91% and 84% of the label gain. Three of
four bars pass; near_center on dec_seq132 misses by 2.4 points.

**The control changed what this means.** I ran omega_rt (same teacher, same
resampling, no rectification) after seeing -52.7%, because without it the number
cannot distinguish "the rectified projection unlocked something" from "a
stronger model was distilled". It recovers 81% / 66% on its own. So teacher
strength is first-order; H14's mechanism is real but second-order (5.7 / 4.9
points) and is also what causes the near_center failure. The report says this
plainly rather than quoting -52.7% on its own.

Two implementation notes worth keeping. `cache_teacher.py` now takes
`--teacher-model`, and its roundtrip arm follows the teacher rather than the
student — otherwise the matched control is impossible to express. And a patch-16
teacher cannot take the 504 px fisheye frame; padding was rejected on the
strength of 02e's dose curve, so it resizes 504 -> 512 -> 504, a 1.6% rescale
against `resample0` measuring one bilinear pass as free.

## 2026-09-08 — H18 cross-room: the decomposition inverts, and I had published the wrong half

Followed the state's own next-steps: seeds, then cross-room. Both mattered, the
second decisively.

**Three seeds.** omega110 -51.5+-2.0 / -22.4+-1.8, omega_rt -47.6+-1.1 /
-17.8+-1.0. Bar 2 passes on all three. Bar 3 (near_center) fails on all three
(+12.4 / +20.1 / +16.9), so that failure is robust rather than a one-seed
artefact — worth knowing, since I had reported it as a 2.4-point miss.

**Cross-room inverts the reading.** On LiteOffice: omega110 -20.0 / -9.0, rect
-28.5 / -10.7, gt -11.8 / -5.0, and **omega_rt +0.8 / -0.1**. The arm that
carried most of the in-room gain transfers nothing at all; every rectified arm
transfers. So:

    in-room:      teacher strength first-order, rectification +3.9 points
    across rooms: teacher strength ~0,          rectification is the whole effect

An hour earlier I published "teacher strength is first-order, H14's mechanism
second-order" with no qualifier. It is true in-room only, and this project has
found repeatedly that the cross-room test is the honest one — so the published
sentence had the mechanism backwards for the test that matters. Section 03b now
carries both halves and says which one was wrong.

Two further things fell out. The **weaker** DA3 rectified teacher transfers
*best* (-28.5 / -10.7) while being far worse in-room, so stronger teacher buys
in-room gain and costs a little transfer. And both rectified arms beat the
labelled ceiling across rooms, extending the earlier cross-room reversal to a
much stronger teacher.

Hypothesis recorded, not tested: the rectified path teaches a systematic radial
relation, which is low-dimensional enough that a 122.9k-parameter LoRA can only
represent it globally; the fisheye path teaches "be like VGGT-Omega on these
images", which is high-dimensional enough to be fitted per room. The next
experiment in the state file tests exactly that.

## 2026-09-08 — H18.2: the mechanism, measured

Followed the state's own next step: test the recorded hypothesis that the
rectified target transfers because it is a low-dimensional radial relation.
Protocol locked first (2a0751e), including the falsification conditions.

Replace the student with a 16-parameter per-theta-bin log-log curve, fit on the
four training sequences only, apply unchanged. near_rim vs frozen: seq136
-16.6%, dec_seq132 -6.5%, DinoToy -26.0%, BlackCeramicBowl -8.9%, against the
LoRA's -51.5 / -22.4 / -20.0 / -9.0. **In-room the LoRA is 3x better; across
rooms 16 numbers match or beat 122,900.** The `global` control (theta dependence
removed) is worse than doing nothing on 3 of 4 sequences, so it is radial
structure and not a rescale — the falsification condition did not fire.

This is the mechanism the cross-room inversion was pointing at, now measured.
The rectified teacher's transferable content is a per-theta recalibration of the
frozen model's depth: a property of the lens, not the room.

The result that pleases me most is unplanned. H9 fit the **same functional form**
from parallax anchors and got -15.4% on seq136 where this gets -16.6%. Two
label-free supervision sources with nothing in common — triangulated metric
anchors from camera motion, and a stronger model's predictions under a change of
projection — land on the same 16-parameter object at the same magnitude. H9
needed per-sequence test-time adaptation; this curve is fit once and moved to
another room unchanged. Two independent lines of this project turn out to have
been estimating the same thing.

Next steps recorded: fit the curve on LiteOffice and compare (is it a lens
constant?), fit it per backbone (or is it a lens x backbone constant?), and
scale the cross-room evidence, which is now carrying two separate claims on 120
frames.

## 2026-09-08 — H18.3/H18.4: the reverse direction, and a second claim withdrawn

Followed the state's own next steps. Both bars were locked before running
(178e5f4), including the falsification lines.

**H18.3 split.** Fitting the curve on LiteOffice and bringing it back to the
Apartment recovers 78% / 95% of the gain — the >=60% bar passes — but the
coefficients differ by mean 0.493 against a <0.10 bar and a >0.20 falsification
line. So the transfer is real and the *numbers* are not a constant.

I then asked, exploratorily, whether that was just the fitting set: a log-log
slope is not identified independently of its depth range, and LiteOffice spans
0.41-4.66 m against the Apartment's 0.44-10 m. Restricting both to 0.5-4.5 m
leaves the Apartment fit **bit-identical** and makes the LiteOffice fit **worse**
(-7.3%, +1.2%). That settles it in the unflattering direction: the LiteOffice
curve is unstable to how its pixels are chosen, so 120 near-static frames cannot
estimate the object. The asymmetry is about the fitting set, not about which end
is the lens's home.

**H18.4 confirmed**: mean |da| 0.297 against a predicted >0.20, so the curve is
backbone-specific — though its *shape* correlates at 0.763 and it is mostly the
level that moves.

**Second withdrawal this session.** One tick ago I wrote "ship a 16-number
radial calibration instead of a LoRA; it transfers better". That claimed a
constant, and it is not one. What stands is a calibration *procedure* per
(lens, backbone) fitted on a few hundred frames with motion. The first
withdrawal was the black-wedge attribution in section 02. Both were published
before the control that would have caught them existed, and in both cases the
control was cheap and ran one tick later — the pattern to fix is publishing the
headline before the matched control, not the controls themselves.

Open, and now the most interesting question in this line: **what makes a fitting
set sufficient?** The Apartment fit is invariant to range matching and the
LiteOffice one is not. Sweeping the number of fitting frames until the curve
stabilises would give the number a calibration procedure has to quote.

## 2026-09-08 — lambda_63 went down; protocol and runner written ahead of it

The box stopped answering SSH partway through this tick (three retries, all
timeouts). It is shared and reboots unannounced — that is in the project notes —
so no GPU work was launched. A single bounded waiter is running rather than a
poll loop.

Used the time for work that does not need it:

* **H18.5 locked** — the question H18.3 left open, and the most interesting one
  in this line: what makes a fitting set sufficient? Frame count 2..240 crossed
  with spread-over-four-sequences vs all-from-one, five draws per cell,
  reporting coefficient stability (mean pairwise |da| between draws, needs no
  held-out data) alongside cross-room transfer. The spread/single split is the
  part that matters: it separates "needs more pixels" from "needs more
  viewpoints", which LiteOffice had confounded, and those have very different
  costs in a deployment calibration.
* **Runner written and committed** (`radial_sweep.py`), one GPU pass then the
  whole sweep in numpy, so it costs a single forward pass over the data when the
  box returns.
* **Drew the curves.** Section 03c argued about coefficient shapes from a table.
  The figure now shows all three fits: the two Apartment ones are smooth and
  unimodal and differ mostly in level (correlation 0.763), while the LiteOffice
  one swings between 0.60 and 1.26 and crosses below 1 in four bins. That makes
  the claim visible — it is not "another lens's curve", it is a curve that was
  not estimated.

## 2026-09-08 — box still down; the report's conclusions caught up with its sections

lambda_63 still unreachable, so no GPU work again. H18.5 stays locked and its
runner stays committed and unrun.

Spent the tick on something that was genuinely owed: **section 08 had not moved
since before the whole H17/H18 block.** Seven new sections had been added
(02b-02e, 03b, 03c) and the conclusions still listed roll augmentation as the
top priority and did not mention the border/backbone finding, the label-free
distillation result, or the 16-number curve at all. Rewritten and regrouped:

* what is settled about the rim penalty itself;
* what is settled about pose and framing — including the two findings with the
  widest reach outside this project, that a hard border's cost is set by
  DISTANCE not area, and that DA3-Large is the most accurate DA3 on a clean
  frame and the worst of four with a border;
* what is settled about label-free adaptation — the in-room/cross-room
  inversion and the 16-number curve, with the constant claim already withdrawn;
* the clean negatives;
* an honest uncertainty list, now including that the 120-frame cross-room set
  carries several claims at once;
* a next-steps list that reflects reality: roll augmentation is explicitly
  **demoted**, with the reason (section 02b says change the backbone, section
  02c says only 1.5% of ADT frames go past 20 degrees), and kept only for the
  extrapolable thing its second bar would measure.

Also added the postscript H9 deserved: the curve it fitted from parallax anchors
is the same object section 03c fitted from a rectified teacher, -15.4% against
-16.6%. Two independent lines of this project were estimating the same thing,
and the reader of section 06 should not have to reach section 03c to find out.

## 2026-09-08 — box down a third tick; wrote the tests the new code never had

lambda_63 still unreachable (~45 min). The bounded waiter exhausted its budget
and was not re-armed — the 20-minute cron re-checks anyway, and a second waiter
would be redundant.

First did an integrity check, since a long outage is exactly when work gets
stranded: working tree clean, `organized` level with origin, zero unpushed
commits, and all 27 h17/h18 result JSONs present in the repo rather than only
on the box. So every conclusion on the report is reproducible from git even if
the box never returns.

Then wrote `h17-roll-prior/code/test_roll.py`. Nine scripts went in this session
and none had a test, which is out of line with how the rest of this repo is
built — and two of them produce numbers quoted as headlines. Twelve tests, all
passing:

* the roll read: a level camera must read zero, a camera rolled by phi must read
  phi back, and the result must not depend on gravity's magnitude or on which
  arbitrary device frame the camera is bolted into. The sign convention is now
  pinned too — `rolls_deg` returns the angle the IMAGE rotates by, which is the
  negative of the camera's roll. Harmless (02c reports |roll|) but it was
  undocumented.
* the radial fit: recovers a generated curve exactly, stays unbiased under
  noise, and applying it inverts the fit.
* the 02e border geometry: the half-edge/corner arithmetic that justifies
  scoring at 30 deg rather than 44, and the black fractions quoted in the text.

The RolledView rotation test (optical axis must not move — that is what makes it
a roll and not a tilt, and it is why the clean arm stays inside the cone) needs
the full env and skips cleanly here.

## 2026-09-08 — seventh tick with the box down; found real work that did not need it

lambda_63 still unreachable (>1.5 h). Rather than hold again, went looking for
research — not chores — that the outage does not block, and there was some.

**H18.6.** The three fitted curves are already committed as JSON, and the figure
drawn two ticks ago makes it obvious that the two Apartment ones are smooth and
the LiteOffice one is not. That is testable without a GPU: fit a quadratic in
theta/theta_max through the eight per-bin coefficients and see how much it
captures. Apartment 0.936–0.993; LiteOffice 0.071–0.590. The contrast is large
and holds across four independent fits.

Two things come out of it, and one of them is more useful than I expected.

* **The 16 numbers are really 6.** Three coefficients for the exponent and three
  for the offset reproduce the working fits to within 3–14% of their own spread.
  A calibration procedure ships six numbers and a smooth form that does not step
  at bin edges.
* **Smoothness is a fit-quality check that needs no held-out data.** This is the
  useful one. H18.3 could only tell a bad fit from a good one by evaluating
  transfer on another room — which a deployment does not have. "Is a quadratic
  enough?" separates them using nothing but the fit itself.

Recorded the caveat honestly: eight points, three parameters, so absolute R^2 is
optimistic and only the contrast carries weight. And locked the prediction it
generates rather than assuming it — if the LiteOffice jaggedness is estimation
noise, constraining that fit to the quadratic form should improve its transfer;
if it does not, the jaggedness is real structure and the fit-quality reading has
to be withdrawn. That runs in the same pass as H18.5 for two extra least-squares
fits.

## 2026-09-08 — eighth tick down; fixed a defect I had introduced by accretion

lambda_63 still unreachable. Checked the report for damage rather than looking
for new work, and found some — mine.

**Section 00 was no longer a one-page summary.** It is titled "一页纸的结论"
and had grown to **14 rows and 6,283 characters**, roughly four or five screens.
Every tick this session I appended a row for whatever had just finished and
never re-read the whole. The section's own title had become false, and the
user's original ask for this document was a clean report with a clear
conclusion.

Rewritten to **7 rows, 1,272 characters**. The five separate roll rows (A, A′,
A″, A‴, A⁗) collapse into two — pose, and framing — which is how they actually
group; the H18 material becomes two rows (the result, and what survives a change
of room); the three clean negatives share one row instead of three. Nothing was
dropped: every number still lives in its own section, and each row names the
sections that expand it. The summary's job is orientation, not completeness.

Worth recording as a pattern, since it is the third self-inflicted problem this
session after the stale-fragment republish and the two withdrawn claims:
**appending to a summary once per tick degrades it monotonically unless
something re-reads the whole.** The build already asserts on known-stale
strings; a length check on section 00 would have caught this one earlier.

## 2026-09-08 — ninth tick down; the build process itself was not in the repo

Same defect class as yesterday's summary bloat, one level up: the **header** —
the first thing a reader sees — still said "2026-09-07" and "six experiments"
when there are ten across sixteen sections. Fixed.

The larger thing found while fixing it: **the build process lived only in a
session scratchpad.** The report is assembled from four fragments with figures
inlined as data URIs, and nothing in the repo could reproduce that. Anyone else
— the other account, or any future session — could edit the fragments and have
no way to publish. It is also what let the stale-fragment republish happen
earlier: the guards existed only as lines I retyped into a shell each time.

`autoresearch/to_human/build_report.py` now carries the whole thing, with the
guards as assertions instead of habits. Each one exists because something got
past me:

* **STALE** — the three strings from withdrawn claims. If one reappears, a
  fragment has been resurrected from an old copy, which is exactly what
  happened when the black-wedge attribution was republished twice after being
  retracted.
* **ONE_PAGE** — section 00 must stay under 2,500 characters and 9 rows. It had
  reached 14 rows and 6,283 characters by being appended to once per work
  session. This is the check I said yesterday would have caught it; it exists
  now rather than as a note.
* **tag balance** — an unbalanced `<div>` has slipped into the hand-edited
  fragments twice this session.
* **missing figures** — fails loudly rather than publishing a broken `src`.

Verified by rebuilding through it: 16 sections, section 00 at 7 rows / 1,213
characters, 2.45 MB.

## 2026-09-08 — eleventh/twelfth tick down; re-read H18.3 and found my own reasoning unchecked

I said last tick that there was no honest work left. That was premature. Going
back over H18.3 I found that the worry I raised there — that a log-log slope is
confounded with the depth range it is fitted over — was **never actually
checked**. I ran the range-matched refit on the strength of it and drew a
conclusion, but never asked whether the confound was operative. It is not.

Restricting the fitting pixels to 0.5–4.5 m moves the Apartment `a` by at most
**0.0057** (4.4% of its own spread) and the LiteOffice one by up to **0.5653**
(84.8%). The Apartment mass was already inside that range, so the two fitting
sets do not differ in effective span at all. The H18.3 conclusion survives; the
reason I gave for it does not.

What actually differs: LiteOffice's fit leans on pixels where the frozen model
predicts **outside the scene's own 0.41–4.66 m range** — on its own gross
errors. Range restriction removes exactly that mass.

Recorded a puzzle instead of explaining it away. Removing that mass makes
LiteOffice's **inner**-bin coefficients look more Apartment-like while making its
transfer **worse**, and the **outer** bins that near_rim actually scores barely
move. So the difference arrives through the evaluation's per-frame scale+shift
alignment — a channel this project has measured before (82% of one sequence's
rim penalty was affine placement). Settling it costs one re-score under
`scale_only` and a frozen affine, and is now queued for the box.

Section 03c of the report carried the wrong reason for two ticks and now carries
the measured one plus the puzzle. The lesson is narrow and worth keeping: **a
caveat I raise is a claim like any other, and running an experiment because of
it is not the same as checking it.**

## 2026-09-08 — thirteenth tick down; audited my own caveats and one of them was wrong

Acting on last tick's lesson — a caveat I raise is a claim like any other — I
went back over the uncertainties recorded this session and asked which are
checkable without the box. One was, and it turned out to be wrong.

§02c said the −2.7° signed median in the roll distribution "could be a real head
tilt or a constant extrinsics offset, and this measurement cannot tell them
apart". It can. A constant offset lives in `T_device_camera` and is **per
device**, so it would be identical on every M1292 sequence. The signed median
ranges **+0.44 to −7.50 across 19 sequences on that one device**, sd 2.38 — an
8-degree spread a calibration constant cannot produce. Hypothesis dead, using
data that was already in the same JSON.

The structure is better than "it varies": **seq142–148 form a contiguous block**
sitting 4–5° further tilted than everything before it. That is a per-session
signature — a different wearer, or the device re-seated — not per-frame
behaviour. And it sharpens the other §02c claim: seq144/145, flagged there for
having 26–30% of frames beyond 10°, are **inside that block**, so what varies is
a per-session mounting offset with ordinary head motion on top, not activity as
such.

Reported the practical value honestly, which is **nil**: a per-session constant
de-roll is far cheaper than per-frame alignment and would remove 5–7° on a third
of the sequences, but §02's curve prices 10° at +2%, so 6° is worth under 1%.
The finding closes an uncertainty and explains a pattern. It does not buy
accuracy, and saying so is the point.

## 2026-09-08 — fourteenth tick down; answered the last checkable caveat, and it strengthened the claim

lambda_63 still unreachable. Continued the caveat audit started two ticks ago.

H18.6 had flagged its own limit: "eight points fitted with three parameters, so
the absolute R^2 values are optimistic; the signal is the contrast". Leave-one-out
answers it directly — the held-out bin never enters the fit, so extra parameters
earn nothing.

**The honest criterion widened the gap rather than closing it.** In-sample the
contrast was 0.94–0.99 against 0.07–0.59. Under LOO it is **0.71–0.96 against
−0.81 to −3.38**, against a constant-predictor reference of −0.306 at n=8. Every
Apartment curve genuinely interpolates a bin it never saw; every LiteOffice curve
predicts a held-out bin *worse than the mean of the other seven*.

That also upgrades the fit-quality check from a red flag to a calibrated one:
**LOO R² above the constant-predictor baseline means there is a curve, below it
means there is not** — and both quantities come from the fit alone, with no
held-out room. That is the form a deployment procedure can actually use.

Operational note worth keeping: the publish timed out mid-upload with "outcome
unknown". Rather than republish blind I read the live artifact back and grepped
it for the new strings — it had the previous tick's content and not this one, so
the upload had genuinely failed. Verifying before republishing costs one call and
is the difference between knowing and guessing.

## 2026-09-08 — H18.5 ran and was refuted; English report published

lambda_63 came back (uptime 33 days — it was a network partition, not the
reboot I had assumed, and the extraction job had in fact finished all 13
sequences during the outage rather than dying at 10). H18.5 launched on GPU0 in
tmux `sweep` and completed, `MARKER_SWEEP=0`.

**The result went against the locked bar.** Stability converges far faster than
predicted (|da| < 0.10 at 8 spread frames, not 60), but "spread beats single at
every matched count" fails 3 of 5, and "single matches spread" was written into
the protocol as the falsification condition. So H18.5 is refuted on its own
terms, and it takes the CPU pre-study's mechanism down with it — the simulation
predicted a large, growing spread/single gap and there is none.

Worth being blunt about the sequencing: I wrote that simulation during the
outage precisely because the box was unavailable, and I flagged it at the time
as a characterisation of the estimator rather than a result about ADT. That
framing is the only reason this is a clean refutation rather than an
embarrassment — the prediction was on record before the data existed.

Two rows in the sweep are degenerate (`spread_240`, `single_60` exhaust their
pools, so 0.000 is arithmetic). Noted in findings.md so nobody quotes them as
convergence.

The usable finding is the practical one: ~30 frames from one ordinary recording
is enough to fit the radial curve.

**Also today:** built and published a separate clean English report at
`autoresearch/to_human/depth-at-the-rim.html`
(artifact c6461af3-0b37-41ac-a32c-0c21aa6f4f69), structured motivation / data /
method / results / conclusion. It is a standalone narrative, not a translation
of the Chinese lab record — the Chinese report stays the dense technical log
with the self-corrections in it. The sweep result is folded into §4.4, and it
replaced a "what we would do next" bullet that the run had just answered.

Next on the box: the affine re-score to settle the -12.9/-7.3 puzzle, the
RolledView test in test_roll.py, and pushing the h18 artefacts to `results`.

## 2026-09-08 (later) — H19: the seductive idea lost, the boring one won

Wrote and locked H19's protocol, then ran it: fit the radial curve on the target
device's own footage, no ground truth, and see if it beats the Apartment-fitted
curve. It does not. It loses by ~10x on DinoToy and ~2.3x on Bowl, in both the
matched and unmatched fitting-range configurations. B2 was the pre-registered
falsification condition and it triggered on both sequences.

B3 passed and explains it: near-static footage gives a fit half as stable per
frame (|da| 0.127 vs 0.060 at 30 frames). That was H18.5's own follow-up
prediction about why *it* had been refuted, and it survived a test it did not
have to.

Two process notes worth keeping.

First, this experiment cost one GPU pass and no training, because the omega110
teacher cache already covered both LiteOffice sequences. Checking what was
already on disk before designing the run turned a day of work into twenty
minutes.

Second, and less comfortable: H19 corrects a recommendation I had already
published in both reports — "about 30 frames, and they can all come from a
single recording". The missing qualifier is "with motion in them". It was in the
*recommendations* section of the English report, which is the worst place to
carry an unqualified claim. Both reports are amended. The general lesson is in
findings.md: "N frames is enough" is never a property of N alone.

The interesting reading is the one H19 did not set out to test. The Apartment
curve transfers to a different lens better than that lens's own static footage
does, which points at the curve being substantially device-independent. If that
holds, calibration is a one-off fit on any well-moving footage rather than a
per-device procedure. Testing it needs moving footage from a third device, which
also breaks the static/device confound H19 cannot break on its own.

B1 (is the native fit radial?) is not established — the margin flips sign with
the fitting-range choice. Recorded as not-a-result rather than quoted.

## 2026-09-08 (later still) — H20: the published claim held, and got sharper

Wrote H20's protocol so that it could overturn what I had published hours
earlier, then ran it. It did not overturn it. All three bars passed.

The reason to run it was that H19's mechanism — "the fit needs motion, not a
lens match" — rested on calling LiteOffice near-static, and I could not find any
measurement of that anywhere in this line. It came from H9's remark about a
static wearer starving the parallax anchors. So a published claim was resting on
an inherited assumption, and that is worth an experiment whichever way it goes.

H20a settled it: 0.105 / 0.210 m of camera spread against 1.52-1.97 m, and 2.4 m
of path against 36-46 m. No overlap. The assumption was correct.

H20b is the part that matters more. Holding the device, room, pool, frame count
and teacher fixed and varying only how far apart the selected frames' cameras
were: stability degrades 5.5x and transfer falls on all four evaluation
sequences. So the mechanism survives a control that closes H19's confound, and
it did so without needing the third device I had written into the H20 candidate.

What I did not expect was the upgrade. Three device-matched points plus the
LiteOffice one lie on a monotone curve across a 10x range of camera motion. The
claim goes from binary to graded, and it acquires a number a person can act on:
the fitting frames should span roughly 2 m of walking. Plotted it in
assets/motion_fig.py and put the figure in both reports.

Two traps recorded in findings.md. A single low-motion fit LOOKS fine — the mean
curve over five draws is a smooth arch while |da| is 0.377 — so H18.6's LOO R^2
check earns its keep a second time. And |da| is not comparable between arms whose
draws vary in different ways, which is why B3 rests on transfer.

Worth noting the pattern: this is the second time a "how much data" number was
published without a qualifier that the next experiment supplied. H18.5's 30
frames needed "with motion"; H19's "motion" needed "about 2 m of it". Recorded
as a pattern rather than a one-off.

## 2026-09-08 (evening) — H21: I was wrong about "a property of the lens"

The H20 candidate said H21 needed moving footage from a third device. That was a
bad judgement about my own tooling. H15's lens family builds lenses over one
fixed cone filling the same disc, so a warp between any two is a pure radial
re-distribution of the same rays — no void, no extrapolation. That holds room,
scene, wearer, motion, teacher and backbone fixed and varies only the radial
mapping, which is a cleaner contrast than a third real device could give, and it
needed no new data at all.

It also cost no teacher inference, because the cache stores range and
grid_between resamples by ray, so range is invariant under a fixed-cone lens
change. Worth remembering as a pattern: check what is invariant before assuming
a re-render means re-running the expensive model.

The result refutes B2 and with it a phrase I have published twice. Cross-lens
application loses 0.84 and 0.47 points against same-lens, both inside H20's
draw-to-draw noise. Matching the lens buys nothing. What does matter is which
geometry the curve was fitted on — a 7.5-point spread, about 9x the matching
effect — and equisolid's curve is the best curve for every lens including the
real one, beating the real lens's own by 5.8 points.

So "a property of the lens" becomes "a property of how the backbone errs against
incidence angle, largely independent of how the lens spreads pixels across it".
Both reports corrected, and both gain a figure.

The useful side is that this is the first recommendation in this line that makes
a result better rather than only fencing it: resample to a rim-stretching
geometry before fitting, then apply the curve to the real images.

Third time now that a published characterisation was tighter than its evidence
and the next experiment supplied the missing qualifier. Promoted from observation
to standing rule in findings.md: before publishing a characterisation, name the
axis it was never varied along. Here that axis was the lens itself, in a claim
whose whole content was about the lens.

## 2026-09-08 (late) — H22: the trend is perfect, and my correction was the right one

H22 existed because H21's analysis stated a mechanism backwards. That was caught
analytically — every lens is normalised so r(theta_max) = R_disc, so the rim's
share of the frame follows from the shape alone, and equisolid gives the rim
FEWER pixels than equidistant, not more. The protocol opened by correcting H21's
analysis.md rather than quietly testing around it.

The corrected direction is right, and cleanly so: Spearman +1.000 between rim
area share and fit quality, on both test recordings, across six geometries
spanning 68% to 87%. All three bars pass.

Two things I made sure not to over-claim.

B2 passed — orthographic beats equisolid by row mean — but on the real Aria
lens, the only target that matters in deployment, they differ by 0.6 points,
inside noise. So the finding is "fit on a rim-compressing geometry", and the
report does not name one.

And the raw diagonal looks better than the off-diagonal, which would appear to
contradict H21's "matching buys nothing". It does not: good fitting geometries
are also easy targets, so row and column effects are correlated and the diagonal
is confounded. Fitting the additive model and looking at the diagonal residual
splits it properly — 0.5 to 1.3 points for the four realistic fisheye shapes
including the real lens (residual sd 2.98), against 8.5 and 8.3 for the two
extremes. H21 holds inside the fisheye family and stops holding outside it,
which is a better result than either "it always holds" or "it never does".

Mechanism is left as a labelled hypothesis with a competing explanation stated
alongside, because I got the last one wrong and the row/column orderings are
degenerate here in a way this experiment cannot separate.

## 2026-09-08 (night) — H23: the second mechanism dies too

H22 deliberately left two explanations standing rather than picking one, and
H23 was built to separate them. It killed the one I favoured.

ADT frames are natively 1408x1408 and this whole line had been reading them at
504, so a rim-expanding warp was inventing detail that was in the file. Feeding
the warp a 1008 source — twice the real detail, identical supervision, nothing
else changed — moves the spread from 18.0 to 18.5 points on one recording and
11.7 to 13.6 on the other. It does not shrink. The bar said falsified below 10%
shrinkage; it grew.

So two mechanisms are now dead for the same effect: "more rim pixels is better"
(H22, the opposite is true) and "rim-expanders lose to interpolation" (H23). The
hypothesis in H22's analysis is withdrawn from findings and from both reports.

I deliberately did not offer a third. The residual is narrower and more
interesting than before — whatever orders these geometries is not about
pixel-level detail at all, since doubling the detail does nothing, so it has to
be about where the angular content sits across the image radius. That rhymes
with the border result. But that is a description of what is left, not a
mechanism, and this effect has now eaten two of my guesses.

Also recorded an experiment NOT to run: scaling the same lens into a smaller
disc would move content inward without changing the radial mapping, which is
exactly the probe wanted — but it introduces a border, and the border result
says that is catastrophic. Confounded by construction, so it goes in findings as
an open problem rather than a queued ticket. Better to write down why a tempting
design is broken than to run it and discover the confound afterwards.

The useful part: H22's recommendation replicates at a second source resolution
with the same slope, so it is not an artefact of the 504 pipeline.

## 2026-09-08 (late night) — H24: the rule travels, the prize does not

After two dead mechanisms I stopped guessing why and asked how far the effect
generalises instead, which is what the recommendation actually depends on.
Everything from H21 to H23 had used DA3-Small.

DA3-Large keeps the ordering: Spearman +0.943 and +1.000, on a model whose
framing sensitivity is nothing like DA3-Small's (+636% vs +106% for a border
alongside the scored region). And it keeps the practical preference — a foreign
rim-compressing curve beats the real lens's own by 9.1 points, more than
DA3-Small's 5.8.

But the size of the prize does not travel at all. On the rearranged-room
recording DA3-Large's best geometry manages -0.7% and every other one makes
things worse. The ordering there is perfectly monotone around zero.

That nearly slipped past the bars. B2 "passed" on that recording at +4.3 points
— which is -2.2% against +2.1%, slightly-helps versus actively-hurts. I have now
seen this three times in this line (H9's anchors, H19's native fits, this), so it
goes into findings as a rule rather than a note: always print the absolute
numbers beside the relative ones, because the relative one is always the one that
looks publishable.

B3 failed on rectilinear in both recordings — global beats radial there — and
that is recorded rather than smoothed over. It fails in exactly the cell where
the method has collapsed for this model, which is the least interesting place for
a sanity check to fail, but it did fail.

The headroom reading is offered as arithmetic, not mechanism: DA3-Large starts
with half the rim error, so a correction that removes a systematic component has
less to remove. Given the last two mechanism guesses, that stays labelled.

Both reports need the recommendation split: it generalises as a rule about which
geometry to pick, not as a promise about the gain.

## 2026-09-08 (very late) — H25: the number this line rests on is an outlier

Thirteen Apartment sequences extracted during yesterday's outage had never been
used for anything. Scoring the unchanged Apartment-fitted radial curve on them
was cheap — no teacher inference, no refitting — and it should have been done
long before now.

The effect is real: all thirteen negative, sd/|mean| = 0.238, and the
2-parameter control worse than nothing on 13 of 13. B1 and B3 pass.

B2 does not. seq136's -16.6%, the in-room number quoted throughout both reports,
is 5.74 standard deviations above the thirteen-sequence mean of -7.02%. The
representative value is 2.4x smaller than the published one. And seq136 is not
an unusually hard or easy recording — its frozen error is +0.51 sd, ordinary. It
is specifically unusually responsive to this correction.

The part I find most uncomfortable is that dec_seq132, which I have been calling
"the harder test" in both reports for days, is the representative recording
(+0.31 sd). The number that read as a pessimistic sanity check was the honest
one; the number that read as the result was the outlier.

Scope limit written into the analysis before anyone can quote past it: only the
radial curve was re-measured. The LoRA's -51.5% was measured on the same
sequence and has not been. So "the adapter is three times better in-room" is now
an untested ratio, since both terms could scale together. Trying to launch that
re-measurement now; the checkpoint search is still running.

Third standing rule, and it subsumes the second: a single designated held-out
sequence quoted everywhere is not a held-out set. Printing absolute numbers
beside relative ones does not help when the absolute number comes from one lucky
recording.

## 2026-09-08 (very late) — H26: both numbers wrong, conclusions firmer

Re-measured the H18 students and the un-rectified control on the same thirteen
recordings H25 used. 52 evaluations, no retraining.

B1 fails: the adapter's -51.5% is 2.60 sd above the thirteen-recording mean of
-38.59%. So it was inflated by seq136 too — but less than half as badly as the
16-number curve's 5.74 sd.

B2 fails in the direction I had not considered. I had worried the adapter-vs-
curve ratio might collapse; instead it GREW, 3.10x to 5.50x, because the curve
was flattered about twice as much as the adapter. The published figure
understated the adapter's in-room advantage.

B3 passes 13 of 13, and it was the one that mattered. The rectified teacher
beats the un-rectified control on every recording, gap -3.66 points against
seq136's -3.9. H18's central in-room comparison had never been checked on more
than one recording; it now holds on thirteen without moving.

Worth carrying forward: the three seeds span 2.9 points while the sequence sd is
about 5. Which recording you score on matters more than which seed you train,
and this line has been spending its variance budget on seeds.

Also noted an incidental that contradicts nothing but could easily be misquoted:
the omega110 students improve the near-centre by 31%, the opposite of the
16-number curve's damage. Different objects — the student learns the teacher
everywhere, the curve is a per-angle fit on rim-dominated statistics — and the
data-ladder table's centre damage comes from different, rim-weighted arms
entirely. Wrote that separation into findings so nobody cross-quotes them.

Still untested and flagged as such: the "89%/82% of what real labels buy" ratio
divides by a GT-trained arm that has not been re-measured.

Net: two headline numbers corrected downward, and every conclusion that rested
on them survived or got stronger.
