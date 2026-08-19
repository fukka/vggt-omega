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
