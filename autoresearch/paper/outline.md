# Paper outline v2 (post-pivot, 2026-08-19 — CVPR target)

One-sentence contribution: **frozen 3D foundation models treat every viewing
angle alike; on egocentric fisheye we measure exactly how that breaks (the
periphery funds pose and is denied depth), prove that the popular fixes at
the input aggravate it, and repair it behind the encoder — a diagnosis-driven
ladder (48-param table → 25k head → LoRA with measured-field losses →
rim-only cross-frame attention at 0.48× FLOPs) that improves near-field rim
depth up to ~60-75% with center and pose provably intact.**

Title direction: "Peripheral Vision for 3D Foundation Models" (working).

## 1. Introduction
- Egocentric large-FOV is where AR/robotics products live; the near-field
  rim (hands, workspace) is precisely the worst-calibrated region (hand
  stats: 80%+ beyond 41°, median 0.26-0.94 m).
- The asymmetry thesis: rim = pose asset (H1.2 masking; externally echoed by
  RayTun3R's Center-PH gap) + depth liability (compression field).
- Contribution bullets: (1) the asymmetry measurements with distance
  control; (2) the where-to-intervene result (four measured refutations);
  (3) the ladder methods incl. peripheral cross-frame attention; (4) the
  radially-honest benchmark protocol (three axes + adaptation-data column).

## 2. Related work
- Adapting geometry FMs to new cameras: RayTun3R (PE TTA), UniDAC (trained
  scale field — their premise IS our measured field), Wid3R (retrain with
  camera token), CalTok, DAC/UniK3D. Our lane: supervised cross-scene,
  tiny-budget, diagnosis-driven.
- Fisheye-specific architectures (PFDepth, OmniDS, DarSwin) vs backbone
  adaptation. Token efficiency (Spark3R saliency-driven pruning) vs our
  geometry-driven rim *enrichment* — stackable, not competing.
- Cross-frame/video depth (ViGeo, NVDS+, PPVD): full-model lanes; ours is a
  bolt-on for frozen FMs, gated at zero-init.

## 3. The asymmetry, measured (analysis I)
- Rim pose value = span-under-real-noise (H1/H1.1, synth control).
- Frozen FMs run pose on the rim (H1.2 center/rim/random-matched masking;
  Aria replication milder but same ordering).
- The depth failure is a radially-modulated range compression (runs
  008b/009: dispersion 2-10%, bias to 3.3×), survives the distance control
  (024A), and context buys the center, not the field (024B).
- Hands: plain occlusion at the near rim (H4/#31), a hygiene item not a module.

## 4. Where to intervene (analysis II — the four refutations)
- Input surgery hurts: Center-PH (+62% near-center on identical pixels,
  49.6% rim coverage), equal-area resampling (+16..+31% everywhere, H8),
  patch undistortion (no-op, ≤0.21px within-patch, H3).
- Geometry-conditioning the adapter is redundant: θ-gated LoRA ties uniform
  at r=8 AND r=4; the free gate stays flat (H7). PE already conditions.
- Output-only recalibration can't invert a many-to-one compression (H2.1).
- ⇒ intervene behind the encoder: features, objectives, or added evidence.

## 5. Method: the ladder (each rung licensed by a measurement)
- Rung 0: 48-param (θ×d̂) table — the floor any method must beat.
- Rung 1: 25k frozen-feature readout head — pose-safe by construction.
- Rung 2: LoRA (last-4 MLPs) + compression-weighted depth loss +
  rim-feature distillation + multi-frame rim consistency. H7 as the
  conditioning ablation.
- Rung 3: peripheral cross-frame attention — rim queries only, zero-init
  gate, depth-head-only feature copy (pose path bit-identical), 0.48× FLOPs
  vs all-token (efficiency.json); all-token control = the H6 ablation.

## 6. Experiments
- Datasets: ADT 6-seq family (within/cross-scene) + 2 held-out scenes
  (seq136, decoration_132) + [ScanNet++ pose anchor if renders stay blocked].
- IMPORTANT benchmark honesty note: held-out scenes carry almost no
  near-field-rim mass (measured; decoration has zero <1m pixels) — they
  certify "no collateral, whole-image parity"; the near-rim claims are
  carried by the six-seq + cross-scene tables. State this openly; it is a
  finding about benchmark design (whole-image means hide the regime).
- Main table rows: frozen (DA3-S/L, VGGT-Ω, UniK3D, DAv2; DAC cited),
  Center-PH, RayTun3R (per-scene TTA, adaptation-data column "test-scene
  RGB"), rung 0/1 (ours, "other-scene GT"), rung 2 full + plain-LoRA
  control, rung 3 rim + all-token control. Three axes each: center depth /
  near-rim depth / pose.
- Efficiency table: params (48 / 25k / 0.49M / 2.96M) + FLOPs (0.48×).
- PENDING GPU: #35 evals (8 JSON), #36 evals (4), #38 v2 rows (4).

## 7. Limitations
- Single lens family at full strength (Aria KB4; ScanNet++ cross-lens only
  partially, renders blocked); decoration boundary = style-transfer limit;
  VGGT-Ω rung-1 collateral (probe-earlier-layers = future work); pose
  improvement (vs preservation) unproven until dense-pair evals.

## Assets status
- Figures ready: h1_family, aria_h13, depth_baseline_h20, feature_head_h22,
  sixseq_h22, trajectory, fig_intervene (paper Fig. 1 candidate),
  fig_centerph, fig_h7h8, fig_baselines, h6 efficiency numbers.
- numbers.md audit current through 2026-08-19 (incl. quarantined #38 v1).
- refs.bib verified 13 + 2 placeholders; ADD: Wid3R 2602.05321, UniDAC
  2603.27105, Spark3R 2605.06270, DAPETR 2606.08680, GIFT 2608.02068,
  ViGeo 2605.30060 (verify via API at insert time).
- main.tex: pre-pivot draft parked; rewrite starts when the three GPU eval
  batches land.
