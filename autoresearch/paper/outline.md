# Paper outline (draft skeleton, 2026-08-22 — drafting starts when #32/#33/#34 land)

Working title: *The Near-Field Rim: Diagnosing and Repairing Fisheye Depth in
Frozen 3D Foundation Models Without Touching What Pose Needs*

Venue shape: ML/vision conference (8 pages); the story is measurement-driven
method design on egocentric fisheye (Aria/ADT).

## 1. Introduction
- Egocentric fisheye is the deployment reality; frozen depth FMs degrade at
  the rim; adapters report whole-image means — the center/periphery Pareto
  question is unasked (survey gap).
- Contributions: (i) a diagnosis: the failure is precise, radially-modulated
  range compression, not noise; (ii) an asymmetry: the rim powers cross-frame
  alignment but receives none of multi-frame fusion's depth gains; (iii) a
  readout-only feature head (25k params, minutes on CPU) that fixes the
  near-field rim with pose untouched by construction — validated on six
  sequences [+ cross-scene #32, + VGGT-Ω #33]; (iv) negative results with
  mechanisms: per-point rim advantage (no), output-indexed recalibration
  (cannot invert compression), patch-content undistortion (≤0.2 px to fix),
  hands-as-corruption (plain occlusion [pending #34]).

## 2. Related work
- Adapter families (RayTun3R, Fisheye3R/CalTokens, DepthFisheye/LoRA,
  OmniVGGT); positioning: all touch backbone computation or input tokens; we
  are readout-only. DrivingDepth (prompt-driven output correction) needs
  sparse depth at test. Diagnosis lineage: UniK3D contraction, DAC/fovbench
  distance control.

## 3. What the periphery is for (measurement I)
- H1 family: quartile bins (no per-point advantage; ideal-noise control),
  span at fixed count (17/17; robustness-not-conditioning via the two arms),
  model masking with area-matched random control (rim load-bearing on 170°
  and Aria; Fig: h1_family + aria_h13).
- The hand-eye bootstrap sidebar (0.77–0.96° gate; 2.33° from factory calib).

## 4. What is actually broken (measurement II)
- Joint (θ×depth) tables with the GT-depth control (Fig: depth_baseline_h20);
  alignment-free bias/dispersion split → compression, precise (run_009).
- Cross-lane: penalty survives control on raw fisheye across 5 models;
  context buys the centre (ticket 024 A/B) → the give/receive asymmetry.
- Hands live at the near-field rim (#28 θ-part; depth numbers per #34).

## 5. The minimal safe repair (method)
- Ladder with each rung justified by the previous failure: 48-param table
  (transfers at rim, near-center collateral, cannot invert compression) →
  feature head (input-conditioned, zero-init, readout-only ⇒ pose invariant
  by construction). Architecture + training (minutes, CPU).
- Loss hygiene: skip dynamic cells (#31/#34).

## 6. Experiments
- Main table: six sequences × two splits, joint tables + three-axis zones
  (Fig: sixseq_h22 → paper table). Comparisons: uncorrected, 48-param table,
  head. [Cross-scene folds #32 → "one head or six". VGGT-Ω #33 →
  backbone-agnostic column.]
- Why-not sections: H3's ≤0.2 px measurement; H1's refutations.
- Caveats table from paper/numbers.md travels into supplementary.

## 7. Limitations
- One lens class (110° KB4); metric scale freed per frame; head granularity
  is the patch grid; per-scene vs cross-scene claim depends on #32; classical
  span saturation at 45° (rim's pose value beyond 45° unproven, n=11).

## Assets status
- Figures ready: h1_family, aria_h13, depth_baseline_h20, feature_head_h22,
  sixseq_h22, h3_resample_example (why-not illustration).
- Numbers audited: paper/numbers.md (all re-derived 2026-08-21).
- Blocking: #32 (claim strength), #33 (generality), #34 (H4 wording).
