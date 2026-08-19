# Provenance of every headline number (audited 2026-08-21)

Every number below was re-derived programmatically from its source JSON on
2026-08-21 and matched the value quoted in findings/analysis/dashboard.
Protocols were git-committed before their runs (`research(protocol)` commits
precede `research(results)` throughout).

| Claim | Value | Source | Notes |
|---|---|---|---|
| H1.1 span at fixed count, wide better | 17/17 pairs, −2.152° median | `experiments/h1-rim-pose-value/results/run_003.json` | ScanNet++ 3f15 local sample, N*=67 |
| H1.2 center-masked ≈ vanilla, rim-masked 2.5× | 4.925 / 5.002 / 12.317° | `run_004.json` | DA3-Small, mask T=45° |
| run_005 area-matched random control | 7.409° | `run_005.json` | patch-level 61% |
| H1.3 hand-eye gate | 0.774° trusted / 0.956° all; angle(C)=40.55° | `run_006.json` | official calib later: 38.44°, Δ=2.33° (`cam3r/data/adt_camera_rgb_calibration.json`, ticket #27) |
| H1.3 Aria masking | 14.8 / 20.0 / 25.3 / 38.5° | `run_007.json` | vanilla/center/random/rim |
| H2.0b bias (0–1 m row) | −0.531 → −1.199 log-depth | `experiments/h2-center-safe-adapter/results/run_009.json` | dispersion everywhere ∈ [0.016, 0.102] |
| H2.1 table near-rim / near-center | −25% / damaged | `run_010_even_odd.json` (+ `_halves`, `_fixedaffine`) | 48 params, 16/48 cells fit |
| H2.2 head near-rim (even/odd) | 1.023 → 0.333 (−67.4%) | `run_011_even_odd.json` | 25k params; halves −50.8% |
| Six-sequence validation (halves) | −21.2% … −71.3% near-rim, 6/6 improve | `autoresearch/data/h22-sixseq/run_011_*_halves.json` | GPU ticket #29, `results` @ 65bb0aa |
| Hands placement (θ distribution) | 0.8–4% of cone; 80%+ θ>41° | `autoresearch/data/h4-stats/*.json` | GPU ticket #28; θ part from segmentation, solid |
| Hands depth (0.26–0.94 m) | **VERIFIED (#34)** — fresh-sample medians 0.306/0.092/0.881 m | `results` @ 1327748 (`autoresearch-h4-provenance/`) | with-skeleton depth contains the person (100% of dyn px differ >5cm from without-variant); #28/#31 used consistent seg+depth provider |
| H4.1 hands ≈ occlusion | Δ(random−gtmask) = −0.13…+1.03° on 10–25° baselines | `results` @ 793e10f (`autoresearch-h4-pose/`) | verified by #34; synthetic-stream counterfactual unsuitable (DA3 domain gap +7…34% swamps hand effect); meal anomaly does not reproduce under synthetic, remains open |
| Cross-lane controlled rim penalty | VGGT-Ω 1.81×, DA3-S 1.25× (raw fisheye) | `results` @ 601fcb22767e (`fovbench-joint-0b0d1d7/ANALYSIS-partA.txt`) | GPU ticket 024, 6 seqs, 300 frames |
| Context buys the centre | pen flat/up with N=3/5/10 frames | same, `ANALYSIS-partB.txt` | ticket 024B |

| Cross-scene (one head) | five clean folds −74.5…−78.0% (≥ within-scene); decoration −18.7% + center collateral | `autoresearch/data/h23-crossscene/run_012_fold_*.json` | GPU #32; genre boundary; 48-param cross-scene control not run (optional) |
| VGGT-Ω head | 12/12 improve, −19.3…−40.6%; halves center collateral up to +49.8% | `autoresearch/data/h24-omega/run_013_*.json` | GPU #33; weaker than DA3; earlier-block probe = future work |

All pending items resolved 2026-08-23; nothing blocks drafting.

Known caveats that must travel with these numbers: one lens per dataset;
ScanNet++ results are one scene; run_006 span n=11; H2.* absolute AbsRel
levels on seq131 are ~2× the six-seq split's (near-field-heavy scene, flagged
in analysis); zone aggregates must always be accompanied by the full joint
table.

## Added 2026-08-19

| number | value | source (file → producer) | caveat |
|---|---|---|---|
| Center-PH near-center on identical px | 0.428→0.695 (+62%) | bench/results/centerph_seq131_odd*.json → centerph_row.py + scratch vanilla_covered.py | seq131 odd, 504px, single scene, EXPLORATORY until held-out repro |
| Center-PH rim coverage | 49.6% of near_rim zone; 75.2% of cone px | centerph_seq131_odd.json | pixel (not solid-angle) coverage |
| H7 verdict | gated==uniform: 0.572/0.567 (r8), 0.754/0.753 (r4); gate |g-1|~0.06 flat | h7-theta-gated-lora/results/*.json + ckpt gate_curve | 252px pilot, seq131 even/odd |
| H8 verdict | equal-area remap: near-rim 1.061→1.394 (+31%), center +16% | h8-equal-area/results/probe_a_seq131.json | 28 frames, 504px, zero-training probe |
| Solid-angle ratio | center patch = 1.73× rim patch; equal-area = 651 vs 973 cone tokens (-33%) | src/solid_angle_probe.py (spherical excess, true KB4) | geometry fact; the EFFICIENCY use died with H8 |
| #37 frozen rows | e.g. DA3-S seq136 whole 0.1376 / near-rim 0.1490 | data/bench/*.json + meta.json → eval_baseline_joint.py on lambda_63 | held-out scenes have ~no <1m rim mass (decoration: zero <1m px) — near-rim numbers there are regime-absent, not model-good |
| #38 v1 rows | ALL QUARANTINED | data/bench/rt3r/*.json | double depth conversion in raytun3r_row.py (fixed 8b5c13d); v2 re-run requested; cite NOTHING from v1 |
| rt3r harness fix check | seq131 vanilla near rows rise to 2.0-2.2 at rim | bench/results/rt3r_seq131_vanilla_fixed.json | consistent with diagnosis-era signature |
| H5/H6 training curves | full 1.19→0.40; plain 0.67→0.19; rim 0.95→0.50; alltok 0.67→0.15 | data/h5-train/*, data/h6-train/* (train_log.json, meta.json) | losses, not eval numbers; evals pending |
