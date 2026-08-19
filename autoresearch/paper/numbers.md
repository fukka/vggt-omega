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
| Hands placement | 0.8–4% of cone; 80%+ θ>41°; median 0.26–0.94 m | `autoresearch/data/h4-stats/*.json` | GPU ticket #28, `results` @ 477b5ac |
| H4.1 hands ≈ occlusion | Δ(random−gtmask) = −0.13…+1.03° on 10–25° baselines | `results` @ 793e10f (`autoresearch-h4-pose/`) | GPU ticket #31; meal_seq131 anomaly open |
| Cross-lane controlled rim penalty | VGGT-Ω 1.81×, DA3-S 1.25× (raw fisheye) | `results` @ 601fcb22767e (`fovbench-joint-0b0d1d7/ANALYSIS-partA.txt`) | GPU ticket 024, 6 seqs, 300 frames |
| Context buys the centre | pen flat/up with N=3/5/10 frames | same, `ANALYSIS-partB.txt` | ticket 024B |

Pending (will be added when they land): #32 cross-scene folds, #33 VGGT-Ω head.

Known caveats that must travel with these numbers: one lens per dataset;
ScanNet++ results are one scene; run_006 span n=11; H2.* absolute AbsRel
levels on seq131 are ~2× the six-seq split's (near-field-heavy scene, flagged
in analysis); zone aggregates must always be accompanied by the full joint
table.
