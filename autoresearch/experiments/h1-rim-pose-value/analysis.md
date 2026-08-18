# H1 analysis

## run_001 (2026-08-18, CONFIRMATORY — matches locked protocol)

Command: `rim_pose_value.py --path ~/Desktop/ADT/scannetpp_example/3f15a9266d
--max-pairs 40 --out results/run_001.json` (seed 0, venv torch 2.2.2 CPU).
16 frames on disk, grid 504×336, θ_max 84.8°, 40 candidate pairs (GT rotation
0.74–27.3°), 19 real / 22 synth pairs survived the all-conditions-answer rule.

| cond | med θ | real med err (°) | real gain | synth med err (°) | synth gain |
|---|---|---|---|---|---|
| all | — | 0.868 | 0.974 | 0.197 | 1.001 |
| q0 | 30.8° | 2.883 | 1.024 | 0.348 | 0.994 |
| q1 | 40.9° | 2.528 | 0.985 | 0.368 | 0.993 |
| q2 | 51.0° | 1.936 | 1.027 | 0.325 | 0.997 |
| q3 | 62.7° | 2.991 | 1.136 | 0.387 | 1.069 |

**The locked prediction is NOT supported.** Error does not decrease with bin θ in
either arm. The synthetic control — same source pixels, GT-consistent targets,
equal 1 px noise, equal count — is flat (0.33–0.39°), i.e. **an outer annulus has
no intrinsic geometric advantage for rotation over an inner one at equal
correspondence count** in this configuration. The real arm is non-monotone (q2
best, rim worst), and rim gain overshoots in both arms (1.14 / 1.07).

What this does establish:

1. Quartering the matches costs ~2–3× error vs using all (0.87 → ~2–3°): **count
   and angular coverage, not per-point rim magic, are where a wide FOV pays** for
   rotation. That reframes N3: the wide-FOV story is "more well-spread bearings",
   not "better bearings at the rim".
2. The rim bin's gain > 1 (systematic rotation overshoot) in both arms is the one
   band-specific signature — worth understanding (MAGSAC threshold geometry at
   high θ? annulus conditioning?) before building anything on it.
3. Caveats: one scene, ~170° DSLR (not Aria's 110°); n = 19 pairs; SIFT features
   only start at θ ≈ 31° here (image content), so a true low-θ band was never
   tested; rotation only (translation direction deferred per protocol).

## run_002 (2026-08-18) — all 43 available pairs + paired rim-minus-center

Same protocol; paired per-pair difference q3−q0 added (exploratory-adjacent —
added after run_001's pooled medians, direction of inference unchanged). The
16-frame sample caps candidates at 43 pairs, so n barely grew (20 real / 23 synth).

Pooled medians reproduce run_001 (real: 2.89/2.39/1.99/3.01; synth:
0.31/0.36/0.30/0.37). Paired: **real q3−q0 median +0.465°, rim better on 9/20;
synth −0.047°, rim better on 14/23 — a coin flip in both arms.**

**Verdict: H1 (naive per-correspondence rim advantage for rotation) is REFUTED**
on this scene/config. What it rules out: adapters or attention schemes premised on
"rim correspondences are individually more informative for rotation". What it
suggests: the wide-FOV pose benefit, if real, lives in (a) total angular span /
conditioning at fixed count, (b) count itself, or (c) translation direction —
none of which this run isolates. → H1.1 (span at fixed count), H1.2 (translation).
The reproducible band-specific signature is the rim bin's gain overshoot
(1.12–1.14 real, 1.06–1.07 synth) — present in the noise-only control, so it is a
property of estimating from a high-θ annulus, not of SIFT.
