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

## run_003 (2026-08-18, CONFIRMATORY — H1.1, protocol-h1.1.md locked in b326e1b)

`span_pose_value.py --path ... --out results/run_003.json`, cumulative disks
θ≤{35,45,55,65,85}°, count-matched per pair (N* median 67), 5 seeded resamples,
17 real / 22 synth pairs.

| cond | real rot (°) | real gain | real t-dir (°) | synth rot (°) | synth t-dir (°) |
|---|---|---|---|---|---|
| t35 | 3.540 | 0.890 | 43.4 | 0.377 | 3.32 |
| t45 | 3.232 | 0.973 | 35.6 | 0.403 | 3.37 |
| t55 | 2.068 | 0.954 | 17.0 | 0.305 | 3.42 |
| t65 | 1.897 | 0.989 | 19.0 | 0.310 | 2.72 |
| t85 | 1.672 | 0.925 | 16.1 | 0.285 | 2.61 |

Paired t85−t35 rotation: real **−2.15° median, wide better on 17/17 pairs**;
synth −0.11°, 15/22.

**H1.1 SUPPORTED, with a mechanism split the protocol's two arms were built to
expose.** At fixed correspondence count, angular span monotonically improves
rotation and (dramatically) translation direction in the real arm; the 65→85°
band still contributes. But the ideal-noise control shows the *conditioning*
effect alone is ~20× smaller (−0.11° vs −2.15°). So span pays mostly by making
the estimate **robust to real feature noise/outliers** — decorrelating errors and
sharpening outlier rejection — not by ideal-geometry conditioning. A theory-only
analysis would have dismissed the periphery; the real-data arm is where its value
lives.

**Bridge to the backbone question (basis for H1.2):** the span-limited classical
estimator at θ≤35° has rotation gain 0.890 — under-reading rotation by 11%. The
repo's prior finding: frozen depth FMs on raw fisheye sit at gain 0.82–0.88.
That similarity suggests the frozen models behave like span-limited estimators —
i.e. **they fail to extract the periphery's (demonstrably recoverable) alignment
value**. H1.2: mask the rim of the input to DA3-Small (CPU-runnable) and compare
its rotation gain masked vs unmasked. If unchanged, the model wasn't using the
rim — and "make the frozen model actually use the rim for alignment" becomes the
adapter's measurable job, with the classical span curve as the attainable target.

Caveats: one scene, one lens (~170° DSLR); real-arm t-dir from 12 pairs; the
θ≤35 disk is small in *pixel* area on this lens (features start ≈31°), so t35
matches cluster in a thin annulus — the both-endpoint variant and an Aria-FOV
replication remain open.

## run_004 (2026-08-18, CONFIRMATORY — H1.2, protocol-h1.2.md; prediction REFUTED)

`model_rim_use.py`, DA3-Small CPU, 16 pairs, mask T=45°: rim-mask kills 61.0% of
pixels, center-mask 39.0%.

| cond | median rot err (°) | gain |
|---|---|---|
| vanilla | 5.002 | 1.398 |
| rim_masked | 12.317 | 2.140 |
| center_masked | **4.925** | 1.375 |

**H1.2's prediction was exactly backwards — and the refutation is the finding.**
The frozen model does not ignore the periphery; its pose estimate *lives* there:
deleting all central content (θ≤45°, 39% of pixels) changes nothing
(4.93 vs 5.00°, per-pair differences within noise on every pair), while deleting
the rim more than doubles error on every single pair. So both classical geometry
(H1.1) and the frozen FM extract their alignment signal from the periphery — the
same region where fisheye depth is worst.

Caveats: (i) not area-matched (61% vs 39% masked) — but the striking arm is
center-masked ≈ vanilla, where *less* information than rim-masked still loses
nothing; a random-mask-61% control would complete the argument. (ii) gains > 1
here are contaminated by six catastrophic pairs (~18–20° err at GT 8–15°,
present in all conditions — likely low-overlap pairs); medians are the robust
read. The prior repo alpha of 0.816 for DA3 on this scene came from a different
pair protocol; do not mix the two numbers. (iii) mean-color masking could have
artifacts, though the model tolerating 61% masked argues against brittleness.

**Synthesis for the research question (updates N3 and N1):** the periphery is
already the alignment workhorse for both classical and frozen-FM pose. The
Pareto goal sharpens to: improve peripheral *depth* without perturbing the
peripheral *features the pose path depends on* — which argues for adaptation
that leaves early/feature layers alone (or is provably identity at init) and
corrects late/geometry readout, and it gives the eval a new required metric:
report pose stability alongside rim depth for any adapter.

## run_005 (2026-08-18, CONFIRMATORY — area-matched control for run_004)

Same 16 pairs, added `random_masked`: random 14 px patches masked at the rim
mask's pixel fraction (61%), seed 0.

| cond | median rot err (°) |
|---|---|
| vanilla | 5.002 |
| center_masked (39%) | 4.925 |
| random_masked (61%) | 7.409 |
| rim_masked (61%) | 12.317 |

**The area confound is closed.** At identical masked area, rim-specific deletion
costs 3× what random deletion costs (+7.3° vs +2.4° over vanilla), and center
deletion costs nothing. The frozen model's pose information is concentrated in
the periphery well beyond its pixel share. H1.2's refutation stands on an
area-matched footing.

## run_006 (2026-08-19, CONFIRMATORY — H1.3 hand-eye + span on real Aria)

`adt_pose_value.py --stages handeye,span`, 28 local seq131 frames (~3.3 s apart),
59 candidate pairs (GT device rotation 2.5–29.9°), θ_max 54.83°.

**Hand-eye bootstrap (verification gate PASSED on second pass).** First pass
(all 15 classical pairs): calibrated median 1.92° — gate failed as locked.
Second pass added a **C-independent** outlier filter (|angle(R̂)−angle(R_dev)| ≤
1.5°, conjugation-invariant, so it cannot be tuned toward the gate): 9/15 pairs
kept, calibrated median **0.774° (trusted) / 0.956° (all 15, incl. the 6 pairs
the fit never saw)** — both under the 1.5° gate. angle(C) = **40.55°**,
consistent with the ~38° device→RGB extrinsic the GPU box measured (plus
convention effects). The conjugation-invariant angle agreement (0.525° median)
independently confirms camera model + trajectory + timestamp matching. The
calibration JSON from the box remains the clean replacement (draft ticket).

**Span (H1.1) replicates on real Aria.** Count-matched cumulative disks
(N* median 44), 11 real / 12 synth pairs:

| cond | real rot (°) | real gain | synth rot (°) |
|---|---|---|---|
| t25 | 10.083 | 1.510 | 0.533 |
| t35 | 2.102 | 1.054 | 0.512 |
| t45 | 1.555 | 1.025 | 0.260 |
| t54.8 | 1.572 | 1.019 | 0.477 |

Paired widest−narrowest: real −2.57° median, wide better 8/11; synth −0.13,
6/12 (flat). Same signature as ScanNet++ (real span value large, ideal-noise
control flat) — **on the target lens**. Two Aria-specific nuances: (i) the θ≤25°
disk is in a failure regime at 44 matches (10° errors, gain 1.51) — the Aria
center alone is *much* worse for classical pose than the equivalent ScanNet++
restriction; (ii) the outermost band (45→54.8°) adds nothing here (1.555→1.572,
n=11) where ScanNet++'s last band still helped — either saturation at Aria's
narrower cone, vignetting-thinned rim features, or small n. Unresolved; matters
for how much of the Aria rim an adapter should protect for pose.

## run_007 (2026-08-19, CONFIRMATORY — H1.3 mask stage, DA3-Small on Aria)

`adt_pose_value.py --stages handeye,mask`, all 59 pairs, mask boundary 35°,
random control area-matched to the rim annulus's within-cone fraction.

| cond | median rot err (°) | gain |
|---|---|---|
| vanilla | 14.761 | 1.100 |
| center_masked | 20.018 | 0.878 |
| random_masked (=rim area) | 25.301 | 0.912 |
| rim_masked | 38.518 | 1.801 |

**The rim-dependence finding replicates on Aria, in a softer form.** Ordering:
rim deletion is by far the worst (+23.8° over vanilla), well beyond the
area-matched random control (+10.5°); center deletion hurts least (+5.3°). So
the rim remains the model's most load-bearing region per unit area on the
target lens. The ScanNet++-strength version ("center contributes literally
nothing") does NOT transfer: on Aria's narrower cone the center carries some
signal — unsurprising, since Aria's θ≤35° "center" is a much larger fraction of
its usable field than θ≤45° was of the DSLR's.

Context numbers: vanilla itself is poor here (14.8° median on GT 2.5–30°,
frames 3.3 s apart) — consistent with the repo's standing finding that frozen
FMs degrade on raw fisheye; classical solved only 15 of these 59 pairs at all,
so many pairs are genuinely hard (low overlap).

**H1.3 verdict: SUPPORTED overall.** Both day-1 findings transfer to the target
lens with quantitative softening: span value saturates by ~45° (run_006), and
rim-dependence is strong but the center is not disposable (run_007). Adapter
design consequence unchanged — protect rim features, correct depth at readout —
plus one new number: on Aria, the pose-relevant rim is roughly the 35–45°+
annulus, and the 45–54.8° ring's pose value is unproven at this n.
