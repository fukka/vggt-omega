# Fold the first real GPU results into raytun3r/README.md

**Owner:** cpu
**Files I may touch:** `raytun3r/README.md`
**Blocked by:** none

## Goal

`raytun3r/README.md` still says every paper number is unverified and lists the
`vggt` backbone as the "faithful target" with no evidence either way. It has now
been run. Update it to state what was measured, and add the two protocol findings
below — both change how any future number should be read.

## Context

Runs on `lambda_63`, VGGT-1B pretrained, UFM matcher, 30 three-frame windows,
300 iters, Adam 1e-3, clip 1.0, 504 max side. Artifacts on the `results` branch.

**ScanNet++ `3f15a9266d`, stride 10** (`results/s10-snpp-3f15a9266d/`):

| method | R° | t° | d_reproj | coverage |
|---|---|---|---|---|
| vanilla | 2.379 | 22.79 | 1.293 | — |
| param_free | 2.377 | 22.64 | 1.304 | — |
| **raytun3r** | **1.858** | 23.85 | 1.196 | — |
| center_ph | 0.378 | 5.46 | 0.445 | 0.66 |
| multi_ph | 0.406 | 5.24 | 0.836 | 1.00 |

Read this honestly: RayTun3R cuts rotation error 2.379° → 1.858°, a **1.28×**
improvement, against the paper's claimed 2–12×. It does not improve `t°`. And
both virtual-pinhole baselines beat it by ~5× on R° — the *opposite* of the
paper's Tab. 1 ordering, where RayTun3R beats Center-PH/Multi-PH. Say so plainly;
do not soften it. Candidate explanations worth listing (none verified): VGGT is
not the paper's primary backbone (DA3-Small is), a 170° full-frame lens is far
outside the 115° the paper ascribes to ScanNet++, and a single scene is a single
sample.

**Finding 1 — the evaluation stride matters more than the method.** The paper
says "evaluate on the full sequence" and filters windows below 2 px of optical
flow. Taken as consecutive frames (stride 1) on this data that admits pairs whose
baseline is ~1.1 cm against ~3 m of scene depth. At that ratio translation
direction is unobservable: MAGSAC++ on UFM matches — as good a geometric
reference as exists here — is itself **11.1°** off the ground-truth translation,
and `d_reproj` stops depending on depth at all (on ADT all five methods landed
within 0.06 px of each other, 6.51–6.57). At stride 10 the baseline is ~9 cm,
inter-frame rotation ~6.6°, and MAGSAC++ agrees with GT to **3.2°**. Stride-1
numbers are in `results/snpp-3f15a9266d/` for comparison; they measure the
protocol, not the method.

**Finding 2 — the GT pose convention is correct.** The nerfstudio→OpenCV
conversion in `data.py` was checked against MAGSAC++, which is independent of it:
median rotation disagreement 0.17°. Worth recording so nobody re-litigates it.

## Steps

1. Rewrite "What is verified, and what is not" — the CPU claims stay; move
   "any number in the paper" into a new measured-results section with the table.
2. Add both findings. Finding 1 belongs in *Interpretation decisions* as a new
   entry on evaluation stride, since the paper genuinely does not specify one.
3. Note in *Data* that ScanNet++'s DSLR is a **full-frame** fisheye whose
   calibration implies ~170° diagonal, not the 115° the paper states — corners
   carry real content and `project∘unproject` round-trips to 1.5e-5 px there.
4. Update the ADT caveat: `T_device_camera` now resolves exactly from `video.vrs`
   via `projectaria_tools`, so ADT pose metrics work. The "poses left unset"
   paragraph is stale whenever that package is installed.

## Done when

- [ ] `python -m pytest raytun3r/tests -q` passes
- [ ] `python raytun3r/smoke_test.py` passes
- [ ] README claims nothing the artifacts do not show
- [ ] pushed to `organized`, issue commented with the sha

## Needs a GPU run afterwards?

no
