# How the field compares, and how we must (2026-08-19)

Human directive: "check how other papers like RayTun3R compare with other
baselines, and eventually we need to compare the same way." Source of truth:
raytun3r/PAPER.md (secs. 5-7), read 2026-08-19.

## RayTun3R's comparison recipe

| Axis | Their choice |
|---|---|
| Baselines | Vanilla (frozen), **Center-PH** (center crop→pinhole), Multi-PH, LoRA (r=8 α=16 on QKV), CalTok (t=4 aux tokens) |
| Backbones | DA3-S/B/L, π³, VGGT (5 frozen FMs) |
| Datasets | ETH3D 110°, ScanNet++ 115°, KITTI-360 185°, TUM-VI 195°, FIORD 200° |
| Pose | R°, t° on consecutive pairs, same pair sampling for all methods |
| Depth | d_reproj (GT-pose reprojection) everywhere; AbsRel + δ1.25 only where dense GT exists (ETH3D, ScanNet++) |
| Protocol | per-scene unsupervised TTA: fit 30 three-frame windows of the test sequence, eval on the full sequence |
| Efficiency | trainable-parameter table (10.8K vs LoRA 147.5K vs CalTok 18.4K) |
| Robustness | GT vs AnyCalib-predicted calibration rerun (their Tab. 5) |

Two honest weaknesses they concede that shape our positioning:
- **RayTun3R does not win depth.** Center-PH beats it on ScanNet++ AbsRel by
  2.5× (0.066 vs 0.108) and wins d_reproj on 4/5 datasets. Their claim is
  pose. The depth-at-the-rim problem is open — exactly our N1 lane.
- Center-PH wins depth *by discarding the periphery*, which costs it pose
  (R° 3.27→1.11, t° 22.77→5.78 on ScanNet++ when RayTun3R keeps the rim).
  This is our H1.2 finding in someone else's table: **the rim is a pose
  asset and a depth liability.** Our paper's thesis is treating the two
  asymmetrically instead of choosing one.

## Gap analysis: our current bench (#37/#38) vs "compare the same way"

| Requirement | Status | Action |
|---|---|---|
| Vanilla frozen rows, multiple backbones | #37 (6 models × 2 held-out ADT scenes) | wait |
| RayTun3R row on our data | #38 (per-scene TTA on ADT, their protocol) | wait |
| **Center-PH baseline** | **MISSING** — the strong depth baseline and the anti-rim foil; without it reviewers will ask | build locally on CPU (DA3-S runs on the Mac), then ticket the held-out-scene run |
| LoRA r=8 α=16 QKV control | #35's plain-LoRA arm is MLP-placed (DA3 DPT head has no Linears; our deviation is documented) | add one sentence in paper; optionally a QKV-LoRA arm later |
| CalTok control | missing; low priority (loses to LoRA almost everywhere in their tables) | cite their numbers, note omission |
| **Public dataset with dense GT** | MISSING — everything is ADT | ScanNet++ first: raytun3r/ already loads it, audit done (docs/research/scannetpp-depth-audit.zh-CN.md), local 3f15a9266d sample on this Mac, **and 3f15… is RayTun3R's own named sequence** → external published anchor (verify-don't-fit). ETH3D terrains second. |
| Pose AND depth per dataset | our eval_lora.py / eval_module.py already emit both | keep |
| Params + FLOPs table | efficiency.json (H6), LoRA count known, head 25k, table 48 | assemble in paper |
| Calibration-robustness rerun | out of scope for v1 | future work |

## Protocol honesty requirement

RayTun3R adapts **unsupervised on the test scene** (TTA). Our H5/H2 lane
trains **supervised on other scenes** and evaluates on held-out scenes
(stricter generalization, but uses depth labels). These are different rows,
not competitors on the same row. The main table needs an explicit
"adaptation data" column: {none / test-scene RGB (TTA) / other-scene GT}.
Claims must be within-column; cross-column comparisons only as discussion.

## Concrete next steps (CPU, this week)

1. Center-PH baseline: rectify ADT KB4 center crop to pinhole (~70° FOV),
   run frozen DA3-S, eval with our joint table on the overlap region + report
   coverage loss (fraction of cone discarded ≈ 58% of solid angle at 110°).
2. ScanNet++ 3f15 row: run our eval_baseline_joint.py path over the local
   sample (needs render_depth — GPU box has the full scene; local sample has
   frames+calib only → ticket the depth part, prep the loader now).
3. After #35-#38 land: assemble the RayTun3R-style main table with the
   adaptation-data column; our method rows = rung1 head, rung2 LoRA-full,
   rung3 peripheral attention (+stack).
