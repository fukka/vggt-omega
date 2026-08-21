# Tickets

One unit of work each, owned by exactly one of the two sessions. `cpu` writes
code on the Mac; `gpu` runs it on the box. **The owner field is the lock** — two
sessions share this working tree, so a ticket that names files another open
ticket also names is a collision waiting to happen (it has happened: `33d3c55`).

Every ticket carries a **Status** line under its owner. Status is derived from
what is on the `results` branch, not from memory — if a run is not there, the
ticket is not done.

Start a new one from [`TEMPLATE.md`](TEMPLATE.md). Archived tickets keep their
number forever; numbers are referenced from commit messages and are never reused.

---

## Live now

Audited against the `results` branch on **2026-08-22**, by
[`tools/ticket_status.py`](../../../tools/ticket_status.py) rather than by hand.
Run it before trusting this table:

```bash
python3 tools/ticket_status.py            # full report
python3 tools/ticket_status.py --stale    # exit 1 if any Status disagrees
```

The previous audit (2026-08-17) had drifted badly: **12 tickets said "open — not
started" while their artefacts were already on `results`**, including all of
#028–#035. That is what the tool exists to prevent — the rule was written down
("Status is derived from what is on the results branch") but was being applied
from memory.

**Actually open, with no artefact behind them:**

| # | | owner | queue |
|---|---|---|---|
| [022](022-fov-on-slam-data.md) (#23) | **The FOV question on real SLAM points.** Radial half is **done** — `slamfov-022-60790fa`, and its oracle control is the result: a model with *no* field dependence reads a 2.84x rim penalty on nymeria, 1.26x once distance is held fixed. All five models beat that null, so the degradation is real. **Step 3 (window) is running on the pod.** | gpu | in flight |
| [036](036-raytun3r-comparison-row.md) (#38) | RayTun3R comparison row on the two held-out scenes | gpu | after the eval batches |
| [037](037-scannetpp-pose-anchor.md) (#39) | ScanNet++ 3f15 pose anchor — external published reference | gpu | queue tail |
| [027](027-adt-hand-pixel-stats.md) | ADT hand-pixel stats | gpu | — |
| [026](026-aria-rgb-calibration-json.md) | Aria RGB calibration JSON | gpu | — |
| [003](003-full-scannetpp-da3-rerun.md) · [006](006-render-scannetpp-depth.md) · [007](007-centerph-fov.md) | raytun3r line — see below | gpu | #006 is **unblocked**: all 5 DAC test scenes have meshes and `renderpy` is already built (issue [#41](https://github.com/fukka/vggt-omega/issues/41)) |

**Two GitHub issues were unblocked on 2026-08-22 without spending GPU time:**

* **[#40](https://github.com/fukka/vggt-omega/issues/40) second scene** — ADT has no Office/bedroom capture, but `adt_egocentric/annotate_rooms.py` left per-frame room labels on the box: `decoration_seq132` frames **1745–2246 are `bedroom`** (502 frames, all with RGB *and* depth), and that sequence is already held out. Same device, same lens, same depth scale — no download and none of the `Lite` set's depth-range confound. It is a **cross-room** probe: cross-sequence < cross-room < cross-building, and it must be named that way.
* **[#41](https://github.com/fukka/vggt-omega/issues/41) Part B** — not blocked. 5/5 DAC `scannetpp_tiny_test` scenes downloaded, 5/5 have `scans/mesh_aligned_0.05.ply`, 0/5 have `render_depth` (the output). `renderpy` is prebuilt for cpython-311 and imports in the `raytun3r` env. Also corrects the ticket: `third_party/` is gitignored, so the DAC splits are on lambda_63, not in a checkout.

**Duplicate ticket numbers** (the README says numbers are never reused, so these
need resolving): `026-aria-rgb-calibration-json.md` / `026-da3-rope-n-prefix-and-checkpointing-drift.md`,
and `028-da3-row-and-the-papers-span.md` / `028-feature-head-six-sequences.md`.
The `autoresearch-h22-sixseq` artefact belongs to **028-feature-head-six-sequences**;
`028-da3-row-and-the-papers-span` has no artefact and is still open.

**#025 is done** — all five checks ran on `lambda_63`, commented on
[#25](https://github.com/fukka/vggt-omega/issues/25#issuecomment-5335209060),
relabeled `cpu`.

**Known hazard on lambda_63:** `fovbench` deadlocks at `dav2_large` at any
`--workers` above 1 — first hit by #023 on 2026-08-17, on code that ran the same
models at `--workers 16` three days earlier. Serial is bit-identical to threaded,
so `--workers 1` is a safe workaround and no ticket is blocked. Needs a `cpu`
ticket; it does not have one yet.

---

## The two experiments

They share a repository and share no protocol, no ground truth and no conclusion
(`tests/test_experiment_separation.py` enforces that). Their tickets are separate
for the same reason.

**ADT-FOV** — *does depth get worse toward the edge of a wide field of view?*

| # | | status |
|---|---|---|
| [008](008-adt-fov-test.md) | The first four-model run | done · `fovbench-main-22c108d` |
| [009](009-adt-fov-rerun-drift.md) | Re-run: the `drift` column measured the wrong thing | done · `fovbench-v2-ef2d50b` — and `drift` was later deleted outright (#017) |
| [010](010-adt-fov-context-and-sequences.md) | Six sequences plus the temporal-context arm | done · `fovbench-v3-24b38e1`, `fovbench-ctx-d351d94` ← **the headline run** |
| [014](014-fovbench-speedup-verification.md) | Prove threading moved no digits | folded into #019 |
| [015](015-adt-fov-da3-small.md) | Add DA3-Small | folded into #019 |
| [019](019-adt-fov-rect-rerun.md) | Re-score on the corrected lens | done · `fovbench-rectfix-393cab9` ← **the headline run**, digests reproduced and the fisheye arm bit-identical (8336 leaves, 0 differing) |
| [023](023-vggt360-on-both-benchmarks.md) | Add our own port to the table | done · `fovbench-023-6fedc20` — `SELF-CHECK PASS`, 8 vanilla cells bit-identical. Level with `vggt_1b` on the pool (0.1198 vs 0.1198) and far flatter across the field (1.04x vs 1.60x), but behind it once the views are cut at equal pixels (0.1276) |
| [024](024-adt-fov-depth-controlled-and-three-frames.md) | The rim penalty with GT depth held fixed, and the context arms on the right split | **open — not started.** Blocks two deck pages |

**SLAM depth on ego-synth 5B** — *how accurate is each model on real egocentric
footage?*

| # | | status |
|---|---|---|
| [012](012-egosynth-calibration-download.md) | Fetch the Aria calibration the release omits | done · aea 143, nymeria 254, oxford 124, all on lambda_63. Oxford is still out of `rect_derect`, now for an unverified rotation and a 1408-vs-2880 convention rather than for missing files |
| [013](013-slambench-harness.md) | Build the harness, run the raw arm | done · `slambench-raw-b1659a0` |
| [016](016-egosynth-depth-convention.md) | Is `d` planar z or range? | done · **z**, both staged takes, `d - z` flat at the float16 floor and the range residual matching `1-cos(theta)` to 4 dp |
| [020](020-slam-baseline-programme.md) | Both baselines → multi-frame | done · `slambench-020-143686a`, digest `61195914f090` across all three steps |
| [022](022-fov-on-slam-data.md) | Where in the field does it degrade, on this data | **open** — code green on CPU, run not started. Carries the distance control the dense experiment does not need: an oracle with no field dependence reads a 1.86x rim effect until distance is held fixed |
| [023](023-vggt360-on-both-benchmarks.md) | Our port as a third lens strategy, beside `raw` and `rect_derect` | done · `slambench-023-6fedc20`, on an A100 pod because the vggt360 arm OOM'd lambda_63 twice. **`rect_derect` beats it on both datasets** (0.1582 vs 0.1885 aea, 0.1305 vs 0.1504 nymeria) while losing less coverage — the ring is the second-best of three lens strategies. Cross-run control PASS on identical support |

---

## Repository work

Not experiments — changes to how the code is arranged. Both landed on
`organized`.

| # | | |
|---|---|---|
| [017](017-separate-fov-and-slam.md) | The FOV package was carrying a second SLAM evaluation. Deleted it, and made the separation seam two-sided | done · `99a68e9` |
| [018](018-one-lens-one-digest.md) | One description of the Aria lens (it had drifted a pixel), one definition of a split digest, and a README that names both experiments | done · `0b4efb7` |

---

## raytun3r

A separate line of work with its own README. **Not audited in the 2026-08-13
FOV/SLAM pass** — the statuses below come from the `results` branch alone, so
check a ticket before picking it up.

| # | | status |
|---|---|---|
| [001](001-readme-first-gpu-results.md) | Fold the first GPU results into the README | not audited |
| [002](002-depth-convention-audit.md) | Audit Eq. 7's depth convention | not audited |
| [003](003-full-scannetpp-da3-rerun.md) | The actual reproduction | open · superseded in practice by #4 |
| [004](004-validate-harness-vanilla.md) | Reproduce VGGT vanilla, validating the harness | done · `vanilla-repro-3f15a9266d` |
| [005](005-protocol-identify.md) | Settle the protocol with every training-free target | done · `protocol-identify-3f15a9266d` |
| [006](006-render-scannetpp-depth.md) | Render ScanNet++ DSLR depth from the mesh | open |
| [007](007-centerph-fov.md) | Identify the paper's Center-PH FOV | open |

---

## [archive/](archive/)

Tickets whose work no longer exists to be done. Kept for the argument, not as
tasks, and never renumbered.

| # | | why |
|---|---|---|
| [011](archive/011-ego-synth-sparse-depth.md) | ego-synth sparse depth, inside `fovbench/` | **void** — #017 deleted everything it built. It never produced a result, and `slambench/` owns that dataset now. Its reasoning was relocated, not discarded: see #013 and #020 |
