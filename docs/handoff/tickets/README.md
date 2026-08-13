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

| # | | owner |
|---|---|---|
| [019](019-adt-fov-rect-rerun.md) | **ADT-FOV: re-score on the corrected lens**, with DA3-Small in the same table. Supersedes #014 and #015. | gpu |
| [020](020-slam-baseline-programme.md) | **The SLAM baseline programme**: settle the depth convention, then both lens strategies, then multi-frame. Successor to #013. | gpu |
| [016](016-egosynth-depth-convention.md) | Is ego-synth's `d` planar z or range? **Step 1 of #020** — nothing else in that evaluation means anything until it is answered. | gpu |

Read #016 before #020, and #020's step ordering before queueing anything: step 1
can invalidate steps 2 and 3, and both later steps must run as single invocations
because `slambench` intersects its scored points across every arm in a run.

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
| [019](019-adt-fov-rect-rerun.md) | Re-score on the corrected lens | **open** |

**SLAM depth on ego-synth 5B** — *how accurate is each model on real egocentric
footage?*

| # | | status |
|---|---|---|
| [012](012-egosynth-calibration-download.md) | Fetch the Aria calibration the release omits | done for aea + nymeria; oxford and egoexo4d outstanding, which is what gates `rect_derect` on those two |
| [013](013-slambench-harness.md) | Build the harness, run the raw arm | done · `slambench-raw-b1659a0` |
| [016](016-egosynth-depth-convention.md) | Is `d` planar z or range? | **open** |
| [020](020-slam-baseline-programme.md) | Convention → both baselines → multi-frame | **open** |

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
| [003](003-full-scannetpp-da3-rerun.md) | The actual reproduction | open |
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
