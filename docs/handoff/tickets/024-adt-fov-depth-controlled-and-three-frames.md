# ADT-FOV: the depth-controlled table, and the context arms on the right split

**Owner:** gpu
**Status:** **done, both halves** — `results/fovbench-joint-0b0d1d7` (2026-08-18). Part A ran on lambda_63 (16/20 cells bit-identical to #019; the four that moved are vggt_1b's known #021 bf16 step), part B's four context arms on space-container's A100s, all at digest `601fcb22767e`. Superseded the "neither half has run" line this file carried until 2026-08-22.
**Files I may touch:** nothing under `fovbench/` — runs only. Results to `results`.
**Blocked by:** none. `organized` @ the commit that adds `geometry.joint_grid`.
**Issue:** https://github.com/fukka/vggt-omega/issues/24 — that, not this file,
is what reaches the box. Its body carried the superseded Part B until
2026-08-17; it now matches this file.
**What is waiting on it:** two deck pages that ship today with no data on them.
`adt_fov_experiment_v6.pptx` p.7 (*FOV–Depth table*) is Part A, and p.8
(*Multi-frame against single-frame*) is Part B. Both currently carry an axis and
a sentence saying the run has not happened.

## The question

**How much of the rim penalty survives once GT depth is held fixed?**

Every metric in this benchmark is relative, and the rim of this lens looks at
nearer surfaces: on the six-sequence real stream the GT median falls 2.96 m on
axis to 1.64 m in the 50–55° band. So a rise along `theta` is partly a change of
subject, and the 1-D table cannot say how much. `fovbench/README.md` already
carries the size of the null — `pen` 1.80 on fisheye, 1.47 on rect from the
scene alone — but a scalar null is not the same as a controlled read.

Part A answers it directly, by scoring the same frozen prediction on a
**(incidence angle × GT depth)** grid. Reading along a row holds the depth band
fixed and varies only where in the field the pixels sit.

Part B fills a hole in the context arm: the deck's multi-frame page wants
1 / 3 / 5 / 10 frames at stride 10, and 3 was never run.

## Context — the code is already written and tested

`geometry.joint_grid` + `geometry.pool_joint`, wired through
`run._score_radial` → `run._reduce_radial` → `results.json["runs"][i]["joint"]`,
printed by `report._joint_note`. Defaults `DEPTH_EDGES = (0, 1, 2, 3, 5, 10)`;
override with `--depth-edges`. Cells thinner than `MIN_JOINT_CELL_PX = 500` are
recorded but not drawn.

**It cannot move a published number.** The grid is another set of masks over the
same per-frame fit, added as a new key. Verified two ways:

* `bin_by` with and without `joint_depth_edges` serialises **bit-identically**
  on everything except the new `joint` key.
* `fovbench/tests` is 141 green, including four tests that pin what the grid
  means: on the closed-form empty room a model with a **depth-only** error
  (fixed 0.12 m offset, no radial behaviour at all) reads **1.94×** across the
  1-D theta table and at most **1.18×** within any depth band — and what
  survives is exactly the within-band depth drift, since AbsRel × the cell's own
  `gt_mean` comes back to 0.12 to 5% everywhere. A theta-driven model keeps its
  gradient inside every band.

So Part A is a re-run for the new key, not a correction of anything.

## Before either part — run single-threaded

**Use `--workers 1`.** #023 hit a hard deadlock on lambda_63: `fovbench` hangs at
`dav2_large` at any thread count above 1 — 498 threads at `--workers 8`, 754 at
16, every one sleeping in `futex_wait`, GPU 0 %, not one frame line ever printed.
Not starvation (469 GB free, load 2.3 across 64 cores) and not the HF Hub
(`dav2_large` alone scores a frame in 6 s, online or `HF_HUB_OFFLINE=1`). The
diagnosis is in `results/fovbench-023-6fedc20/meta.json`.

It is not a correctness risk. `_ordered_map` pools rows in split order, so serial
is bit-identical to threaded; `tests/test_end_to_end.py` pins that, and #023's
self-check confirmed it against a reference produced at `--workers 16`.

**Part A carries `dav2_large`, so Part A must be `--workers 1`.** Part B's model
list does not, so it is worth one attempt at `--workers 16` — but if no frame
line appears within a few minutes, kill it and drop to 1. Every cost figure in
this ticket is a single-threaded figure already, so nothing here assumes
threading.

This is a CPU-side regression, not this ticket's job:
`results/fovbench-rectfix-393cab9` ran these same 300 frames and these same
models at `--workers 16` on 2026-08-14 without hanging, so something under the
harness moved in three days. **Do not wait for a fix.**

## Part A — the joint table on the headline split

Same split as #019, digest `601fcb22767e`, rebuilt from `--adt-root` and **not**
from `--manifest`:

```bash
git -C <repo> pull --ff-only origin organized
python -m fovbench.run --adt-root "$ADT" --protocols radial \
  --models vggt_1b,vggt_omega,dav2_large,da3_large,da3_small \
  --n-frames 50 --workers 1 --out eval_out/fovbench-joint/partA_6seq \
  2>&1 | tee eval_out/fovbench-joint/partA.log
```

**Stop and report if the digest is not `601fcb22767e`.**

Every pre-existing number in this file must reproduce #019's
`results/fovbench-rectfix-393cab9/partA_6seq/results.json` exactly. Check before
anything else — a diff here means the additive claim above is wrong and the
joint table should not be read:

```bash
python - <<'EOF'
import json
a = json.load(open("eval_out/fovbench-joint/partA_6seq/results.json"))
b = json.load(open("<path to fovbench-rectfix-393cab9 partA_6seq>/results.json"))
def strip(p):
    return [{k: v for k, v in r.items() if k != "joint"} for r in p["runs"]]
print("identical:", json.dumps(strip(a), sort_keys=True)
                    == json.dumps(strip(b), sort_keys=True))
EOF
```

**Cost: about half an hour — run all five.** The hedge this ticket used to carry
("if it is too long to queue, the minimum is `--models vggt_omega`") was written
before the timing was known, and it is moot. `fovbench-ctx-d351d94`'s ITEM 5
measured this exact run — six sequences, both views, N=1, single-threaded — as
`fovA6`: `vggt_1b` 517 s, `vggt_omega` 307, `dav2_large` 404, `da3_large` 386,
plus ~300 for `da3_small`. That is ~1900 s of forward pass, and the joint grid
adds none: it is another set of masks over the same frozen fit.

**Part A is the cheap half of this ticket and it unblocks a whole deck page.
Queue it first.**

## Part B — the context arms on the **six-sequence** split

**Rewritten 2026-08-17. The old Part B asked for 3 frames on seq131, to join the
existing `N=1/5s/10s` there. Do not run that.** The seq131 context arms are the
wrong split to begin with, and a fourth point on them would deepen the problem
rather than fix it.

Why: the deck plots those curves next to the Result page, which is the
six-sequence split (`601fcb22767e`). seq131 is not a representative sixth of it.
Per-sequence `pen` (`results/fovbench-ctx-d351d94/ANALYSIS.txt` and the
`perseq_seq13*` runs), real stream:

| | seq131 | 132 | 133 | 134 | 135 | 136 |
|---|---|---|---|---|---|---|
| VGGT-Omega fisheye | **1.48** | 2.02 | 2.86 | 2.06 | 2.21 | 1.86 |
| DA3-Large fisheye | **1.45** | 2.08 | 2.20 | 2.05 | 1.80 | 2.25 |
| VGGT-Omega rect | **0.98** | 1.34 | 2.12 | 1.47 | 1.79 | 1.45 |
| DA3-Large rect | **1.27** | 1.45 | 2.00 | 1.80 | 1.54 | 2.26 |

seq131 is lowest of the six in 5 of 6 model x view cells and second-lowest in
the sixth. It is not sample size: `partA_seq131_200f` (200 frames of the same
sequence) moves `pen` *further* from the pooled value, not toward it — rect
VGGT-Omega 0.97 at 50 frames, 0.86 at 200.

So run the context arms on the same split the Result page uses. Context does not
enter the digest by design (`fovbench/split.py`), so these come back as
`601fcb22767e` and are directly comparable to #019's `partA_6seq`:

```bash
for N in 3 5 10; do
  python -m fovbench.run --adt-root "$ADT" --protocols radial \
    --models vggt_1b,vggt_omega,da3_large,da3_small \
    --n-frames 50 --context-frames $N --context-stride 10 \
    --out eval_out/fovbench-joint/partB_6seq_${N}s \
    2>&1 | tee eval_out/fovbench-joint/partB_${N}s.log
done
```

`--n-frames` is **per sequence**, so 50 over six sequences is the 300-frame
split. `dav2_large` is out on purpose: monocular, no context path.

**Cost.** Scaling the seq131 timings in `ANALYSIS.txt` item 5 by the measured
6x frame ratio: roughly 3.9 h for `vggt_1b` across the three arms and ~0.8 h
each for the other three, **~6.3 h total**. `d351d94` predates `--workers`, so
those timings are already single-threaded and the `--workers 1` constraint above
does not inflate this number. If that is too much in one queue,
**drop the 3-frame arm first** (~1.5 h) — it is a nice-to-have; 5 and 10 are
what the deck pages plot. Dropping a *model* instead is worse: all four are on
the page.

**Do not** re-run `N=1` on this split. #019's `partA_6seq` already is it, and
re-running would risk two `N=1` curves that differ.

## Done when

- [ ] Part A's non-`joint` payload is byte-identical to #019's, and it is said so
      in the issue comment
- [ ] `report.txt` carries a `JOINT` block for vggt_omega on both views
- [ ] Part B's three runs all report digest `601fcb22767e` — **stop and report if
      any does not**, that is the whole point of the ticket
- [ ] all `results.json` on the `results` branch under
      `results/fovbench-joint-<sha>/`, with a `meta.json` recording the
      `--workers` actually used and whether the deadlock recurred
- [ ] issue commented with the sha and the digests

**Partial is useful here, and the halves are independent.** Part A alone
unblocks deck p.7 and costs half an hour; do not hold it behind Part B's six.
Push and comment as each half lands rather than waiting for both.

## Needs a GPU run afterwards?

This *is* the GPU run. Hand back to `cpu` when the results are pushed — the
figures and the deck page are CPU work.
