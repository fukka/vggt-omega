# ADT-FOV: the depth-controlled table, and the context arms on the right split

**Owner:** gpu
**Status:** **open** — part A unchanged; **part B rewritten 2026-08-17**, it is
now the six-sequence split and not seq131. Code is on `organized`; the runs are
the ticket.
**Files I may touch:** nothing under `fovbench/` — runs only. Results to `results`.
**Blocked by:** none. `organized` @ the commit that adds `geometry.joint_grid`.

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

## Part A — the joint table on the headline split

Same split as #019, digest `601fcb22767e`, rebuilt from `--adt-root` and **not**
from `--manifest`:

```bash
git -C <repo> pull --ff-only origin organized
python -m fovbench.run --adt-root "$ADT" --protocols radial \
  --models vggt_1b,vggt_omega,dav2_large,da3_large,da3_small \
  --n-frames 50 --out eval_out/fovbench-joint/partA_6seq \
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

**If the full grid is too long to queue, the minimum that unblocks the deck is
`--models vggt_omega`** — the figure is VGGT-Omega on `fisheye` and `rect`,
real stream. Run the rest when there is room; the joint key is free for every
model once the forward pass is happening anyway.

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
each for the other three, **~6.3 h total**. If that is too much in one queue,
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
      `results/fovbench-joint-<sha>/`
- [ ] issue commented with the sha and the digests

## Needs a GPU run afterwards?

This *is* the GPU run. Hand back to `cpu` when the results are pushed — the
figures and the deck page are CPU work.
