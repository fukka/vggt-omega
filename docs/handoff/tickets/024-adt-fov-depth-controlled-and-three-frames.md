# ADT-FOV: the depth-controlled table, and the missing 3-frame point

**Owner:** gpu
**Status:** **open** — two runs, both on splits that already exist. Code is on
`organized`; the runs are the ticket.
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

## Part B — 3 frames at stride 10

The seq131 context split, digest `8ca25fd0ebd2`, one run:

```bash
python -m fovbench.run --adt-root "$SEQ131_ROOT" --protocols radial \
  --models vggt_1b,vggt_omega,da3_large,da3_small \
  --n-frames 50 --context-frames 3 --context-stride 10 \
  --out eval_out/fovbench-joint/partB_3s 2>&1 | tee eval_out/fovbench-joint/partB.log
```

This joins `N=1`, `5s` and `10s` from `results/fovbench-rectfix-393cab9`, which
were measured on code this commit changes only by addition — so the four points
form one curve. `dav2_large` is not in the list on purpose: it is monocular and
has no context path.

## Done when

- [ ] Part A's non-`joint` payload is byte-identical to #019's, and it is said so
      in the issue comment
- [ ] `report.txt` carries a `JOINT` block for vggt_omega on both views
- [ ] both `results.json` on the `results` branch under
      `results/fovbench-joint-<sha>/`
- [ ] issue commented with the sha and the two digests

## Needs a GPU run afterwards?

This *is* the GPU run. Hand back to `cpu` when the results are pushed — the
figures and the deck page are CPU work.
