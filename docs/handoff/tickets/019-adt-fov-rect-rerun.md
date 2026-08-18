# ADT-FOV: re-score on the corrected lens, with DA3-Small in the same table

**Owner:** gpu
**Status:** **RUN, 2026-08-14** — `results/fovbench-rectfix-393cab9` @ `393cab9`,
both digests reproduced, and the fisheye arm came back **bit-identical** (8336
leaves compared, 0 differing). It is the FOV experiment's headline run and
`fovbench/README.md:237` points at it. Supersedes #014 and #015.
**Left open only for the issue comment** — issue
[#19](https://github.com/fukka/vggt-omega/issues/19) has never been commented.
**Files I may touch:** nothing under `fovbench/` — runs only. Results to `results`.
**Blocked by:** none. `organized` @ `0b4efb7` or later.

## The question

**Nothing new is being asked. The published rect numbers were measured through a
lens description that was a pixel wrong, and this makes the table one table
again.**

#018 found that `finetune/data/rectify.py` placed the rotated principal point at
`cx' = W - cy` where the fisheye arm used `cx' = (H-1) - cy` — so the two arms of
this experiment, whose entire purpose is that `rect` and `fisheye` be comparable,
described the same camera 1.000 px apart at every resolution. It is fixed. The
measured cost, on the synthetic scene:

| | before | after | |
|---|---|---|---|
| rect AbsRel, per bin | — | — | **+0.1 % to +0.7 %** |
| rect `pen` | 1.2309 | 1.2262 | **−0.385 %** |
| fisheye, everything | — | — | **0.000 %** |

**No conclusion in `fovbench/README.md` changes at that size.** What changes is
that `results/fovbench-ctx-d351d94` is now a mixture: its fisheye column matches
the current code and its rect column does not. A published comparison should not
need a footnote saying which half of it is current.

## What to run

One invocation per split, the **full grid**, on `organized` @ `0b4efb7`+.

**1. The headline six-sequence run**, digest `601fcb22767e`:

```
python -m fovbench.run --adt-root $ADT --protocols radial \
  --models vggt_1b,vggt_omega,dav2_large,da3_large,da3_small \
  --n-frames 50 --out eval_out/fovbench-rect-fix/partA_6seq
```

**2. The seq131 context grid**, digest `8ca25fd0ebd2` — five runs, as #010:
N=1, then `--context-frames 5,10` x `--context-stride 1,10`.

Rebuild both splits from `--adt-root` (and the seq131 symlink root for the
second), **not** from `--manifest`: the digest excludes the context by design, so
rebuilding reproduces it, and a manifest would silently pin the context to 1.
Stop and report if either digest differs.

`da3_small` is folded in here rather than run separately (#015) so it lands in
the same table as everything else instead of starting a third comparison. It is
`on_device=True`, so check it loads before committing to the grid:

```
python -m fovbench.run --adt-root $ADT --models da3_small --protocols radial \
  --n-frames 3 --out eval_out/da3s_smoke
```

## The control, which is the point

**The fisheye arm must come back bit-identical to `fovbench-ctx-d351d94`.**

It is the same code path on the same frames — the lens fix touches only the
rectifier — so any movement there is a bug in this run, not a finding. Diff
`results.json` numerically, not by eye, and report the largest absolute
difference in the fisheye rows.

That control also discharges **#014**, which wanted proof that the threaded
harness moves no digits. #014 can no longer be run as written (it asked for exact
reproduction of a run whose rect arm is now superseded), but the check it wanted
is exactly this one: if the fisheye arm reproduces to the last digit across a
different worker count, the threading is clean. Record `--workers` in the report.

## Report

1. Both digests, confirmed equal to `601fcb22767e` and `8ca25fd0ebd2`.
2. **Fisheye arm: max absolute difference from the published run.** Expected 0.
3. rect `pen` per (model × stream), old beside new — 8+ cells, and whether any
   of `fovbench/README.md` items 3, 3b, 5 or 7 moves by more than its stated
   margin. They should not; say so with numbers rather than asserting it.
4. `da3_small` beside `da3_large` on `pen` and on level, both views, both
   streams — the capacity cut #015 was written for.
5. Timings and `--workers`.

## Done when

- [x] both digests reproduce — `meta.json` `digests_reproduced: true`
- [x] the fisheye arm is bit-identical to `fovbench-ctx-d351d94` — 8336 leaves,
      0 differing, and at serial → 16 threads rather than the 8 → 16 #014 asked for
- [x] `fovbench/README.md`'s headline run pointer is updated to the new run id,
      with a line recording that the previous run's rect arm predates #018
- [x] pushed to `results` — `48e4d90`
- [ ] issue commented with the sha — **not done**

## Not in scope

The six-sequence **context** grid (#010 left it undone and the README calls it
affordable) and the thirteen unextracted ADT sequences. Both widen the base; this
ticket makes the base internally consistent first. Raise them separately once
this table is one table.
