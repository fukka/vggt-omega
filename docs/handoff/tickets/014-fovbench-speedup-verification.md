# fovbench is 2x faster — prove it did not move a single number

**Owner:** gpu
**Files I may touch:** nothing under `fovbench/` — one run and one comparison.
Result (a short note plus the two `results.json`) to `results`.
**Blocked by:** none. `organized` @ `d2d2cfc`.

## The question

**Does the threaded harness reproduce `fovbench-ctx-d351d94` exactly, on real
weights?**

Not "closely". Exactly — same digits in `results.json`. Anything else and the
speedup gets reverted, because every published ADT-FOV number was measured on
the serial path and a benchmark whose answer depends on its thread count is not
a benchmark.

## Why this needs the box

You reported the harness sitting at 25–30% GPU with ~2 of 64 cores busy. That is
now fixed: `--workers N` scores frames on N threads (numpy and cv2 release the
GIL), with the forward pass serialised behind a lock and the rows pooled **in
split order** — float addition is not associative, so completion-order pooling
would move the last digits of everything. Plus the per-pixel θ/radius maps and
the rectifier's remap grids are memoised instead of rebuilt every frame.

Measured on CPU here: **2.07×** end to end at 8 workers on 12 cores, bit-identical
at every worker count, 125 tests green.

**What I could not check is the only thing that matters to you.** I have no
weights on this machine, so bit-identity is verified with the *analytic*
stand-in only. The argument for the real heads is that a forward pass under a
lock is deterministic given the same input — which is an argument, not a
measurement. You have the weights.

## What to run

**1. The identity check.** Re-run the seq131 Part A split, which you already have
at `8ca25fd0ebd2`, twice — once serial, once threaded:

```
python -m fovbench.run --adt-root /tmp/adt_seq131 --n-frames 50 \
  --models vggt_1b,vggt_omega,dav2_large,da3_large --protocols radial \
  --workers 1 --out eval_out/fov_w1
python -m fovbench.run --adt-root /tmp/adt_seq131 --n-frames 50 \
  --models vggt_1b,vggt_omega,dav2_large,da3_large --protocols radial \
  --workers 8 --out eval_out/fov_w8
```

Then compare, `config.workers` excluded — it is the one field that must differ:

```python
import json
a = json.load(open("eval_out/fov_w1/results.json"))
b = json.load(open("eval_out/fov_w8/results.json"))
for d in (a, b): d["config"].pop("workers")
print(json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True))
```

**Report `False` immediately and stop** — do not tune, do not re-run hoping, do
not report a tolerance. A mismatch is a defect in my change and I want the diff,
not a workaround. Print the first ten differing paths.

**2. Also check it against the published run.** `fov_w1` should equal your stored
`partA_seq131/results.json` too, since nothing about the serial path changed.
If **1** passes and **2** fails, the memoisation moved something and that is a
different bug from the threading — say which.

**3. Time it.** Report seconds per model at `--workers` 1, 8, 16 and 32. The box
has 64 cores and I measured saturation at 8 on 12; where it saturates on your
hardware is a real question and my default (`min(8, cpu_count)`) is a guess made
on the wrong machine. If 16 or 32 wins clearly, say so and I will raise it.

Note the forward pass is serialised, so the speedup ceiling is set by how much
of a cell is GPU: ~120 ms of ~430 ms in your Part A timings, which caps this at
roughly 3.5× however many cores you throw at it. If you measure much more than
that, something is wrong, not wonderful.

## What not to do

**Do not start the six-sequence context grid or the remaining 13 sequence
extractions.** Both were considered and explicitly deferred by the owner; this
ticket is the prerequisite, not the trigger. If the identity check passes, the
next ticket decides what to spend the speedup on.

## Report

Four lines is enough: identity vs `--workers 1` (true/false), identity vs the
published `partA_seq131` (true/false), the timing table, and your recommended
default worker count. Push both `results.json` files so the comparison can be
re-run without the box.
