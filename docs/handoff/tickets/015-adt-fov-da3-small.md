# ADT-FOV: add DA3-Small to the baseline line-up

**Owner:** gpu
**Files I may touch:** nothing under `fovbench/` — runs only. Results to `results`.
**Blocked by:** #19 (the speedup identity check) should land first, since this
uses the threaded harness.

## The question

**Where does Depth-Anything-3 Small sit on the field-position curves?**

The line-up is four models, all large: VGGT-1B, VGGT-Omega 1B/512, DAv2-Large,
DA3-Large. Nothing in it says whether the periphery penalty is a property of
these architectures or of *capacity* — a small model of the same family, scored
identically, is the cheapest cut at that.

`da3_small` is already in the zoo (`model_zoo.py:146`) and already runnable:
`fovbench.models.load_model` goes through `zoo.get_specs`, and its `da3` kind is
a `BackboneAdapter`, so it also takes a frame stack. It just has never been run
here. `organized` @ `5a1f021` adds it to `CONTEXT_CAPABLE` so `--context-frames`
does not refuse it.

## What to run

**Check it loads first** — it is `on_device=True` in the spec, so unlike
`da3_large` it may pull weights on first use:

```
python -m fovbench.run --adt-root $ADT --models da3_small --protocols radial \
  --n-frames 3 --out eval_out/da3s_smoke
```

**Then both arms, on the two splits that already exist**, so it drops straight
into the published figures rather than starting a third comparison:

1. **The six-sequence baseline**, digest `601fcb22767e` — 50 frames × 6
   sequences, `--protocols radial`. This is the one the three headline figures
   are drawn from.
2. **The seq131 context grid**, digest `8ca25fd0ebd2` — 50 frames of seq131 at
   N=1, and at `--context-frames 5,10` × `--context-stride 1,10`. Five runs.

**The digests must come out equal to those two.** Rebuild the splits from
`--adt-root` (and the `/tmp/adt_seq131` symlink root for the second) — **not**
from `--manifest`, which silently overrides `--context-frames`; the driver now
refuses that conflict, but the point is that rebuilding gives the same digest by
design because the digest excludes context. Stop and report if either differs.

Expect ~1 min per run on the threaded harness (DA3-Small is 6x smaller than
DA3-Large, and the harness is CPU-bound anyway, so the model size will barely
show).

## Report

1. The two digests, confirmed equal to `601fcb22767e` / `8ca25fd0ebd2`.
2. `pen` per (view, stream) beside DA3-Large's, on the six-sequence run.
3. Whether DA3-Small's context behaviour matches DA3-Large's — the large model
   got *worse* with 10 consecutive frames and better with 10 strided (rect/real
   0.0774 → 0.0478). Same direction or not.
4. Timings.

**Do not tune anything and do not add other models.** If DA3-Small is simply
worse everywhere, that is the result; the question is the *shape* against field
position, not the ranking.

## Not in scope

The six-sequence context grid and the remaining 13 sequence extractions are
still deferred by the owner. This ticket adds one model to two existing splits
and nothing else.
