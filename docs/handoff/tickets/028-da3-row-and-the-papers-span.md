# DA3's raytun3r row, and the whole table at the paper's own span

**Owner:** gpu
**Files I may touch:** nothing under `raytun3r/` — runs only. Results to `results`.
**Blocked by:** the 026 re-check (5 min, do it first — same session is fine).
**Blocks:** closing #4's reading.

## Goal

Two holes left after `results/rt3r/*-222d4a3`:

1. **DA3's `raytun3r` row is missing** — and DA3 is the paper's primary
   backbone. The hook that raised is fixed (see ticket 026's RESOLVED section);
   the row is now one fit + one eval.
2. **Nothing adapted has ever been measured at the paper's protocol.** Ticket 10
   identified the span: stride 60 is where vanilla reproduces Tab. 2 on two
   backbones at once (7.242 vs 7.21 VGGT, 6.392 vs 6.17 π³). Our whole
   post-audit table is stride 10 — internally consistent, but not comparable to
   the paper's *absolute* adapted numbers (raytun3r 0.93, center_ph 2.45 on
   VGGT). Re-evaluating the already-fitted adapters at stride 60 makes every
   row line up with Tab. 2 directly, for the price of eval only.

No new fits except DA3's. The vggt/π³ adapters from `*-222d4a3` are reused —
the adaptation set is the method's input, the eval stride is the protocol's,
and the paper itself fits once and evaluates everywhere.

## Steps

### 0. The 026 re-check first

`python -m pytest raytun3r/tests -q -k "da3 or rope"` on the real package —
expect green where it failed before. Then the controlled checkpointing A/B from
ticket 026's RESOLVED section (`--windows-cache`). Paste both into #26.

### 1. DA3 fit + eval, stride 10 (fills the missing row)

Same protocol as the vggt/π³ runs in `*-222d4a3` (30 windows, 3000 iters,
seed 0), so the row drops into the existing table:

```bash
python -m raytun3r.train --backbone da3 --variant small --weights pretrained \
  --dataset scannetpp --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d \
  --stride 10 --windows 30 --iters 3000 --seed 0 --out runs/rt3r/3f15-da3-fit
python -m raytun3r.eval --backbone da3 --variant small --weights pretrained \
  --dataset scannetpp --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d \
  --adapter runs/rt3r/3f15-da3-fit/adapter.pt --stride 10 \
  --methods vanilla,center_ph,raytun3r --out runs/rt3r/3f15-da3-s10.json
```

Note DA3 does not support grad checkpointing through our wrapper the way VGGT
does; if it OOMs, GPU0 has the headroom (π³ needed the same).

### 2. All three backbones, eval at stride 60

Adapters unchanged — vggt/π³ from `results/rt3r/*-222d4a3` (`args.adapter`
paths in their `meta.json`), DA3 from step 1:

```bash
python -m raytun3r.eval --backbone {vggt,pi3,da3} ... --stride 60 \
  --methods vanilla,center_ph,raytun3r --out runs/rt3r/3f15-{bb}-s60.json
```

### 3. Read against Tab. 2 (VGGT and π³ only; DA3 has no per-scene paper row)

| | paper vanilla | paper center_ph | paper raytun3r |
|---|---|---|---|
| VGGT | 7.21 | 2.45 | **0.93** |
| π³ | 6.17 | 2.28 | **0.78** |

The vanilla cells should land near 7.2 / 6.4 (that is the protocol match doing
its job — if they don't, stop and say so, the eval windows sample differently
than protocol_identify's pairs and that difference is then the finding). The
open question is the other two columns: at the span where vanilla matches the
paper, do center_ph and raytun3r land anywhere near 2.45 / 0.93?

Reading rules from #4 stand: do not tune toward the targets; a raytun3r cell
that beats center_ph here would be the paper's ordering appearing for the first
time, and the first suspect is still the TV fix, not a reproduction.

## Recording

`results/rt3r/s60-3f15a9266d-<sha>/` plus DA3's stride-10 run beside the other
two; meta.json per run as usual. Comment #4 with the table.

## Done when

- [ ] 026 re-check pasted into #26 (pytest green + controlled A/B)
- [ ] DA3 raytun3r row exists at stride 10
- [ ] stride-60 table for all three backbones, commented on #4
- [ ] pushed to `results`; hand back to cpu
