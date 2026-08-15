# VGGT-1B is the only backbone running in fp32, and it costs 3.3x

**Owner:** unassigned — the decision below is the user's, not a ticket's
**Status:** **measured, not acted on.** Nothing in `raytun3r/` was changed.
Found while investigating why #020's grid took 2h55m.
**Files this would touch:** `raytun3r/backbones.py` (shared with fovbench —
see the blast radius below). The measurement itself touched nothing:
`slam020_profile.py`, `slam020_precision.py`, `slam020_bf16_run.py` and
`slam020_prec_diff.py` are standalone.
**Blocked by:** nothing.

## The finding

Three of the four multi-view backbones run their aggregator in bf16. One
does not.

| backbone | how it handles precision |
|---|---|
| `DA3Backbone.forward` | explicit `torch.autocast(cuda, bfloat16)` |
| `VGGTOmegaBackbone.forward` | `vggt_omega`'s own forward opens bf16 autocast |
| **`VGGTBackbone.forward`** | **`self.model(images)`, no autocast at all** — and the vendored `vggt_visfeat` then runs its heads under `autocast(enabled=False)` |

On top of that torch defaults `matmul.allow_tf32=False`, so VGGT-1B's fp32 is
*true* fp32 and never reaches the tensor-core path.

VGGT-1B and VGGT-Omega are 1191M and 1144M parameters doing the same job.
Forward cost, ms per call, RTX 6000 Ada:

| model | S=1 | S=3 | S=5 | S=10 |
|---|---|---|---|---|
| **vggt_1b** | **198** | **591** | **1096** | **2840** |
| vggt_omega | 51 | 106 | 184 | 407 |
| da3_large | 39 | 104 | 188 | 395 |
| da3_small | 18 | 27 | 36 | 66 |

3.9x at one frame, **7.0x at ten**. On #020's grid that made VGGT-1B **87% of
all model time**.

## What each lever buys, and what it costs

Measured end-to-end on a real 80-frame subset of the #020 split, both arms,
contexts 1 and 10, `vggt_1b` only:

| setting | wall clock | speedup | max ΔAbsRel | ordering |
|---|---|---|---|---|
| fp32 — as published | 9m41s | — | — | — |
| +tf32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, **no code change**) | 7m17s | 1.33x | **0.04%** | unchanged |
| **+bf16** (what the other three already do) | **3m24s** | **2.84x** | **0.51%** | unchanged |

bf16 subsumes tf32: with bf16 on, tf32 adds nothing.

The depth *map* moves more than the score does — ~0.2% median, 3% at p99,
10% at max under bf16 — because a per-frame affine is fitted before scoring, so
a change uniform in scale cancels and only structured change survives. That is
why the table above reports the scored metric and not the map, and why the map
alone would have overstated the risk.

**0.51% is not nothing.** #020's findings are 8-21% effects, so none of them is
threatened, but the published `vggt_1b` column would move in its third decimal
if re-run.

## Why this is also a fairness problem

The published #020 table times a fp32 VGGT-1B against three bf16 models. The
*accuracy* comparison is unaffected — if anything fp32 favours VGGT-1B — but the
**timings were never like-for-like**, and `results/slambench-020-143686a/meta.json`
quotes VGGT-1B's 13.7x context scaling as though it were a property of the
architecture. It is substantially a property of the dtype.

## The blast radius, which is why this is not just done

`raytun3r/backbones.py` is reached by both experiments through
`finetune/eval/baselines/model_zoo.py`. Making `VGGTBackbone` bf16 there would
move **fovbench's published VGGT-1B numbers too**, as a side effect of speeding
up the SLAM run. Every fovbench artefact on `results` carries a VGGT-1B column.

So there are three positions and they are not equivalent:

1. **bf16 by default.** All four backbones match, VGGT-1B gets 3.3x, timings
   become comparable. Both experiments' VGGT-1B columns need re-running to stay
   consistent with the code that produced them.
2. **tf32 by default.** 1.33x for a 0.04% move, no code change at all — it is an
   environment variable. Does not fix the fairness problem.
3. **Opt-in only.** `slam020_bf16_run.py` wraps the forward without touching
   `raytun3r/`. Nothing published moves; the speed is there when asked for.

## The harness, which is the smaller half

decode 6.9 ms/frame, `read_points` 23 ms/frame, rectify 9.9 ms/frame — about
5 min per model per run. That is **74% of `da3_small`'s runtime and 17% of
`vggt_1b`'s**, and it holds two redundancies that are numerically exact to
remove:

* the same 374 clips are decoded **once per model**, so 12 times across the
  three runs of #020;
* the `rect_derect` arm rectifies **19 frames per scored frame** where the
  windows only contain 10 distinct ones.

Both live in `slambench/run.py`, which #020 fenced off ("runs only"). Removing
them changes no number.

## What was already done

`slam020_lane.sh` carries the corrected step-3 partition. The run's own
partition was balanced on step-2 timings and left lane 1 idle for 105 minutes;
`vggt_1b` alone in one lane is 2.0x instead of the 1.3x step 3 got. That fix is
independent of everything above and is already on `organized`.
