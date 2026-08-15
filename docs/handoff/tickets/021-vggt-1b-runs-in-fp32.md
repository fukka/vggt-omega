# VGGT-1B was run against its own specification, and it cost 2.9x

**Owner:** gpu
**Status:** **FIXED.** `VGGTBackbone.forward` now opens the bf16 autocast the
model is written to expect. This is a **correctness** fix, not an optimisation —
VGGT-1B was the only caller in the repo running it against its own
specification — and the 2.9x is a consequence. Found while investigating why
#020's grid took 2h55m.
**Deliberately NOT re-run**, by owner decision: `results/slambench-020-143686a`
and every fovbench artefact keep their published VGGT-1B columns, which were
produced by the fp32 path. See "What is now stale" — this is the cost of the
decision and it is recorded rather than hidden.
**Files touched:** `raytun3r/backbones.py` (one call site). The measurement
touched nothing: `slam020_profile.py`, `slam020_precision.py` and
`slam020_prec_diff.py` are standalone.
**Blocked by:** nothing.

## The verdict: fp32 was wrong, not merely slow

The question "what precision is each model supposed to run at?" has a checkable
answer, and four of five already had it right.

| model | its reference prescribes | the repo did | |
|---|---|---|---|
| **vggt_1b** | bf16 aggregator, fp32 heads | **no autocast at all** | ✗ |
| vggt_omega | bf16 aggregator, fp32 heads — `vggt_omega.py:41` / `:51` | same | ✓ |
| da3_small/large | `depth_anything_3/api.py:126` autocast, bf16 with fp16 fallback | same idiom | ✓ |
| dav2_large | HF `AutoModelForDepthEstimation`, fp32 | fp32 | ✓ |

**The evidence is inside the model itself.**
`VGGT-360-fisheye/vggt_visfeat/models/vggt.py:72` wraps VGGT's heads in
`autocast(enabled=False)`. That line is a **no-op unless a caller has opened an
autocast** — it exists precisely because the model is written expecting a bf16
aggregator and fp32 heads. Running with no autocast did not "keep more
precision"; it silently made line 72 meaningless and ran a configuration the
authors never specified.

Every other VGGT call site in this repo already supplied it:

* `VGGT-360-fisheye/main_adt.py` — `--dtype` **defaults to `bf16`**, and warns
  loudly on a bf16→fp16 fallback because "fp16 quality is a KNOWN VGGT failure
  mode (official repo recommends bf16)";
* `VGGT-360-fisheye/main_erp_upstream.py:58` — hardcodes the autocast.

`raytun3r.backbones.VGGTBackbone.forward` was the only one that did not.

## Verified, not assumed

The fix was measured through the real CLI, not the wrapper that discovered it —
the wrapper covered the whole `forward` and the fix covers only the model call,
so its numbers were not inherited:

| | wall clock | speedup | max ΔAbsRel | ordering |
|---|---|---|---|---|
| fp32, as published | 9m41s | — | — | — |
| **the fix** | **3m18s** | **2.93x** | **0.51%** | unchanged |

80-frame subset of the #020 split, both arms, contexts 1 and 10. The fix and the
exploratory wrapper agree **to four decimals on all eight rows**, which is the
check that the narrower scope changed nothing: the extra code the wrapper
covered is `pose_encoding_to_extri_intri`, and slambench scores depth.

`pytest tests slambench/tests` → 98 passed. Two failures in
`raytun3r/test_raytun3r.py` (`vggt`, `vggt_omega`) are **pre-existing and
identical before and after** — verified by stashing the change — and that file
is untracked, not part of the committed suites.

## What is now stale, and why that is a choice

Every published VGGT-1B column reached this backbone through
`finetune/eval/baselines/model_zoo.py` → `BackboneAdapter` → `build_backbone`,
so all of them were produced in fp32 and no current checkout reproduces them:

* `results/slambench-020-143686a` — #020, all three runs
* `results/slambench-raw-b1659a0` — #013
* `results/fovbench-*` — every FOV artefact carrying a `vggt_1b` row

The accuracy consequence is bounded: **≤0.51% on AbsRel with no ordering flip**,
against findings of 8–21%, so no published conclusion moves. The *timing*
consequence is not bounded the same way — `meta.json` in
`slambench-020-143686a` quotes VGGT-1B's 13.7x context scaling as a property of
the architecture, and it was substantially a property of the dtype. Any future
run is 2.9x faster on VGGT-1B and its timings are finally like-for-like against
the other three.

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

## What the other lever was worth

`tf32` (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, no code change) gives 1.33x for a
max 0.04% AbsRel move. It is **subsumed**: with bf16 on, tf32 adds nothing, and
it would not have fixed the specification mismatch anyway. Recorded because it
is the cheapest thing to reach for if a future backbone is genuinely meant to
run fp32 and is still slow.

The depth *map* moves further than the score does — ~0.2% median, 3% at p99,
10% at max — because a per-frame affine is fitted before scoring, so change that
is uniform in scale cancels and only structured change survives. Reporting the
map alone would have overstated the risk, which is why the tables above report
the scored metric.

## The harness, which is the smaller half

decode 6.9 ms/frame, `read_points` 23 ms/frame, rectify 9.9 ms/frame — about
5 min per model per run. That is **74% of `da3_small`'s runtime and 17% of
`vggt_1b`'s**, and it holds two redundancies that are numerically exact to
remove:

* the same 374 clips are decoded **once per model**, so 12 times across the
  three runs of #020;
* the `rect_derect` arm rectifies **19 frames per scored frame** where the
  windows only contain 10 distinct ones.

Both live in `slambench/run.py`, which #020 fenced off ("runs only").
**Left alone by owner decision** — removing them changes no number, so this is
deferred work rather than a risk being carried. Whoever owns `slambench/` next
inherits it.

## What was already done

`slam020_lane.sh` carries the corrected step-3 partition. The run's own
partition was balanced on step-2 timings and left lane 1 idle for 105 minutes;
`vggt_1b` alone in one lane is 2.0x instead of the 1.3x step 3 got. That fix is
independent of everything above and is already on `organized`.
