# This run's `vggt_1b` column was produced in fp32 (#021)

Added after publication. Nothing in this directory was edited — the numbers,
logs and `meta.json` are exactly as the run left them. This file records
something discovered afterwards that a reader of the artefact alone would
otherwise not know.

## What was wrong

`raytun3r.backbones.VGGTBackbone.forward` was the one VGGT call site in the
repo that did not open the bf16 autocast the model is written to expect.
`VGGT-360-fisheye/vggt_visfeat/models/vggt.py:72` wraps VGGT's heads in
`autocast(enabled=False)` — a no-op unless a caller has opened one — so the
model is written for a **bf16 aggregator and fp32 heads**, and it was run
entirely in fp32.

The other four models were already on their own reference precision:
`vggt_omega` opens bf16 internally (`vggt_omega.py:41`), DA3 uses the same
dtype expression as `depth_anything_3/api.py:126`, and `dav2_large` is HF
`AutoModelForDepthEstimation` at fp32 by design.

Fixed on `organized` at `66f5da3`. **This run was deliberately not repeated**,
so no current checkout reproduces it.

## What that costs, precisely

**Accuracy — bounded, and no conclusion here moves.** Measured on an 80-frame
subset of this same split, both arms, contexts 1 and 10: **max ΔAbsRel 0.51%,
Δδ1 0.20%, ΔRMSE 0.14%, and the model ordering by AbsRel is unchanged in every
arm.** This run's findings are 8–21% effects — rectification helping `vggt_1b`
by 20.6%, context ratios spanning 0.866–1.161 — so none of them depends on a
margin that small.

**Timing — not bounded the same way, and `meta.json` overstates one thing.**
`meta.json` reports VGGT-1B's cost under a 10-frame window growing **13.7×**
against VGGT-Omega's 4.4×, and presents it as a property of the architecture.
It is substantially a property of the dtype: VGGT-1B was fp32 while the other
three were bf16. Concretely, VGGT-1B was **3.9× VGGT-Omega at one frame and
7.0× at ten** on 1191M against 1144M parameters. **Any cross-model timing
comparison in this artefact is invalid.** The lane-imbalance analysis in
`meta.json` and in `slam020_lane.sh` is still correct as a description of what
this run did, but the corrected partition it recommends was derived from
timings that will not recur — a re-run on the fixed code is 2.93× faster on
`vggt_1b` and would want its lanes rebalanced again.

Same applies to `results/slambench-raw-b1659a0` and every `results/fovbench-*`
carrying a `vggt_1b` row: all of them reached this backbone through
`finetune/eval/baselines/model_zoo.py`.
