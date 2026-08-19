# H7 analysis — REFUTED (2026-08-19, CPU pilot, all three predictions failed)

Pilot mirrors the H5 pilot exactly (seq131 even/odd, 252px, 10 epochs, same
seed, same losses); the ONLY difference per arm is the gate and/or rank.

| arm | params | near_rim | near_center | center | far |
|---|---|---|---|---|---|
| uniform r=8 (H5 pilot, anchor) | 123k | **0.567** | 0.492 | 0.291 | 0.212 |
| gated r=8 | 124.4k | 0.572 | 0.478 | 0.292 | 0.212 |
| uniform r=4 | 61k | 0.753 | 0.418 | 0.304 | 0.227 |
| gated r=4 | 62k | 0.754 | 0.433 | 0.302 | 0.226 |

- P1 (gated r=8 beats uniform r=8): FAILED — ties on every zone.
- P2 (gate learns theta-dependence): FAILED — |g(theta)-1| ~ 0.058-0.062
  across the whole range (flat; only +5% at the rim). The gate had full
  freedom (zero-init tanh) and chose to stay uniform.
- P3 (gated r=4 matches uniform r=8): FAILED — 0.754 vs 0.567; rank is a
  real constraint but gating does not substitute for it (gated r=4 ==
  uniform r=4 exactly).

Mechanism (why, not just what): LoRA's update is already input-dependent
per token, and the backbone's positional embedding injects position into
every token — so a spatially-varying correction is ALREADY inside uniform
LoRA's hypothesis class, and SGD finds it without an explicit theta input.
Explicit geometry conditioning is redundant at the adapter level. This
matches DAPETR's "learned adaptation vs explicit geometric reparameterization
can conflict" and RayTun3R's naive-PE-remap ablation from the other side.

Paper use: ablation row justifying plain LoRA placement in the H5 method;
kills the "why not condition the adapter on the camera?" reviewer question
with a measured answer. No GPU ticket will be spent on H7.

Provenance: results/h7_eval_r8.json, results/h7_eval_r4.json,
results/h5_eval_r4.json (uniform r=4 control); gate curve inside the r=8
checkpoint (scratch h7_pilot_r8/gated_lora_last.pt, gate_curve key).
