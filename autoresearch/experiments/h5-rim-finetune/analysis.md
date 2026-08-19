# H5 analysis

## CPU pilot (2026-08-26, EXPLORATORY — not the protocol claim)

Setting deliberately reduced: seq131 only, even-frames train / odd-frames
eval (frame-level split, NOT the protocol's scene-level holdout), 252 px,
10 epochs, defaults (α=2, λ_f=1, λ_m=0.5). Purpose: direction + hyperparameter
sanity while the box queue is dark. Staged via symlink split dirs.

Depth (held-out odd frames, protocol-of-record eval):

| zone | before | after | Δ |
|---|---|---|---|
| near rim | 1.408 | 0.567 | **−59.7%** |
| near center | 0.715 | 0.492 | −31.2% |
| center | 0.399 | 0.291 | −27.0% |
| far | 0.268 | 0.212 | −21.2% |

Every zone improves; no near-center collateral (the failure mode the H2.1
table had). Training curve healthy: total 1.44→1.03 over 10 epochs, the
rim-feature term rising to ~0.14 then flat (LoRA moving rim features but
bounded by the distillation pull — behaving as designed), mv term stable.

Pose: unchanged within noise (rot median 76.9°→75.7°, RRA@15 0.154→0.154,
n=13) — but these pilot pairs are 100-frame-spaced (~60–80° rotations), the
saturated regime the eval addendum already flagged; this says "no collapse",
not "no effect". The informative pose read needs the box's dense pairs.

Read: mechanism works end to end across frames; defaults are sane; nothing
here justifies changing the #35 configuration. The protocol claims (scene-
level holdout, pose improvement, plain-LoRA control comparison) remain
GPU-gated.
