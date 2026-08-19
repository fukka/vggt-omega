# H7 — theta-gated LoRA (protocol, locked before any run)

Hypothesis (state file H7): gating the low-rank update by incidence angle,
y = base(x) + B(g(theta) ⊙ A x)·s, matches or beats uniform LoRA at equal
rank, because the measured failure field (run_009) is a smooth function of
theta and uniform LoRA must spend shared capacity to approximate a
theta-dependent correction.

## Design
- GatedLoRALinear = h5 LoRALinear + per-token rank-wise gate
  g(theta) = 1 + tanh(W2 relu(W1 [theta/theta_max, (theta/theta_max)^2])),
  W2 zero-init ⇒ g ≡ 1 at init = exactly uniform LoRA at start (and the
  teacher path stays bit-identical via the inherited `enabled` toggle).
- Special (non-patch) tokens get g = 1 always.
- Same placement as H5 (ViT blocks 8-11 MLP fc1/fc2), same losses, same
  defaults (alpha=2, lambda_f=1, lambda_m=0.5). Gate adds ~1.5k params on
  top of ~123k (+1.2%); zero inference cost beyond a precomputed (N,r) map.

## Pilot (CPU, EXPLORATORY — direction only, mirrors the H5 pilot exactly)
- seq131 even-frames train / odd-frames eval, 252px, 10 epochs, same seed.
- Uniform-LoRA anchor = the existing H5 pilot run (identical settings):
  near_rim 1.408→0.567, near_center 0.715→0.492, center 0.399→0.291,
  far 0.268→0.212.
- Arm A: gated r=8. Arm B (only if A shows direction): gated r=4 vs the
  uniform r=8 anchor ("half the rank, same quality" efficiency claim).

## Predictions (locked)
- P1: gated r=8 near_rim AbsRel <= uniform's 0.567, with near-center no
  worse than uniform's 0.492 (no collateral).
- P2 (mechanism): the learned gate magnitude ||g(theta)-1|| increases with
  theta — the model spends its conditioning where the measured field is.
- P3 (stretch, arm B): gated r=4 within 5% of uniform r=8 on near_rim.

## Decision rule
P1 holds → H7 graduates to a #35-style held-out-scene GPU ticket (after the
current queue drains). P1 fails but P2 holds → analyze where the gate went.
Both fail → H7 refuted, record why, uniform LoRA stands.
