# H12 — lens-Jacobian conditioning: the pre-registered kill criterion fired

**Verdict: STOP.** `jac` does not beat `shuffled` on either held-out sequence.
This is the fourth controlled negative in the rim-intervention line, and by the
criterion written down *before* the run it ends the H12 method here. It does not
proceed to ScanNet++.

## The design

Three arms, identical in architecture, parameter count, data, seed, optimiser
and loss. The **only** difference is what the FiLM conditioner is shown:

| arm | field |
|---|---|
| `jac` | real per-token `(log_area, log_aniso, theta/theta_max)` |
| `shuffled` | the same values under a fixed permutation of token positions |
| `theta` | `theta` only, zero-padded to equal width |

`shuffled` is the arm that decides it: same value distribution, same capacity,
no spatial correspondence. FiLM's output layer is zero-initialised, so at step 0
every arm is bit-identical to the unconditioned network. Loss is the **plain**
depth loss — rim weighting is what already lost to its own control in H5.

Trained on seq131/133/134/135, 20 epochs, 504 px. Held out: seq136 and
decoration_seq132 — the same split as #35/#36.

## The decider

near-rim AbsRel after training (lower is better; `before` is shared):

| sequence | `jac` | `shuffled` | `theta` | winner |
|---|---|---|---|---|
| seq136 | 0.2477 | **0.2354** | 0.2400 | shuffled |
| decoration_seq132 | 0.2394 | 0.2337 | **0.2334** | theta |

**`jac` is the worst of the three on near-rim, on both sequences.** The real
geometry field loses to a scrambled one carrying identical values.

Full zone table, before → after:

| seq | arm | near_rim | near_center | center | far |
|---|---|---|---|---|---|
| seq136 | jac | −81.78 % | −52.04 % | −52.42 % | −57.86 % |
| seq136 | shuffled | **−82.69 %** | −50.38 % | −53.79 % | −58.99 % |
| seq136 | theta | −82.34 % | −50.30 % | −53.40 % | −58.67 % |
| seq132 | jac | −33.75 % | **+29.97 %** | **+0.31 %** | −37.09 % |
| seq132 | shuffled | −35.33 % | +57.49 % | +16.73 % | −37.90 % |
| seq132 | theta | −35.40 % | +50.82 % | +12.10 % | −37.12 % |

## The one thing that is not nothing

On decoration_seq132 the arms differ far more in the **centre** than at the rim,
and there `jac` is much the best: centre **+0.31 %** against shuffled's
**+16.73 %**, and near-centre **+29.97 %** against **+57.49 %**. Every arm makes
the centre worse on that sequence; the real geometry field makes it worse by
much less.

So the conditioning is not inert — it does something, and it is not what it was
built for. It **protects the centre** rather than helping the rim. That is a
finding, not a consolation: it is consistent with the field's largest *gradient*
being in the mid-field, and with the rim band being the region where the
conditioning is least informative because `log_aniso` turns over and changes
sign there (README, `test_aria_field_is_NOT_monotone…`).

It does not rescue the hypothesis. The claim was about the rim, the criterion
was about the rim, and at the rim the scrambled control wins.

## What this adds to the line

Four rim-targeted interventions have now failed against their own controls:

| | intervention | control | result |
|---|---|---|---|
| H5 | rim-weighted losses | plain LoRA | control wins (−80.6 % vs −83.5 %) |
| H6 | rim-restricted KV | all-token | control wins (−52.2 % vs −75.9 %) |
| — | centre/rim dual-expert MoE | — | killed by H7+F2+F4 |
| **H12** | **lens-Jacobian conditioning** | **shuffled field** | **control wins** |

H12 was the strongest form of the idea available: it gave the network the
*geometry* rather than telling it where to try harder, and it still lost. The
reframing that motivated it — the rim deficit is a global lens-prior mismatch,
not a region-shaped capacity problem — survives, and is now supported by an
experiment designed to exploit it and failing.

## Not claimed

* **One seed, one run per arm.** The margins are small (seq136: 0.2477 vs
  0.2354 is 5 % relative) and no error bars were computed. What licenses the
  stop is not the size of the gap but its DIRECTION plus the pre-registration:
  `jac` had to beat `shuffled` and it did not, on either sequence.
* That conditioning *cannot* work — only that this injection point (FiLM on the
  last ViT block), at this capacity, on this data, does not.
* Anything about transfer to a different lens. That was the reason for the
  design; it was never reached.
* The centre effect above is from two sequences and was not pre-registered.
  It is reported because it is real in this data, not as a salvaged result.
