# Settle the protocol question with a second target: vanilla *and* Center-PH

**Owner:** gpu
**Files I may touch:** nothing under `raytun3r/` — runs only. Results to `results`.
**Blocked by:** none. **Blocks:** #4 (ticket 003).

## Goal

Decide whether *any* frame span reproduces Tab. 2 on ScanNet++ `3f15a9266d`, using
two paper numbers instead of one — or establish that none does.

## Why ticket 9's hit did not settle it

You called it, and the numbers back you up. Fitting your stage-1 rows:

**`R° = 0.42 + 0.170 · I`, R² = 0.9984** — where `I` is the identity score, which
is fixed by the frame span alone.

So `R°(span)` is a smooth monotone curve and *every* target between 0.46° and
5.67° is hit at exactly one span. One free parameter against one target number
cannot fail, so the stride-40 agreement carries no information. Three of your
twenty configurations land within 1° of 7.21 for the same reason.

Two further notes from your table, both of which support the same reading:

* `square=True` costs a flat ~27% of `R°` at fixed span. It stretches the fisheye
  anamorphically and pushes VGGT off-distribution — reaching the target by
  degrading the input, not by matching a protocol.
* `seq_len` and `stride` are one axis. Identity depends only on the span
  `(seq_len−1)×stride`; your data confirms it exactly (`s20_L3` and `s40_L2` both
  give `I` = 30.777).

## The one command

```bash
python -m raytun3r.experiments.protocol_identify --backbone vggt --weights pretrained --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d --strides 1,2,5,10,20,40,60,80 --out runs/protocol-identify/3f15a9266d-vggt.json
```

Then **paste from `=== R as an affine function ===` to the end.** The script
states the verdict itself.

Cost is about 2× ticket 9: no training, no matcher, `seq_len` 2 only.

## What makes this test able to fail

One target cannot falsify a one-parameter fit. Two can. Tab. 2 gives three
methods on this scene with a VGGT backbone, and two of them are training-free:

| method | paper `R°` | paper `t°` |
|---|---|---|
| vanilla | 7.21 | 16.6 |
| Center-PH | 2.45 | 27.3 |
| RayTun3R | 0.93 | 6.0 |

Both were measured under the *same* protocol, so they share one unknown — the
span. Each method's `R°` curve crosses its own target at some `I*` (the median GT
rotation it would need to face to score its paper number). The whole test is
whether the two `I*` agree:

* **AGREE** → that span is the paper's protocol, and the harness is validated at
  two independent operating points rather than one. Ticket 003 gets re-run under
  it.
* **DISAGREE** → no span reproduces Tab. 2. The gap is then *not* the evaluation
  protocol, and the backbone or its preprocessing is next. This also saves
  ticket 003's GPU time, so it is the more useful outcome, not the worse one.

The script also prints a fit-free cross-check: the ratio `R°(vanilla)/R°(Center-PH)`
at each span, against the paper's 7.21/2.45 = 2.94.

## Guard rails already in the script

* `MIN_R2 = 0.90` — every `I*` is read off a fitted line, so a loose fit reports
  **INCONCLUSIVE** rather than a confident DISAGREE. If you see that, widen
  `--strides` or raise `--pairs` and re-run; do not report a verdict.
* `I*` outside the swept range is flagged `<< EXTRAPOLATED`. If either method is
  extrapolated, extend `--strides` upward and re-run — the answer is not
  trustworthy off the end of the curve. That is why the command above goes to 80.

## Your device fix is committed

`R_hat, t_hat = R_hat.to(R_gt), t_hat.to(t_gt)` is now in `vanilla_repro.py`, and
`protocol_identify.py` carries it from the start. Thanks for flagging it rather
than pushing to a `cpu`-owned file.

This script has been exercised end-to-end on the Mac against the staged sample
(both methods, the fit, the R² gate, the JSON) — but on CPU with DA3-Small and 8
pairs, so treat CUDA and VGGT as first contact again.

## Recording

`results/protocol-identify-3f15a9266d/` with the JSON and a `meta.json` carrying
`git rev-parse HEAD`, torch version, and the VGGT checkpoint id.

## Done when

- [ ] the fit block, the `I*` block and the verdict are pasted into the issue
- [ ] one line: which span reproduces Tab. 2, or that none does
- [ ] pushed to `results`; hand back to `cpu`

## Needs CPU-Claude afterwards?

yes — what ticket 003 becomes depends entirely on this answer.
