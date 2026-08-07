# Settle the protocol question with every training-free target, not one

**Owner:** gpu
**Files I may touch:** nothing under `raytun3r/` — runs only. Results to `results`.
**Blocked by:** none. **Blocks:** #4 (ticket 003).

## Goal

Decide whether *any* frame span reproduces the paper's ScanNet++ numbers — or
establish that none does — using seven training-free targets across three
backbones instead of the single number ticket 9 fitted.

## Why ticket 9's hit did not settle it

You called it, and the numbers back you up. Fitting your stage-1 rows:

**`R° = 0.42 + 0.170 · I`, R² = 0.9984** — where `I` is the identity score, which
is fixed by the frame span alone.

So `R°(span)` is smooth and monotone, and *every* target between 0.46° and 5.67°
is hit at exactly one span. One free parameter against one target number cannot
fail. Three of your twenty configurations land within 1° of 7.21 for that reason.

Two further notes from your table:

* `square=True` costs a flat ~27% of `R°` at fixed span. It stretches the fisheye
  anamorphically and pushes VGGT off-distribution — reaching the target by
  degrading the input, not by matching a protocol. I would not read your
  `square=True` unanimity as settling resolution.
* `seq_len` and `stride` are one axis. Identity depends only on the span
  `(seq_len−1)×stride`; your data confirms it exactly (`s20_L3` and `s40_L2` both
  give `I` = 30.777).

## I under-counted the targets, and that was the real mistake

Tickets 004/005 said "DA3-Small has no per-scene vanilla number, so only VGGT can
be checked". Literally true, but it justified one backbone against one number —
the exact degeneracy above. The actual inventory:

| source | backbone | scene | training-free rows |
|---|---|---|---|
| Tab. 2 | VGGT | `3f15` **named** | vanilla 7.21, Center-PH 2.45 |
| Tab. 2 | **π³** | `3f15` **named** | vanilla 6.17, Center-PH 2.28 |
| Tab. 1 | DA3-Small | ScanNet++ mean | vanilla 10.21, Center-PH 3.27, **Multi-PH 1.66** |

Seven numbers, not one. Tab. 2's four are tight (named sequence, one protocol).
Tab. 1's three cannot be matched in absolute terms from one scene — but they are
the only row with Multi-PH, and **ratios between methods survive both the unknown
span and the unknown scene set**, which is exactly the property absolute `R°` lacks.

## The commands

**π³ needs installing first** — it is new in `3a38e22`, not on PyPI:

```bash
pip install git+https://github.com/yyfz/Pi3.git
```

Needs Python ≥ 3.10 and torch ≥ 2.3 (`torch.nn.attention`); lambda_63's 2.11 is
fine. If pip is awkward, clone it beside the repo or set `$PI3_ROOT` — the loader
tries both. Then:

```bash
python -m raytun3r.experiments.protocol_identify --backbone vggt --weights pretrained --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d --strides 1,2,5,10,20,40,60,80 --out runs/protocol-identify/3f15-vggt.json
```

```bash
python -m raytun3r.experiments.protocol_identify --backbone pi3 --weights pretrained --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d --strides 1,2,5,10,20,40,60,80 --out runs/protocol-identify/3f15-pi3.json
```

```bash
python -m raytun3r.experiments.protocol_identify --backbone da3 --weights pretrained --methods vanilla,center_ph,multi_ph --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d --strides 1,2,5,10,20,40,60,80 --out runs/protocol-identify/3f15-da3.json
```

**Paste, for each, from `=== R as an affine function ===` to the end.** The script
states its own verdict. No training, no matcher; the third is slower because
Multi-PH runs five virtual views per frame.

If π³ will not install, run the other two and say so — they carry the tight
targets and the ticket still concludes.

## What makes this able to fail

Each method's `R°` curve crosses its own paper target at some `I*` — the median GT
rotation it would need to face to score that number. All methods in a table share
one unknown, the span. So:

* **AGREE** → that span is the paper's protocol, and the harness is validated at
  several independent operating points. Ticket 003 gets re-run under it.
* **DISAGREE** → no span reproduces the table. The gap is then *not* the
  evaluation protocol, and the backbone or its preprocessing is next. This saves
  ticket 003's GPU time, so it is the more useful outcome, not the worse one.

VGGT and π³ are independent replications of the same question on the same
sequence: if they disagree with each other about `I*`, that alone is decisive.

## Guard rails already in the script

* `MIN_R2 = 0.90` — every `I*` is read off a fitted line, so a loose fit reports
  **INCONCLUSIVE** rather than a confident DISAGREE. If you see that, widen
  `--strides` or raise `--pairs`; do not report a verdict.
* `I*` outside the swept range is flagged `<< EXTRAPOLATED`. That is why the
  commands go to 80.
* The `da3` run prints **NO VERDICT on I\***, because Tab. 1 is a mean over unnamed
  scenes and the absolute level carries a scene-composition offset this cannot
  separate from a protocol offset. Its pairwise ratio block is the signal there.

## Exercised, but not on CUDA

Both scripts ran end-to-end on the Mac against the staged sample — all three
methods, the fit, the R² gate, the extrapolation flags, the pairwise ratios, JSON
out — but on CPU with DA3-Small and a handful of pairs. π³'s pose convention is
pinned by a test (it emits camera-**to**-world, and the wrapper inverts it;
verified to 4e-6 against π³'s own `points = camera_poses @ local_points`), but the
π³ *checkpoint* has never been loaded anywhere. Treat CUDA and the real weights as
first contact. 51 pytest + 35 smoke pass. Your device fix is committed.

## Recording

`results/protocol-identify-3f15a9266d/` with the three JSONs and a `meta.json`
carrying `git rev-parse HEAD`, torch version, and the VGGT/π³ checkpoint ids.

## Done when

- [ ] the fit block, `I*` block and verdict pasted for each backbone that ran
- [ ] one line: which span reproduces the paper, or that none does
- [ ] pushed to `results`; hand back to `cpu`

## Needs CPU-Claude afterwards?

yes — what ticket 003 becomes depends entirely on this answer.
