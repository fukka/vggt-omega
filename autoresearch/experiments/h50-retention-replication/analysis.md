# H50 — the run is VOID by its own gate, and one measurement survives it

## The verdict

**VOID.** `protocol.md` fixed three gates before the run and said, in the same
sentence, what a failure costs: *"A cell that fails a gate is not scored; the
run is void and is reported as void."* One did fail.

| gate | result |
|---|---|
| plumbing (flip twice = identity) | **PASS** — worst 0.000000% |
| resampling preserved the content | **PASS** — every arm inside ±15% |
| resolving power ≥ 2.0× | **FAIL** — `da3:small` in the ScanNet++-content cell, **1.56×** |

The rule was not applied. For the record, and because hiding it would be worse
than printing it, `analyze_h50.py` also reports what the rule *would* have
returned had the gate passed: **CONTENT REPLICATES** — retention 0.62 / 0.41 /
0.46 / 0.64, row > column 4/4. That sentence is not a verdict and is not cited
as one.

## The table

| backbone | Aria/Aria | Aria ct / SN++ lens | SN++ ct / Aria lens | retention | SN++/SN++* |
|---|---|---|---|---|---|
| da3:small  | +173.5% | +107.2% | −5.8% | 0.62 | −4.1% |
| da3:large  | +296.8% | +121.4% | −5.2% | 0.41 | +23.1% |
| vggt       |  +54.5% |  +24.9% | +5.6% | 0.46 |  +6.1% |
| vggt_omega |  +31.2% |  +19.9% | +1.6% | 0.64 |  +0.3% |

\* not region-matched, in no clause of the rule — declared in `protocol.md`
before the run.

Data: ADT `seq144`–`seq150` and the eight ScanNet++ scenes at positions 9–16 of
the depth-carrying list. H48 used `seq136`–`seq143` and positions 1–8. Disjoint.

## What survives the void, and why it is separable

**Clause 1 — retention — does not touch the failing cell.** It is
(Aria content, ScanNet++ lens) ÷ (Aria content, Aria lens), and both of those
cells clear the gate everywhere:

| cell | floor | da3:small | da3:large | vggt | vggt_omega |
|---|---|---|---|---|---|
| Aria / Aria lens (7 seqs)      | 0.4404 | 3.81× | 9.05× | 6.16× | 8.15× |
| Aria / ScanNet++ lens (7 seqs) | 0.4363 | 2.79× | 4.47× | 3.74× | 4.48× |
| ScanNet++ / Aria lens (8 scenes) | 0.2255 | **1.56×** | 2.54× | 2.68× | 3.77× |

So the retention numbers are a gate-clean measurement, and they are the
striking part of this run:

| backbone | H48 retention (scenes 1–8, seq136–143) | H50 retention (held out) |
|---|---|---|
| da3:small  | 0.62 | **0.62** |
| da3:large  | 0.41 | **0.41** |
| vggt       | 0.46 | **0.46** |
| vggt_omega | 0.44 | 0.64 |

Three of four agree **to two decimals on disjoint recordings**, and they do so
while *both* of the numbers making the ratio moved: `da3:small`'s native effect
went +154.8% → +173.5%, `da3:large`'s +261.4% → +296.8%. The effect size is a
property of the recording; **the fraction that survives a change of optics
looks like a property of the backbone.** That is a stronger statement than H48
could make and it is not what the rule was written to test.

`vggt_omega` is the one that moved (0.44 → 0.64), and it is the backbone whose
native effect is smallest (+29.9 / +31.2%) — the instability a ratio has on a
small denominator, which is why H51's rule asks 3-of-4 rather than 4-of-4.

## Why the gate fired, and what it does not mean

`da3:small` on these eight ScanNet++ scenes sits at AbsRel 0.1444 against a
know-nothing floor of 0.2255. It is barely better than a fronto-parallel plane
there, so its cell value (−5.8%) is a near-zero measured by a model with almost
no purchase on the scene — precisely the situation H48's first read walked into
and the reason this gate exists.

It is **not** a code failure and not a property of the pipeline: the same cell
gave 2.40× on H48's eight scenes. It is these scenes. ScanNet++'s central 28°
cone is often close to fronto-parallel, and eight scenes is a small draw.

## What this costs and what to do about it

The clean fix is another draw of scenes, with the selection rule fixed in
advance — for example *"the next held-out scenes whose GT depth spread over the
scored cone exceeds a stated value"*, which is a **content** property measurable
without running any model, and therefore selectable without touching the
outcome. Choosing scenes by their floor **after** seeing which ones failed would
be selection on the result, and is not available.

Until that is run, the honest state of the mirror thread is:

- The content-versus-lens attribution has **one** confirmatory test outstanding.
  H48 (exploratory) and H50 (void by gate) both point the same way, and H50's
  retention figures replicate H48's on disjoint data, but no pre-registered
  rule has yet returned a verdict.
- Nothing here weakens H45/H47 — the effect itself was established on 13
  recordings and is re-measured here at +173.5 / +296.8 / +54.5 / +31.2% on
  seven more.
