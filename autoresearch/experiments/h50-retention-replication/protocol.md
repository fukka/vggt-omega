# H50 — does H48's content verdict replicate on held-out data?

**Locked before the held-out data was touched.** Written the same day H48's
result was published, and committed before the run.

## Why this exists

H48 answered the mirror thread's central question — the effect travels with the
content, not the lens — but it had to be reported as **exploratory**, because
its own pre-registered rule did not fire. That rule asked whether both DA3
cells exceeded **100%** in the Aria-content column; they came in at **+96.3%**
and **+106.5%**. `analyze_h48.py` therefore returned its else-branch, whose
stated interpretation the same session's floor table separately refutes.

The rule was not moved. The diagnosis instead: an **absolute** threshold was
the wrong statistic to pre-register. It is not scale-free, so it asks the same
question of a backbone whose native effect is 261% and one whose native effect
is 30%, and `vggt` / `vggt_omega` could never have satisfied it in any cell.

H50 fixes the statistic, not the threshold, and re-runs on data neither arm has
seen.

## Honest statement of what this is and is not

**This is a replication on held-out data, not a blind pre-registration.** The
rule below was written *after* seeing H48's numbers, so its thresholds are
informed by them. What is genuinely unseen is the **data**: seven ADT sequences
and eight ScanNet++ scenes that neither H46, H47 nor H48 used. A rule fixed
before new data still tests something real — whether the effect is a property
of these datasets or of the particular eight scenes and seven sequences H48
happened to draw. It does not test what a blind pre-registration would.

Said plainly here so it cannot be quietly upgraded later.

## The statistic

For each backbone *m*, from the same four cells as H48:

    retention_m = mirror(Aria content, ScanNet++ lens) / mirror(Aria content, Aria lens)
    row_m       = mirror(Aria content, ScanNet++ lens)
    col_m       = mirror(ScanNet++ content, Aria lens)

`retention` is scale-free by construction: it asks what fraction of *that
backbone's own* effect survives a change of optics, so a backbone with a 30%
native effect and one with a 261% native effect are asked the same question.

## The rule (fixed here, before the run)

**CONTENT REPLICATES** iff both hold:

1. **retention_m ≥ 0.30 for all four backbones.** H48 measured 0.41–0.62; 0.30
   is a real bar that a null result (retention near 0) fails and the observed
   effect clears with room.
2. **row_m > col_m for all four backbones, 4/4, no exceptions.** This is the
   row-beats-column claim itself, per backbone, and it needs no threshold.

**CONTENT DOES NOT REPLICATE** if either fails. There is no partial verdict and
no third branch to retreat into: a 3/4 on clause 2, or one backbone below 0.30,
is a failure to replicate and will be reported as one.

Both clauses are computed on the **ratio of means**, this line's citable
statistic since H37 — never the mean of per-frame ratios.

## Gates, checked before the rule is applied

A cell that fails a gate is not scored; the run is void and is reported as void.

- **Plumbing.** Flip twice = identity, worst deviation `< 0.001%` in every cell.
  H45's bar, which converted H44's void into a result.
- **Resolving power.** Know-nothing floor ÷ baseline AbsRel `≥ 2.0` for every
  backbone in every cell, measured over every scene actually used. H48 measured
  2.4–4.7×. A cell below this cannot show an effect and its number means
  nothing — the mistake H48's first read made.
- **Resampling preserved the content.** GT depth spread ((p95−p5)/p50) over the
  scored cone must agree between the native and resampled arms of the same
  content within **±15%**. This is the new standing rule from H48's void arm,
  where 1.270 became 0.398 and nothing complained. Now recorded in every result
  file and gated on.

## Held-out data

Neither the scenes nor the sequences below appear in H46, H47 or H48.

- **Aria content** (reciprocal arm): ADT `seq144–seq150`, seven sequences.
  H48 used `seq136–seq143`.
- **ScanNet++ content** (forward arm): the eight scenes at positions 9–16 of the
  sorted scene list. H48 used positions 1–8.

Native cells are re-measured on the same held-out data rather than quoted from
H47/H46, since `retention` is a ratio between two cells and both must come from
the same recordings.

## What a failure would mean

Not "the effect is not real" — H45/H47 established it on 13 recordings. It
would mean the *content-versus-lens attribution* does not hold outside the
scenes H48 drew, which would put the lens/warp back among the live candidates
and make the honest statement "we do not know which of four".

## Cost

Evaluation only. 4 backbones × 4 cells × 20 frames × (7 sequences + 8 scenes),
two GPUs, no training.
