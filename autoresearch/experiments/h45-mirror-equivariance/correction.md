# Correction to H45 and H47 — the citable statistic, and the absolute numbers

Found by applying two of this line's own standing rules to its own freshest
results, on the Mac, with no new data.

## 1. The reported numbers were the mean of per-frame ratios

`mirror_curve.py` reports `mean(mirror_i / normal_i)` over frames. This line
already wrote down, twice, that this is the wrong statistic:

* **H33**: multi-view baselines near zero made per-frame ratios explode, one to
  `nan`, and would have read as "VGGT is 5× asymmetric".
* **H37**: *"Do not quote the mean of the per-recording ratios… the ratio of
  means is the citable form."*

The same inflation is present here. Both statistics, over the same frames:

| | | normal | mirror | **ratio of means** | mean of ratios (reported) |
|---|---|---|---|---|---|
| **H45** Aria 89°, 13 rec | `da3:small` | 0.1309 | 0.4115 | **+214.3%** | +255.1% |
| | `da3:large` | 0.0578 | 0.2264 | **+291.7%** | +350.0% |
| | `vggt` | 0.0732 | 0.1154 | **+57.6%** | +67.2% |
| | `vggt_omega` | 0.0477 | 0.0714 | **+49.6%** | +58.0% |
| **H47** Aria 60° border-free, 7 rec | `da3:small` | 0.1192 | 0.3038 | **+154.8%** | +211.4% |
| | `da3:large` | 0.0512 | 0.1851 | **+261.4%** | +352.4% |
| | `vggt` | 0.0722 | 0.1053 | **+45.7%** | +57.3% |
| | `vggt_omega` | 0.0567 | 0.0736 | **+29.9%** | +41.9% |
| **H46** ScanNet++ rectified | all four | — | — | +29.1 / −2.1 / −26.0 / +2.2% | +28.7 / +0.0 / −15.2 / +2.9% |

On Aria the inflation is **10 to 60 percentage points**. On ScanNet++ the two
statistics agree, because the effects are small — so **H46's non-replication is
unaffected by which statistic is used.**

## 2. H47's "unchanged" was too strong

On the citable statistic the border-free 60° view retains **60–90%** of the 89°
effect (0.72, 0.90, 0.79, 0.60), where the inflated statistic made it look like
0.83–1.00, i.e. "unchanged".

**H47's conclusion survives and its wording does not.** Removing the border
leaves most of the effect, so **the border is not the mechanism** — but it is
*"most of it survives"*, not *"it is unchanged"*. And the 60° arm also changes
the field of view, so even that 10–40% drop is not attributable to the border
alone.

## 3. The absolute numbers, which were never printed

`standing_rule_2` says to print them beside the relative ones, and H45 did not:

| backbone | normal AbsRel | mirror AbsRel |
|---|---|---|
| `da3:small` | 0.131 | **0.412** |
| `da3:large` | 0.058 | **0.226** |
| `vggt` | 0.073 | **0.115** |
| `vggt_omega` | 0.048 | **0.071** |

**Nothing collapses.** A scale-shift-aligned AbsRel above ~0.5 is not a usable
depth map, and none of these reaches it. So the honest verb is **degrades
substantially** — for DA3-Small, from a decent map to a poor one — not *breaks*.

## What does not change

The family split (both DA3 far above both VGGT), the plumbing bar
(0.000000% everywhere), H46's non-replication, and H47's conclusion that the
black corners are not the mechanism. **The direction and the ordering are
untouched; the magnitudes come down and the verb softens.**

## How it was found

By applying `standing_rule_2` ("always print the absolute numbers beside the
relative ones") and H37's ratio rule to results that were already committed. No
GPU, no new data, and the box was unreachable at the time. **Third instance of
CPU-only arithmetic on committed artefacts producing a correction** — after
H39b's decomposition and H43's origin account.

## Two ways to aggregate, both citable, ≤4 points apart

The table above pools every frame from every recording. The report and
`summary.json` instead give the **mean over recordings of each recording's own
ratio of means, with an sd** — which is how the rest of this line reports
things:

| backbone | pooled over frames | mean over 13 recordings |
|---|---|---|
| `da3:small` | +214.3% | **+214.5% ± 29.3** |
| `da3:large` | +291.7% | **+295.5% ± 62.2** |
| `vggt` | +57.6% | **+59.0% ± 18.7** |
| `vggt_omega` | +49.6% | **+50.1% ± 14.2** |

They differ by at most 4 points. **The per-recording form is the one quoted**,
because it carries a spread; the difference is recorded here so the two are not
silently mixed.

`analyze_h45.py` now computes the citable statistic and keeps the as-reported
one beside it, and the report's figure is regenerated from it — the figure and
the table had disagreed for one publish.
