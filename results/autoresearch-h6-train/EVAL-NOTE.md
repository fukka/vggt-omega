# #35 / #36 eval runs — 2026-08-22

The training halves landed 2026-08-19; these are the eval halves both issues
were still open for. Run on lambda_63 @ `8f8b574`, `--size 504 --max-frames 60`
(the same as `results/autoresearch-bench/meta.json`, so the frame budget matches
the frozen rows).

## What unblocked them

`AriaLocalPairs` globbed `videos_rgb/*.jpg`. Every ADT sequence is PNG and none
has a single `.jpg`, so every eval died with "no frames" on data that was
sitting right there. Tickets 033/034 told the runner to apply a "`*.png` glob
workaround" by hand — folklore rather than a fix. Fixed in `8f8b574`.

## #35 — the rim-targeted losses do not beat plain LoRA

near-rim AbsRel (`<=2m, >=38deg`), before -> after, on the two HELD-OUT sequences:

| run | seq136 | decoration_seq132 |
|---|---|---|
| **full** (all rim losses) | 1.3595 -> 0.2638 (**-80.6%**) | 0.3614 -> 0.2410 (**-33.3%**) |
| **plain** (LoRA only, control) | 1.3595 -> 0.2238 (**-83.5%**) | 0.3614 -> 0.2402 (**-33.5%**) |

**The control matches or beats the method on both sequences.** Pose too: plain
improves median rotation 13.51->11.52 deg on seq136 where full moves it the
wrong way (13.51->13.73).

The rim gain is real and large — but it is bought by LoRA fine-tuning per se,
not by `--depth-alpha`/`--lambda-f`/`--lambda-m`. That is the control doing
exactly its job.

## #36 — the all-token control beats the rim-restricted module

near-rim AbsRel, before -> after:

| run | seq136 | decoration_seq132 |
|---|---|---|
| **rim** (627/1296 tokens) | 0.7294 -> 0.3488 (**-52.2%**) | 0.2519 -> 0.2601 (**+3.2%**) |
| **alltok** (control) | 0.7294 -> 0.1757 (**-75.9%**) | 0.2519 -> 0.2364 (**-6.2%**) |

H6.1's exploratory finding was "rim-KV == full-KV on near_rim". **On held-out
sequences it does not hold**: all-token is clearly better on seq136, and the rim
arm makes decoration_seq132 slightly worse. The efficiency claim (0.48x FLOPs)
is not free.

## The magnitudes are NOT comparable to `results/autoresearch-bench/`

`eval_lora.py`/`eval_module.py` are internally consistent — `Seq.gt_range`
converts ADT planar-z GT to euclidean **range**, and the prediction is asserted
`require_convention("range")`, so both sides match. But
`eval_baseline_joint.py` scores under the **planar-z** contract. The two are
different conventions, which is why this file's "before" near-rim on seq136
(1.3595) sits far from the frozen `da3_small` row (0.1490).

Reconciling them is not done here and is a `cpu` ticket. It does **not** affect
either conclusion above: within each table both arms share one code path, one
frame set, one alignment and one GT, so full-vs-plain and rim-vs-alltok are
clean comparisons. Only cross-table magnitude claims are blocked.

## Not claimed

* Single run per arm, no seeds, no bootstrap CI drawn here — the `per_frame` key
  is present in every JSON so error bars can be computed, but they were not.
* Two held-out sequences of one apartment on one device. This is a
  cross-SEQUENCE result, not cross-scene.
* Why the two eval families disagree in magnitude is recorded, not explained.
