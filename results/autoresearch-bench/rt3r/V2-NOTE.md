# #38 v2 — the RayTun3R rows, re-run with the double-conversion fix

Issue #38 was reopened on 2026-08-19: `ADTSequence.depth()` already converts ADT
planar-z to euclidean range on load (`raytun3r/data.py:389-392`), and
`raytun3r_row.py` divided by `cos(theta)` **again**, inflating rim GT by up to
1.73x. Fixed in `8b5c13d`. The fitted adapters were unaffected, so only the four
row evals were re-run (lambda_63, 2026-08-22, `--max-frames 60`).

## v1 vs v2

near-rim = count-weighted mean AbsRel over joint cells with `theta_bin_mid >= 38`
and GT depth `<= 2m`, the same definition `autoresearch-bench/meta.json` uses.

| seq | arm | whole v1 -> v2 | near_rim v1 -> v2 |
|---|---|---|---|
| seq136 | vanilla | 0.1718 -> **0.2978** | 0.5417 -> **0.5660** |
| seq136 | adapted | 0.2288 -> **0.2764** | 0.5771 -> **0.4776** |
| decoration_seq132 | vanilla | 0.0793 -> **0.0701** | 0.1295 -> **0.1226** |
| decoration_seq132 | adapted | 0.1262 -> **0.1032** | 0.2555 -> **0.1688** |

## The fix changed the conclusion, not just the magnitude

v1's consumption comment recorded "adaptation hurts on both held-out ADT
scenes". With the double conversion removed that is **no longer true**:

| seq | whole | near_rim |
|---|---|---|
| **seq136** | 0.2978 -> 0.2764 (**-7.2%**) | 0.5660 -> 0.4776 (**-15.6%**) |
| **decoration_seq132** | 0.0701 -> 0.1032 (**+47.3%**) | 0.1226 -> 0.1688 (**+37.6%**) |

RayTun3R adaptation **helps on seq136 and hurts on decoration_seq132** — mixed,
not uniformly negative. The direction flipped on one of the two scenes purely
from correcting the GT convention, which is why the reopen was right and why no
v1 number should be quoted.

## Not claimed

* Two scenes, one adapter fit each, raytun3r defaults (30 windows / 300 iters).
* Why the two scenes disagree in direction is not established here. seq136's
  vanilla whole AbsRel (0.2978) is much worse than decoration_seq132's (0.0701),
  so seq136 is the harder scene for this backbone before any adaptation — but
  that is an observation, not an explanation.
* `cos_t` is now computed and unused in `raytun3r_row.py:74` (dead after the
  fix). Harmless, worth a tidy in a `cpu` ticket.
