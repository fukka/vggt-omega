# H48 — content or pipeline? The 2×2, read with its own decision rule

## The table

Mirror cost (ratio of means; positive = reflecting the input and reflecting the
answer back costs accuracy), four backbones × four cells:

| backbone | Aria content / Aria lens (H47) | ScanNet++ content / **Aria lens** | Aria content / **ScanNet++ lens** | ScanNet++ / ScanNet++ (H46) |
|---|---|---|---|---|
| da3:small  | +154.8% | +24.3% | **+96.3%**  | +13.5% |
| da3:large  | +261.4% |  +6.5% | **+106.5%** |  +3.3% |
| vggt       |  +45.7% |  −3.2% | **+21.0%**  |  −9.3% |
| vggt_omega |  +29.9% | +11.2% | **+13.3%**  |  +1.6% |

Read it as a 2×2 and the pattern is the same for all four backbones: the two
cells with **Aria content** are large, the two with ScanNet++ content are small,
and which lens the content was pushed through moves the number far less than
whose content it is.

Stated as retention rather than as raw size, which is scale-free and therefore
comparable across backbones that differ tenfold in effect:

| backbone | Aria effect kept through a **foreign lens** | ScanNet++ effect gained through **Aria's lens** |
|---|---|---|
| da3:small  | 62.2% (154.8 → 96.3)  | +10.8 pp (13.5 → 24.3) |
| da3:large  | 40.7% (261.4 → 106.5) |  +3.2 pp (3.3 → 6.5) |
| vggt       | 46.1% (45.7 → 21.0)   |  +6.1 pp (−9.3 → −3.2) |
| vggt_omega | 44.3% (29.9 → 13.3)   |  +9.6 pp (1.6 → 11.2) |

Aria content keeps **41–62%** of its mirror cost when re-imaged through
ScanNet++'s optics. ScanNet++ content pushed through Aria's optics picks up
**3–11 percentage points** — small, but positive for all four backbones, which
is more than chance would give.

## Both cells can resolve an effect — checked, not assumed

The first run of this 2×2 was read as "no effect either way" and that reading
was wrong twice over: one arm was void, and neither arm had been shown capable
of showing an effect at all. So the licence to read the table is measured here
rather than assumed, with the same know-nothing floor as
`VOID_reciprocal_run1.md` — the AbsRel an aligned *constant* prediction scores
— over every scene and sequence actually used, not one of them:

| cell | floor (know-nothing) | GT depth spread | da3:small | da3:large | vggt | vggt_omega |
|---|---|---|---|---|---|---|
| ScanNet++ ct / Aria lens (8 scenes)  | 0.2145 | 0.671 | 2.40× | 2.78× | 3.45× | 4.73× |
| Aria ct / ScanNet++ lens (7 seqs)    | 0.4400 | 1.185 | 2.60× | 3.95× | 3.44× | 3.81× |

Headroom is floor ÷ baseline AbsRel: how much better than a fronto-parallel
plane each backbone actually is in that cell. Every cell clears 2.4×, and the
four backbones' baselines separate (A: 0.089 / 0.077 / 0.062 / 0.045;
R: 0.169 / 0.111 / 0.128 / 0.116) instead of collapsing onto one number the way
the void arm's did. Both cells are informative.

One caution this replaces: the diagnostic's first pass used a single ScanNet++
scene (`00dd871005`, floor 0.101, depth 1.13–1.66 m) and suggested the forward
arm was nearly fronto-parallel and barely above its floor. Over all eight
scenes the floor is 0.2145 — that scene is not typical, and the caution does
not survive its own measurement.

## What the pre-registered rule says, and why it is reported unchanged

`code/analyze_h48.py` was committed before either arm ran. Its rule: call it
CONTENT if **both** DA3 cells exceed **100%** in the Aria-content column and
neither does in the other. The Aria-content column came in at **+96.3%** and
**+106.5%**. One cell misses the threshold by 3.7 points, so the rule
mechanically returns its else-branch — printed here verbatim, unedited:

> NEITHER CELL LARGE — the resampling itself destroys whatever carries the
> effect.

**The threshold did not fire; that branch's stated interpretation is separately
refuted.** The resampling demonstrably does *not* destroy what carries the
effect: the diagnostic's know-nothing floor travels with the content row and
ignores the lens column (Aria content 0.4888 / 0.4836; ScanNet++ content
0.1010 / 0.1005 — see `VOID_reciprocal_run1.md`), and a cell at +96.3% has not
had its effect destroyed by anything.

The rule is not being retuned after the fact. An absolute 100% threshold was
the wrong statistic to pre-register — it is not scale-free, so it asks the same
question of a backbone whose native effect is 261% and one whose native effect
is 30%, and vggt/vggt_omega could never have satisfied it in any cell. That is
a design fault in the rule, visible now only because the numbers landed near
it. Moving it to 90% today would buy a "CONTENT" verdict at the cost of the
thing pre-registration is for.

So the finding is reported at the strength the evidence actually supports:

- **Confirmatory, by the pre-registered rule:** the rule did not fire. No
  confirmatory verdict.
- **Exploratory, from the table:** the effect travels predominantly with the
  **content**, not the lens or the warp — same direction and same ordering in
  all four backbones, 41–62% retained across a change of optics, against
  3–11 pp picked up in the other direction.

A clean confirmatory test is cheap and is named here so it is not quietly
skipped: pre-register **retention** (Aria effect through a foreign lens, as a
fraction of native) with a threshold, and run it on sequences and scenes not
used here.

## What this does and does not eliminate

H46 left four candidate differences between the datasets standing —
**content, capture, ground truth, scored region**. This 2×2 was designed to
split the first from the rest, and it does not split all four:

- **Eliminated as the main carrier: the lens and the warp.** Aria's optics are
  not what produces the effect; the effect survives their removal at ~half
  strength, and ScanNet++ content does not acquire much by being pushed
  through them.
- **Not separated: content from capture.** Named in `protocol.md` before the
  run, and it still holds. A resample carries each dataset's *sharpness*,
  motion blur and exposure through the other's geometry — the geometry changes,
  the photometry does not. "Content" here means "the imagery", not "the scene".
- **Not addressed: ground truth and scored region.** Both cells use each
  dataset's own ground truth, and the scored region is matched at θ ≤ 28°.

So the honest statement of what H48 buys: **of the four candidates, one — the
lens/warp — is now the least likely, and the live question narrows to imagery
versus capture.** That is one difference eliminated out of four, which is what
the protocol promised at most.

## Provenance

- Forward arm: 8 ScanNet++ scenes, `results/A_*.json`.
- Reciprocal arm: 7 ADT sequences, `results/R_*.json` — **rerun** after the
  first run was found void (`VOID_reciprocal_run1.md`); the void run's outputs
  are kept under `results/void_R_cornerdepth/`.
- Plumbing bar (flip twice = identity) passes at **0.000000%** in every cell of
  both arms, the same bar H45 used to convert H44's void into a result.
- Native cells quoted from H47 (Aria/Aria) and H46 (ScanNet++/ScanNet++).
