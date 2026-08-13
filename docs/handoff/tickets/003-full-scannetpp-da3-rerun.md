# Does the adapter close the gap to classical geometry? The actual reproduction

**Owner:** gpu
**Status:** **open** — raytun3r. Supersedes #9/#10/#12; no run under `results/`.
**Files I may touch:** nothing under `raytun3r/` — runs only. Results to `results`.
**Blocked by:** nothing. This supersedes #9, #10 and #12.

## Goal

Measure whether RayTun3R moves the frozen backbone's **rotation gain** from ~0.85
toward the ~0.97 that classical geometry reaches on the same pairs — on
ScanNet++ `3f15a9266d`, with DA3-Small, VGGT and π³.

## What changed, and why the last three tickets were the wrong shape

Tickets 9, 10 and 12 all asked the same question: *what settings make our number
equal the paper's?* That is fitting, not verifying. `R°` is an **absolute** angle,
so its value is set by which pairs you evaluate — and the paper never says. Some
stride, some resolution, some FOV was always going to land on 7.21. Agreement
proved nothing; disagreement diagnosed nothing.

**The harness is now verified directly, with no paper number involved.**
`harness_verify.py` runs SIFT on two real frames, unprojects the matches through
*our* camera model, recovers the pose with MAGSAC++, and compares to *our* ground
truth — a loop containing every component that was in doubt. It recovers the
ground truth to **0.14–0.73°** across GT rotations from 1° to 23°.

The evaluation is correct. Your measurements from #10 were all sound; they were
answering a question that could not be answered.

## The reference this gives us

Same pairs, same ground truth, on `3f15a9266d`:

| | median error | rotation gain |
|---|---|---|
| SIFT + MAGSAC | 0.31° | **0.97** |
| DA3-Small vanilla | 2.63° | 0.72 |
| DA3-Small Center-PH | 1.16° | 0.82 |

The fisheye damage is real — a frozen DA3 is 8× worse than a 1990s algorithm on
identical pairs. Center-PH removes about half. **Classical geometry is the bar,
and closing that gap is what the adapter claims to do.** That is a reproduction
target that does not depend on the paper's pair selection.

## Read gain, not `R°`

A model recovering a fraction `α` of every rotation scores exactly `(1−α)·I`. So
`R°` slides with pair separation while `α` does not. **Gain is comparable across
strides, scenes and datasets; bare `R°` is comparable across none of them.** Quote
`R°` only with the pair separation printed beside it — both scripts now do.

## Step 1 — establish the reference on the full sequence (cheap, no training)

```bash
python -m raytun3r.experiments.harness_verify --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d --backbone da3 --weights pretrained --out runs/verify/3f15-da3.json
```

Repeat with `--backbone vggt` and `--backbone pi3`. Paste the three summary blocks.

**Stop here and report if the verdict is `HARNESS SUSPECT`** — that would mean
classical geometry cannot recover our ground truth at scale, which contradicts the
staged-sample result and would have to be understood before anything else runs.

## Step 2 — the adapter, measured against that reference

Fit RayTun3R per the paper's Sec. 4.3 (30 three-frame windows, 2 px static filter
on the *adaptation set only*, evaluate on the full sequence) and report, for each
backbone, gain and median `R°` for **vanilla / Center-PH / RayTun3R** on the same
pairs as step 1.

Use `--stride 10` unless step 1 suggests otherwise. The value is not critical any
more — gain is stride-invariant, which is the point — but it must be **identical
across the three methods and stated in the report**.

Iteration count is the one hyperparameter Sec. 4.3 never gives, and App. D quotes
2–3 h per scene, so 300 is likely far too few. Run `--iters 300` and `--iters 3000`
for DA3-Small; if they differ materially, say so and stop rather than sweeping
further.

## How to read the result

* **Gain rises toward ~0.97** → the paper's mechanism reproduces. That is the
  headline, and it does not require matching 0.93.
* **Gain rises but stalls below Center-PH's** → RayTun3R underperforms a pinhole
  crop on our harness. Given Center-PH already reaches 0.99 on VGGT/π³, this is a
  real possibility and a legitimate finding, not a failure to report.
* **Gain does not move** → the adapter is not learning; then `--iters` and the
  loss terms are the next suspects, not the protocol.

## What NOT to do

Do not tune anything to bring a number closer to 7.21, 2.45 or 0.93. Those are a
sanity check on *mechanism* — does fisheye cost rotation gain, does rectification
restore it, does the adapter restore it — not targets. Three tickets went into
chasing them and produced one withdrawn conclusion.

## Recording

`results/adapter-3f15a9266d/` with the JSONs, `adapter.pt`, and a `meta.json`
carrying `git rev-parse HEAD`, torch version, checkpoint ids and the stride used.

## Done when

- [ ] step 1's three summary blocks pasted, with the `HARNESS OK` verdicts
- [ ] a gain table: vanilla / Center-PH / RayTun3R × three backbones
- [ ] one line on whether the adapter closes the gap to classical geometry
- [ ] pushed to `results`; hand back to `cpu`

## Needs CPU-Claude afterwards?

yes — folding this into README.md (#1), which has been waiting for a number
worth publishing since the start.
