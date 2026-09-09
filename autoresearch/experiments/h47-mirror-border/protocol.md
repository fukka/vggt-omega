# H47 — was H45's mirror effect the black corners?

## What H46 found, and why it cannot be left there

H46 ran H45's mirror test on ScanNet++ with a **matched rectified 89° view** —
same construction, same arms, same plumbing bar (0.000000% everywhere), baselines
in the same range as Aria's. The effect **did not replicate**:

| backbone | H45 (Aria, 89° view) | H46 (ScanNet++, 89° view) |
|---|---|---|
| `da3:small` | **+254%** | +3.5% |
| `da3:large` | **+350%** | +0.4% |
| `vggt` | **+67%** | −13.1% |
| `vggt_omega` | **+58%** | +0.7% |

Two datasets, one construction, and a 100× difference. Something specific to the
Aria pipeline is doing the work, and there is an obvious candidate.

## The candidate

**Aria's rectified 89° view has black corners; ScanNet++'s does not.** A square
89° view has a corner ray of 62.9°, which is outside Aria's 54.83° cone and
inside ScanNet++'s 84.8° one. And Aria's principal point sits **4.5 px off the
frame centre**, so that black pattern is **not exactly symmetric** — mirroring
moves it, putting black where content was and content where black was, at the
rim, where this project has measured errors to be largest.

The numbers fit that story uncomfortably well. H45's ordering and magnitudes look
like the **border** results, not like anything else:

| backbone | H45 mirror | H37 border | H40 border at matched angle |
|---|---|---|---|
| `da3:small` | +254% | +59% | +105.7% |
| `da3:large` | **+350%** | **+269%** | — |
| `vggt` | +67% | +48% | — |
| `vggt_omega` | +58% | +6% | +6.6% |

Same ordering, DA3-Large worst in both, same family gap. **H45 may have measured
the border a third time.**

## The control

Run H45's mirror test on Aria with a **60° view instead of 89°**. Its corner ray
is 42.4°, inside the 54.83° cone, so the view is **fully populated — no black
anywhere**. Nothing else changes: same recordings, same frames, same arms, same
plumbing bar, same scoring (capped at θ ≤ 28°, inside the 60° view's inscribed
disc).

## Bars, locked before running

* **B1 — the plumbing bar.** `twice` equals `normal` under 0.01% everywhere.
  Non-optional; without it nothing here is a measurement.
* **B2 — does the effect survive without black corners?** DA3-Small's mirror
  cost at 60° stays above **+100%** (H45 measured +254% at 89°). **Falsified
  below**, and then **H45's finding is withdrawn**: it was the border, not model
  handedness, and the recommendation that came with it goes with it.
* **B3 — is the drop specific, or does everything shrink?** If B2 fails, the
  drop must be large for the two DA3 variants *and* leave the VGGT variants
  proportionally less changed — because that is what a border explanation
  predicts, given that VGGT-Omega pays +6% for a border while DA3-Large pays
  +269%. Reported either way; not a pass/fail on its own.

## What each outcome means

* **B2 passes** → H45 stands. The black corners are not the mechanism, and the
  ScanNet++ non-replication needs a different explanation.
* **B2 fails** → H45 is withdrawn, promptly and in both reports, and the
  underlying finding becomes *another* instance of the border effect this line
  has now measured four ways. That is not a small correction: H45 was published
  one tick ago with a practical recommendation attached.

**This is written before the run.** The outcome that requires retracting my own
most recent result is the one the design is pointed at.
