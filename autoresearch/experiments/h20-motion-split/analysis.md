# H20 — all three bars pass. The claim survives, and gets a dose-response.

`results/motion_split.json`. One GPU pass, 5 draws per arm, seed 0.

## H20a — "near-static" is now a number

Camera centre per frame from GT pose, `C = -R^T t`.

| sequence | spread from centroid | path length | bbox diagonal |
|---|---|---|---|
| Apartment seq131 | 1.524 m | 38.22 m | 6.26 m |
| Apartment seq133 | 1.941 m | 45.65 m | 7.57 m |
| Apartment seq134 | 1.967 m | 38.88 m | 7.28 m |
| Apartment seq135 | 1.609 m | 36.10 m | 6.35 m |
| **DinoToy** | **0.105 m** | **2.34 m** | 1.37 m |
| **BlackCeramicBowl** | **0.210 m** | **2.43 m** | 1.34 m |

**B1 passes with no overlap.** 9-19x in spread, 15-20x in path length. The
Apartment wearer walks ~40 m; the LiteOffice wearer moves 2.4 m in total. The
label was inherited rather than measured, and it turns out to have been right —
but it is only now on the record as a measurement.

Incidental: only 55 of DinoToy's and 45 of Bowl's 60 frames have a trajectory
sample within the 2 ms window. Does not affect H19, which needed no pose.

## H20b — the within-device control

Same device, same room, same 240-frame Apartment pool, same 30-frame count,
same teacher. The only variable is how far apart the selected frames' camera
positions are.

| arm | camera spread | stability \|da\| | seq136 | dec132 | DinoToy | Bowl |
|---|---|---|---|---|---|---|
| `high` | 2.021 m | **0.069** | −20.0 ± 1.1 | −9.4 ± 1.4 | −30.5 ± 1.0 | −9.3 ± 0.2 |
| `low`  | 0.499 m | **0.377** | −11.5 ± 9.0 | −1.4 ± 9.4 | −21.1 ± 7.6 | −8.9 ± 0.7 |

**B2 passes**: 5.5x worse stability. **B3 passes**: worse transfer on all four
sequences, and the draw-to-draw spread explodes — ±9.0 and ±9.4 against ±1.1 and
±1.4. Low-motion fitting is both worse and unreliable.

**So the mechanism published on 2026-09-08 survives a control that holds the
device fixed.** It did not need withdrawing. It is now supported by something
stronger than the cross-device comparison it was originally read off.

## What upgrades the claim: it is a dose-response, not a binary

Rim gain on DinoToy against how far the camera travelled. The first three rows
are the **same camera in the same room**:

| fitting set | spread | → DinoToy |
|---|---|---|
| Apartment, 30 most spread out | 2.021 m | −30.5% |
| Apartment, all 240 | 1.768 m | −26.0% |
| Apartment, 30 least spread out | 0.499 m | −21.1% |
| LiteOffice Bowl, its own 60 | 0.210 m | −2.6% |

Monotone in camera motion across a 10x range, three of the four points
device-matched. H19 said "motion beats a lens match"; H20 says how much motion,
and that the relationship is graded rather than a threshold.

Practical form: **the fitting frames should span roughly 2 m of walking.** That
is a specification someone can follow, which "30 frames with motion in them" was
not.

## Two things not to over-read

**The `low` arm's mean curve looks fine.** Its mean `a(θ)` over five draws is a
smooth arch (1.454 → 1.630 → 1.412). Averaging hides the instability that
|da| = 0.377 reports. Anyone eyeballing a single low-motion fit could easily
conclude it worked.

**`low`'s |da| is not directly comparable to LiteOffice's.** `low` re-draws a
random anchor each time, so its five fits sample different corners of the
apartment; LiteOffice has only one place to be, so its draws agree with each
other while agreeing on something wrong (H19's plotted curves sit below 1.0).
|da| measures draw-to-draw spread, and the two arms differ in what varies
between draws. The **transfer** numbers are the clean comparison, and they are
the ones B3 rests on.

## Limitation, as recorded before the run

The cache holds every ~10th source frame, so `low` is "30 frames from a short
stretch of the walk" (0.499 m) rather than 30 consecutive video frames. It is
still 2.4-4.8x more motion than either LiteOffice sequence. That the effect
shows up anyway at this weaker contrast makes the result conservative.
