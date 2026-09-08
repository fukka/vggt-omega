# H36 — VOID on its own question. But it banked a real finding and an honest gap.

Three runs, three B3 failures. The external-domain question is **not answered**.

| run | convention | cone | whole-image AbsRel | rim/centre |
|---|---|---|---|---|
| 1 | `z` | full ~190° | **2.640** | 1.23 |
| 2 | `range` | full ~190° | **5.993** | 4.28 |
| 3 | `range` | capped 80° | **3.150** | 7.17 |

B3's bar was AbsRel < 0.6. **Every run missed it by 4–10×**, so none of the
rim/centre ratios means anything.

## The number I am not going to quote

Runs 2 and 3 give rim/centre of **4.28** and **7.17**, both far above Aria's
2.2. Written up carelessly that reads *"the rim penalty is two to three times
worse in automotive fisheye than indoors"* — a striking, quotable, entirely
unsupported claim, because the denominator it is built on is a measurement B3
says is broken.

**This is the third time this session** that a broken measurement produced an
attractive-looking number: H34's mirrored asymmetry "flipped" 7/8 from a 4×-wrong
baseline, H36 run 2 produced 1.5×10⁶-metre predictions, and now this. In all
three the sanity bar is the only thing standing between the artefact and the
report.

## What it did establish, and this part is solid

**The depth-convention machinery assumes a sub-90° cone.** `range` divides
planar z by cos θ, and cos crosses zero at 90°; on a ~190° lens the prediction
reaches **1.5×10⁶**. `z` instead compares planar z against range ground truth.
**Neither is correct past 90°**, and Aria's 54.83° cone meant it never surfaced
in twenty experiments. Anyone reusing this evaluation code on a wide automotive
fisheye hits it in the first run.

Diagnostic that settled it: ground-truth median 1.2–1.9 m against prediction
median 2.2–2.4 (sane), prediction **maximum** 1.1–2.1×10⁶ (not), and
`corr(1/pred, gt)` negative so it is not inverse depth.

## Why the truncated run did not rescue it

Capping at 80° removes the cos-divide explosion — AbsRel drops 5.99 → 3.15 — but
the **centre** zone, where distortion is mild and the conversion is
unambiguous, still reads **0.875**. An 87% error in the easy part of the frame
is not a rim problem. Either DA3 is genuinely far out of domain on synthetic
automotive fisheye, or a residual plumbing difference remains (image
normalisation, for instance). **Three runs were not enough to separate those and
I am stopping rather than spending a fourth.**

## The gap this leaves, stated plainly

**Nothing in this line has been validated outside Aria.** H25–H35 re-measured
extensively, but all within one camera family, one apartment and one small
office. H36 was the attempt to change that and it did not produce a usable
measurement.

Both reports should say so, because "verified across thirteen recordings and
four backbones" reads like external validation and is not.
