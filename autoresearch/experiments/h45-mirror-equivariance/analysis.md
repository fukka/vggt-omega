# H45 — these models are a long way from mirror-equivariant, and it splits by family

**Status: B1, B2 and B3 all PASS.** Thirteen recordings, 20 frames, four
backbones, no roll — only the reflection.

`D(flip(I))` should equal `flip(D(I))` for a model with no preferred handedness.
Measured as the extra whole-image AbsRel from reflecting the model's input and
reflecting the answer back, against the same frames untouched:

| backbone | mirror cost | sd | range |
|---|---|---|---|
| `vggt_omega` | **+58.0%** | 13.2 | +27.1 … +80.2 |
| `vggt` | **+67.0%** | 16.8 | +39.8 … +99.4 |
| `da3:small` | **+254.2%** | 39.8 | +182.7 … +304.5 |
| `da3:large` | **+350.1%** | 74.7 | +222.6 … +464.3 |

**B3 first, because nothing else counts without it.** The flip-twice arm — flip
and unflip before predicting, which is the identity — differs from the untouched
arm by **0.000000%** on every recording and every backbone. The plumbing is
exact; these are measurements, not artefacts. That bar is what turned H44's void
into this result.

**B1: the two families do not overlap.** Every DA3 value ≥ +182.7%, every VGGT
value ≤ +99.4%. A 4–6× gap on the means, same direction as roll robustness and
as the border ordering. **B2**: DA3-Small's +254% clears its +100% bar four-fold.

## What is worth saying about it

**A mirrored room is still a room**, and every one of these models is trained on
scenes that would look perfectly ordinary reflected. Yet reflecting the input
multiplies a single-image model's depth error by **3.5×**. Whatever handedness
cue they rely on, they rely on it heavily.

**DA3-Large is the worst again.** It is the most accurate DA3 on a clean frame
and the most fragile of the four to a border (H37: +269%, worst 13/13) and now
to a reflection (+350%, worst of four). Roll is the exception — H42 found it
slightly *flatter* than DA3-Small there. So: **most accurate, most fragile on
two of the three perturbation axes measured.**

**The practical consequence is immediate.** Horizontal-flip augmentation is a
reflex in vision pipelines, and mirroring a preview or a front-facing view is
common. On these depth models, either one is not a free symmetry: it costs
**+58% at best and +350% at worst**. If a pipeline mirrors anything upstream of
the depth model, un-mirror it first.

## What this does not say

* **Nothing about the roll asymmetry.** H44 set out to use the mirror as a
  discriminator for it and was void precisely because the mirror moves the error
  regime by 2–5×. This measures the mirror; it does not explain the asymmetry.
* **No mechanism.** Chirality cues in natural photographs are the obvious guess
  — text, common object orientations, room layouts — and nothing here tests it.
* **The transform is a reflection of the model's input, not specifically a
  left–right mirror in the model's own frame.** `forward_z` applies a fixed
  quarter turn, so flipping the rendered view lands as a reflection about a
  different axis once it reaches the backbone. That is still a reflection and
  the equivariance question is unchanged, but the axis is not the one a reader
  would assume.

## Labelling

**CONFIRMATORY, direction already seen.** Four recordings of pilot data came out
of the void H44 run before it was stopped and already showed the family split.
The bars were written down before this run; it was not a blind test and is not
presented as one.

## Limits

* One rectified 89° view of one fisheye, indoors, per-frame scale-shift
  alignment. The magnitude is large enough that an independent check on a
  different dataset would be worth having.
* Four backbones, one lens, thirteen recordings from one apartment.
