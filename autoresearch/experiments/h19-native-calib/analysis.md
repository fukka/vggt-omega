# H19 — result: B2 refuted, B3 confirmed. Motion beats device match.

Two configurations (`native_unmatched.json`, `native_matched.json` with
`--fit-pred-range 0.5,4.5`). They agree on every direction that matters.

## The table that decides it

Near-rim gain vs the frozen model. **The eval frames are identical across
rows** — the only thing that changes is which footage the curve was fitted on.
No ground truth enters any fit.

| fitted on | device | footage | → DinoToy | → Bowl |
|---|---|---|---|---|
| `apartment`, 240 fr | **wrong** | walking | **−26.0%** | **−8.9%** |
| `bowl`, 60 fr | right | near-static | −2.6% | — |
| `dino`, 60 fr | right | near-static | — | −3.8% |
| `lite_both`, 120 fr | right | near-static | −18.1%\* | −3.9%\* |

\* contaminated: `lite_both` contains the evaluation sequence's own frames, so
these are an upper bound, not a result. Even so they do not reach `apartment`.

## Bars

**B2 — does native beat foreign? FAILED, on both sequences, in both configs.**
This was the pre-registered falsification condition. Native cross-sequence
fitting loses to the foreign curve by roughly **10x on DinoToy** (−2.6% against
−26.0%) and by **2.3x on Bowl** (−3.8% against −8.9%).

**B3 — is near-static footage worse per frame? PASSED.** Mean pairwise |da| for
a 30-frame fit:

| fitting pool | |da| |
|---|---|
| one Apartment sequence (H18.5, walking) | **0.060** |
| BlackCeramicBowl (near-static) | 0.100 / 0.111 |
| DinoToy (near-static) | 0.127 / 0.132 |

Roughly **2x less stable** on the same number of frames. H18.5 explained its own
refutation by saying an ADT sequence is a person walking, so consecutive frames
already sweep a wide depth range, and predicted that the spread advantage should
reappear on genuinely static footage. It did. That prediction did not have to
survive, and it did.

**B1 — is the native fit radial? Not cleanly established.** Radial beats its
2-parameter `global` control on both cross-sequence cells in the unmatched
config (by 3.0 and 2.9 points, against a 2-point threshold), but in the matched
config `bowl`→DinoToy inverts and radial is 3.5 points *worse* than global.
Margins that flip with a fitting-range choice are not a result. Do not quote B1
in either direction.

## What this establishes, and what it does not

The direct comparison is clean because the evaluation is held fixed. Wrong
device + moving footage beats right device + static footage, on the same frames.
So **what the fit needs is motion in the fitting set, not a lens match.** B3
supplies the mechanism, measured rather than argued: static footage gives a
fit that is half as stable per frame.

What this cannot separate: LiteOffice is the *only* second device available, and
it is also the *only* near-static footage available. "Target-device fitting is
bad" and "this particular footage is static" are confounded as properties of the
fitting set. The reason to read it as staticness rather than device identity is
that B3 measures staticness directly and the same lens's own curve is the one
that fails.

## It corrects something already published

Both reports said the calibration costs "about 30 frames, and they can all come
from a single recording". That is now wrong as stated. It needs the qualifier
**30 frames with motion in them** — and, more usefully, footage from the right
device does not compensate for the lack of it. Both reports have been amended.

## What it opens

Fitting on moving footage from a *third* device would separate the confound. It
would also test the more interesting reading of the whole H18/H19 line: that the
curve may be substantially device-independent, and that "calibration" is closer
to a one-off fit on any well-moving footage than to a per-device procedure.
`apartment` already transfers to a different lens better than that lens's own
static footage does, which is evidence in that direction.
