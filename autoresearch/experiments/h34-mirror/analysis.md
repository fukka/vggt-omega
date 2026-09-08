# H34 — VOID. B3 caught a design error in my own construction.

## What happened

**B3 failed, and it was the bar that mattered.** The level-0° whole-image error
is **0.1545 un-mirrored against 0.6145 mirrored** — a **4× degradation** from
mirroring alone, before any roll is applied. On seq136 the near-rim error goes
from 0.329 to 1.537.

B3 was written as *"a large gap would mean mirroring alone damages the input, and
the comparison would be confounded."* That is exactly what it found. **The test
is void and its numbers say nothing about the asymmetry.**

## The error, named

I mirrored the source image and **did not mirror the ground truth**. The model
therefore predicted depth for a mirrored scene and was scored against the
unmirrored scene. Every number in the mirrored arm is that mismatch, not a
property of roll.

The apparent "flip" — 7 of 8 recordings changing sign, mirrored mean −5.09 — is
an artefact of that: with a level baseline already four times too large, rolling
the camera reduces the measured error, so every difference goes negative.
**B1 "passing" at 7/8 is meaningless**, and it would have looked like a clean
result if B3 had not been there.

## Why the obvious fix is not cheap

Mirroring the ground truth too is not a one-line change. The mirror is applied in
the **fisheye source** frame; the depth is scored in the **pinhole view** frame,
with the rig's warp and the roll in between. Because the principal point is
off-centre and the roll is applied inside the rig, flipping the output back is
not the inverse of flipping the input. Doing this properly means mirroring the
source and the GT together *before* the rig, which is a real change to how
`zones` obtains its ground truth.

## Decision: stop here

The asymmetry is a second-order detail of the roll line. It has now consumed
H32 (found it), H33 (could not attribute it), and H34 (test void). Meanwhile the
**cross-room exposure** — the largest open weakness in the whole line — is
blocked on a data decision that has been sitting with the user for several ticks.

**Continuing to spend runs on this while that sits is bad prioritisation.** The
asymmetry rests where H33 left it, which is a defensible resting state:

* it reproduces on **DA3-Small** (7/8 here, 11/13 in H32);
* **DA3-Large shows none** (4/8), so it is not a DA3-family property;
* all four backbones lean the same way on average, so **some part of it may be
  our own rendering, and we could not say how much**.

Both reports already say exactly that. Nothing needs changing; this experiment
adds no evidence in either direction.

## What is worth keeping

**B3 earned its place.** It was written as a routine sanity check and it is the
only reason this did not go out as "the asymmetry flips under mirroring — it
travels with the scene", which is what B1 and B2 alone would have suggested at
7/8. Any test that transforms the input needs a bar on the untransformed
baseline.
