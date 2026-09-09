# H48 — is it the content or the pipeline?

## The question H46/H47 left in exactly one sentence

The mirror effect is **large on Aria and not the black corners** (H47: +211 to
+350%, unchanged on a border-free 60° view) and **absent on ScanNet++ under the
same construction** (H46: +13.5 to −9.3%).

Four things differ, and H46's analysis listed them without ranking: content,
capture, ground truth, scored region. **One experiment separates the first from
the other three.**

## The construction

`autoresearch/data/scannetpp_aria.py` already resamples a ScanNet++ frame **into
Aria's lens**, with the void kept explicit and never filled — it was written for
exactly this kind of question. So:

1. Take ScanNet++ frames (content: a room from a tripod-like viewpoint).
2. Resample RGB **and** planar-z depth into Aria's 504×504 camera.
3. Run **H47's pipeline unchanged** on the result: a 60° co-axial view, three
   arms (`normal`, `mirror`, `twice`), scored at θ ≤ 28°.

Everything downstream of step 2 is byte-identical to H47. The only thing that
changes is whose photons are in the frame.

**The void does not reach this experiment.** ScanNet++'s vertical half-FOV is
51.53° against Aria's 54.83°, so the uncovered crescent sits at 51.57–54.83° —
and a 60° view's corner ray is 42.4°. **Nothing in the scored view is void.**

## What each outcome means

* **The effect appears** → it is the **pipeline or the lens**, not the content.
  ScanNet++ content pushed through Aria's optics behaves like Aria.
* **The effect stays absent** → it is the **content**. The same pipeline, the
  same lens, the same scoring, and only egocentric imagery produces it.

Either way it eliminates three of the four candidates or one of them, which is
more than any other single run available.

## Bars, locked before running

* **B1 — the plumbing bar, non-optional.** `twice` equals `normal` under 0.01%
  everywhere. This bar has travelled with the method since H44 and it stays.
* **B2 — does the effect appear?** DA3-Small's mirror cost exceeds **+100%**
  (H47 measured +210.9% on Aria content, H46 measured +13.5% on ScanNet++
  content through ScanNet++'s own lens). **Falsified below**, which is the
  content answer.
* **B3 — the family ordering.** If B2 passes, both DA3 variants exceed both
  VGGT variants, as on Aria. If B2 fails, this is not evaluated.

## Addendum 2026-09-09 — make it a 2×2 with the reciprocal arm

`AriaRemap.build(src_cam, dst_cam, hw)` is generic in both cameras, so the
reverse resample costs nothing extra to write: **Aria content through
ScanNet++'s lens.** With it the design becomes a proper factorial:

| | Aria lens | ScanNet++ lens |
|---|---|---|
| **Aria content** | **H47** (+154.8 / +261.4 / +45.7 / +29.9%) | **new arm** |
| **ScanNet++ content** | **H48** (this) | **H46** (+13.5 / +3.3 / −9.3 / +1.6%) |

Two cells are already measured. The two resampled cells decide it:

* effect follows the **row** (Aria content) → **content**;
* effect follows the **column** (Aria lens) → **lens or warp**;
* effect in neither → the resampling itself destroys whatever carries it, and
  the 2×2 cannot answer the question. **That outcome is possible and is stated
  now**, because a resample is not a null operation.

**Geometry checked before writing any code** (with the *corrected* corner-ray
formula — see `../h47-mirror-border/CORRECTION.md`): the reciprocal arm renders
a 60° co-axial view whose corner ray is **39.2°**, and Aria's content only exists
out to **54.83°**, so the region with no Aria source never enters the scored
view. Same argument as the forward arm, other direction.

## What it cannot do, either way

It cannot separate **capture** (motion blur, auto-exposure, 30 Hz video) from
**lens** (the optics and the warp), because a resample carries each dataset's
sharpness through the other's geometry — the geometry changes and the sharpness
does not. So "follows the column" means *"the lens or the warp"*, not *"the
lens"*. Named here so it is not quietly dropped later.

## Cost

Evaluation only: 4 backbones × 3 arms × 20 frames × 8 scenes, one GPU.
