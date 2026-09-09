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

## What it cannot do

It cannot separate **capture** (motion blur, auto-exposure, 30 Hz video) from
**lens** (Aria's KB4 optics and the warp), because resampling carries
ScanNet++'s sharp stills through Aria's geometry — the geometry changes and the
sharpness does not. So a positive result means *"the lens or the warp"*, not
*"the lens"*. That distinction is named here so it is not quietly dropped later.

## Cost

Evaluation only: 4 backbones × 3 arms × 20 frames × 8 scenes, one GPU.
