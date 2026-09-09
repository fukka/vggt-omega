# H47 — it was not the black corners

**Status: B1 and B2 both PASS.** Seven recordings, four backbones, a 60° Aria
view whose corner ray is 42.4° — **fully inside the 54.83° cone, so no black
anywhere**. Everything else identical to H45.

| backbone | 60°, no border | sd | H45 at 89° | ratio |
|---|---|---|---|---|
| `da3:small` | **+210.9%** | 47.6 | +254% | 0.83 |
| `da3:large` | **+350.1%** | 123.3 | +350% | **1.00** |
| `vggt` | **+56.8%** | 16.0 | +67% | 0.85 |
| `vggt_omega` | **+41.9%** | 7.9 | +58% | 0.72 |

Plumbing exact everywhere (0.000000%).

**The candidate is dead.** H46's non-replication had an obvious suspect: Aria's
89° view has black corners (corner ray 62.9° against a 54.83° cone) and its
principal point sits 4.5 px off centre, so mirroring moves an asymmetric black
pattern across the rim — and H45's ordering and magnitudes looked uncomfortably
like the border results this line has measured three times.

Remove the border entirely and the effect is **unchanged**: ratios 0.83, 1.00,
0.85, 0.72. DA3-Large is identical to two decimal places. **H45 did not measure
the border.**

## Where that leaves H45

Two solid results that point in different directions:

* **On Aria the effect is real, large, and not the border** (H47).
* **On ScanNet++ it is 10–100× smaller under a matched construction** (H46).

So H45's *measurement* stands and H45's *framing* does not. It is not "these
models are far from mirror-equivariant"; it is "**on this egocentric footage
they are, and on ScanNet++ they are not**". The recommendation that came with it
— *if anything upstream mirrors, un-mirror before the depth model* — has to
become **check it on your own data**, because on one of the two datasets tried
it would have bought nothing.

## What differs between the two, unranked

Named because the next experiment has to choose one, and not ranked because
nothing here distinguishes them:

* **Content.** Aria is egocentric: hands, arms, the wearer's own body and near
  surfaces, all strongly lateralised. ScanNet++ is a room captured from a
  tripod-like viewpoint.
* **Capture.** Aria frames are 30 Hz video, motion-blurred, auto-exposed.
  ScanNet++ frames are DSLR stills.
* **Ground truth.** ADT rendered depth against ScanNet++ mesh-rendered depth.
* **Scored region.** Aria is capped at θ ≤ 44° (28° here) in the fisheye frame;
  ScanNet++ scores the whole valid region.

**The cheapest test of the first one is already on disk**: ADT ships
segmentation, so hand pixels can be excluded from the score. If the effect is
carried by the wearer's hands it should shrink; if it is the capture or the
lens it should not.

## Honest note on the size of this correction

H45 was published one tick ago, with a section in the report and a bullet in the
recommendations. H46 and H47 were run in the tick immediately after, and both
were designed with the outcome that damages H45 as the pre-registered pass
condition. The correction lands in the same tick it was found.
