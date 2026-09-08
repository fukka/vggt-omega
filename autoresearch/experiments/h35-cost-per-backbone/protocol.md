# H35 — what does roll cost for the model you would actually ship?

## The gap

H32 computed the expected cost of real head roll for the first time: **1.46% ±
1.09**. That was **DA3-Small only** — and DA3-Small is the *most* roll-sensitive
model in the set.

H17's headline is that **multi-view-pretrained models are 4–5× less
roll-sensitive** at 30°. This report also recommends **VGGT-Omega** as the
teacher. So the number a reader would act on is not 1.46% — it is whatever the
model they would deploy costs, and nobody has computed that.

H33 ran all four backbones but on the coarse grid `-30,-20,0,20,30`. The gap
between 0 and 20 is exactly where **90% of the roll mass sits**, so integrating
H33's curves would reintroduce the interpolation flaw H32 existed to fix.

## Design

`roll_boundary.py --models da3:small,da3:large,vggt,vggt_omega` on the **dense**
grid `-30,-20,-15,-10,-5,0,5,10,15,20,30`, six recordings, `pinhole` arm,
integrated against H17's 2.5° histogram over 60,105 frames exactly as H32 did.

Six recordings rather than thirteen because VGGT-Omega at patch 16 is the
expensive arm and the quantity is a mean over a smooth curve, not a tail
statistic.

## Bars, locked before running

* **B1 — does H17's sensitivity gap survive integration?** VGGT-Omega's expected
  cost is **under half** DA3-Small's. H17 measured 4–5× at 30°; the integral
  weights the flat region far more heavily, so the ratio may compress —
  **anything from "under half" upward counts, and equal costs would falsify**.
* **B2 — sanity.** DA3-Small's expected cost on this six-recording subset falls
  within H32's **1.46% ± 1.09**. If it does not, the subset or the runner
  disagrees with H32 and B1 should not be quoted.
* **B3 — is the tail finding backbone-specific?** H32 found frames beyond ±20°
  carry 26% of the cost while being 1.5% of the data. Does that hold for the
  multi-view models, whose curves are much flatter? **If their tail share is
  materially lower, "handle the rare large roll" is DA3-specific advice, not
  general advice**, and both reports should say so.

## What each outcome means

* **B1 passes** → the deployment number is smaller than the one published, and
  the report can give a per-model figure instead of one measured on its most
  sensitive model.
* **B1 fails** → H17's sensitivity gap does not survive weighting by how often
  each angle actually occurs, which would be a real correction to the roll
  line's central claim.
* **B3 fails** → the tail advice narrows to DA3-like models.

## Cost

11 angles × 4 backbones × 6 recordings × 20 frames. No training, no new data.
