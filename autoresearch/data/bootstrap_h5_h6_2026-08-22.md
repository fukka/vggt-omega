# Paired bootstrap on the #35/#36 eval deliveries (Mac, 2026-08-22)

Source: `results` branch commit aef0f04, per_frame['after'].near_rim of each
eval JSON. Paired per-frame difference (method − control), 20k resamples,
seed 0, percentile 95% CI. Pure-python (this Mac has no torch/numpy env of
record — see POLICY.md 2026-08-22 correction).

| family | seq | n frames | mean diff | 95% CI | verdict |
|---|---|---|---|---|---|
| H5 full−plain | seq136 | 60 | +0.0382 | [+0.0172, +0.0592] | rim losses significantly WORSE |
| H5 full−plain | dec_seq132 | 60 | +0.0080 | [−0.0178, +0.0352] | tie |
| H6 rim−alltok | seq136 | 59 | +0.1769 | [+0.1625, +0.1926] | rim-KV significantly worse |
| H6 rim−alltok | dec_seq132 | 59 | +0.0268 | [+0.0147, +0.0390] | rim-KV significantly worse |

Both refutations (H5 refuted-by-control, H6 refuted-on-held-out) are outside
error bars on seq136; H6's also on dec_seq132. Positive diff = method worse
than control (near-rim AbsRel, lower is better).

Reproduce: extract the eight JSONs from `origin/results` and run the paired
bootstrap over frames common to both arms (frame ids match exactly; one frame
lacking a near_rim value in h6 is dropped pairwise).
