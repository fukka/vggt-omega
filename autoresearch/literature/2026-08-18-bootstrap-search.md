# Bootstrap literature search — 2026-08-18

Complements `docs/research/fisheye-wide-fov-adaptation.md` (2026-07-29, verified,
1283 lines — the primary survey; not duplicated here). These are papers surfaced by
four fresh searches keyed to the novelty axes. Status: **found via web search,
abstracts only — not yet verified against primary sources** (the survey's standard).

## Axis N1 — peripheral accuracy without hurting center

- **WideDepth: Millimeter-Accurate Benchmark for Fisheye Depth Estimation**
  ([arXiv:2605.24074](https://arxiv.org/abs/2605.24074)) — candidate external
  benchmark for peripheral depth claims. TODO: check camera model, FOV, license.
- **Extending Foundational Monocular Depth Estimators to Fisheye Cameras with
  Calibration Tokens** ([arXiv:2508.04928](https://arxiv.org/abs/2508.04928)) —
  token-based adaptation for *monocular depth* specifically (Fisheye3R's cousin on
  the DAv2 side). Aligns latent embeddings of fisheye to the perspective
  distribution. TODO: do they report radial breakdown? If yes, first Pareto datapoint.
- **Fisheye Stereo Vision: Depth and Range Error**
  ([arXiv:2602.02973](https://arxiv.org/abs/2602.02973)) — analytic peripheral
  precision-loss model; useful for the "expected" radial error floor.

## Axis N2 — distortion-adaptive computation

- **DarSwin: Distortion Aware Radial Swin Transformer**
  ([arXiv:2304.09691](https://arxiv.org/abs/2304.09691), ICCV23) — radial-azimuth
  patch partitioning keyed to the lens curve; the closest prior art to "tokenize by
  distortion level". Classification-era; no depth, no frozen-backbone story.
- **Multi-level distortion-aware deformable network for omnidirectional SR**
  ([arXiv:2512.17343](https://arxiv.org/abs/2512.17343)) — deformable conv + attn
  keyed to distortion level, super-resolution domain.
- **Vision Transformer with Deformable Attention** (CVPR22) / **Quadrangle
  Attention** ([arXiv:2303.15105](https://arxiv.org/abs/2303.15105)) — generic
  deformable-attention machinery, no camera awareness.
- Note: survey already covers PanoFormer (tangent-domain tokens + token flow), SPE
  (sector patches), SphereNet (sampling-location adaptation). N2's gap: nobody keys
  *token budget / sampling density* to the KB4 Jacobian on a frozen depth FM.

## Axis N3 — periphery for cross-frame alignment

- **LF-VISLAM** ([arXiv:2209.05167](https://arxiv.org/abs/2209.05167)) — SLAM for
  >180° FOV; explicit statement that outer-FOV features converge easily due to large
  parallax. The folklore citation for H1.
- **A*SLAM dual-fisheye** ([arXiv:1911.04063](https://arxiv.org/abs/1911.04063)),
  **BundledSLAM** ([arXiv:2403.19886](https://arxiv.org/abs/2403.19886)) — wide-FOV
  robustness evidence, classical pipelines.
- Gap: no measurement of *per-band pose value* for feed-forward 3D FMs. H1 is novel
  as far as these searches show.

## Axis N4 — dynamics / hands in egocentric

- **POMATO: Pointmap Matching + Temporal Motion for Dynamic 3D Reconstruction**
  ([arXiv:2504.05692](https://arxiv.org/abs/2504.05692)) — dynamic-aware pointmap
  matching; the DUSt3R-family answer to moving content.
- **ReViV: Reconstructing the Viewer and the View in 4D from Monocular Egocentric
  Video** ([reviv4d.github.io](https://reviv4d.github.io/)) — joint hands + camera +
  depth; heavyweight, opposite of our limited-finetuning constraint, but the eval
  protocol may be reusable.
- **HOT3D** ([arXiv:2406.09598](https://arxiv.org/abs/2406.09598)) — 833 min of
  Aria/Quest3 hand-object interaction with GT poses; candidate data for H4 beyond ADT.
- **HaWoR** ([arXiv:2501.02973](https://arxiv.org/abs/2501.02973)) — world-space
  hand motion from egocentric video; hand-mask machinery worth borrowing.
- ADT itself ships GT segmentation + depth for every frame — H4's measurement needs
  no new data.

## Sources (search result pages)

- https://arxiv.org/abs/2605.24074 · https://arxiv.org/abs/2508.04928 ·
  https://arxiv.org/abs/2602.02973 · https://arxiv.org/abs/2304.09691 ·
  https://arxiv.org/abs/2512.17343 · https://arxiv.org/abs/2303.15105 ·
  https://arxiv.org/abs/2209.05167 · https://arxiv.org/abs/1911.04063 ·
  https://arxiv.org/abs/2403.19886 · https://arxiv.org/abs/2504.05692 ·
  https://reviv4d.github.io/ · https://arxiv.org/abs/2406.09598 ·
  https://arxiv.org/abs/2501.02973
