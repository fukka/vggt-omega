# H2.0 — The depth baseline: how bad is the Aria rim, measured properly?

Locked before running. This is the "before" measurement the H2 adapter will be
judged against, produced inside this workspace rather than inherited.

**Question.** On real Aria (local ADT seq131, 28 frames with GT planar-z depth),
how does DA3-Small's depth error vary with incidence angle θ — and how much of
that radial curve survives the **distance control** (the repo's standing
confound: the rim of this lens looks at nearer surfaces, so error-vs-θ is
partly subject-vs-θ)?

**Predictions.**
1. Uncontrolled: AbsRel rises from center to rim (the folklore curve).
2. Controlled (θ × GT-depth joint table, reading along rows = fixed depth
   band): a genuine radial degradation remains, but smaller than the
   uncontrolled spread. (Repo prior on ego-synth: an oracle with no field
   effect reads 1.86× uncontrolled — so shrinkage is expected; the open
   question is what fraction survives.)
3. The surviving radial curve is the adapter's target; its magnitude sets the
   size of the prize for axis N1.

**Method.**
- Model: DA3-Small (CPU), same install as runs 004–007, `depth_convention="range"`.
- Data: all 28 local seq131 frames as stored (native orientation), 504²;
  GT: `depth_npy` (planar z, established by the repo's
  `check_gt_depth_domain.py`), same pixel layout as the stored JPGs (verify at
  load by shape; nearest-neighbour resize to 504²; mm→m if values look like mm).
- Scoring domain: range (DAC protocol) — GT z is converted once at the boundary
  via `1/cos θ`; the conversion factor reaches 2.15× at the rim, so domain
  mixing would masquerade as exactly the radial effect under study.
- Alignment: **one scale per frame** (`median(gt/pred)`, the repo's
  `scale_only`), frozen before any binning. Report per-frame scale stats.
- Bins: θ in 8 equal bins over [0°, 54.83°]; GT depth (range domain) edges
  (0, 1, 2, 3, 5, 10) m matching fovbench defaults. Valid: GT>0, GT range ≤10 m,
  inside the imaged cone. Cells under 500 px recorded but flagged.
- Report: (a) per-θ-bin AbsRel (uncontrolled curve) with **spread max/min**,
  not rim-over-center (U-shape artifact of frozen-affine binning); (b) the
  joint θ×depth AbsRel table; (c) per-row spread = the controlled radial read;
  (d) pixel counts everywhere.

**Refutation / surprise handling.** If the controlled rows are flat, the
"rim depth is bad" premise is mostly a depth confound on this data and the N1
prize shrinks to whatever the rows show — that would redirect H2 from "fix the
rim" to "fix the depth-dependent miscalibration", a different adapter target.

**Not claimed.** One scene, one model; no multi-frame fusion; metric scale
(scale freed per frame). The 28 frames share one apartment — depth statistics
are correlated across frames.
