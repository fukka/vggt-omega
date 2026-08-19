# BENCH — the public-dataset baseline matrix (locked skeleton)

Human-directed: no CVPR submission without concrete evaluations on public
datasets with published baselines, on BOTH pose and depth. This file locks
the matrix before any number is produced; cells fill in as runs land.

## Datasets (public)

| dataset | lens | role |
|---|---|---|
| ADT (6-seq split + 2 held-out) | Aria KB4 110° | primary; train/holdout as in H5 |
| ScanNet++ DSLR | fisheye ~170° | cross-lens generalization |
| KITTI-360 fisheye | 185° | outdoor, if compute allows |
| TUM-VI | 195° | pose-centric, if compute allows |
| WideDepth | fisheye, mm GT | adopt if released publicly (check first) |

## Methods

| method | source | tasks |
|---|---|---|
| DA3 (frozen) | released weights | depth+pose |
| VGGT-Ω (frozen) | released weights | depth+pose |
| UniK3D | released weights (baselines/ infra) | depth |
| DAC | released weights (third_party/) | depth |
| RayTun3R | in-repo reproduction (raytun3r/) | depth+pose |
| Fisheye3R | in-repo reproduction (fisheye3r/) — needs training | depth+pose |
| CAM3R | in-repo reproduction (cam3r/); paper's own ADT numbers (RRA 99.0 / RTA 95.0) as reference | pose(+depth) |
| H2.2 head | ours (baseline rung) | depth |
| H5 finetune | ours | depth+pose |
| H6 video module | ours | depth+pose |

## Metrics and protocol

- Depth: AbsRel, δ1 (whole-image, per repo protocol of record) PLUS the joint
  (θ×GT-depth) table and pen_ctl — our radial breakdown is itself a
  contribution of the benchmark section.
- Pose: RRA@15, RTA@15, median rotation error, rotation gain (span-invariant)
  on fixed published pair lists (pair spacing declared per dataset; the
  RRA-saturation trap from tick 19 recorded).
- Every number's provenance lands in paper/numbers.md as before; comparisons
  against published numbers quote the original papers' protocols and flag
  any mismatch rather than tuning toward them (verify-don't-fit).

## Order of execution

1. ADT matrix (frozen models + ours) — mostly existing machinery.
2. ScanNet++ (loaders exist in raytun3r/) — depth via rendered GT is blocked
   upstream (mesh renders absent from download; known issue), so ScanNet++
   contributes pose + the classical harness; depth there only if renders
   become available.
3. KITTI-360 / TUM-VI — stretch; decide after 1–2.

## Addendum 2026-08-19: Center-PH baseline (locked before any run)

Why: the comparison-recipe study (paper/comparison-protocol.md) showed
Center-PH is RayTun3R's strongest *depth* baseline (wins ScanNet++ AbsRel
2.5x over RayTun3R by discarding the rim) and the structural foil for our
thesis — it cannot have rim depth at all. Without it, our near-rim gains
have no "just crop the problem away" control.

Definition (ADT/Aria, matches our eval grid):
- Rectify the KB4 image to a pinhole: f = 252 px, principal point (252, 252),
  output 504x504 (axis coverage theta<=45.0 deg, diagonal <=54.7 deg ~ the
  Aria cone edge). Bilinear sampling from the fisheye frame via the exact
  KB4 forward projection (shared raytun3r/cameras.py, post-bisection-fix).
- Run the frozen backbone on the rectified image with the Pinhole camera
  installed, depth_convention="range" (same install path as all other rows).
- Evaluate ON THE FISHEYE PIXEL GRID: each fisheye pixel's ray is projected
  into the pinhole; covered pixels sample the prediction bilinearly,
  uncovered pixels (rim band the crop discards) are excluded AND counted.
  One scale_shift affine per frame fitted on covered/valid pixels only.
  Same joint (theta x GT-depth) table, same zones, plus per-zone coverage.

Predictions (locked):
- P1: center zone AbsRel <= vanilla-fisheye center (rectified input is
  closer to the backbone's pretraining distribution).
- P2: near_rim zone coverage collapses (38-45 deg sliver only, plus
  diagonal wedges to 54.7 deg); covered-part AbsRel may look good — the
  honest report is (coverage x error), never error alone.
- P3: solid-angle coverage = 1.840/2.651 sr = 69.4% of the cone; the
  discarded 30.6% is exactly where H5/H6 claim their gains.

Local anchor first (this Mac): frozen DA3-S, seq131 odd frames (the
run_010/011 eval split) so the row sits directly beside the existing anchor
table. Held-out-scene rows go to the box later with the same script.
