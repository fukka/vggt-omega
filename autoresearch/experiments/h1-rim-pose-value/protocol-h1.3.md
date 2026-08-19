# H1.3 — Do the span (H1.1) and rim-dependence (H1.2) findings hold on real Aria?

Locked before running. Closes the lens-generalization caveat: runs 001–005 were
one ScanNet++ scene with a ~170° DSLR; the target domain is Aria 110° KB4.

**Data.** The 28 local ADT seq131 fisheye JPGs (1408², stored in native sensor
orientation, ~3.3 s apart), GT device poses from `groundtruth/aria_trajectory.csv`
(nearest-timestamp match via the filename ns stamp), camera = Aria 214-1 KB4
constants from `cam3r.cameras` lifted into a `raytun3r.KannalaBrandt` at working
resolution, `rotated=False` (frames used as stored), θ_max = 54.83°.

**The extrinsics problem, and the bootstrap.** No device→RGB-camera calibration
is available locally (no `online_calibration.jsonl`, no projectaria_tools), and
GT relative rotations in the *device* frame differ from camera-frame rotations by
a fixed conjugation `R_cam = C R_dev C^T` (the docstring in `cam3r/adt.py`
records ~38° for this term on the GPU box — fatal if ignored). Plan:

1. Estimate classical camera-frame rotations `R̂_i` (SIFT+MAGSAC++, full FOV) on
   all usable pairs.
2. Solve rotation-only hand-eye for fixed `C`: rotation angles are conjugation-
   invariant (checkable per pair), and axes obey `axis(R̂) = C·axis(R_dev)` —
   Wahba/Kabsch on angle-weighted axis pairs.
3. **Verification gate (no paper number involved):** median rotation error of
   classical vs `C R_dev C^T` must drop below ~1.5° (the harness sanity line);
   per-pair angle agreement |angle(R̂)−angle(R_dev)| must be small independently
   of C. If the gate fails, STOP and file the GPU ticket for the calibration
   JSON instead of proceeding.
   Consistency note (not a fit target): angle(C) should be in the vicinity of
   the ~38° the GPU box measured, composed with any convention rotation.
4. C is estimated from classical poses, so classical's error vs the calibrated
   GT is partially circular — report it as a residual, not a result. The MODEL
   arm (independent of C's construction) is where C is load-bearing.

**Predictions (H1.1/H1.2 transferred to the Aria cone).**
- Span: cumulative disks θ≤{25,35,45,54.8}°, count-matched, real arm: rotation
  error decreases with T; the outermost Aria band (45→54.8°) still contributes.
  Synthetic arm expected near-flat (ideal-noise conditioning is weak; run 003).
- Masking (DA3-Small): with the Aria cone as "full", masking the 35–54.8° rim
  annulus hurts much more than masking θ≤35° center at whatever area each has
  (report fractions; add the run_005-style area-matched random patch control).

**Refutation.** If on Aria the span curve is flat or the rim annulus is not
disproportionately load-bearing for the model, the day-1 narrative is a property
of ultra-wide DSLR fisheye, not of the target domain — N3's method implications
would need to be re-scoped to >120° lenses.

**Not claimed.** Translation direction (lever arm unknown locally; t-dir shown
flagged-approximate only). Depth (separate experiment). One scene, 28 frames —
statistics will be thin; report n everywhere.
