# H1 — Does the fisheye rim carry pose value per correspondence?

**Hypothesis (locked before running).** On wide-FOV fisheye, correspondences in the
peripheral band recover relative rotation *better per correspondence* than central
ones, because bearing changes at high incidence angle disambiguate rotation
(large-parallax folklore from wide-FOV SLAM, LF-VISLAM et al., never measured
per-band for this setting).

**Prediction.** Splitting matched keypoints into incidence-angle (θ) quartile bins
and estimating pose from each bin alone (equal match count by construction),
median rotation error decreases with bin θ; the top-θ bin beats the bottom-θ bin.
Secondary prediction: a synthetic-bearing control (GT poses, ideal points, equal
pixel noise per bin) shows the same ordering — if real SIFT shows the reverse while
the control shows the geometric advantage, rim *feature quality*, not rim geometry,
is the bottleneck (which would redirect axis N3 toward N2).

## Method

- Data: local ScanNet++ sample `~/Desktop/ADT/scannetpp_example/3f15a9266d`
  (24 frames, full transforms.json, OPENCV_FISHEYE ~170° DSLR; the harness already
  recovers GT to 0.14–0.73° here, so every component is pre-verified). ADT/Aria
  generalization is a follow-up arm, not this run.
- Machinery: reuse `raytun3r` (ScanNetPPFisheye, camera.unproject, Matches,
  relative_pose_magsac, rotation_error_deg). New script:
  `autoresearch/experiments/h1-rim-pose-value/code/rim_pose_value.py`. No backbone —
  classical only; CPU.
- For each frame pair (strides 1,2,5,10,20; ~20 pairs/stride): SIFT match as in
  `harness_verify._sift_pose` (nfeatures 6000, ratio 0.8, MAGSAC thresh 0.5°).
  Compute θ_i = incidence angle of the *source*-frame keypoint through the
  calibrated camera. Split matches into θ quartiles **of that pair's matches**
  (equal count per bin by construction). Run MAGSAC++ per bin. Also run "all"
  (the harness's own condition) as anchor.
- Denominator discipline: a pair enters the cross-bin summary only if **every** bin
  (and "all") produced a pose. Report per-bin median rotation error, rotation gain
  (slope of predicted-on-GT magnitude), n pairs, and each bin's median θ and match
  count. Translation direction reported only for strides ≥ 20 (below that it is at
  or worse than chance per prior repo finding), and not part of the claim.
- Synthetic control: for the same pairs, sample 3D points from GT-consistent
  geometry (unproject source pixels at plausible depths — depth value is irrelevant
  for pure-rotation bearing geometry only if translation is zero, which it is not;
  so instead: use matched pairs' *triangulated* points where available, else sample
  pixels uniformly per bin, assign depths from a fixed range, project into both
  frames with GT poses, add N(0, σ=1px) noise), and run the same per-bin MAGSAC.
  This isolates bin *geometry* from feature quality at equal noise and count.
- Randomness: fixed seed 0 for any subsampling.

## What would refute H1

Top-θ bin not better than bottom-θ bin in BOTH real and synthetic arms (no
geometric advantage → axis N3's premise is wrong for this FOV range). Real worse
but synthetic better = partial refutation, redirect to feature quality (N2).

## Not claimed / known limits

- One scene, one lens (~170° DSLR, wider than Aria's 110°); Aria arm pending.
- Source-frame θ binning (a match may change band between frames); a both-endpoint
  variant is logged as a secondary table, not the primary.
- Annulus-restricted bearings could be a weak geometry for MAGSAC independent of θ;
  the synthetic control carries that structure identically, which is why it exists.
- SIFT on raw distorted rim may localize worse — that is *part of the measured
  quantity* in the real arm (deliberately), and excluded in the control arm.
