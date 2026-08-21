# Do hand/body pixels disrupt pose, and should the head ignore them?

**Owner:** gpu
**Status:** **done** — `results/autoresearch-h4-pose` (meta `ticket: 29`).
**Files I may touch:** new script(s) under
`autoresearch/experiments/h4-dynamics/code/` (create), results to `results`
under `results/autoresearch-h4-pose/`.
**Blocked by:** none — the three skeleton sequences from ticket #28 are
already on lambda_63, and the masking machinery exists
(`autoresearch/experiments/h1-rim-pose-value/code/adt_pose_value.py`, mask
stage; `model_rim_use.py::masked`).
**What is waiting on it:** ticket #28's statistics place dynamic pixels at
median 0.26–0.94 m and 80%+ beyond θ=41° (single-skeleton sequences) — i.e.
inside both the worst-calibrated depth cells (run_008b/009) and the rim band
the model's pose relies on (runs 004–007). Whether they actually *hurt* is
unmeasured.

## The task

On the three #28 sequences (subsample ~40 frame pairs each, consecutive-ish):

1. **Pose arm** — DA3-Small rotation error per pair under three inputs:
   (a) vanilla; (b) GT-dynamic pixels masked to mean color (masks from the
   skeleton segmentation, as in #28); (c) area-matched random patch mask
   (seeded), same fraction as (b) per frame. GT relative pose from the
   trajectory + the #27 calibration JSON (`cam3r/data/
   adt_camera_rgb_calibration.json` — use its T_device_camera, no hand-eye
   needed). Report median rotation error and gain per condition per sequence.
2. **Depth arm** — on frames with dyn_frac > 2%: AbsRel of DA3-Small on
   STATIC pixels only, vanilla vs dynamic-masked input. Does removing hands
   from the *input* improve static depth around them? (Protocol of record:
   scale_shift per frame on static pixels, range domain.)

Prediction (recorded here so the run can confirm or refute): masking dynamic
pixels improves pose more than the area-matched random mask on high-dyn_frac
pairs (they are near, fast-moving, and sit in the pose-critical band); if
instead (b) ≈ (c), hands are just occlusion, not signal corruption, and H4
reduces to "the head should skip dynamic cells in its loss".

## Acceptance

- JSONs + a comment: per-sequence table of the three pose conditions, the
  static-depth before/after, and one sentence each on the prediction.
