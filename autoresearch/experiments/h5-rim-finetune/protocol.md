# H5 — Rim-targeted, pose-preserving finetuning (method track 1)

Locked before any training run. Human-directed method phase; every component
below is dictated by a measured finding, cited inline.

## Hypothesis

LoRA-scale finetuning of a depth backbone with the three losses below closes
≥ half of the distance-controlled rim penalty on HELD-OUT scenes, IMPROVES
relative pose (RRA/RTA@15) over the frozen model, and beats both the H2.2
readout head and RayTun3R's per-scene adaptation under the same budget.

## Method

Student = backbone with LoRA (rank 8) on the dense/DPT head and the last 4
transformer blocks' MLPs (attention untouched — the pose path reads attention
geometry; measured rim-feature dependence, runs 004–007). Teacher = the same
backbone frozen.

Total loss `L = L_depth + λ_f·L_rimfeat + λ_m·L_mv` with:

1. **Compression-weighted depth loss** `L_depth`: L1 on log-range against GT,
   pixel-weighted by `w(θ, d_gt) = 1 + α·|b(θ, d_gt)|` where `b` is the
   measured bias map (run_009, bilinearly interpolated over the 8×5 grid),
   α=2. Rationale: spend gradient where the compression is (runs 008b/009),
   keep nonzero weight everywhere so the center is not abandoned.
2. **Rim-feature preservation** `L_rimfeat`: L2 between student and teacher
   final-block patch tokens on rim patches (θ > 35°). Rationale: the pose
   path demonstrably lives on rim features (center-masked ≈ vanilla,
   rim-masked catastrophic; runs 004–007). This is the pose-safety mechanism
   once we no longer have readout-only safety by construction.
3. **Multi-frame rim consistency** `L_mv`: for training frame pairs with GT
   relative pose, warp predicted depth i→j through the KB4 camera and
   penalize log-range disagreement in the overlap, weighted by θ of the
   source pixel (rim-weighted). Rationale: parallax evidence exists and is
   currently spent on the center (ticket 024B: "context buys the centre");
   this term routes it to the rim at train time. Inference stays single-image
   (or video via H6 later) — no extra inputs at test.

λ_f, λ_m: start 1.0/0.5; a 4-point sweep is allowed and will be reported in
full (no silent tuning; verify-don't-fit applies to OUR OWN targets too).

## Training / evaluation

- Backbone 1: DA3-Small (CPU-smokeable, GPU-trainable); backbone 2 (after
  B1 works): VGGT-Ω (largest measured headroom, 1.81×).
- Data: the six-sequence ADT split — train on 4 clean sequences, hold out 1
  clean + decoration entirely (scene-level holdout; harder than H2.2's
  frame-level splits).
- Eval (the three-axis protocol of record + pose): joint (θ×depth) AbsRel
  tables on held-out scenes; RRA/RTA@15 on held-out pairs with the factory
  calibration; comparisons: frozen backbone, H2.2 head, H2.1 table,
  RayTun3R adaptation (in-repo reproduction), and the LoRA-without-our-losses
  control (plain L1, no weighting/preservation/mv — the ablation that shows
  the design matters, not just "finetuning on ADT").
- Success gate: controlled rim penalty (pen_ctl, ticket-024A definition)
  toward 1.0 by ≥ half on held-out scenes; RRA/RTA ≥ frozen (improvement is
  the goal, non-regression is the floor); center column within noise.

## Refutation / failure readings

- If plain-L1 LoRA matches ours: the design story dies; report honestly.
- If pose regresses despite L_rimfeat: raise λ_f sweep; if still regressing,
  the attention-untouched choice was insufficient and the paper's
  pose-preservation claim needs restating.
- If rim depth improves but only within-genre (decoration fails like H2.3):
  genre generalization becomes the limitation, matching the head's boundary.

## Division of labor

CPU (this machine): loss module + LoRA injection + tests + 2-step smoke on
real frames; GPU ticket: training runs + eval matrix. BENCH protocols are a
separate workstream.
