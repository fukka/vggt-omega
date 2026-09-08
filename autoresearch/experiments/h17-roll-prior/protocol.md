# H17 — The roll prior: is it universal, does egocentric data violate it, and is gravity worth feeding in?

Locked 2026-09-07, before H17.2 / H17.3 / H17.4 were run. H17.1 is a
measurement, not a hypothesis test, and is labelled EXPLORATORY throughout.

## Where this comes from

h16's `roll_controls.py` established, on DA3-Small with a clean protocol (the
virtual camera is rolled and re-imaged, so no black wedge is ever fed to the
model): 0 deg 0.1545 all-image AbsRel, +13% at 20 deg, +46% at 30 deg, +131% at
40 deg. The rim/centre ratio does not move until 40 deg — centre and rim
degrade together, which is what a broken gravity prior looks like and not what a
fisheye-specific problem looks like.

[Breaking the Horizontal Prior](../../literature/2608.00678-breaking-horizontal-prior.md)
(arXiv 2608.00678) independently reports the same effect on Marigold,
GenPercept, DAv2 and DistillAD, and attributes it to a long-tailed orientation
bias in the training data. It evaluates only perspective, non-egocentric
datasets; it uses no gravity or IMU signal; and its best *algorithmic* roll
estimator is off by 25.9 deg on average.

Four questions follow, and they are the four sub-hypotheses here.

## H17.1 — What is the roll distribution of a real head-mounted camera? (EXPLORATORY, done)

Read, not estimated: Aria's MPS closed-loop trajectory carries the device pose
in a gravity-aligned world frame plus the gravity vector, per frame at 30 Hz.
`code/roll_distribution.py` converts that to the roll of the RGB camera about
its optical axis in the frame the model is shown.

Result (60,105 frames, 19 Apartment + 2 LiteOffice sequences): median |roll|
3.7 deg, p90 10.2, p99 21.8, max 34.5; 1.5% of frames beyond 20 deg, 0.1%
beyond 30 deg, **none beyond 40 deg**. Roll rate median 5.4 deg/s, p99 40.7.

Bonus, and independent of any loss: of the four quarter-turn offsets only 270
deg puts the distribution on zero (median 3.7 deg vs 87-176 for the others),
which confirms `UPRIGHT_K=3` from geometry.

Boundary: ADT wearers perform scripted household activities, upright, standing
or walking. This is the mild end of egocentric. Per-sequence spread is already
visible (seq144/145: 26-30% of frames beyond 10 deg; decoration_seq132: 0.8%),
so the distribution is activity-dependent and this measurement does not
transfer to, say, a wearer lying down or working under a car.

## H17.2 — Is the roll prior a property of the pretraining data, or of the model?

**Hypothesis.** The horizontal prior comes from single-image photo collections.
Backbones pretrained on *multi-view geometry* (VGGT, VGGT-Omega), which see
cameras at many orientations by construction, should be measurably less
roll-sensitive than single-image depth models (DA3-Small, DA3-Large) — even
where their absolute accuracy is better.

**Method.** The `pinhole` arm of `h16-orientation/code/roll_controls.py`
(89 deg co-axial view, virtual camera rolled, fill 1.000, scored on the same
theta <= 44 deg disc at every angle) run on all four backbones, seq136, 20
frames, angles -40..40 in steps of 10.

**Prediction (locked).** Order the four by *relative* AbsRel rise from 0 deg to
30 deg. Prediction: `vggt` and `vggt_omega` rise strictly less than both `da3`
variants. Numerically: DA3-Small is +46%; the prediction is that both VGGT
variants come in under +30%.

**Falsified if** either VGGT variant rises by more than DA3-Small, or all four
land within 5 percentage points of each other (in which case the sensitivity is
architectural or intrinsic to the depth task, not a data artefact).

**Secondary read, not a bar.** Whether rim/ctr stays flat for all four, as it
does for DA3-Small. If it does, "the rim penalty is roll-independent under 30
deg" generalises across backbones.

## H17.3 — Is roll already legible in the frozen features?

**Why this is the decisive question for the user's proposal.** H15 established
in this project that conditioning the network on geometry it can already derive
from the image is inert (the real Jacobian field lost to a position-shuffled
fake 10/16, and to no field at all 10/16), while conditioning it on *wrong*
geometry hurts (16/16). If roll is already linearly decodable from frozen
features, then feeding measured roll in as an extra input is predicted to be
inert for the same reason, and the fix has to change the *function* (augmentation,
or de-rolling the input) rather than add an input. If roll is NOT decodable,
conditioning has real headroom — and note that 2608.00678's algorithmic roll
estimator being off by 25.9 deg is weak evidence for "not decodable".

**Method.** Frames are rolled by a known angle drawn uniformly from
[-45, 45] deg, **by rolling the virtual camera and re-imaging** (no black
wedge — otherwise the probe reads the wedge orientation, not the scene). Frozen
DA3-Small tokens from the last encoder block, mean-pooled over the grid. Ridge
regression from features to (sin roll, cos roll), fit on 4 training sequences,
tested on seq136 and decoration_seq132. Report mean absolute angular error.

**Control arm (this is what makes it interpretable).** The same probe on
frames rolled the naive way — rotating the *picture*, black wedges and all. The
gap between the two arms is exactly how much of roll-legibility is the wedge
rather than the scene.

**Prediction (locked).** Camera-rolled arm: mean absolute error < 10 deg, i.e.
roll is legible from the scene alone, therefore roll conditioning will be inert
and augmentation is the right fix. Picture-rolled control: error at least 2x
lower than the camera-rolled arm (the wedge is a giveaway).

**Falsified if** the camera-rolled probe error exceeds 20 deg, which would mean
the frozen features genuinely do not encode gravity and an IMU-derived roll
input carries information the model lacks. That outcome promotes H17.4b from
"predicted inert" to the main event.

## H17.4 — Given a measured roll, what is the best way to spend it?

Run only after H17.2/H17.3. Four arms, all evaluated on frames whose roll is
drawn uniformly from [-45, 45] deg via camera-rolled re-imaging, on both
held-out sequences, primary sequence `decoration_seq132`:

| arm | what it does | needs IMU at test time |
|---|---|---|
| `raw` | nothing | no |
| `derotate` | use the measured roll to re-image the frame upright, predict, map back | yes |
| `condition` | roll fed to the network (FiLM, as in H15) | yes |
| `augment` | LoRA trained with random camera-rolled inputs, no test-time signal | no |

**Prediction (locked), conditional on H17.3.** If the probe reads roll to
< 10 deg: `derotate` >= `augment` > `condition` ~ `raw`. If the probe fails
(> 20 deg): `condition` becomes competitive with `augment`.

**Bars.** (i) `augment` must cut the 30 deg near_rim rise (currently +39%) by at
least half, to <= +20%; (ii) `augment` must not lose more than 2% at 0 deg
against the un-augmented control — the gravity prior is *useful* at 0 deg and
augmentation may erase it; (iii) both held-out sequences.

**What the answer is worth even if every arm fails.** H17.1 says ADT's p99 roll
is 21.8 deg and the model's flat zone is +-20 deg. Those two numbers were
measured independently and they coincide. If they generally coincide, the
horizontal prior is not a bug on this data — it is a prior matched to its
deployment distribution, and the interesting claim becomes about which
egocentric activities fall outside it.
