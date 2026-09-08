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

## H17.5 — the boundary or the projection? (locked 2026-09-08, before running)

### The open question this closes

h16 left one thing undecomposed, and the report says so: the clean `pinhole`
arm rises +48% (all-image AbsRel) at 30 deg while `fisheye_frame` rises +82%
and `fisheye_disc` — which holds its black region FIXED at every angle — rises
+146%. The `pinhole` arm differs from the fisheye arms in *two* ways at once:
it has no hard black boundary, **and** it is a rectified projection. Nothing
run so far separates them.

The obvious fourth arm (a virtual fisheye whose cone sits strictly inside its
frame) does not actually help: it still shows the model a hard black annulus,
so it changes which confound is present rather than removing one. The geometry
is unavoidable — Aria's imaged cone does not fit inside its square frame, so
*any* roll performed in a fisheye frame either clips content or introduces an
annulus.

### The design that does separate them

Go the other way: put the hard boundary **into the clean arm**.

Mask the 89 deg pinhole view to its inscribed disc, at every angle. The
arithmetic makes this a near-exact analogue of `fisheye_disc`:

* an 89 deg square view has a half-edge ray of 44.5 deg and a corner ray of
  54.3 deg, so its **inscribed disc is exactly the theta <= 44.5 deg cap**;
* everything is already scored on the theta <= 44 deg disc, so the mask removes
  only content *outside the scored region* — which is precisely what
  `fisheye_disc` does when it drops the 1.7% of the cone outside its inscribed
  circle.

So `pinhole_masked` is `pinhole` plus one hard black boundary and nothing else.
The projection, the resampling, the scored pixels and the rolled-camera
construction are identical.

### Arms

| arm | projection | hard black boundary | 30 deg rise |
|---|---|---|---|
| `pinhole` | rectified | no | +48% (measured) |
| `pinhole_masked` | rectified | **yes** | this experiment |
| `fisheye_disc` | fisheye | yes | +146% (measured) |
| `fisheye_frame` | fisheye | yes, and it moves | +82% (measured) |

### Prediction (locked)

**If the hard boundary is what makes the fisheye curves steep**, then adding one
to the rectified arm should reproduce most of the gap: `pinhole_masked` rises
**at least +100%** at 30 deg (against `pinhole`'s +48%).

**If the fisheye projection is what makes them steep**, the rectified arm should
stay gentle even with a boundary: `pinhole_masked` **at most +70%** at 30 deg.

An outcome between +70% and +100% means both contribute and neither dominates;
that is a real possibility and will be reported as such rather than rounded to
whichever story is tidier.

### Secondary, and independently useful

The 0 deg cost of the mask. `fisheye_disc` pays +25% all-image / +38% near_rim
at 0 deg for masking 1.7% of the cone. If `pinhole_masked` pays a comparable
0 deg cost, then "a hard black border is a large insult" is a **general**
property of this backbone rather than something about fisheye framing — which
is the form the standing rule in `research-state.yaml` claims, and it has not
been tested outside the fisheye setting.

### Not a bar, but recorded

DA3-Small only, seq136, 20 frames, one seed — same scale as every other arm in
h16/h17, so the comparison is like-for-like. This does not establish the
boundary effect across backbones; H17.2 showed roll sensitivity itself is
backbone-dependent, so the boundary sensitivity may be too.

## H17.6 — is border sensitivity backbone-dependent too? (locked 2026-09-08, before running)

### Why this one has a payoff, not just an answer

H17.2 found roll sensitivity is a **pretraining-data** property: multi-view
backbones rise +11% at 30 deg where single-image ones rise +46-52%. The dose
curve explicitly says nothing about VGGT — it is DA3-Small only — so the
standing rule ("a hard border wrecks the band next to it") is, as measured, a
rule about one backbone family.

If it turns out to be a DA3 property rather than a general one, one specific
door opens. **H14's central tension was accuracy vs coverage**: the 95 deg
teacher is accurate but sees 70% of the rim band; the 110 deg teacher covers
100% of the cone but 22.5% of its frame is black and it inverted in every zone.
Section 02e now says that inversion was the black corners sitting *against*
`near_rim`. A backbone that tolerates a border adjacent to the zone of interest
could therefore run the 110 deg teacher and get **100% rim coverage with a
teacher that is still accurate** — which is exactly what H14 needed and could
not have.

### Method

Two measurements per backbone, both from code already written:

1. **Adjacent** (`roll_boundary.py`, extended with `--models`): mask the 89 deg
   rectified view to its inscribed disc, score on theta <= 44 deg, i.e. right
   against the border. DA3-Small pays +106% all-image / +141% near_rim at 0 deg.
2. **Distant** (same run, 30 deg roll included): whether the border also
   amplifies roll for that backbone, as it does for DA3-Small (+47.6% -> +84%).

View size rounded to each backbone's patch (624 for VGGT-Omega, 630 otherwise),
as in H17.2; the disc is defined on the frame so it is the same angular cap
either way.

### Prediction (locked)

**Both VGGT variants pay under +40% all-image at 0 deg for the adjacent
border**, against DA3-Small's +106% — the same direction and roughly the same
factor as the roll result.

**Falsified if** either VGGT variant pays more than DA3-Small, or if all four
land within 20 percentage points (which would make border sensitivity a
property of the depth task rather than of pretraining, unlike roll).

### The decision this feeds

If both VGGT variants come in under +40%, the next H14 experiment is
**a 110 deg VGGT teacher**: full cone coverage, no coverage/accuracy trade-off,
still label-free. That would be tested against the existing `rect`
(95 deg DA3) and `roundtrip` arms on the same two held-out sequences, with
`decoration_seq132` primary. If they come in high, the 110 deg door stays shut
and H14's next version remains "narrower teacher + rim-band mask" as recorded.

Not a bar, recorded: one sequence, 20 frames, one seed, same scale as every
other arm in h16/h17.

## H18 — the 110 deg VGGT-Omega teacher (locked 2026-09-08, before training)

### Why this exists

H17.6's decision rule fired: VGGT-Omega is the one backbone measured to tolerate
a hard border adjacent to the zone of interest (+29% where DA3-Small pays
+106%). The pre-check now confirms the consequence, and the margin is not
subtle. Same 110 deg view, same 22.5% black frame, only the teacher's backbone
changes:

| teacher | cone cov | rim-band cov | frame fill | near_rim vs raw DA3-Small |
|---|---|---|---|---|
| DA3-Small 110 deg | 100% | 100% | 0.775 | **+33.3% / +39.3%** (inverts) |
| VGGT-Omega 110 deg | 100% | 100% | 0.775 | **−65.0%** |
| VGGT-Omega 95 deg | 83.7% | 71.9% | 0.987 | −70.4% |
| DA3-Small 95 deg (H14's shipped config) | 83.7% | 70.4% | 0.987 | −14.7% |

H14's whole difficulty was that the teacher's rim advantage was thin
(−11.5…−14.7%) and covered only 70% of the band. The new teacher is **4.4x
stronger and covers 100%**. The accuracy-vs-coverage trade-off is dissolved,
not traded along.

### What changes about the claim

This is **cross-model** distillation, not self-distillation. It is still
completely **label-free** — no depth ground truth touches the teacher, the
targets, or the student — but "the model teaches itself" becomes "a stronger
frozen model teaches a smaller one, through a projection the smaller one cannot
use directly". Both are useful; they are not the same claim and the report must
not blur them.

### Arms

| arm | teacher | student | uses labels |
|---|---|---|---|
| `omega110` | VGGT-Omega, 110 deg view, 100% cone | DA3-Small + LoRA on raw fisheye | no |
| `rect` (existing) | DA3-Small, 95 deg view, 70% rim | same | no |
| `roundtrip` (existing) | DA3-Small on the fisheye, same resampling | same | no |
| `gt` (existing) | dense depth ground truth | same | **yes** |

Everything except the teacher is held at H14's shipped settings: 4 training
sequences x 60 frames, 20 epochs, seed 0, LoRA r=8 on blocks 8-11.

### Bars (locked)

1. **Primary.** `omega110` must beat `roundtrip` at near_rim on **both** held-out
   sequences. This is H14's original P1, which `rect` failed on
   decoration_seq132 (+4.7%).
2. **Magnitude.** It must recover at least **half** of the `gt` arm's near_rim
   gain on each sequence (gt is −57.9% on seq136, −27.2% on dec_seq132). `rect`
   recovered 22.8% and a negative fraction respectively. Half is a deliberately
   demanding bar; the teacher is 4.4x stronger, so a weak result would mean the
   student cannot absorb it, which is itself worth knowing.
3. **Centre not sacrificed.** near_center must not degrade by more than 10%.
   `rect` degraded it +43.3% on seq136, inherited from its teacher's own
   near_center offset. VGGT-Omega's 110 deg teacher is −46.0% at near_center, so
   this failure mode should be gone; if it is not, the damage is coming from the
   student or the transfer, not from the teacher.
4. **decoration_seq132 is primary.** seq136 is a sanity check only — the data
   ladder showed it behaves like a training sequence.

### Falsification and what each outcome means

* All four bars pass -> H14's idea was right and its teacher was the problem.
  The label-free line becomes live again.
* Bar 1 passes, bar 2 fails -> the student is the bottleneck, not the teacher.
  Next lever is student capacity or the loss, not the teacher.
* Bar 1 fails on dec_seq132 with a −65% teacher -> the transfer itself does not
  survive a change of room, which would be a much stronger negative result than
  H14's original one and would close the label-free line properly.

### Boundary conditions, recorded now

One seed. The pre-check is 60 frames on two sequences. VGGT-Omega runs
single-frame here, so its multi-frame machinery is idle and the comparison is
representation-vs-representation. Teacher frame is 624 px (patch 16) against
DA3's 630 (patch 14) — the same 110 deg field, 1% fewer pixels.
