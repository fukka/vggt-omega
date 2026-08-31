# H14 — rect-teacher → fisheye-student distillation

**Status:** code written, CPU tests written. Nothing trained. Pre-registration
below is written *before* the first run, as in H12.

## The one measurement this rests on

Ticket 024 Part A (`results` digest `601fcb22767e`, 6 ADT sequences, 300
frames) measured the controlled rim/centre AbsRel ratio on the **same pixels**,
after the GT-depth control:

| input | VGGT-Ω | DA3-Large | DA3-Small | VGGT-1B |
|---|---|---|---|---|
| raw fisheye | 1.81 | 1.66 | 1.25 | 1.33 |
| rectified | ≈1.0 (−0.5…0.2 of the log-penalty survives) | | | |

Same weights, same scene, same pixels. **The information the model needs at the
rim is already there; what it cannot do is read it out of a fisheye image.**

Four rim-targeted interventions have since lost to their own controls — H5's
rim-weighted losses to plain LoRA, H6's rim-KV to all-token, the centre/rim MoE
to H7+F2+F4, and H12's lens-Jacobian FiLM to a shuffled field. Every
intervention that *has* helped is a global lens operation: plain LoRA,
`rect_derect` on slambench, the whole-image feature head.

So this experiment designs no new mechanism. It takes the answer the frozen
backbone **already gives in pinhole space** and distils it back into the
fisheye domain — and it does so **without any depth labels**, which is also the
structural answer to the project's standing external-validity problem (one
apartment, one device: `findings.md`, F8).

## Method

```
        ┌── warp into a co-axial 110° pinhole ──┐
frame ──┤                                        ├─ frozen backbone ─ depth ─┐
        └── (teacher path, no gradients) ────────┘                           │
                                                        transport back ──────┤
                                                        to the fisheye grid  │
frame ── frozen backbone + LoRA (student) ── depth ──── log-L1 ──────────────┘
```

* **Teacher**: one forward pass of the *same frozen backbone* on a co-axial
  virtual pinhole that images the **whole** cone (110° ≥ 2·54.83°), rendered at
  700 px so the centre is not downsampled (`rect_teacher.virtual_pinhole`
  documents the 0.72× trap and `test_size_700_restores_centre_sampling_parity`
  pins the arithmetic).
* **Transport**: pure resampling, **no depth conversion**. For a co-axial lens
  re-parameterisation both planar z and euclidean range are functions of the
  ray alone, so both are invariant. That licence to convert nothing is exactly
  where `raytun3r_row.py` went wrong in the other direction (#38 v1, four rows
  quarantined), so it is asserted numerically, not argued.
* **Student**: plain LoRA on the same four MLP blocks as H5/H12, same rank.
  Deliberately nothing new — the novelty under test is the *supervision*, not
  the parameterisation, and reusing the standing baseline's parameter set is
  what makes the comparison against it legitimate.

## Arms

| arm | target | labels? |
|---|---|---|
| `rect` | teacher run in the virtual pinhole, transported back | **no** |
| `roundtrip` | same teacher run on the **raw fisheye**, sent through the same pinhole and back | **no** |
| `gt` | ground-truth depth (= the plain-LoRA row that won #35) | yes |

`roundtrip` is the arm that decides it. It carries the same resampling budget,
the same coverage mask and the same loss geometry; what it does not carry is
the change of image formation. A student that improves from it has gained from
self-distillation and we would have learned nothing about the lens prior.

One asymmetry, stated rather than hidden: `rect` resamples the image once and
the depth once; `roundtrip` resamples the depth twice and the image never. The
blur budgets are comparable, not identical.

## The pre-check that can kill this before any training

`cache_teacher.py --score-teacher` scores the teacher against GT on the zones
of record, on the same pixels as the raw model.

> **If the rect teacher's `near_rim` AbsRel is not better than the raw model's,
> stop.** 024A was measured with a different harness, at ~85°, over the
> fovbench model set. If it does not reproduce for DA3-Small at this
> configuration, the premise does not transfer here and no student trained on
> this teacher can help. Record it and spend nothing on training.

The harness note that makes this non-trivial: fovbench rectifies at focal
`0.55·max(H,W)` — an **84.6°** pinhole that reaches only 42.3° on axis
(52.1° in the corners). It does **not** image the whole rim.
`test_the_fovbench_85_deg_rectification_does_not_cover_the_cone` exists so that
number can never be quietly reused as if it did.

## The standing warning about distillation, and why this design is its answer

H13's registration carries a Mac warning drawn from `slamfov` #23: a student
that matches a fisheye-space teacher everywhere **inherits that teacher's rim
behaviour**, and VGGT-Ω was measured as the steepest rim error field of the
five models benchmarked (1.83/1.96 standardised, against dav2_large's
1.41/1.57). The conclusion attached to it was that the rim must be supervised
by GT or geometry, not by a teacher.

That warning is about a teacher run **in fisheye space**, and it is exactly why
this experiment's teacher is not one. 024A's whole content is that the same
weights do not have the rim deficit when they are run on rectified input, so
the `rect` teacher is not a copy of the model's fisheye rim behaviour — it is
the model's *pinhole* behaviour, transported.

The warning does apply, in full force, to the **`roundtrip` control**: that arm
is a fisheye-space teacher and should therefore lock in the rim error rather
than reduce it. So the standing warning becomes a prediction about the control,
which is the strongest position a control can be in.

The warning's second half is also respected: teacher confidence is **not**
cached and **not** used. The DAv2 Phase-A gate already measured conf-weighted
training as worse than ungated (a1 0.0474 vs a0 0.0469), because conf is high
on easy central texture and low exactly on the band this project exists to fix.

## Pre-registered predictions

**P1 (the decider).** `rect` beats `roundtrip` on `near_rim` on **both**
held-out sequences (seq136, decoration_seq132). If it does not, H14 dies as the
**fifth** controlled negative and the write-up says so.

**P2 (what would make it a method).** `rect` recovers **≥ 1/3** of the `gt`
arm's near-rim gain over the frozen model. Below that it is a real effect and a
weak method; above it, a label-free adaptation worth a ScanNet++ cross-lens
row.

**P3 (the constraint every method here must satisfy).** `rect` does not damage
`near_center` or `center` by more than the `gt` arm does. Centre collateral is
what killed H2.1, and zone aggregates hide it — the full (θ × depth) joint
table is read, not the pooled zones.

**P4 (pose, the mandatory third axis).** `eval_lora.py` reports rotation on the
same run. `rect` must not move pose by more than the `gt` arm does. Adapters
that perturb rim features break the pose path (runs 004–007).

## What a win would NOT be

* **A global rescale.** The eval of record aligns scale *and* shift per frame
  before binning, so an affine gain is invisible by construction. The target is
  additionally scale-aligned to the frozen model's own per-frame output
  (`--scale-align`, on by default), so no LoRA capacity is spent on that
  constant. For `roundtrip` that alignment is a no-op by construction, and the
  cache manifest's `log_offset_median` is the check that it is not doing the
  work.
* **A coverage artefact.** Teacher and raw model are scored on identical
  pixels. Center-PH looked reasonable on ADT until its 49.6% near-rim coverage
  was measured.
* **Error bars.** One seed per arm, as in H12. Direction plus pre-registration
  licenses a stop; a *claim* would need the paired frame bootstrap that
  `data/bootstrap_h5_h6_2026-08-22.md` established as this project's standard.

## Files

| file | what |
|---|---|
| `code/rect_teacher.py` | the transport: virtual pinhole, both warp grids, coverage. Geometry only, no backbone import. |
| `code/test_rect_teacher.py` | 10 CPU tests, no weights, no data |
| `code/cache_teacher.py` | builds one arm's cache + runs the pre-check |
| `code/train_student.py` | the three arms |
| eval | **reused verbatim**: `../h5-rim-finetune/code/eval_lora.py --seq … --lora …`, so the numbers land in the same table as #35/#36 and H12, with pose included |
