# slambench: a proper baseline evaluation on real egocentric SLAM data

**Owner:** gpu (#016's fetch is done, and it ran on cpu)
**Status:** **RUN. Steps 2 and 3 are measured** on `143686a`, digest
`61195914f090`, published to `results` @ `b10b087` as
`results/slambench-020-143686a`. Issue
[#22](https://github.com/fukka/vggt-omega/issues/22). See "What it says" below.
Both pre-flight items the issue carries are done: `verify_camera --takes 8`
passes 8/8 on aea and nymeria at 0.30 px, and the rectification was done by
**`projectaria_tools`**, not the silent cv2 fallback.
**Files I may touch:** nothing under `slambench/` except what #016 adds — runs only.
Results to `results`. Nothing under `slambench/` was touched.
**Blocked by:** nothing. **Step 1 is answered — `d` is planar z (#016,
2026-08-14) — and steps 2 and 3 are released.** #012 has also landed all three
calibration sets on lambda_63, so `rect_derect` on aea + nymeria has no
remaining technical gate.

> **Provenance, added after the run (#021).** This run's `vggt_1b` column was
> produced in **fp32**, because `raytun3r.backbones.VGGTBackbone.forward` was
> the one VGGT call site in the repo not opening the bf16 autocast the model is
> written to expect. That was fixed after publication and the run was
> **deliberately not repeated**, so no current checkout reproduces this
> artefact. The accuracy cost is bounded — ≤0.51% on AbsRel with no ordering
> flip, against findings of 8–21% — so nothing below moves. The **timings**
> are not comparable across models here: VGGT-1B was fp32 while the other
> three were bf16, and the 13.7× context scaling quoted below is
> substantially a property of the dtype, not the architecture.

## What it says

**Rectifying is nearly free here, and this ticket expected it not to be.** The
trap section below warns that a 110 deg pinhole cannot cover the fisheye cone
and will strip the rim from both arms. Measured, it strips **1.25% on aea and
1.04% on nymeria** — MPS semi-dense points are concentrated centrally enough
that the pinhole reaches almost all of them. Against the published raw-only run
the raw column moves by **−0.1% to −0.9%**, all improvements. The shrinkage is
real, and it is an order of magnitude smaller than the warning implies.

**Step 2 — rectify or not** (AbsRel, raw → rect_derect, ratio):

| model | aea | nymeria |
|---|---|---|
| vggt_1b | 0.199 → 0.158 (**0.794**) | 0.144 → 0.131 (0.912) |
| vggt_omega | 0.151 → 0.148 (0.979) | 0.121 → 0.125 (**1.040**) |
| dav2_large | 0.158 → 0.150 (0.952) | 0.144 → 0.139 (0.966) |
| da3_large | 0.207 → 0.194 (0.938) | 0.172 → 0.164 (0.953) |
| da3_small | 0.240 → 0.235 (0.979) | 0.205 → 0.208 (1.013) |

**VGGT-1B is the only model rectification clearly helps; VGGT-Omega is the only
one it hurts** — and Omega is already the best raw model on both datasets. So
the answer to "is rectifying worth it" is per-model, not global, and it is
worth most to the model that needs it most.

**Step 3 — what a second frame is worth** (ratio to the 1-frame arm at 10
frames; <1 means more frames helped):

| model | baseline | stride 1 aea / nym | stride 10 aea / nym |
|---|---|---|---|
| vggt_1b | raw | 1.110 / **1.161** | 1.088 / 1.085 |
| vggt_1b | rect_derect | 1.023 / 1.026 | 1.022 / 0.955 |
| vggt_omega | raw | 1.012 / 1.015 | 0.991 / 0.969 |
| vggt_omega | rect_derect | 0.965 / 0.966 | 0.933 / 0.943 |
| da3_large | raw | 1.013 / 0.963 | 0.976 / 0.892 |
| da3_large | rect_derect | 1.071 / 0.971 | 0.990 / **0.866** |
| da3_small | raw | 1.037 / 1.025 | 1.001 / 0.934 |
| da3_small | rect_derect | 1.020 / 1.011 | 0.972 / 0.923 |

**The FOV experiment's finding survives on real footage.** Consecutive frames
hurt, strided frames help, in 14 of 16 arms. The exception is worth naming:
**`vggt_1b`'s raw arm is not rescued by striding** (1.085–1.088), while its
rectified arm is (0.955 on nymeria). Both of VGGT-1B's context failures are in
the unrectified arm, which is the same model and the same axis on which
rectification helped it most.

**VGGT-Omega does not behave like VGGT-1B**, which report item 4 asks directly.
Omega gains from context in every arm at stride 10 and never degrades past
1.015; VGGT-1B degrades in three of four.

### The stride gap measures to exactly zero

`--context-stride` is a single int, so stride 1 and stride 10 are separate runs
and each intersects its own support. Their `context=1` arm is the *identical
computation on the identical frames*, so any difference between those two rows
is the support moving underneath — the floor below which no stride effect is
resolvable. **It is 0.00% on all 16 cells, with coverage identical to four
decimals.**

The reason is worth keeping: `rect_derect` gives up the *same* rim at every
context size, because the rim it cannot reach does not depend on how many frames
it was shown. So the intersection is set by `rect_derect` alone and is constant
across the whole sweep. The two tables are therefore directly comparable —
measured, not assumed — and widening `--context-stride` to a list would have
bought nothing. That is a property of this grid, not a theorem: a run whose
context arms lost points of their own would move the support, and
`ANALYSIS.txt` section 2 recomputes the check rather than restating the result.

### Both GPUs, sharded by model

`slambench.run` computes the support intersection **inside** its per-model loop,
over that model's own arms and contexts, so two processes carrying disjoint
model lists produce the numbers one process would. `tools/merge_slambench_shards.py`
does not take that on faith — it refuses to merge unless the shards agree on
digest, root and the whole `config` block, and refuses overlapping model sets.
All three merges passed and each run's two `manifest.json` were byte-identical.

1.40× overall. Step 2 got 1.83×; step 3 ran lopsided because the partition was
balanced on step-2 timings and **`vggt_1b`'s cost under a 10-frame window grows
13.7× where `vggt_omega`'s grows 4.4×** (334 s → 4571 s against 206 s → 910 s).
`slam020_lane.sh` now carries the corrected split: `vggt_1b` alone in one lane.

### One thing this run needed that was not in the ticket

`pytest tests slambench/tests` was **red** on `bd1a86e`: #016's new
`slambench/verify_depth_convention.py` imports `gzip`, which was missing from
the stdlib allowlist in `tests/test_experiment_separation.py` though `zlib`
beside it was there. #016 reported green because it ran `pytest slambench/tests`,
the narrower path, which never sees that file. Fixed in `143686a`; 98 passed on
the box in the run's own env.

## The goal

**A complete baseline table on real egocentric footage: the right depth
convention, both lens strategies, and what a second frame is worth.**

`results/slambench-raw-b1659a0` (digest `321f55a96bd4`) is what exists: 5 models
× {aea, nymeria, oxford}, 8 takes each, 25 frames per take, **raw baseline only,
no context**. That is one cell of the grid the harness was built for.

Three things are missing, and they are ordered because the first one changes what
the other two mean.

---

## Step 1 — is `d` planar z? (#016) — **ANSWERED: yes. Proceed.**

**Done 2026-08-14. The verdict is `z`, planar camera-frame Z**, on both staged
takes, decisively. Nothing already published moves and steps 2 and 3 are
released. Full table and method in #016; the short version:

* `|d - z| / z` reads **0.0002 at every incidence angle** on both datasets —
  the float16 floor of the stored `d`, and *flat*, which is what a radial
  convention error cannot be;
* `|d - range| / range` matches **`1 - cos(theta)` to four decimal places in all
  eight bins**, which is the signature of `d` *being* z rather than merely of
  range being wrong. (Note the algebra: each reading predicts the *other's*
  residual, and `d=z => 1-cos(theta)` while `d=range => sec(theta)-1`. They are
  different curves — 0.36 against 0.56 at 50 deg.)
* an independent reading that matches nothing agrees: reconstructed under z the
  points land 0.0007-0.0016 m from the cloud and flat; under range,
  0.0065-0.0216 m and growing with theta.

Measured on **two takes, one per dataset** — the limit of the local sample, the
same limit `verify_camera` has. `oxford` is unmeasured, and is out of step 2 for
other reasons anyway.

The check was built so it *could* fail: matching is on ray direction alone, and
a test requires it to return **range** on a synthetic take built as range.

---

## Step 2 — both lens strategies, in one run

**Is rectifying worth it, once you have paid to map the depth back?** This is the
question `slambench/` was built to answer and it has never been asked.

```
python -m slambench.run --egosynth-root $EGOSYNTH --calib-root $EGOSYNTH_CALIB \
  --datasets aea,nymeria --baselines raw,rect_derect \
  --models vggt_1b,vggt_omega,dav2_large,da3_small,da3_large \
  --takes 8 --n-frames 25 --out eval_out/slambench-rect/main
```

**`rect_derect` is limited to aea and nymeria.** `camera.VERIFIED_ROTATION` holds
only those two, and `require_verified` refuses the rest by design — a quarter-turn
error does not make a score worse, it scores a different part of the image.
Oxford was in the raw run and cannot join this one until `verify_camera` passes
on it, which needs its calibration (#012 left oxford and egoexo4d outstanding).
Either run the verifier on oxford first and add it by hand, or report the rect
table on two datasets and say why the third is absent. **Do not pass
`--allow-unverified` to make the table look complete.**

### The trap in this step

**You cannot paste a new rect column beside the published raw column.**
`run.py` intersects the support across *every* arm in the run:

```python
support = np.ones(len(pts), bool)
for p in preds.values():           # every (baseline, context) pair
    support &= np.isfinite(p) & (p > 0)
```

A 110 deg pinhole cannot cover the whole fisheye cone, so adding `rect_derect`
removes the rim points from **both** arms. The raw numbers in this run will not
equal `slambench-raw-b1659a0`'s, and that is correct — they are raw scored on the
points both arms could answer for. Report the new run's own raw column, and
report `kept` (the share each arm could answer for before intersecting) beside
it, which is what makes the shrinkage visible rather than mysterious.

---

## Step 3 — what a second frame is worth, including VGGT-Omega

The FOV experiment found that ten *consecutive* frames are worse than one, and
ten *strided* frames are better by up to 31% — a second viewpoint helps, a second
sample of the same instant does not. That was measured on a digital twin. This
asks it on real footage.

```
python -m slambench.run --egosynth-root $EGOSYNTH --calib-root $EGOSYNTH_CALIB \
  --datasets aea,nymeria --baselines raw,rect_derect \
  --models vggt_1b,vggt_omega,da3_small,da3_large \
  --context-frames 1,3,5,10 --context-stride 1 \
  --takes 8 --n-frames 25 --out eval_out/slambench-ctx/stride1
```

then the same with `--context-stride 10`.

* **`dav2_large` is dropped from this step, not silenced.** It is monocular and
  `predict_stack` raises rather than scoring the target alone — a table reading
  "context does not help" when nothing was tried is the one wrong answer that
  looks like a result.
* **The context sweep is one model load**, which is why `--context-frames` takes
  a list. Do not run the four sizes as four invocations.
* The same support intersection applies *across contexts too*, so all four
  context arms must be in one run to be comparable.
* `vggt_omega` needs `--omega-checkpoint` or `$VGGT_OMEGA_CKPT`; its weights are
  gated. Check it loads before committing to the grid.

---

## Report

1. **Step 1's verdict** — z, range, or undecided — with the per-theta-bin table.
   Everything below is conditional on it.
2. The digest, confirmed the same across steps 2 and 3 (the context is excluded
   from the digest by design, so all of these are one split).
3. **raw vs rect_derect**, per (model × dataset), on the intersected support,
   with `kept` beside each so the reader can see what each arm gave up.
4. **Context**, per (model × baseline): 1 / 3 / 5 / 10 at stride 1 and stride 10.
   Say explicitly whether the FOV experiment's "strided beats consecutive"
   survives on real footage, and whether VGGT-Omega behaves like VGGT-1B.
5. `gt_median` per dataset alongside every score. Every metric here is relative
   and grows with depth, and Oxford is outdoors at ~5.3 m median against AEA's
   ~1.2 m indoors — "worse on Oxford" is not yet a statement about a model.
6. Timings, and which models were skipped and why.

## Done when

- [x] #016 has reported a verdict and it is recorded here — **z**, planar
      camera-frame Z, both staged takes
- [x] `python -m pytest tests slambench/tests -q` passes on the run's commit —
      **98 passed** on `143686a`, on the box, in the run's own env. It was red
      on `bd1a86e`; see the `gzip` note above.
- [x] steps 2 and 3 share one digest — **`61195914f090`** across all three
      runs, with root and the whole `config` block identical apart from the
      context axis. Checked by `ANALYSIS.txt` section 1, not by eye.
- [x] the tables above are in the issue, `kept` included — `coverage` is beside
      every row in `report.txt` and in the tables above (raw 1.000,
      `rect_derect` 0.988 aea / 0.990 nymeria)
- [x] pushed to `results` — `b10b087`, `results/slambench-020-143686a`
- [ ] issue commented with the sha

## Not in scope

* **egoexo4d.** Paused upstream; its rotation has never been measured.
* **Widening the raw arm** to more takes. This ticket completes the grid on the
  sample that exists rather than growing the sample — a wider raw arm that still
  has no rect column answers nothing new.
* **Changing `slambench/`.** If step 1 comes back "range", or if `rect_derect`
  turns out to need oxford, those are tickets written after this one reports.
