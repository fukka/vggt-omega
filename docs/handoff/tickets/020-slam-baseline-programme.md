# slambench: a proper baseline evaluation on real egocentric SLAM data

**Owner:** gpu (with #016's fetch on cpu)
**Status:** **open** — this is the SLAM evaluation's live ticket. Successor to #013.
**Files I may touch:** nothing under `slambench/` except what #016 adds — runs only.
Results to `results`.
**Blocked by:** nothing. **Step 1 is answered — `d` is planar z (#016,
2026-08-14) — and steps 2 and 3 are released.** #012 has also landed all three
calibration sets on lambda_63, so `rect_derect` on aea + nymeria has no
remaining technical gate.

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
- [ ] `python -m pytest tests slambench/tests -q` passes on the run's commit
- [ ] steps 2 and 3 share one digest
- [ ] the tables above are in the issue, `kept` included
- [ ] pushed to `results`, issue commented with the sha

## Not in scope

* **egoexo4d.** Paused upstream; its rotation has never been measured.
* **Widening the raw arm** to more takes. This ticket completes the grid on the
  sample that exists rather than growing the sample — a wider raw arm that still
  has no rect column answers nothing new.
* **Changing `slambench/`.** If step 1 comes back "range", or if `rect_derect`
  turns out to need oxford, those are tickets written after this one reports.
