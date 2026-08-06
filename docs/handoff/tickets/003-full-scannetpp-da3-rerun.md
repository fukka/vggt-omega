# Reproduce one paper number first, then scale out: DA3-Small on ScanNet++

**Owner:** gpu
**Files I may touch:** nothing under `raytun3r/` — this is runs only. Results go
to the `results` branch.
**Blocked by:** none. Pull `organized` first; everything below is already in it.

> **Read [`raytun3r/PAPER.md`](../../../raytun3r/PAPER.md) — the paper and only the
> paper**: every hyperparameter, all six result tables, all four ablation tables,
> the named sequences, the twelve things it never specifies, six errata. And
> [`raytun3r/reproduction.md`](../../../raytun3r/reproduction.md) — what is *ours*:
> how each gap was resolved and how to read a run. Neither needs the PDF open.

## Goal

We can say, with a committed number, whether RayTun3R reproduces on ScanNet++:
`raytun3r` on scene `3f15…` with DA3-Small lands near the paper's **0.40° `R°`**
(Tab. 5), or it does not and we know which of three explanations is responsible.

**Work the phases in order.** Phase 1 is the actual reproduction and costs ~15
GPU-minutes; Phases 2 and 3 only make sense once it has an answer. The previous
version of this ticket led with a full-dataset sweep, which cannot reproduce any
paper number — Tab. 1 and Tab. 3 are means over scene sets the paper never names.

## Why this supersedes the earlier runs

Three bugs since `bfdb47e`, all now fixed; the first two invalidate `d_reproj` on
the `results` branch. **None of them affect `R°`/`t°`** — pose comes from the
camera head — so the rotation findings still stand.

1. **The depth convention was inconsistent between methods.** The virtual-pinhole
   baselines were converted planar z → euclidean range, the direct fisheye path
   (`vanilla`, `param_free`, `raytun3r`) was not. Worth **~0.99 px of `d_reproj`**
   on ScanNet++ geometry against a method-to-method spread of 0.10 px. Backbones
   now declare `native_depth`, `install(depth_convention=...)` converts once at
   the boundary, consumers call `require_convention`.
2. **`d_reproj` was averaged the wrong way.** Eq. 8 carries `w_ij` inside the sum;
   **Eq. 16 carries no weights at all.** We were computing `sum(w·e)/sum(w)` — the
   mean over the *confidently matched* subset — where Eq. 16 is an unweighted mean
   over all of Ω. UFM's covisibility collapses exactly where reprojection error is
   worst, so the old number dropped the hardest pixels and renormalised by their
   absence. On synthetic 170° geometry the two differ by **1× to 170×** depending
   on how much of Ω the matcher abandons. `eval.py` now reports **both**:
   `d_reproj` (Eq. 16, the only one comparable to the paper) and `d_reproj_conf`
   (the old behaviour, so pre-fix runs stay interpretable).
3. **`da3` never worked** — the paper's primary backbone, and Tab. 1/4/5/7b are
   DA3-Small numbers. Four bugs, now fixed and covered by a CPU test against real
   `depth_anything_3` 0.1.1: the public `forward` is wrapped in `torch.no_grad()`
   (the adapter could never have trained), the DPT hook targeted
   `_apply_pos_embed` instead of `_add_pos_embed`, a wrong helper import path, and
   `load()` fetched GIANT regardless of `--variant`.

---

## Phase −1 — audit the evaluation data first (seconds, no GPU)

**Phase 1 ran and missed (see below), so this now precedes everything.** `vanilla`
is the easiest thing in the paper to reproduce — frozen backbone, no adapter, no
training, no randomness — and ours does not match: paper 7.21° `R°` on this scene
with VGGT, we measure 0.554 at stride 1 and 2.379 at stride 10. Until that is
explained, no adapter result means anything.

```bash
python -m raytun3r.experiments.data_audit --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d --json runs/audit/3f15a9266d.json
```

Reads `transforms.json` only — no images, no weights, no GPU. Report the whole
output on the issue. It answers four things:

1. **Keys the loader ignores.** `test_frames`, `applied_transform`,
   `applied_scale`, and per-frame **`mask_path`**. ScanNet++ ships per-frame DSLR
   masks and we ignore them — if those masks define the valid fisheye region, then
   the paper's Ω is the mask and ours is the whole rectangle, which would explain
   the 115°/170° disagreement outright.
2. **The FOV the intrinsics actually imply**, at frame edges and corner
   (horizontal / vertical / diagonal). One of those may be the paper's 115°.
3. **Whether the poses are metric.** If the camera bbox is ~1 rather than metres,
   every distance we have quoted — including the 1.09 cm baseline — is in the
   wrong unit.
4. **The identity-predictor score per stride.** This is the important one.
   `R°` is an *absolute* angular error, so a model predicting identity scores
   exactly the median GT rotation. **Two runs' `R°` are only comparable when their
   GT rotation distributions match**, and `--stride` changes that distribution
   directly. If our stride-1 median GT rotation is ~0.5°, then `vanilla`'s 0.554°
   is the identity score and that evaluation carries no information at all.

Then the pixel-level questions, still without moving any pixels:

```bash
python -m raytun3r.experiments.data_probes --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d --json runs/audit/3f15a9266d-probes.json
```

* **Probe 1 — is the frame corner real content or dead vignette?** Intensity mean
  and std per incidence-angle bin. Near-zero std in the outer bins means the image
  circle is smaller than the sensor, Ω is a circle, and the paper's 115° is right;
  std like the inner bins means content out to the corner and our ~170° is right.
* **Probe 2 — what is `mask_path`?** A lens mask is radially symmetric, identical
  across frames, and cuts at a fixed incidence angle; an anonymisation mask is
  small, irregular and frame-varying. If it is a lens mask, **Ω should come from
  it** and we have Ω wrong.
* **Probe 3 — planar z or euclidean range?** Regresses `log(D)` on
  `log(cos θ)`: planar z gives slope ≈ 1, range gives ≈ 0. Verified on a synthetic
  scene with a known answer (0.95 → `z`, −0.05 → `range`). Only matters for
  `d_reproj`/AbsRel, never for `R°`/`t°`.

**Both scripts are read-only, need no GPU or weights, and emit JSON.** Commit the
two JSON files to `results` — they are *derived statistics*, which is what makes
them safe: **ScanNet++ is under Terms of Use that forbid redistribution and this
repo is public, so no frame, mask or depth map from it may ever be committed.**
That is also why these probes return numbers instead of crops.

**Decision rule.** If some stride puts the median GT rotation near 7°, that is the
operating point the paper is at and Phase 1 should be re-run there. If *no* stride
reaches it, then our frame set differs from the paper's — that is the finding, and
it is worth more than any further sweep. Either way, stop and report before
spending GPU time.

## Phase 0 — setup and two cheap checks (~10 min)

```bash
git -C /user/f.zhang2/projects/vggt-omega-organized pull --ff-only origin organized
```

```bash
pip install depth-anything-3
```

**Check A — the geometry of the pairs being scored.** Our one real run has
*unadapted* numbers far better than the paper's (vanilla `R°` 2.379 vs 7.21,
Center-PH 0.378 vs 2.45) and only the *adapted* one worse. That is the signature
of an evaluation easier than theirs: there is only 2.4° of error available to
remove where they had 7.2°.

```bash
python -c "
from raytun3r.data import ScanNetPPFisheye
import torch
s = ScanNetPPFisheye('/netapp/datasets/scannetpp/data/3f15a9266d')
P = [s.pose(i) for i in range(len(s))]
ok = [i for i,p in enumerate(P) if p is not None]
C = {i: (-P[i][0].T @ P[i][1]) for i in ok}
d = torch.tensor([ (C[b]-C[a]).norm() for a,b in zip(ok, ok[1:]) ])
print('frames', len(s), 'posed', len(ok))
print('consecutive baseline  median %.4f m  mean %.4f m' % (d.median(), d.mean()))
"
```

We measured ~1.1 cm at stride 1 against ~3 m of scene depth, which is why
`--stride 10` exists. ScanNet++ DSLR is a *sparse* handheld capture of a few
hundred images per scene — 1.1 cm between consecutive shots is not what that
should look like. If it comes back that small, check whether `transforms.json` is
read in capture order and whether we are picking up a denser image set than the
DSLR one. **If our pairs are much closer together than the paper's
consecutive-pair protocol, that alone could explain the too-good vanilla**, and it
is cheaper to fix than anything in Phase 2.

**Check B — the FOV disagreement.** Print the half-angle at the frame corner. The
paper says ScanNet++ DSLR is 115°; we measure ~170°. Confirm which it is on the
actual download before Phase 2 spends GPU time on it.

---

## Phase 1 — the reproduction (~15 GPU-min, one scene)

**This is the run that answers "does it reproduce".** Tab. 5 (the AnyCalib
appendix) is the only place in the paper with **per-sequence DA3-Small** numbers,
and `3f15…` is a scene we have. So it is the single tightest target available —
tighter than Tab. 2, and on the right backbone.

Targets, DA3-Small + GT calibration on ScanNet++ `3f15…`, as `R° / t° / d_reproj`:

| method | paper (Tab. 5) |
|---|---|
| **`raytun3r`** | **0.40 / 2.2 / 1.7** |
| `lora` (r=8, α=16) | 4.22 / 23.0 / 2.9 |
| `caltok` (t=4) | 3.09 / 20.0 / 4.4 |

All three are implemented, and `lora`/`caltok` each need their own fit — they
share the objective, so only `--method` changes. Three fits plus one eval:

```bash
S=/netapp/datasets/scannetpp/data/3f15a9266d
for M in raytun3r lora caltok; do
  python -m raytun3r.train --backbone da3 --variant small --weights pretrained \
    --dataset scannetpp --path $S --method $M --stride 10 \
    --out runs/rt3r/tab5-3f15-da3s-$M
done
```

```bash
python -m raytun3r.eval --backbone da3 --variant small --weights pretrained \
  --dataset scannetpp --path $S --stride 10 \
  --adapter runs/rt3r/tab5-3f15-da3s-raytun3r/adapter.pt \
  --methods vanilla,param_free,raytun3r,center_ph,multi_ph \
  --out runs/rt3r/tab5-3f15-da3s/results.json
```

Then evaluate `lora` and `caltok` against **their own** checkpoints — `--adapter`
is applied only to the method it was fitted for, and feeding one method's state to
another is a hard error:

```bash
python -m raytun3r.eval --backbone da3 --variant small --weights pretrained \
  --dataset scannetpp --path $S --stride 10 \
  --adapter runs/rt3r/tab5-3f15-da3s-lora/adapter.pt --methods lora \
  --out runs/rt3r/tab5-3f15-da3s/results_lora.json
```

(and the same for `caltok`).

**Report `d_reproj` and `d_reproj_conf` side by side.** Their ratio is the first
real measurement of how much of Ω UFM gives up on, and it tells us immediately
whether bug 2 above explains the old 14–30× `d_reproj` gap.

**Phase 1 succeeds** if `raytun3r` lands near 0.40° and beats `lora`/`caltok` by
the margin Tab. 5 shows. If so, say so on the issue and go straight to Phase 3 —
Phase 2 is only for a miss.

---

## Phase 2 — only if Phase 1 misses (~4 GPU-h)

Two orthogonal explanations, both cheap. **Run 2a before 2b**: an undertrained
adapter would also make the FOV sweep unreadable.

### 2a — is 300 iterations simply too few?

```bash
python -m raytun3r.experiments.iters_sweep --backbone da3 --variant small --weights pretrained --dataset scannetpp --path /netapp/datasets/scannetpp/data/3f15a9266d --out runs/iters-sweep/3f15a9266d-da3s
```

The step count is the one hyperparameter Sec. 4.3 never states. We guessed
`--iters 300` (~3 min); App. D quotes **2–3 h per ScanNet++ scene**. Even allowing
that their figure covers full-sequence evaluation and every baseline, three
minutes of fitting does not sit comfortably inside two hours. If `R_deg` is still
falling at 300, that is the answer and everything else needs redoing at the elbow.

### 2b — does the FOV disagreement explain the baseline ordering?

```bash
python -m raytun3r.experiments.fov_sweep --backbone da3 --variant small --weights pretrained --dataset scannetpp --path /netapp/datasets/scannetpp/data/3f15a9266d --out runs/fov-sweep/3f15a9266d-da3s
```

Center-PH beat RayTun3R ~5× on `R°`, the reverse of the paper. If ScanNet++'s DSLR
is really ~170°, re-projecting to a 110° pinhole discards the hard pixels rather
than handling them. `--max-fov` narrows Ω only — images untouched — so this
isolates where a method is *scored*. Gap closes as the cone narrows → FOV is the
answer; flat → look elsewhere.

---

## Phase 3 — scale out, once Phase 1 has an answer

These produce our own aggregates. They are **not** comparable to Tab. 1/3 (means
over unnamed scene sets), so they are for *our* error bars, not for reproduction.

```bash
python -m raytun3r.experiments.scannetpp_all --backbone da3 --variant small --weights pretrained --root /netapp/datasets/scannetpp/data --out runs/rt3r/snpp-all-da3s --workers 4
```

Sanity-check with `--limit 2` first; ~3 min fit plus eval per scene, one scene per
GPU. Same command with `--backbone vggt` for the VGGT aggregate, which makes the
earlier single-scene VGGT run comparable and separates "the method does not
reproduce" from "VGGT is not the paper's backbone".

Then, cheap and worth doing:

* **`--backbone vggt_omega` on one scene.** Expected to barely help — DINOv3,
  RoPE-only, 20 adapter parameters. It is the direct test of Tab. 7b's "RoPE only"
  row (19.52° vs 0.48°). A *large* improvement would contradict the paper and
  would be the most interesting result available.
* **The queued `s10-adt-seq131` run**, now that conventions are fixed.
* **Find ScanNet++ `render_depth/`.** It was absent, so `AbsRel`/`δ₁.₂₅` (Tab. 3)
  have never been measured on ScanNet++ at all. If it is not in the download, say
  so and Tab. 3 stays out of scope for this dataset.

---

## Protocol notes

Defaults already match the paper — Adam 1e-3, clip 1.0, 300 iters, 30 three-frame
windows, 2 px flow filter, 504 max side, UFM, LoRA r=8 α=16, CalTok t=4 — with one
deliberate deviation: **`--stride 10`, not 1**. At stride 1 the baseline is ~1.1 cm
against ~3 m of depth, translation direction is unobservable (MAGSAC++ is itself
11.1° off ground truth) and `d_reproj` stops depending on depth. The paper does not
specify a stride; Check A above may change this. Do not change it back without
saying so in `meta.json`.

**"Same number of GPUs as the paper" is settled.** App. D: single RTX A6000/A4000
cards, "independent scenes, sequences, and baselines executed in parallel when
possible", ~180–250 GPU-h total. The paper's setting *is* one GPU per scene with
scenes in parallel — what `--workers` does. Nothing to shard within a scene; a
shared adapter across scenes would be a different method.

## How to read the numbers — three ways to call a reproduction a failure

* **RayTun3R is not supposed to win `d_reproj`.** It loses that column to
  Center-PH or Multi-PH on **4 of 5 datasets** in the paper's own Tab. 1, and
  Center-PH wins ScanNet++ depth outright (AbsRel 0.066 vs 0.108, Tab. 3). The
  claim is about **pose**.
* **Do not tune toward Tab. 4a.** The full model has the second-worst `R°` in its
  own component ablation — six of seven ablated variants beat it on rotation. It
  is selected on `d_reproj`.
* **Parameter-free corrections hurting alone reproduces the paper.** Tab. 8 has
  them making FIORD Kitchen worse (28.09 → 39.04 `R°`); they only help combined
  with the learned residual.

## Recording

Per POLICY.md, results to the `results` branch, JSON and trimmed logs only.

* **Pin the commit each run actually used**, not one per batch. Last time all
  three runs were labelled `bfdb47e`, but the two stride-1 runs predate it — they
  lack the `coverage` key it introduced and carry the 117 px `center_ph`
  `d_reproj` it fixed. Record `git rev-parse HEAD` per run.
* Record `depth-anything-3` and `torch` versions alongside the matcher, and the
  `--stride` / `--max-fov` actually used.

## Done when

- [ ] Phase 0 Check A reported on the issue (frames, posed, median baseline)
- [ ] **Phase 1: `raytun3r` / `lora` / `caltok` on `3f15…` with DA3-Small, scored
      against Tab. 5**, with `d_reproj` and `d_reproj_conf` both reported
- [ ] a plain statement of whether Phase 1 reproduced 0.40° or not
- [ ] Phase 2 sweeps, if Phase 1 missed
- [ ] Phase 3 aggregates for DA3-Small and VGGT
- [ ] pushed to `results`, this issue commented with the branch sha
- [ ] handed back to `cpu` for the README rewrite against real numbers

## Needs CPU-Claude afterwards?

yes — README rewrite, and interpretation of Phase 2 if it runs.
