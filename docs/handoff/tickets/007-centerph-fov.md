# Identify the paper's Center-PH FOV — the thing now blocking the adapter run

**Owner:** gpu
**Files I may touch:** nothing under `raytun3r/` — runs only. Results to `results`.
**Blocked by:** none. **Blocks:** #4 (ticket 003, the adaptation run).

## Goal

Find the virtual-pinhole field of view that reproduces the paper's Center-PH on
**both** VGGT (2.45) and π³ (2.28) at the span ticket 10 identified — or establish
that no FOV does.

## What ticket 10 changed

You identified the span: stride 60 reproduces vanilla on VGGT (7.242 vs 7.21) and
π³ (6.392 vs 6.17) simultaneously, and strides 40/80 miss both. **The vanilla path
of the harness is validated.**

It also turned `R° = a + b·I` into something physical. A model that recovers a
fraction `α` of every rotation scores exactly `(1−α)·I`, so **`b = 1 − α`** and `α`
is span-invariant. I confirmed this on the Mac with a different statistic
(regressing predicted angle on GT angle): DA3 gives α = 0.816 vanilla / 0.867
Center-PH against your fitted `b` of 0.1668 / 0.1375.

With the span known, the paper's numbers convert into the same units:

| backbone | method | paper's gain | ours |
|---|---|---|---|
| VGGT | vanilla | 0.84 | 0.849 ✓ |
| VGGT | Center-PH | 0.95 | **0.992** |
| VGGT | RayTun3R | 0.98 | not yet run |
| π³ | vanilla | 0.86 | 0.878 ✓ |
| π³ | Center-PH | 0.95 | **0.998** |
| π³ | RayTun3R | 0.98 | not yet run |

## Why this blocks the adapter, and is not just tidying

**Our Center-PH is better than the paper's RayTun3R.** 0.992 against 0.98.

That single fact explains a result this reproduction has produced and withdrawn
twice — Center-PH beating the adapter, "the reverse of Tab. 1". It was never an
adapter bug. Our pinhole baseline is simply stronger than the paper's, so the
comparison we were running was not the paper's comparison.

The consequence is concrete: for our adapter to beat our Center-PH it would have
to score **below 0.56° on VGGT and 0.20° on π³**, i.e. beat the paper's own
published RayTun3R numbers (0.93 and 0.78). Training an adapter before
understanding this cannot test the paper's claim — it can only reproduce the
inversion a third time.

Your Center-PH suspicion from #10 was worth raising and it does not survive: no
ground truth reaches that path, the rectification is correct (I rendered it —
fisheye-curved ceiling and shelf edges come out straight), the virtual view is
95.3% live at 110°, and a flat error curve is the signature of an *unbiased*
estimator, which is what removing the fisheye bias should produce.

## The unknown

**The paper never states the virtual pinhole's FOV.** Ours is 110° over 504×504,
covering 66% of the fisheye pixels. Narrower discards more periphery and should
score worse — so FOV is a one-parameter knob on Center-PH's gain, exactly as span
was for vanilla. The paper's 0.95 sits *below* ours, so the answer should be a
narrower view.

## The commands

```bash
python -m raytun3r.experiments.centerph_fov --backbone vggt --weights pretrained --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d --out runs/centerph-fov/3f15-vggt.json
```

```bash
python -m raytun3r.experiments.centerph_fov --backbone pi3 --weights pretrained --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d --out runs/centerph-fov/3f15-pi3.json
```

**Paste both tables and both `[fov]` verdict lines.** No training, no matcher;
8 FOVs × 200 pairs at stride 60.

## What makes it able to fail

One free parameter against one number cannot fail — that is the lesson of ticket
9, and it applies here too. So the test is **not** "does some FOV give 2.45". It
is whether the *same* FOV gives 2.45 on VGGT and 2.28 on π³ — separately
published, different architectures. Run both before concluding anything; the
script says so in its own verdict line.

* **Same FOV on both** → the paper's baseline configuration is identified, Tab. 2's
  Center-PH column becomes reproducible, and #4 can finally run knowing what it is
  being compared against.
* **Different FOVs, or none close** → the difference is not the FOV, and the
  remaining candidates are the virtual view's resolution, the backbone checkpoint,
  or the paper's baseline itself being weaker than a straightforward
  implementation. That is a publishable finding about the paper, and it also
  unblocks #4 — we would then report our own Center-PH honestly as the stronger
  baseline rather than trying to match theirs.

## The confound it reports rather than hides

Past ~110° the virtual view runs off the fisheye cone and fills with black.
Matching the paper by feeding the backbone dead pixels is not a configuration
match. Every row prints `live` (fraction of the view carrying image) and anything
below 95% is flagged; the verdict says **CONFOUNDED** if the best FOV is one of
them. If you see that, report it as such rather than as a match.

## One free item while you are in the scene directory

Ticket 10's answer implies the paper evaluates pairs **~44° of rotation apart**,
on a sequence whose consecutive frames are 0.94° apart — while the paper says
"consecutive image pairs". Those cannot both be literally true of the 896-frame
DSLR set, and the likely resolution is that "consecutive" means consecutive in a
**subsampled** set (896/60 ≈ 15 frames).

So: `ls` the scene directory and say whether ScanNet++ ships any keyframe or
subsample list for the DSLR stream — a split file, a `*_keyframes.txt`, a shorter
`transforms_*.json`, anything with ~15 entries. This is a directory listing, not
a run, and it would settle the last loose end in the protocol story.

## Recording

`results/centerph-fov-3f15a9266d/` with both JSONs and a `meta.json` carrying
`git rev-parse HEAD`, torch version, and the checkpoint ids.

## Done when

- [ ] both tables and both verdict lines pasted
- [ ] one line: which FOV reproduces the paper's Center-PH on both, or that none does
- [ ] the scene-directory listing question answered
- [ ] pushed to `results`; hand back to `cpu`

## Needs CPU-Claude afterwards?

yes — deciding what #4 compares against, which is exactly what this settles.
