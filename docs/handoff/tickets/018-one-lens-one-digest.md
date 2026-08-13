# One description of the lens, one definition of a digest, and a way in

**Owner:** cpu — **done**, landed on `organized`.
**Files touched:** `finetune/aria_calibration.py` (new), `finetune/eval/manifest.py` (new),
`tests/test_shared_foundations.py` (new), `finetune/data/rectify.py`, `finetune/test_run.py`,
`finetune/eval/baselines/aria_fisheye.py`, `VGGT-360-fisheye/utils/fisheye_cam.py`,
`fovbench/split.py`, `slambench/split.py`, `README.md`
**Follows:** ticket 017. Candidates C and D of the architecture review.

## Goal

The Aria lens is described in one place, the split digest is defined in one
place, and the repository root says what the two experiments are and how to run
them.

## C — the lens, and the pixel it had drifted by

Four files carried the same five calibration numbers, three of them with their
own spelling of the ADT storage rotation. Two agreed. Two did not:

    fisheye_cam.py        cx' = (H-1) - cy        the FOV experiment's fisheye arm
    aria_fisheye.py       cx' = (H-1) - cy        the baselines
    rectify.py            cx' = W - cy            the FOV experiment's RECTIFIED arm
    finetune/test_run.py  cx' = W - cy            the data/model sanity tool

**`(H-1)` is right, and it is not a matter of taste.** `np.rot90(m, 3)` — the
rotation every ADT loader in this repo applies — sends source pixel `(u, v)` to
`((H-1) - v, u)`; you can read that off a labelled 9x9 grid, which is what
`tests/test_shared_foundations.py` now does rather than trusting any of the four
formulas. A pixel's centre is at its integer coordinate, so the last column of an
`H`-wide frame is at `H-1`, not `H`.

So the two arms of the ADT-FOV experiment — the experiment whose entire purpose
is that `rect` and `fisheye` be comparable — described the same lens **exactly
1.000 px apart, at every resolution**.

### What it cost, measured

Correcting it moves the rectified arm and leaves the raw arm bit-identical:

| | before | after | |
|---|---|---|---|
| rect AbsRel, per bin | — | — | **+0.1 % to +0.7 %** |
| rect `pen` | 1.2309 | 1.2262 | **−0.385 %** |
| fisheye, everything | — | — | **0.000 %** |

Real, and far below the effects the benchmark reports (`pen` 1.0–2.9, differences
of tens of percent). **No published conclusion changes**; a rect-arm re-run would
move the fourth significant figure. Worth doing when the box is next free, not
worth blocking on.

Note *why* it survived: a lateral shift of the principal point displaces the
image sideways, so within an incidence-angle ring half the pixels gain and half
lose. Ring-averaged checks see a mean bias near zero. It shows up as **scatter** —
in the sphere probe the rect arm's p95 |theta error| ran 0.14–0.19 deg against
the fisheye arm's 0.035–0.047, a 4x difference that had been sitting in plain
sight.

### What was done

`finetune/aria_calibration.py` holds the five constants and `intrinsics(H, W,
rotated)`, with the rotation derived in the docstring. Four consumers import it.
Deliberately **no geometry**: the KB4 ray LUTs, the two angular limits and the
rectifier's remap stay with their consumers. Only what was duplicated moved,
because duplication is where a divergence goes unseen.

The "self-contained subproject" justification for vendoring is retired, because
it was already void: `VGGT-360-fisheye/main_adt.py` puts the repository root on
`sys.path` specifically to import `finetune.eval.metrics`, "the shared scoring
protocol". The subproject depends on the root for the definition of a metric;
depending on it for the definition of the lens is the same arrangement.
`fisheye_cam.py` does its own `sys.path` bootstrap because `checks/*.py` add only
the subproject directory.

## D — the splits: the classes stay, the rules do not

The review said "do not merge the splits" and that was right about the classes
and wrong about what sits underneath them. `Split` and `Frame` are genuinely
different — different fields, different context-window semantics, different
`build`, and separate `PROTOCOL` tokens precisely so a digest from one experiment
can never be mistaken for the other's. Merging them would recreate the coupling
ticket 017 removed.

But the **digest rule was byte-identical in both**, and so was `_evenly_spaced`.
That is the part where duplication is dangerous rather than merely repetitive:
the digest is the definition of "these two runs measured the same thing", and two
copies of it can drift into meaning different things while both still looking
like digests — a failure with no symptom.

`finetune/eval/manifest.py` owns those two rules. Both splits import them; both
keep their own `PROTOCOL`, `Frame` and `build`.

`tests/test_shared_foundations.py` pins the extracted digest against the literal
body both splits used to carry, so it cannot have moved a published digest
(`601fcb22767e`, `8ca25fd0ebd2`, `fcc6c600f83b`).

## Follow and run

`README.md` had no mention of either experiment — a newcomer met the upstream
VGGT-Omega readme and no route to `fovbench/` or `slambench/`. It now opens with
what the two experiments ask, the ground truth each uses, one command each, the
weight-free stand-in for both, and the one test command:

```bash
python -m pytest tests fovbench/tests slambench/tests -q
```

plus a table of the shared substrate — metrics, manifest, model zoo, calibration
— so "where does this belong" has a visible answer.

## Done when

- [x] `python -m pytest tests fovbench/tests slambench/tests -q` — 200 passed, 4 skipped
- [x] `raytun3r` unchanged at its pre-existing 4 failures (`cv2.USAC_MAGSAC` on
      OpenCV 4.4, `huggingface_hub` absent — neither related)
- [x] the ADT fingerprint differs from the pre-fix state by **exactly** the
      isolated 1-px probe, bit-identically, with the fisheye arm untouched
- [x] the lens constants appear once in code; the digest rule appears once

## Still duplicated, deliberately or out of scope

* **`cam3r/cameras.py`** carries its own copy of the constants. A separate line of
  work, outside this ticket's scope, and it should be pointed at
  `finetune/aria_calibration.py` when someone next touches it.
* **`slambench/camera.py`** derives `610.94, 715.11, 716.71` independently, from
  the 2880 sensor calibration via projectaria_tools #322. That agreement is the
  only cross-check these constants have and it must **not** be collapsed into an
  import — a test that reads the value it is checking checks nothing.
* **Test literals** in `tests/test_shared_foundations.py` and
  `slambench/tests/test_camera.py`, for the same reason.
