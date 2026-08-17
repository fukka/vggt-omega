# Copyright (c) 2026.
"""Depth evaluation against real egocentric SLAM ground truth.

**This is not the ADT-FOV experiment**, and the two must not be read as one.
`fovbench/` asks *where in the field of view does depth degrade*, on a synthetic
twin whose ground truth is a dense depth map. This package's headline question is
different and simpler — *how accurate is each model on real egocentric footage* —
against ego-synth 5B's semi-dense MPS SLAM points.

``run.py`` is that question and has no eccentricity axis; three published
artefacts were produced under that contract and it does not change.

``run_fov.py`` asks the **FOV question of this data**, which is a third thing
again: the same question ``fovbench`` asks, of different ground truth, under a
protocol that had to be rebuilt for it. Sparse points need a different estimator
(sums pooled across frames, not per-frame bin means) and one control the dense
experiment does not need at all — distance falls 3.6x from this field's centre to
its rim, so an uncontrolled error-against-eccentricity curve here is partly a
distance curve. ``fov.py`` is where that is written down.

So the three share a repository, some models and a definition of AbsRel. They do
not share ground truth, and only the two FOV drivers share a question. **Do not
read a number from one as if it came from another**; in particular ``run.py``'s
tables and ``run_fov.py``'s are not two views of one measurement, because the
second pools by points and the first by frames.

Nothing here imports `fovbench` and nothing here reads ADT.
``tests/test_experiment_separation.py`` enforces both mechanically, from the
repository root so that it can see each package from outside — the earlier guard
lived in this package and checked only the direction *into* it, which is how a
duplicate ego-synth reader grew on the other side and stayed green for months.

    python -m pytest tests fovbench/tests slambench/tests -q

The shape of the evaluation
---------------------------
Ground truth is used in the form the producer captured it: the SLAM points as
projected into the **raw fisheye** frame. The harness performs no rectification,
no warping and no depth-convention conversion — it reads a frame, hands it to a
baseline, receives one depth per ground-truth point, and scores that.

    harness    raw fisheye frame + native GT points
                 -> baseline.predict(frames) -> depth per point
                 -> metrics

**The lens is the baseline's business.** Two baselines differ only in what they
do behind that one call:

    raw           feed the fisheye frame to the model as it is
    rect_derect   rectify, run the model, map its depth back onto the fisheye
                  points -- all inside the baseline

which is what makes "is it worth rectifying first, once you have paid to map
back?" a question this harness can answer rather than presuppose.

**One frame or several.** Four of the five models are multi-view, and
``--context-frames 1,3,5,10`` hands them a window of preceding frames in one
forward pass. Exactly one frame of the window is scored, so the arms measure
identical points and only the evidence moves; the context is therefore kept out
of the split digest, which exists to say two runs scored the same points.

The depth convention, and how it stopped being an assumption
-----------------------------------------------------------
Every number here is scored against ego-synth's ``d`` read as **planar z** about
the camera axis. If it were euclidean range instead, every score would carry a
``1/cos(theta)`` error — 1.00 on axis and 1.74 at 55 deg, radial, and so not
absorbable by the per-frame affine.

For a while that rested on the data card alone (gotcha 4). It no longer does:
``verify_depth_convention.py`` (ticket 016, closed 2026-08-14) measured it on
both staged datasets and read **z**, with the residual at 0.0002 and flat across
incidence angle — the float16 noise floor of the stored value — and the range
hypothesis wrong by exactly ``1 - cos(theta)`` in all eight bins. The check can
also return "range": ``tests/test_depth_convention.py`` builds a synthetic take
under each convention and requires it to recover the one it was built under,
because a check that could only ever say "z" would make agreement with the card
worthless.

What is still worth naming is *why* it needed its own check. Two things in this
package **look** like checks of the convention and are structurally incapable of
being one: the rectified/fisheye depth agreement cited in ``baselines`` (range is
equally invariant under a co-axial rectification) and ``verify_camera``'s
sub-pixel reprojection (projection reads only the ray's direction, which both
readings share). Believing either one is how this ends up unnoticed — and it is
the trap to remember if a fifth dataset is ever added.

Layout
------
    camera.py     the Aria FISHEYE624 model, per take
    data.py       reading ego-synth: takes, clips, points, frames
    baselines.py  the two lens strategies
    metrics.py    per-point scoring, under each model's own alignment protocol
    split.py      the frozen frame manifest and its digest
    run.py        the driver — the published one, no eccentricity axis
    report.py     tables

    fov.py        the FOV question on this data: the eccentricity axis, the
                  distance control it needs, and the re-aimed window
    run_fov.py    its driver. Shares everything above and owns only the binning
    fov_report.py its tables

Importing this package puts the repo root and ``VGGT-360-fisheye/`` on
``sys.path``. The root is for ``finetune.eval.metrics`` and
``finetune.eval.baselines.model_zoo`` — shared infrastructure belonging to
neither experiment — and the repo is not installed. ``VGGT-360-fisheye/`` is for
the ``vggt360`` baseline arm, which runs that port's pipeline rather than
reimplementing it; the directory has a hyphen, so its ``utils`` package can only
be reached by path. ``fovbench/__init__.py`` does the same, for the same reasons.
"""
import os as _os
import sys as _sys

_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_REPO, _os.path.join(_REPO, "VGGT-360-fisheye")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
