# Copyright (c) 2026.
"""Depth evaluation against real egocentric SLAM ground truth.

**This is not the ADT-FOV experiment.** `fovbench/` asks *where in the field of
view does depth degrade*, on a synthetic twin, by binning error against
eccentricity. This package asks a different and simpler question — *how accurate
is each model on real egocentric footage* — against ego-synth 5B's semi-dense
MPS SLAM points, and it has no eccentricity axis at all. The two share a
repository, some models and a definition of AbsRel; they share no protocol, no
ground truth and no conclusion. Do not read a number from one as if it came from
the other.

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

What is not verified
--------------------
One assumption underneath every number here rests on a document rather than a
measurement: that ego-synth's ``d`` is **planar z** about the camera axis, which
is the data card's own statement (gotcha 4) and is what the models are scored
against. If it were euclidean range instead, every score would carry a
``1/cos(theta)`` error — 1.00 on axis and 1.74 at 55 deg, radial, and so not
absorbable by the per-frame affine.

It is worth naming because two things in this package *look* like checks of it
and are structurally incapable of being one: the rectified/fisheye depth
agreement cited in ``baselines`` (range is equally invariant under a co-axial
rectification) and ``verify_camera``'s sub-pixel reprojection (projection reads
only the ray's direction, which both readings share). Believing either one is
how this ends up unnoticed. Ticket 016 specifies the check that can fail, which
needs the source MPS points and therefore the box.

Layout
------
    camera.py     the Aria FISHEYE624 model, per take
    data.py       reading ego-synth: takes, clips, points, frames
    baselines.py  the two lens strategies
    metrics.py    per-point scoring, under each model's own alignment protocol
    split.py      the frozen frame manifest and its digest
    run.py        the driver
    report.py     tables

Importing this package puts the repo root on ``sys.path`` so that
``finetune.eval.metrics`` and ``finetune.eval.baselines.model_zoo`` — shared
infrastructure belonging to neither experiment — can be reached; the repo is not
installed.
"""
import os as _os
import sys as _sys

_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
