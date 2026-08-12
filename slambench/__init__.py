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

Nothing here imports `fovbench`, and
``tests/test_separation.py::test_slambench_does_not_import_fovbench`` enforces
that mechanically rather than by good intentions.

The shape of the evaluation
---------------------------
Ground truth is used in the form the producer captured it: the SLAM points as
projected into the **raw fisheye** frame. The harness performs no rectification,
no warping and no depth-convention conversion — it reads a frame, hands it to a
baseline, receives one depth per ground-truth point, and scores that.

    harness    raw fisheye frame + native GT points
                 -> baseline.predict(frame) -> depth per point
                 -> metrics

**The lens is the baseline's business.** Two baselines differ only in what they
do behind that one call:

    raw           feed the fisheye frame to the model as it is
    rect_derect   rectify, run the model, map its depth back onto the fisheye
                  points -- all inside the baseline

which is what makes "is it worth rectifying first, once you have paid to map
back?" a question this harness can answer rather than presuppose.

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
