# Copyright (c) 2026.
"""The boundary between the two experiments, enforced rather than intended.

`slambench/` and `fovbench/` answer different questions against different ground
truth under different protocols. They drifted together once already — the SLAM
data was implemented *inside* the FOV experiment, and by the time that was caught
`fovbench/geometry.py` carried a function kept alive only because the SLAM path
mirrored it, and `fovbench/report.py` branched on the dataset in eight places.

A comment saying "keep these apart" would not have caught that. This does.
"""
from __future__ import annotations

import ast
import os

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN = "fovbench"


def _module_files():
    for root, _dirs, files in os.walk(PKG):
        if "__pycache__" in root:
            continue
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(root, f)


def _imported_names(path: str):
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                yield node.module


def test_slambench_does_not_import_fovbench():
    """The SLAM evaluation must not depend on the FOV experiment, at all.

    Not "should not" — a shared import is how the last entanglement started, and
    it is invisible until someone changes the other package and this one's
    numbers move.
    """
    offenders = {}
    for path in _module_files():
        bad = sorted({m for m in _imported_names(path)
                      if m == FORBIDDEN or m.startswith(FORBIDDEN + ".")})
        if bad:
            offenders[os.path.relpath(path, PKG)] = bad
    assert not offenders, (
        f"slambench imports {FORBIDDEN}: {offenders}. These are two experiments "
        f"with different ground truth and different protocols; share code only "
        f"through finetune/eval/, which belongs to neither.")


def test_the_shared_infrastructure_is_the_only_cross_package_dependency():
    """What slambench is allowed to reach for, stated positively.

    ``finetune/eval/metrics.py`` and ``finetune/eval/baselines/model_zoo.py`` are
    repository-wide and predate both experiments; depending on them is the
    intended arrangement, and pinning it here means a new cross-package import
    has to be argued for rather than merely added.
    """
    # Each of these is allowed for a stated reason, so that a new one has to be
    # argued for rather than merely added:
    #   finetune.*          repository-wide eval code, older than both experiments
    #   projectaria_tools   the REFERENCE implementation of Aria's FISHEYE624
    #                       model; optional, and preferred over the fallback in
    #                       camera.py wherever a sub-pixel answer matters
    #   scipy               nearest-neighbour search, for the point-cloud
    #                       statistic verify_camera.py is built on
    allowed_prefixes = ("finetune.", "slambench", "numpy", "cv2", "torch",
                        "projectaria_tools", "scipy")
    stdlib_ok = {"os", "sys", "json", "math", "glob", "ast", "csv", "io",
                 "time", "struct", "zlib", "argparse", "hashlib", "dataclasses",
                 "typing", "collections", "subprocess", "pytest", "__future__"}
    unexpected = {}
    for path in _module_files():
        for m in _imported_names(path):
            top = m.split(".")[0]
            if top in stdlib_ok or m.startswith(allowed_prefixes):
                continue
            unexpected.setdefault(os.path.relpath(path, PKG), set()).add(m)
    assert not unexpected, f"unexpected cross-package imports: {unexpected}"


def test_no_eccentricity_vocabulary_leaked_into_the_scoring_path():
    """This experiment has no field-of-view axis, and its code should not imply
    one. ``pen``, ``drift`` and incidence-angle binning are the FOV experiment's
    apparatus; a reader who finds them here would reasonably conclude the two
    measure the same thing."""
    banned = ("anchored_ratio", "raw_scale_ratio", "theta_edges", "radius_edges")
    hits = {}
    for path in _module_files():
        if os.path.basename(path).startswith("test_"):
            continue
        with open(path) as fh:
            src = fh.read()
        found = [b for b in banned if b in src]
        if found:
            hits[os.path.relpath(path, PKG)] = found
    assert not hits, (f"FOV-experiment vocabulary in the SLAM evaluation: {hits}")
