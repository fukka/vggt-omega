# Copyright (c) 2026.
"""The boundary between the two experiments, enforced from both sides.

``fovbench/`` and ``slambench/`` answer different questions against different
ground truth under different protocols. They drifted together once already — the
SLAM data was implemented *inside* the FOV experiment, and by the time that was
caught ``fovbench/geometry.py`` carried functions kept alive only because the
SLAM path mirrored them, and ``fovbench/report.py`` branched on the dataset.

That was fixed by half. ``slambench/data.py`` was written as a clean ego-synth
reader; ``fovbench/datasets_egosynth.py`` stayed. The check that was supposed to
prevent exactly this could not see it, because it asked the wrong question:

    it asked    does slambench IMPORT fovbench?
    it needed   does either package READ the other's dataset?

An import graph cannot see a second copy. ``fovbench`` never imported
``slambench`` — it reimplemented it, 700 lines of it, and the guard stayed green
for months.

So the invariant here is **dataset ownership**, which catches a duplicate as well
as a dependency:

    fovbench     reads ADT, and nothing else
    slambench    reads ego-synth 5B, and nothing else
    both         may share finetune/eval/, which belongs to neither

and this file lives at the repository root rather than inside one of the two
things it keeps apart, so it is nobody's to quietly relax.

    python -m pytest tests fovbench/tests slambench/tests -q
"""
from __future__ import annotations

import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Names that only appear in code that is *reading* a dataset — array keys,
#: directory names, CLI flags. Deliberately not prose: both packages discuss
#: each other in docstrings, and they should be able to.
ADT_MARKERS = ("depth_npy", "videos_synthetic", "videos_rgb", "adt_root",
               "Apartment_release")
EGOSYNTH_MARKERS = ("sparse_depth", "inv_dist_std", "egosynth", "EGOSYNTH",
                    "nymeria", "egoexo4d", "ego-synth")

#: package -> (the dataset it owns, the markers it must not carry)
OWNERSHIP = {
    "fovbench": ("ADT", EGOSYNTH_MARKERS),
    "slambench": ("ego-synth 5B", ADT_MARKERS),
}


def _sources(pkg: str):
    """Every non-test module of a package."""
    base = os.path.join(ROOT, pkg)
    for root, _dirs, files in os.walk(base):
        if "__pycache__" in root or os.path.basename(root) == "tests":
            continue
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(root, f)


def _docstring_nodes(tree: ast.AST):
    """The Constant nodes that are docstrings, which are prose and not code."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, (ast.Str, ast.Constant))):
            out.add(id(body[0].value))
    return out


def code_text(path: str) -> str:
    """The module's *code*: identifiers, attributes, imports and real string
    literals, with docstrings and comments dropped.

    The distinction is the whole point of scanning at all. ``slambench/data.py``
    says "unlike this repo's ADT loaders" in a docstring and should keep saying
    it; what must not appear is ``depth_npy`` in something it opens.
    """
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    skip = _docstring_nodes(tree)
    parts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parts.append(node.name)
        elif isinstance(node, ast.arg):
            parts.append(node.arg)
        elif isinstance(node, ast.Import):
            parts.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                parts.append(node.module)
            parts.extend(a.name for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip:
                parts.append(node.value)
    return "\n".join(parts)


@pytest.mark.parametrize("pkg", sorted(OWNERSHIP))
def test_each_experiment_reads_only_its_own_ground_truth(pkg):
    """The invariant the import check could not express.

    A package that grows a reader for the other's dataset has started the second
    implementation, whether or not it ever writes an import — and a second
    implementation of a data card is two places for the next correction to land
    and one place for it to be forgotten.
    """
    owns, forbidden = OWNERSHIP[pkg]
    offenders = {}
    for path in _sources(pkg):
        hits = sorted({m for m in forbidden if m in code_text(path)})
        if hits:
            offenders[os.path.relpath(path, ROOT)] = hits
    assert not offenders, (
        f"{pkg} owns {owns} and nothing else, but these modules carry another "
        f"dataset's vocabulary in code: {offenders}. If that dataset needs "
        f"evaluating, the package that owns it already reads it — share through "
        f"finetune/eval/, which belongs to neither experiment.")


@pytest.mark.parametrize("pkg,other", [("fovbench", "slambench"),
                                       ("slambench", "fovbench")])
def test_neither_experiment_imports_the_other(pkg, other):
    """Now symmetric. It was not, and the asymmetry was the whole problem: the
    unguarded direction is where the duplicate grew."""
    offenders = {}
    for path in _sources(pkg):
        names = []
        with open(path) as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names.append(node.module)
        bad = sorted({n for n in names
                      if n == other or n.startswith(other + ".")})
        if bad:
            offenders[os.path.relpath(path, ROOT)] = bad
    assert not offenders, (
        f"{pkg} imports {other}: {offenders}. Two experiments with different "
        f"ground truth and different protocols; share only through "
        f"finetune/eval/.")


def test_the_shared_infrastructure_is_the_only_cross_package_dependency():
    """What each package is allowed to reach for, stated positively, so a new
    cross-package import has to be argued for rather than merely added."""
    allowed = ("finetune.", "fovbench", "slambench", "numpy", "cv2", "torch",
               "matplotlib", "PIL",
               # the REFERENCE implementation of Aria's FISHEYE624 model, and
               # the nearest-neighbour search verify_camera's statistic needs
               "projectaria_tools", "scipy",
               # the KB4 camera and view rendering the FOV experiment is built
               # on; vendored, and the subject of its own review candidate
               "utils.",
               # this repo's ADT loader -- fovbench reading its OWN dataset
               "datasets.")
    stdlib_ok = {"os", "sys", "json", "math", "glob", "ast", "csv", "io", "re",
                 "time", "struct", "zlib", "argparse", "hashlib", "dataclasses",
                 "typing", "collections", "subprocess", "threading", "itertools",
                 "contextlib", "functools", "warnings", "concurrent",
                 "pytest", "__future__"}
    unexpected = {}
    for pkg in sorted(OWNERSHIP):
        for path in _sources(pkg):
            with open(path) as fh:
                tree = ast.parse(fh.read(), filename=path)
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    mods = [node.module]
                for m in mods:
                    if m.split(".")[0] in stdlib_ok or m.startswith(allowed):
                        continue
                    unexpected.setdefault(os.path.relpath(path, ROOT), set()).add(m)
    assert not unexpected, f"unexpected cross-package imports: {unexpected}"


def test_no_eccentricity_vocabulary_leaked_into_the_slam_evaluation():
    """The SLAM evaluation has no field-of-view axis and its code should not
    imply one. ``pen``, ``drift`` and incidence-angle binning are the FOV
    experiment's apparatus; a reader who found them here would reasonably
    conclude the two measure the same thing."""
    banned = ("anchored_ratio", "raw_scale_ratio", "theta_edges", "radius_edges")
    hits = {}
    for path in _sources("slambench"):
        with open(path) as fh:
            src = fh.read()
        found = [b for b in banned if b in src]
        if found:
            hits[os.path.relpath(path, ROOT)] = found
    assert not hits, f"FOV-experiment vocabulary in the SLAM evaluation: {hits}"
