# Copyright (c) 2026.
"""The ADT-FOV test: how monocular depth degrades with position in a wide FOV.

Four vanilla models (VGGT, VGGT-Omega, Depth-Anything V2, Depth-Anything 3) on
ADT's Aria fisheye, scored on rectified perspective *and* raw fisheye pixels,
from synthetic *and* real input, with the error resolved by where in the field of
view it happened.

``README.md`` is the protocol; ``run.py`` is the entry point.

Importing this package puts the repo root and ``VGGT-360-fisheye/`` on
``sys.path``. Both are needed and neither is importable by name: the repo is not
installed, and ``VGGT-360-fisheye`` has a hyphen, so its ``utils`` and
``datasets`` packages can only be reached by path. Doing it once here keeps the
bootstrap out of every module (``checks/depth_probe.py`` carries its own copy for
the same reason).
"""
import os as _os
import sys as _sys

_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_REPO, _os.path.join(_REPO, "VGGT-360-fisheye")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
