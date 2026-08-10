# Copyright (c) 2026.
"""Put the repo root on ``sys.path`` so ``import fovbench`` works.

Needed before the package's own bootstrap can run, and so ``pytest fovbench/tests``
works from anywhere without the repo being installed. Importing ``fovbench`` then
adds ``VGGT-360-fisheye/`` (see ``fovbench/__init__.py``).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
