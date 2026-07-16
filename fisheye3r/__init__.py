"""Reproduction of "Fisheye3R: Adapting Unified 3D Feed-Forward Foundation
Models to Fisheye Lenses" (arXiv:2603.28896) on the VGGT-Omega backbone.

The official repository (github.com/android-xr/fisheye3r) is an empty
placeholder as of 2026-07; this package re-implements the paper from its
description. See fisheye3r/README.md for the paper->code map and the list of
deliberate simplifications.
"""

from fisheye3r.distortion import KannalaBrandtCamera, sample_kb_cameras, distort_images, undistort_dense
from fisheye3r.model import Fisheye3R

__all__ = [
    "KannalaBrandtCamera",
    "sample_kb_cameras",
    "distort_images",
    "undistort_dense",
    "Fisheye3R",
]
