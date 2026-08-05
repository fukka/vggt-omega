"""RayTun3R -- online camera adaptation in 3D foundation models (reproduction).

Re-implementation of Sinitsyn, Araslanov and Cremers, "RayTun3R: Online Camera
Adaptation in 3D Foundation Models" (arXiv:2607.02711), on the backbones this
repo carries. No official code exists as of 2026-08, so everything here is
reconstructed from the paper; see ``README.md`` for the paper-to-code map and
the list of places where an interpretation was required.
"""

from .adapter import RadialAngularPE, RadialRoPE, RayTun3RAdapter
from .backbones import BACKBONES, Backbone, Prediction, build_backbone
from .cameras import (Camera, EUCM, KannalaBrandt, OpenCVFisheye, Pinhole,
                      from_aria, from_opencv_fisheye, from_scannetpp)
from .losses import LossWeights, total_loss
from .matching import build_matcher, relative_pose_magsac

__all__ = [
    "RayTun3RAdapter", "RadialAngularPE", "RadialRoPE",
    "Backbone", "Prediction", "build_backbone", "BACKBONES",
    "Camera", "Pinhole", "KannalaBrandt", "EUCM", "OpenCVFisheye",
    "from_aria", "from_opencv_fisheye", "from_scannetpp",
    "LossWeights", "total_loss",
    "build_matcher", "relative_pose_magsac",
]

__paper__ = "arXiv:2607.02711"
