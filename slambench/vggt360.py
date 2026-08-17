# Copyright (c) 2026.
"""VGGT-360-fisheye on ego-synth: the lens adapter and the per-take map cache.

``VGGT-360-fisheye`` was written against Aria's KB4 model on ADT. ego-synth ships
a per-take **FISHEYE624** (``FisheyeRadTanThinPrism``) calibration instead, which
is not the same lens description and not even the same *shape* of one: KB4 is
radially symmetric, FISHEYE624 adds tangential and thin-prism terms that are not.

There were two ways to bridge that and only one of them is honest.

*Fit a KB4 to the take's FISHEYE624.* Cheap, and the residual would sit inside
the fisheye-to-tangent warp, which is the operation the method consists of. An
error there is not noise on the result, it is a lens error dressed as a model
result — and this repository has already paid once for a geometry mistake that
looked like a mediocre model (CONTEXT.md). The size of that residual is the
non-radial part of the model, and it is **measured** rather than argued about:
``tests/test_vggt360.py`` puts it at about 1.4 px at the 896 frame on the suite's
test calibration, against the 0.29 px ``verify_camera`` accepts a camera at. It
is calibration-dependent and a given take may be milder; what it is not is known
to be negligible.

*Give the pipeline the real lens.* ``utils/fisheye_views.persp_maps`` and
``utils/fisheye_fusion.fuse_views_to_fisheye`` take a ``project`` hook and a
``ray_lut`` — the only two places KB4 is touched — so the take's own camera can
answer both. That is what :class:`Fisheye624Lens` does, and nothing is fitted,
approximated or assumed anywhere in this file.

Cost, and why the cache exists
------------------------------
The layout is thirteen views at 512 px rendered at ``supersample=3``, which is
about 21 million rays per frame. FISHEYE624 has no closed-form projection: it is
a degree-13 radial polynomial plus tangential and prism terms. Paying that per
frame would dominate the run.

It does not have to be paid per frame. The maps depend on the lens and the aim
and never on the pixels, and a take has one lens — so they are built once per
``(take, view)`` and reused across every frame of every clip in it. The nine base
views hit the cache on every frame after the first; the adaptive neighbours are
drawn from a small candidate set and converge into it within a few frames.

The cone
--------
The layout must be sized against the angle the lens actually images, and a
FISHEYE624 calibration carries no valid radius. :meth:`Fisheye624Lens.theta_max`
derives it (``camera.max_imaged_theta``) rather than assuming Aria's 54.83 deg,
and :func:`layout_report` prints how far the ring reaches into it — because
"the 60 deg model" is a layout designed for one cone, and running it on another
lens without saying so is how a confound gets published.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from slambench import _REPO  # noqa: F401  (import registers sys.path)

from slambench.camera import Fisheye624  # noqa: E402

#: Views whose maps are held per take. Thirteen is the layout's own cap
#: (``max_views``); the adaptive neighbours draw from a small candidate set, so
#: a little headroom holds them all without the cache becoming a leak.
MAX_CACHED_VIEWS = 64


class Fisheye624Lens:
    """A take's FISHEYE624 camera, wearing the surface the pipeline expects.

    ``utils/pipeline`` asks a lens for three things — ``H``, ``W`` and
    ``theta_max()`` — and takes its geometry through the ``project`` / ``ray_lut``
    hooks. This supplies all five, and no KB4 coefficient exists anywhere on it:
    an attribute error is the intended outcome if some future code path reaches
    for ``.k``, rather than a plausible number from a lens this is not.
    """

    def __init__(self, cam: Fisheye624, theta_max_deg: Optional[float] = None):
        self.cam = cam
        self.H = int(cam.height)
        self.W = int(cam.width)
        self._theta_max = (math.radians(theta_max_deg)
                           if theta_max_deg is not None
                           else float(cam.max_imaged_theta()))
        self._ray_lut: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._maps: Dict[tuple, tuple] = {}

    # -- the surface ``utils/pipeline`` reads ------------------------------- #
    def theta_max(self) -> float:
        """Usable half-FOV in radians — derived, never assumed. See the module."""
        return self._theta_max

    @property
    def theta_max_deg(self) -> float:
        return math.degrees(self._theta_max)

    def project(self, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ray field ``(..., 3)`` -> pixel maps, through the take's own lens."""
        return self.cam.project_bulk(vec)

    def ray_lut(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(rays[H,W,3], inside_cone[H,W])`` for this frame grid, memoised.

        The fisheye analogue of the ERP ray grid: fusion iterates over *output*
        rays, so this is where the lens must be inverted. FISHEYE624's inverse is
        two nested fixed points over the whole 896x896 grid — worth doing once
        per take and never again.
        """
        if self._ray_lut is None:
            us, vs = np.meshgrid(np.arange(self.W, dtype=np.float64),
                                 np.arange(self.H, dtype=np.float64))
            rays = self.cam.unproject_bulk(us, vs).astype(np.float32)
            theta = np.arccos(np.clip(rays[..., 2], -1.0, 1.0))
            cone = theta <= (self._theta_max - 1e-6)
            self._ray_lut = (rays, cone.astype(bool))
        return self._ray_lut

    def cos_theta(self) -> np.ndarray:
        """Per-pixel ``cos(theta)``: euclidean range <-> planar z on this grid."""
        return self.ray_lut()[0][..., 2].astype(np.float32)

    # -- the map cache ------------------------------------------------------ #
    def maps_for(self, azimuth: float, tilt: float, fov: float,
                 size: int) -> tuple:
        """``persp_maps`` for one view, built once per take. See the module."""
        from utils.fisheye_views import persp_maps

        key = (round(float(azimuth), 4), round(float(tilt), 4),
               round(float(fov), 4), int(size))
        m = self._maps.get(key)
        if m is None:
            if len(self._maps) >= MAX_CACHED_VIEWS:
                # The layout is bounded, so this cannot happen with the shipped
                # configuration; say so rather than growing without limit if a
                # future sweep varies the aim per frame.
                raise RuntimeError(
                    f"the view-map cache holds {len(self._maps)} entries for one "
                    f"take, past the {MAX_CACHED_VIEWS} a bounded layout needs. "
                    f"Something is varying the view aim per frame.")
            m = persp_maps(self, azimuth, tilt, fov, height=size, width=size,
                           theta_max=self._theta_max, project=self.project)
            self._maps[key] = m
        return m


def layout_report(cfg, lens: Fisheye624Lens) -> str:
    """One line saying how the ADT-designed layout sits on *this* lens.

    The 60 deg / tilt-26 / 8-ring layout is sized by the rule
    ``tilt + fov/2 >~ theta_max`` against Aria's 54.83 deg usable cone. ego-synth
    is Aria too, but a per-take calibration is not the nominal one, and a lens
    whose cone is wider leaves an unmeasured annulus at the rim covered by the
    centre view's corners alone — which would read as the method degrading at
    eccentricity when it is the layout not reaching. Printed per take so it is in
    the log beside the numbers rather than inferred later.
    """
    reach = cfg.ring_tilt + cfg.fov / 2.0
    frac = cfg.covers_cone(lens.theta_max_deg)
    verdict = ("ring reaches the rim" if frac >= 0.98 else
               f"ring stops {lens.theta_max_deg - reach:.1f} deg short of it")
    return (f"lens cone {lens.theta_max_deg:.2f} deg | layout reaches "
            f"{reach:.1f} deg ({frac:.0%}) | {verdict}")
