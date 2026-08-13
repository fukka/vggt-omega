# Copyright (c) 2026.
"""The Aria 214-1 RGB lens, described once.

Three modules in this repository describe the same physical camera, for three
different jobs, and each of them used to carry its own copy of these five
numbers and its own spelling of the storage rotation:

    VGGT-360-fisheye/utils/fisheye_cam.py   the FOV experiment's fisheye arm
    finetune/eval/baselines/aria_fisheye.py the baselines' ERP and cone masks
    finetune/data/rectify.py                the training loader and the FOV
                                            experiment's rectified arm

They now all read this module. Nothing here does geometry — the KB4 ray LUTs,
the cone limits and the rectifier's remap stay where their consumers are. What
lives here is only what was *duplicated*, which is the part where a divergence
is invisible: constants, and the one rule for reading them in a rotated frame.

Why that is worth a module of its own
-------------------------------------
``fisheye_cam.py`` used to say *"Constants and the KB4 math are vendored from
this repo's finetune/eval/baselines/aria_fisheye.py ... If you change the
calibration, change it in both places."* A seam written in prose is a seam that
does not hold, and this one did not: ``rectify.py`` rotated the principal point
as ``cx' = W - cy`` where the other two use ``cx' = (H-1) - cy``, so the two arms
of the ADT-FOV experiment described the same lens one pixel apart.

That is small — see :func:`intrinsics` for the derivation and the measured cost —
but nothing in the repository would have told you it was there. A one-pixel
disagreement between two arms whose whole purpose is to be comparable is exactly
the class of error a shared constant exists to prevent.

The self-containment argument, and why it does not apply
--------------------------------------------------------
``fisheye_cam.py``'s docstring justified the vendoring by keeping VGGT-360-fisheye
"a self-contained subproject". Its own entry point already contradicts that:
``main_adt.py`` puts the repository root on ``sys.path`` specifically so it can
import ``finetune.eval.metrics``, "the shared scoring protocol". The subproject
depends on the root for the definition of a metric; depending on it for the
definition of the lens is the same arrangement, and it is the arrangement that
stops the two descriptions drifting.

Provenance
----------
The calibration is Aria 214-1 at the 1408 stream resolution, validated to
<0.22 deg against ``projectaria_tools``. ``f, cx, cy`` are the same numbers
``projectaria_tools`` issue #322 reports for the binned RGB stream, which is what
``slambench/camera.py`` derives independently from the 2880 sensor calibration —
the two agree, and that agreement is the only cross-check these constants have.
"""
from __future__ import annotations

from typing import Tuple

#: The frame the calibration is expressed in: Aria's binned RGB stream.
NATIVE: float = 1408.0

#: Focal length and principal point, in pixels of the :data:`NATIVE` frame, as
#: the sensor stores them — i.e. *before* any of the rotations below.
F_NATIVE: float = 610.94
CX_NATIVE: float = 715.11
CY_NATIVE: float = 716.71

#: Kannala-Brandt radial coefficients, ``theta_d = theta*(1 + k1 th^2 + k2 th^4
#: + k3 th^6 + k4 th^8)``. Dimensionless, so invariant to both resize and
#: rotation — which is why only ``f, cx, cy`` move in :func:`intrinsics`.
KB4: Tuple[float, float, float, float] = (0.3852, -0.4442, 0.5591, -0.3254)


def intrinsics(H: int, W: int, rotated: bool = True
               ) -> Tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` in pixels of an ``H x W`` frame.

    ``rotated=True`` means the frame has had ADT's 270-degree-CCW rotation
    applied — Aria stores frames 90 degrees CW, and every loader in this repo
    undoes that with ``np.rot90(frame, k=3)`` before anything else touches it.

    **The rotation rule, derived rather than asserted.** ``np.rot90(m, 3)``
    sends source pixel ``(u, v)`` to ``(u', v') = ((H-1) - v, u)``; you can read
    that straight off numpy with a labelled 9x9 array. So::

        fx, fy  ->  fy, fx                 (the axes swap)
        cx      ->  (H - 1) - cy
        cy      ->  cx

    It is ``H - 1`` and not ``H`` because a pixel's centre sits at its integer
    coordinate, so the last column of an ``H`` wide frame is at ``H - 1``.
    ``rectify.py`` used ``H`` here, which put its principal point exactly 1.000 px
    to the right of the other two consumers' at every resolution. Correcting it
    moves the ADT-FOV rectified arm's per-bin AbsRel by 0.1-0.7% and its ``pen``
    by -0.4%, measured on the synthetic scene — real, well under the effects that
    benchmark reports, and not something anyone should have had to discover.

    Both scale factors are carried separately so a non-square frame is described
    rather than silently assumed away; every Aria frame in this repo is square.
    """
    sx, sy = W / NATIVE, H / NATIVE
    fx, fy = F_NATIVE * sx, F_NATIVE * sy
    cx, cy = CX_NATIVE * sx, CY_NATIVE * sy
    if rotated:
        fx, fy = fy, fx
        cx, cy = (H - 1) - cy, cx
    return fx, fy, cx, cy


def centered(H: int, W: int) -> Tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` with the principal point forced to the frame centre.

    For renders whose fisheye circle is centred and whose per-device principal
    point is unknown — the EgoExo4D ego renders, used qualitatively, where a few
    pixels of offset does not carry the argument.
    """
    f = F_NATIVE * (max(H, W) / NATIVE)
    return f, f, W / 2.0, H / 2.0
