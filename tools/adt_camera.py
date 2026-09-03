"""Build a KB4 camera from the calibration a sequence was actually recorded with.

WHY THIS EXISTS
---------------
`raytun3r.cameras.from_aria()` hard-codes one device's lens -- the Apartment
scene's M1292 -- and every experiment in this repo has used it because every
experiment has used that one scene. ADT's second scene, `LiteOffice`, is
recorded on a different device, and the two lenses are not interchangeable:

                       f (at 1408)     k1        k2        k3        k4
    Apartment M1292        610.94    0.3852   -0.4442    0.5591   -0.3254
    LiteOffice 61283       594.5     0.4215   -0.6047    0.5485   +0.5628*

    (*) k4 flips SIGN. The LiteOffice figure is from its raw-sensor
        FISHEYE624 and is not directly comparable term by term -- which is the
        point of fitting rather than reading coefficients across devices.

Feeding LiteOffice frames through the Apartment's lens would put a smooth
radial error into every number, and a smooth radial error is exactly what "the
model is worse at the rim" looks like. That is the class of bug that
invalidated #38 v1 and cost a four-row re-run, so the camera comes from the
sequence, not from a constant.

THE FIT, AND WHY IT IS A FIT
----------------------------
ADT's calibration is FISHEYE624 (six radial terms plus tangential and thin
prism). This repo's protocol, its `theta_max`, its incidence grids and H12/H15's
Jacobian fields are all written against KB4. The Apartment's KB4 of record is
itself a four-term fit of that device's FISHEYE624 -- so fitting LiteOffice the
same way is what makes the two scenes comparable. `--compare` checks the fitter
by refitting the Apartment and holding the result against the constant the
repo already trusts.

The tangential and thin-prism terms are dropped, as they are for the Apartment.
They are order 1e-4 here and they are not radially symmetric, so no radial model
can carry them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: The Apartment's KB4 of record, for `--compare` only. Never used to build a
#: camera here -- that is the whole point of this module.
_APARTMENT_KB4 = (0.3852, -0.4442, 0.5591, -0.3254)


def radial_fisheye624(theta: np.ndarray, params: Sequence[float]) -> np.ndarray:
    """``d(theta)`` for FISHEYE624, normalised by focal length.

    ``params`` is ``[f, cx, cy, k1..k6, p1, p2, s1..s4]``; only ``k1..k6`` are
    radial. Returns ``theta * (1 + k1 th^2 + ... + k6 th^12)``, i.e. image
    radius in units of ``f``, which is the same normalisation KB4 uses here.
    """
    k = np.asarray(params[3:9], dtype=np.float64)
    t2 = theta * theta
    poly = np.ones_like(theta)
    p = np.ones_like(theta)
    for ki in k:
        p = p * t2
        poly = poly + ki * p
    return theta * poly


def fit_kb4(params: Sequence[float], theta_max: float, n: int = 4001
            ) -> Tuple[float, float, float, float]:
    """Least-squares KB4 coefficients matching FISHEYE624 over ``[0, theta_max]``.

    Fitted on ``d(theta)/theta - 1`` against ``[th^2, th^4, th^6, th^8]``, which
    weights the residual by 1/theta and so does not let the outer angles -- where
    d is largest -- swamp the fit. The rim is where this project lives; a fit
    that is good at the rim and wrong on axis would be worse than useless.
    """
    th = np.linspace(1e-6, float(theta_max), n)
    d = radial_fisheye624(th, params)
    y = d / th - 1.0
    A = np.stack([th ** 2, th ** 4, th ** 6, th ** 8], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return tuple(float(c) for c in coef)


def camera_from_json(cam_json: Path, size: int, rotated: bool = False):
    """A `KannalaBrandt` at ``size x size`` from an extracted `camera.json`."""
    from raytun3r.cameras import KannalaBrandt
    from finetune.eval.baselines.aria_fisheye import usable_max_incidence

    c = json.loads(Path(cam_json).read_text())
    p = c["params"]
    w0, h0 = int(c["width"]), int(c["height"])
    f0, cx0, cy0 = float(p[0]), float(p[1]), float(p[2])

    # theta_max from the DEVICE's own frame, then the lens is rescaled. Doing it
    # the other way round would compute the imaged cone from a resized principal
    # point and quietly move the rim.
    k_native = fit_kb4(p, math.radians(70.0))
    tmax = float(usable_max_incidence(k_native, f0, f0, cx0, cy0, h0, w0))
    k = fit_kb4(p, tmax)

    sx, sy = size / w0, size / h0
    fx, fy = f0 * sx, f0 * sy
    cx, cy = cx0 * sx, cy0 * sy
    if rotated:
        fx, fy = fy, fx
        cx, cy = (size - 1) - cy, cx
    return KannalaBrandt(fx=fx, fy=fy, cx=cx, cy=cy, width=size, height=size,
                         k=k, theta_max=tmax)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-json", default=None)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--compare", action="store_true",
                   help="refit the Apartment and hold it against the constant")
    a = p.parse_args(argv)

    if a.compare:
        # The Apartment's own FISHEYE624, as the provider reports it at 1408.
        apt = [610.9412, 715.1147, 716.7148,
               0.38520, -0.44420, 0.55910, -0.32540, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print("NOTE: run with --camera-json on an extracted Apartment sequence "
              "for the real 15-vector; this built-in is the KB4 of record "
              "padded, so a perfect match here only checks the algebra.")
        got = fit_kb4(apt, math.radians(54.83))
        print("  refit  ", [round(x, 5) for x in got])
        print("  record ", list(_APARTMENT_KB4))
        print("  max |d|", max(abs(g - r) for g, r in zip(got, _APARTMENT_KB4)))
        return

    if not a.camera_json:
        raise SystemExit("pass --camera-json or --compare")
    cam = camera_from_json(Path(a.camera_json), a.size)
    print(f"size {a.size}  fx {cam.fx:.3f}  cx {cam.cx:.3f}  cy {cam.cy:.3f}")
    print(f"k    {[round(x, 5) for x in cam.k]}")
    print(f"theta_max {math.degrees(cam.theta_max):.3f} deg")


if __name__ == "__main__":
    main()
