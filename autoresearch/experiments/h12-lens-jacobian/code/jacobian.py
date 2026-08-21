"""The local Jacobian of a radial lens map, as a per-pixel conditioning field.

WHY THIS EXISTS
---------------
Three rim-targeted interventions in this repo have now failed against their own
controls: H5's rim-weighted losses lose to plain LoRA (-80.6% vs -83.5% near-rim
on seq136), H6's rim-restricted KV loses to the all-token control (-52.2% vs
-75.9%), and the centre/rim dual-expert MoE was killed by H7+F2+F4. Meanwhile
the interventions that DO work are global lens operations: `rect_derect` beats
native fisheye on slambench, and plain LoRA beats everything on ADT.

Read together those say the rim deficit is not a region-shaped problem. It is a
symptom of a global lens-prior mismatch: the backbone's features were learned on
near-pinhole image statistics, and a fisheye's local image formation departs
from that prior by an amount that grows with incidence angle. The departure is
loudest at the rim, which is why it LOOKS like a rim problem, but adding
capacity where the symptom is loudest has not helped three times running.

So instead of telling the network WHERE to try harder, this hands it the
geometry it currently has to infer from content: at every pixel, how is the
image locally stretched, and how anisotropically?

THE QUANTITY
------------
A radial lens maps a ray at incidence ``theta`` and azimuth ``phi`` to the image
point ``(r(theta) cos phi, r(theta) sin phi)``. In the local orthonormal frame on
the unit sphere the map is diagonal, with two magnifications:

    radial (meridional)   m_rad = r'(theta)
    tangential (sagittal) m_tan = r(theta) / sin(theta)

Both tend to the on-axis focal length as ``theta -> 0``, so normalising by it
removes the focal length entirely and leaves a pure *shape* descriptor of the
lens. Writing ``r = f * d(theta)``:

    log_area  = log( d'(theta) * d(theta) / sin(theta) )     area scale
    log_aniso = log( d'(theta) * sin(theta) / d(theta) )     stretch ratio

Both are 0 on axis by construction, for every lens. That is the point: the field
says "how far from the on-axis (locally similar) case is this pixel", which is
exactly the quantity the pinhole-trained prior is blind to.

`f` cancelling matters -- it means the conditioning does not encode the sensor,
only the lens shape, so a model conditioned on it has some hope of transferring
to a lens it never saw. That is the claim this experiment exists to test, and
its control is the same network conditioned on a SHUFFLED field.

Pure numpy on purpose: no torch, so it is verifiable on a machine with no
deep-learning stack (this Mac currently has none -- see the 2026-08-22 note in
docs/handoff/POLICY.md).
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

__all__ = ["kb4_d", "kb4_dprime", "pinhole_d", "pinhole_dprime",
           "equidistant_d", "equidistant_dprime",
           "log_area_aniso", "theta_from_radius", "jacobian_field"]

# ---------------------------------------------------------------- lens shapes

def kb4_d(theta: np.ndarray, k: Tuple[float, float, float, float]) -> np.ndarray:
    """Kannala-Brandt radial polynomial, normalised: r(theta) = f * d(theta)."""
    k1, k2, k3, k4 = k
    t2 = theta * theta
    return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))


def kb4_dprime(theta: np.ndarray, k: Tuple[float, float, float, float]) -> np.ndarray:
    """d/dtheta of :func:`kb4_d`, in closed form."""
    k1, k2, k3, k4 = k
    t2 = theta * theta
    return 1.0 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))


def pinhole_d(theta):    return np.tan(theta)
def pinhole_dprime(theta): return 1.0 / np.cos(theta) ** 2
def equidistant_d(theta):  return np.asarray(theta, dtype=float)
def equidistant_dprime(theta): return np.ones_like(np.asarray(theta, dtype=float))


# ---------------------------------------------------------------- the field

def log_area_aniso(theta: np.ndarray,
                   d: Callable[[np.ndarray], np.ndarray],
                   dprime: Callable[[np.ndarray], np.ndarray],
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """``(log_area, log_aniso)`` for a radial lens shape ``d``.

    Both are exactly 0 at ``theta == 0``. The on-axis limit is taken explicitly
    rather than left to floating point: ``d(theta)/sin(theta)`` is 0/0 there and
    evaluating it naively gives a NaN in the one pixel that is definitionally
    the reference point.
    """
    theta = np.asarray(theta, dtype=float)
    small = theta < 1e-8
    th = np.where(small, 1.0, theta)          # dummy, masked out below
    dv, dp = d(th), dprime(th)
    m_tan = dv / np.sin(th)
    log_area = np.log(np.maximum(dp * m_tan, 1e-300))
    log_aniso = np.log(np.maximum(dp / np.maximum(m_tan, 1e-300), 1e-300))
    return np.where(small, 0.0, log_area), np.where(small, 0.0, log_aniso)


def theta_from_radius(r_norm: np.ndarray,
                      d: Callable[[np.ndarray], np.ndarray],
                      theta_max: float,
                      n: int = 200_001) -> np.ndarray:
    """Invert ``r = d(theta)`` numerically on ``[0, theta_max]``.

    Monotonicity is asserted rather than assumed: a KB4 fit can turn over inside
    the imaged field, and past the turnover the inverse is not a function at all.
    Interpolating through a turnover silently returns the wrong angle, which is
    the kind of error that looks like a model being bad at the rim.
    """
    grid = np.linspace(0.0, float(theta_max), n)
    rv = d(grid)
    if not np.all(np.diff(rv) > 0):
        bad = int(np.argmax(np.diff(rv) <= 0))
        raise ValueError(
            f"lens radial map is not monotone on [0, {theta_max:.4f}] -- it "
            f"turns over at theta={grid[bad]:.4f} rad. Inverting past a "
            f"turnover returns a plausible wrong angle.")
    return np.interp(np.asarray(r_norm, dtype=float), rv, grid)


def jacobian_field(h: int, w: int, fx: float, fy: float, cx: float, cy: float,
                   d: Callable[[np.ndarray], np.ndarray],
                   dprime: Callable[[np.ndarray], np.ndarray],
                   theta_max: float) -> np.ndarray:
    """Per-pixel ``(log_area, log_aniso, theta/theta_max)``, shape ``(h, w, 3)``.

    Pixels whose ray exceeds ``theta_max`` are outside the imaged cone; their
    third channel saturates at 1 and the first two hold the value at
    ``theta_max``. They are not silently extrapolated -- extrapolating a KB4 fit
    past the field it was fitted on is how a lens description drifts from the
    physical lens (see docs/research/scannetpp-camera-reference.md).
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    rx, ry = (xs - cx) / fx, (ys - cy) / fy
    r = np.hypot(rx, ry)
    r_max = float(d(np.array([theta_max]))[0])
    theta = theta_from_radius(np.minimum(r, r_max), d, theta_max)
    la, ln = log_area_aniso(theta, d, dprime)
    return np.stack([la, ln, theta / theta_max], axis=-1)
