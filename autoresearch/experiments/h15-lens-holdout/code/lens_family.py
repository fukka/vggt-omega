"""A family of radial lenses over ONE fixed cone, and the warps between them.

WHY THIS EXISTS
---------------
H12 asked whether conditioning a backbone on the local lens-Jacobian field
beats conditioning it on a position-SHUFFLED copy of that field, and answered
no. But the pilot trained and tested on a SINGLE lens, and on a single lens the
two arms are information-equivalent by construction:

    with one lens the field is a fixed function of token position, so a fixed
    permutation of it is an equally learnable fixed function of token position.

Nothing in that setting can separate "the network used the geometry" from "the
network memorised a per-position modulation". The margin H12 measured (~5%
relative, one seed) is what that degeneracy looks like. The kill criterion was
pre-registered and fired honestly; the setting could not decide it.

H12's actual claim was never about one lens. `log_area` and `log_aniso` are
normalised so the focal length CANCELS -- the field describes the LENS, not the
sensor -- and the whole point of that was transfer to a lens the model never
saw. This module builds the setting where that claim is decidable: many lenses
at training time, and a held-out lens at test time, where a shuffled or
mismatched field carries no usable information but the real one still does.

THE CONSTRUCTION, AND WHY IT HAS NO VOID
----------------------------------------
Every lens in the family images EXACTLY the same cone as the source camera and
fills exactly the same disc:

    f = R_disc / d(theta_max)

so the map is a pure radial re-distribution of the same rays. Consequences that
matter more than they look:

* **No void.** Every pixel inside the disc has a ray inside the source cone, so
  nothing is missing and nothing is filled in. Warping to a WIDER lens would
  leave a hole (`docs/research/dataset-scope-2026-08.md` sec. 3.2 measures 3.3%
  median / 21.6% worst on the ScanNet++ route); warping to a NARROWER one would
  throw the rim away, which is the region under study.
* **Planar z is invariant.** Same ray in both cameras, and z is a function of
  the ray and the world point alone. GT is resampled, never converted -- the
  rule `dataset-scope-2026-08.md` sec. 3.4(2) states and the error class that
  invalidated #38 v1 in the other direction.
* **The augmentation is exactly the quantity being conditioned on.** The field
  (log_area, log_aniso) IS the derivative of this re-distribution. Nothing else
  about the image changes: same content, same cone, same rays.

Resampling density does differ per lens -- a stereographic warp upsamples the
rim, an orthographic one compresses it -- but every arm sees the identical
warped images, so it cannot confound the comparison BETWEEN arms.

Torch for the cameras, numpy for the field (shared with H12's `jacobian.py`,
which is pure numpy on purpose).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "h12-lens-jacobian" / "code"))

from raytun3r.cameras import Camera, pixel_grid  # noqa: E402
import jacobian as J  # noqa: E402

__all__ = ["SHAPES", "RadialShapeCamera", "make_lens", "lens_family",
           "grid_between", "warp", "token_field", "TRAIN_LENSES",
           "HELDOUT_LENSES"]

OUT_OF_RANGE = -2.0


# ---------------------------------------------------------------- lens shapes
# Closed-form d and d' for each shape, in BOTH torch (for the camera) and numpy
# (for the Jacobian field). One definition per shape, used by both, so the
# camera the image is warped through and the field the network is shown can
# never describe different lenses -- which would silently make the treatment
# arm a mismatched arm.

def _kb4(k):
    def d(t, xp):
        t2 = t * t
        return t * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))

    def dp(t, xp):
        t2 = t * t
        return 1.0 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])))
    return d, dp


#: name -> (d(theta, xp), d'(theta, xp)); ``xp`` is ``torch`` or ``np`` so one
#: expression serves both.
SHAPES: Dict[str, Tuple[Callable, Callable]] = {
    "equidistant": (lambda t, xp: t,
                    lambda t, xp: xp.ones_like(t)),
    "equisolid": (lambda t, xp: 2.0 * xp.sin(t / 2),
                  lambda t, xp: xp.cos(t / 2)),
    "stereographic": (lambda t, xp: 2.0 * xp.tan(t / 2),
                      lambda t, xp: 1.0 / xp.cos(t / 2) ** 2),
    "orthographic": (lambda t, xp: xp.sin(t),
                     lambda t, xp: xp.cos(t)),
    "rectilinear": (lambda t, xp: xp.tan(t),
                    lambda t, xp: 1.0 / xp.cos(t) ** 2),
}

#: Aria's own KB4 and two rescalings of it. Rescaling the coefficients moves the
#: lens along the axis the field measures while keeping it a plausible KB4 fit,
#: which is what makes the family a family rather than a list of textbook
#: projections.
_ARIA_K = (0.3852, -0.4442, 0.5591, -0.3254)
for _s in (0.5, 1.0, 1.5):
    _name = "aria_kb4" if _s == 1.0 else f"kb4x{_s:g}"
    SHAPES[_name] = _kb4(tuple(c * _s for c in _ARIA_K))

#: The split. Held-out lenses are chosen to be INTERPOLATIVE in field space --
#: their (log_area, log_aniso) at the rim sit inside the training family's range
#: -- so the test is transfer, not extrapolation. `test_lens_family.py` asserts
#: that rather than trusting this comment.
TRAIN_LENSES: Tuple[str, ...] = ("aria_kb4", "equidistant", "orthographic",
                                 "rectilinear", "kb4x0.5", "kb4x1.5")
HELDOUT_LENSES: Tuple[str, ...] = ("stereographic", "equisolid")


# -------------------------------------------------------------------- camera

@dataclass
class RadialShapeCamera(Camera):
    """A central radial camera for any shape in :data:`SHAPES`.

    ``theta_of_r`` is bisection rather than a closed form: the family has to
    accept an arbitrary monotone ``d``, and a per-shape inverse is one more
    place for a shape and its inverse to disagree. 40 halvings on a bracket
    that is verified monotone at construction is exact to ~1e-12 rad and is
    computed once per (H, W) by ``Camera.ray_grid``'s cache.
    """

    shape: str = "equidistant"
    #: upper bracket for the inverse; set by :func:`make_lens`, never guessed
    theta_bracket: float = math.pi / 2

    def _d(self, t: Tensor) -> Tensor:
        return SHAPES[self.shape][0](t, torch)

    def r_of_theta(self, theta: Tensor) -> Tensor:
        return self._d(theta.clamp(0.0, self.theta_bracket))

    def theta_of_r(self, r: Tensor) -> Tensor:
        lo = torch.zeros_like(r)
        hi = torch.full_like(r, self.theta_bracket)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            go_up = self._d(mid) < r
            lo = torch.where(go_up, mid, lo)
            hi = torch.where(go_up, hi, mid)
        return 0.5 * (lo + hi)

    def _params(self) -> dict:
        return {**super()._params(), "shape": self.shape,
                "theta_bracket": self.theta_bracket}


def _first_turnover(shape: str, hi_cap: float, n: int = 20001) -> float:
    """Largest angle below ``hi_cap`` on which ``d`` is strictly increasing.

    A KB4 fit read past its turnover stops describing the physical lens
    (`docs/research/scannetpp-camera-reference.md`), and inverting past one
    returns a plausible WRONG angle -- which reads downstream as "the model is
    bad at the rim", the exact conclusion this project is trying to measure.
    """
    ts = np.linspace(1e-6, hi_cap, n)
    dv = SHAPES[shape][0](ts, np)
    bad = np.nonzero(np.diff(dv) <= 0)[0]
    return float(ts[bad[0]]) if len(bad) else hi_cap


def make_lens(shape: str, size: int, theta_max: float,
              margin_px: float = 1.0) -> RadialShapeCamera:
    """A lens of the given ``shape`` whose cone exactly fills the disc.

    ``f = R_disc / d(theta_max)``: the focal length is a CONSEQUENCE of the
    cone and the frame, not a free parameter. That is what makes every lens in
    the family image the same rays into the same pixels-worth of image, so a
    warp between any two of them is a pure radial re-distribution.
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown lens shape {shape!r}; have {sorted(SHAPES)}")
    hi = _first_turnover(shape, min(theta_max * 1.35, math.radians(89.0)))
    if hi <= theta_max * 1.01:
        raise ValueError(
            f"lens {shape!r} turns over at {math.degrees(hi):.2f} deg, which is "
            f"not clear of the {math.degrees(theta_max):.2f} deg cone. Pixels "
            f"outside the disc could not be told apart from rim pixels.")
    r_disc = size / 2.0 - margin_px
    f = r_disc / float(SHAPES[shape][0](np.array([theta_max]), np)[0])
    return RadialShapeCamera(fx=f, fy=f, cx=(size - 1) / 2.0, cy=(size - 1) / 2.0,
                             width=size, height=size, theta_max=theta_max,
                             shape=shape, theta_bracket=hi)


def lens_family(names: Sequence[str], size: int, theta_max: float
                ) -> Dict[str, RadialShapeCamera]:
    return {n: make_lens(n, size, theta_max) for n in names}


# --------------------------------------------------------------------- warps

def grid_between(src: Camera, dst: Camera) -> Tuple[Tensor, Tensor]:
    """``(grid, valid)`` resampling a map defined on ``src`` onto ``dst``'s grid.

    For every ``dst`` pixel: take its ray, and find where ``src`` images that
    same ray. Valid requires the ray to be inside BOTH cones and to land inside
    the ``src`` frame. Invalid addresses are pushed out of range so
    ``grid_sample(padding_mode="zeros")`` returns 0 -- never border-replicated,
    which would be a fabricated observation the model then gets scored on.
    """
    uv = pixel_grid(dst.height, dst.width, dtype=torch.float32)
    rays = dst.unproject(uv)
    theta = torch.acos(rays[..., 2].clamp(-1.0, 1.0))
    at = src.project(rays)
    valid = ((theta <= dst.theta_max) & (theta <= src.theta_max)
             & (rays[..., 2] > 1e-6) & torch.isfinite(at).all(-1))
    valid &= ((at[..., 0] >= 0) & (at[..., 0] <= src.width - 1)
              & (at[..., 1] >= 0) & (at[..., 1] <= src.height - 1))
    g = torch.stack((2.0 * (at[..., 0] + 0.5) / src.width - 1.0,
                     2.0 * (at[..., 1] + 0.5) / src.height - 1.0), dim=-1)
    g = torch.where(valid[..., None], g, torch.full_like(g, OUT_OF_RANGE))
    return g, valid


def warp(src: Tensor, grid: Tensor, mode: str = "bilinear") -> Tensor:
    """Resample ``src`` (H,W) / (C,H,W) / (N,C,H,W) with a ``(Ho,Wo,2)`` grid.

    RGB is bilinear; GT depth must be ``nearest`` -- bilinear across a depth
    discontinuity invents a surface that is in neither the near nor the far
    object, and the near-field rim is where those discontinuities live.
    """
    x, squeeze = src, 0
    if x.dim() == 2:
        x, squeeze = x[None, None], 2
    elif x.dim() == 3:
        x, squeeze = x[None], 1
    g = grid.to(x.device, x.dtype)[None].expand(x.shape[0], -1, -1, -1)
    out = torch.nn.functional.grid_sample(
        x, g, mode=mode, padding_mode="zeros", align_corners=False)
    for _ in range(squeeze):
        out = out[0]
    return out


# --------------------------------------------------------------------- field

def token_field(cam: RadialShapeCamera, size: int, patch: int = 14) -> Tensor:
    """Per-token ``(log_area, log_aniso, theta/theta_max)``, shape ``(P, 3)``.

    Same construction as H12's `token_jacobian_field`, generalised off KB4:
    theta comes from the CAMERA (authoritative) and the two Jacobian channels
    from the shape's own closed forms, clamped to the imaged cone before the
    lens function is evaluated. H12's first box smoke printed
    ``log_area[-690.776, 0.577]`` without that clamp -- one saturated corner
    token sets the FiLM MLP's input scale for the entire run.
    """
    theta = cam.incidence_grid(size, size).cpu().numpy().astype(np.float64)
    tmax = float(cam.theta_max)
    tc = np.minimum(theta, tmax)
    d, dp = SHAPES[cam.shape]
    la, ln = J.log_area_aniso(tc, lambda t: d(t, np), lambda t: dp(t, np))
    gh = gw = size // patch

    def pool(a):
        return a.reshape(gh, patch, gw, patch).mean((1, 3)).ravel()

    f = np.stack([pool(la), pool(ln), pool(tc) / tmax], axis=-1)
    if not np.isfinite(f).all() or np.abs(f).max() > 50.0:
        raise SystemExit(
            f"[h15] token field for {cam.shape!r} out of range: "
            f"min {f.min():.3f} max {f.max():.3f}. A saturated value sets the "
            f"FiLM input scale for the whole run; refusing to train on it.")
    return torch.from_numpy(f).float()
