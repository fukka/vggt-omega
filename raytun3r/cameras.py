"""Differentiable central camera models for RayTun3R.

Paper: Sec. 3 "Camera projection", Eq. 1, 3, 4.

A camera here is a *central* model with a forward projection ``kappa`` mapping a
3D point in camera coordinates to a pixel, and an inverse ``kappa_inv`` mapping a
pixel to a unit-norm viewing ray.

Two asymmetries drive the implementation:

* ``kappa`` sits inside the reprojection loss (Eq. 8) applied to *predicted* 3D
  points, so it must be differentiable w.r.t. its input. All forward models here
  are closed-form and autograd-friendly.
* ``kappa_inv`` is only ever evaluated at the fixed pixel grid. Camera intrinsics
  are not trained by RayTun3R (only PE/RoPE residuals are), so the ray table is
  constant across the whole fit and is computed once and cached. That lets the
  fisheye models invert their radial polynomial with Newton iterations without
  worrying about backprop through the solver.

Conventions: pixel coordinates are continuous with ``(0, 0)`` at the *centre* of
the top-left pixel, matching ``torch.nn.functional.grid_sample``'s
``align_corners=True`` convention after normalisation. Rays are unit-norm and
point away from the optical centre with +z forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import Tensor

__all__ = [
    "Camera",
    "Pinhole",
    "KannalaBrandt",
    "EUCM",
    "OpenCVFisheye",
    "pixel_grid",
    "from_opencv_fisheye",
    "from_aria",
    "from_scannetpp",
    "DEPTH_CONVENTIONS",
    "convert_depth",
]

#: The two readings of a depth map. ``"z"`` is planar z -- distance along the
#: optical axis, what a z-buffer and every pinhole-trained depth head emit.
#: ``"range"`` is euclidean distance along each pixel's own ray. They are related
#: by ``range = z / cos(theta)`` and are *not* interchangeable on a wide lens:
#: the factor runs to 1.74x at the Aria rim and ~11x at a 170 deg frame corner,
#: and being radially varying it cannot be absorbed by any global scale.
DEPTH_CONVENTIONS = ("z", "range")


def pixel_grid(height: int, width: int, *, device=None, dtype=torch.float32) -> Tensor:
    """``(H, W, 2)`` grid of pixel centres as ``(u, v)``."""
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((u, v), dim=-1)


@dataclass
class Camera:
    """Base central camera. Subclasses implement the radial distortion law.

    ``fx, fy, cx, cy`` are in pixels; ``width, height`` give the image size the
    intrinsics refer to. ``theta_max`` is the largest incidence angle the lens
    actually images, in radians -- pixels whose ray exceeds it are invalid and
    are excluded from ``valid_mask`` (and hence from ``Omega`` in every loss).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    theta_max: float = math.pi / 2
    _ray_cache: dict = field(default_factory=dict, repr=False, compare=False)

    # -- subclass hooks ---------------------------------------------------

    def r_of_theta(self, theta: Tensor) -> Tensor:
        """Distorted image radius (in normalised units) for incidence angle."""
        raise NotImplementedError

    def theta_of_r(self, r: Tensor) -> Tensor:
        """Inverse of :meth:`r_of_theta`. Only called on the fixed pixel grid."""
        raise NotImplementedError

    # -- projection -------------------------------------------------------

    def project(self, xyz: Tensor) -> Tensor:
        """``kappa``: camera-frame points ``(..., 3)`` -> pixels ``(..., 2)``.

        Differentiable. Points behind the imaged cone still produce a value; call
        sites mask with :meth:`valid_mask` or with the returned finite check.
        """
        x, y, z = xyz.unbind(-1)
        xy_norm = torch.sqrt(x * x + y * y).clamp_min(1e-12)
        theta = torch.atan2(xy_norm, z)
        r = self.r_of_theta(theta)
        scale = r / xy_norm
        u = self.fx * scale * x + self.cx
        v = self.fy * scale * y + self.cy
        return torch.stack((u, v), dim=-1)

    def unproject(self, uv: Tensor) -> Tensor:
        """``kappa^-1``: pixels ``(..., 2)`` -> unit rays ``(..., 3)``."""
        u, v = uv.unbind(-1)
        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r = torch.sqrt(mx * mx + my * my)
        theta = self.theta_of_r(r)
        sin_t = torch.sin(theta)
        # At r == 0 the direction in the image plane is undefined; the ray is +z.
        safe_r = r.clamp_min(1e-12)
        dirs = torch.stack(
            (sin_t * mx / safe_r, sin_t * my / safe_r, torch.cos(theta)), dim=-1
        )
        return torch.nn.functional.normalize(dirs, dim=-1)

    # -- cached grids -----------------------------------------------------

    def ray_grid(self, height: Optional[int] = None, width: Optional[int] = None,
                 *, device=None, dtype=torch.float32) -> Tensor:
        """``(H, W, 3)`` unit rays, computed once per (H, W, device, dtype)."""
        height = self.height if height is None else height
        width = self.width if width is None else width
        key = ("rays", height, width, str(device), str(dtype))
        if key not in self._ray_cache:
            cam = self.resized(width, height) if (height, width) != (self.height, self.width) else self
            uv = pixel_grid(height, width, device=device, dtype=dtype)
            with torch.no_grad():
                self._ray_cache[key] = cam.unproject(uv)
        return self._ray_cache[key]

    def incidence_grid(self, height: Optional[int] = None, width: Optional[int] = None,
                       *, device=None, dtype=torch.float32) -> Tensor:
        """``(H, W)`` incidence angle theta of each pixel's ray, in radians."""
        rays = self.ray_grid(height, width, device=device, dtype=dtype)
        return torch.acos(rays[..., 2].clamp(-1.0, 1.0))

    def valid_mask(self, height: Optional[int] = None, width: Optional[int] = None,
                   *, device=None, dtype=torch.float32) -> Tensor:
        """``(H, W)`` bool mask of the imaged disc -- the set ``Omega``."""
        return self.incidence_grid(height, width, device=device, dtype=dtype) <= self.theta_max

    def resized(self, width: int, height: int) -> "Camera":
        """Intrinsics rescaled to a new image size (aspect change allowed)."""
        sx, sy = width / self.width, height / self.height
        clone = self.__class__(**{**self._params(), "fx": self.fx * sx, "fy": self.fy * sy,
                                  "cx": (self.cx + 0.5) * sx - 0.5,
                                  "cy": (self.cy + 0.5) * sy - 0.5,
                                  "width": width, "height": height})
        return clone

    def cropped(self, u0: float, v0: float, width: int, height: int) -> "Camera":
        """Intrinsics for a crop whose top-left pixel centre was ``(u0, v0)``."""
        return self.__class__(**{**self._params(), "cx": self.cx - u0, "cy": self.cy - v0,
                                 "width": width, "height": height})

    def with_max_fov(self, fov_deg: float) -> "Camera":
        """Same lens, cone restricted to ``fov_deg`` total field of view.

        Only ever *narrows*: asking for more than the lens images is a no-op, so
        a sweep can be written as one ascending list without special-casing the
        native FOV.

        This is the knob for the paper's stated 115 deg ScanNet++ FOV versus the
        ~170 deg its released calibration actually implies. Because ``theta_max``
        defines ``valid_mask``, and ``valid_mask`` is ``Omega``, narrowing it
        propagates to every loss (Eq. 8, 10) and every metric (Eq. 16-18) without
        touching the images -- the model still sees the full frame, so this
        isolates *where the method is scored* from what it is shown.
        """
        return self.__class__(**{**self._params(),
                                 "theta_max": min(self.theta_max,
                                                  math.radians(fov_deg / 2.0))})

    def _params(self) -> dict:
        out = {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
               "width": self.width, "height": self.height, "theta_max": self.theta_max}
        return out

    # -- pinhole companion ------------------------------------------------

    def pinhole_focal(self, fov_deg: float) -> float:
        """Focal length of a virtual pinhole with the given horizontal FOV."""
        return (self.width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

    def to_pinhole(self, fov_deg: float = 110.0, width: Optional[int] = None,
                   height: Optional[int] = None) -> "Pinhole":
        """The virtual pinhole used by the Center-PH baseline and by the
        fisheye->pinhole map behind the parameter-free corrections (Sec. 4.2)."""
        width = self.width if width is None else width
        height = self.height if height is None else height
        f = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        return Pinhole(fx=f, fy=f, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
                       width=width, height=height,
                       theta_max=min(self.theta_max, math.radians(89.0)))

    def backproject_jacobian(self, height: Optional[int] = None, width: Optional[int] = None,
                             *, normalized: bool = False, device=None,
                             dtype=torch.float32) -> Tensor:
        """``(H, W, 3, 2)`` Jacobian ``d kappa^-1 / d(u, v)`` -- Eq. 2 and 4.

        Used by the pinhole-bias diagnostic (Fig. 2): for a pinhole this is
        constant over the image, for a fisheye it varies with radius.

        ``normalized=False`` (default) uses the paper's ``z = 1`` convention from
        Eq. 1 and 3, under which the pinhole Jacobian is *exactly* constant.
        ``normalized=True`` differentiates the unit ray instead, which adds a
        mild radial term even for a pinhole.

        Computed by central differences in float64 rather than autograd:
        :meth:`unproject` divides by the image radius, whose derivative is NaN
        exactly at the principal point, and that is precisely the pixel the
        radius-dependence plot starts from.
        """
        height = self.height if height is None else height
        width = self.width if width is None else width
        cam = self.resized(width, height) if (height, width) != (self.height, self.width) else self
        uv = pixel_grid(height, width, device=device, dtype=torch.float64)

        def rays_at(p: Tensor) -> Tensor:
            r = cam.unproject(p)
            return r if normalized else r / r[..., 2:3].clamp_min(1e-8)

        h = 1e-4
        cols = []
        for axis in range(2):
            step = torch.zeros(2, device=uv.device, dtype=uv.dtype)
            step[axis] = h
            cols.append((rays_at(uv + step) - rays_at(uv - step)) / (2 * h))
        return torch.stack(cols, dim=-1).to(dtype)   # (H, W, 3, 2)


@dataclass
class Pinhole(Camera):
    """Eq. 1. ``r(theta) = tan(theta)``."""

    def r_of_theta(self, theta: Tensor) -> Tensor:
        return torch.tan(theta.clamp(max=math.radians(89.5)))

    def theta_of_r(self, r: Tensor) -> Tensor:
        return torch.atan(r)


@dataclass
class KannalaBrandt(Camera):
    """Kannala-Brandt (KB4) [38]: ``r = theta + k1 th^3 + k2 th^5 + k3 th^7 + k4 th^9``.

    This is the model behind ADT/Aria (``finetune/data/rectify.py`` coefficients),
    KITTI-360's fisheye cameras, and OpenCV's ``fisheye`` module.
    """

    k: Sequence[float] = (0.0, 0.0, 0.0, 0.0)

    def r_of_theta(self, theta: Tensor) -> Tensor:
        k1, k2, k3, k4 = self.k
        t2 = theta * theta
        return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def theta_of_r(self, r: Tensor) -> Tensor:
        # Bisection-safeguarded Newton on f(th) = r_of_theta(th) - r,
        # bracketed inside [0, turnover] where the polynomial is monotone.
        #
        # The previous plain-Newton version initialised at theta = r; with
        # Aria's k1 > 0 that overshoots into the flat region near the
        # turnover, where df -> 0 makes the step explode and the iteration
        # lands up to ~1 px wrong for rays in the outermost ~5 deg of the
        # cone (caught 2026-08-23 by an identity-warp test in the H5 losses;
        # theta-binned aggregates were too coarse to notice — a 1 px slip is
        # ~0.2 deg against 6.9 deg bins — but differentiable warping is not).
        k1, k2, k3, k4 = self.k

        def df_of(theta: Tensor) -> Tensor:
            t2 = theta * theta
            return 1.0 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))

        # Upper bracket: the turnover (df == 0) if one exists below pi, found
        # once by coarse scan + refinement on the scalar polynomial.
        hi_val = math.pi / 2 + 0.35
        import numpy as _np
        ts = _np.linspace(1e-6, hi_val, 4097)
        dfs = 1.0 + ts**2 * (3 * k1 + ts**2 * (5 * k2 + ts**2 * (7 * k3 + ts**2 * 9 * k4)))
        neg = _np.nonzero(dfs <= 0)[0]
        if len(neg):
            hi_val = float(ts[neg[0]])
        lo = torch.zeros_like(r)
        hi = torch.full_like(r, hi_val)
        r_hi = self.r_of_theta(hi)
        # Radii beyond the monotone range clamp to the turnover angle.
        theta = (r / r_hi.clamp_min(1e-8) * hi_val).clamp(0.0, hi_val)
        for _ in range(30):
            f = self.r_of_theta(theta) - r
            lo = torch.where(f < 0, theta, lo)
            hi = torch.where(f >= 0, theta, hi)
            step = f / df_of(theta).clamp_min(1e-8)
            cand = theta - step
            inside = (cand > lo) & (cand < hi)
            theta = torch.where(inside, cand, 0.5 * (lo + hi))
        return theta.clamp(0.0, hi_val)

    def _params(self) -> dict:
        return {**super()._params(), "k": self.k}


@dataclass
class EUCM(Camera):
    """Enhanced Unified Camera Model [39], parameters ``alpha, beta``.

    ``r(theta) = sin(theta) / (alpha * d + (1 - alpha) * cos(theta))`` with
    ``d = sqrt(beta * sin^2(theta) + cos^2(theta))``.
    """

    alpha: float = 0.5
    beta: float = 1.0

    def r_of_theta(self, theta: Tensor) -> Tensor:
        s, c = torch.sin(theta), torch.cos(theta)
        d = torch.sqrt(self.beta * s * s + c * c)
        return s / (self.alpha * d + (1.0 - self.alpha) * c).clamp_min(1e-8)

    def theta_of_r(self, r: Tensor) -> Tensor:
        # Closed form: EUCM unprojection (Khomutenko et al., Eq. 13).
        a, b = self.alpha, self.beta
        r2 = r * r
        disc = 1.0 - (2 * a - 1.0) * b * r2
        disc = disc.clamp_min(0.0)
        mz = (1.0 - b * a * a * r2) / (a * torch.sqrt(disc) + (1.0 - a)).clamp_min(1e-8)
        return torch.atan2(r, mz)

    def _params(self) -> dict:
        return {**super()._params(), "alpha": self.alpha, "beta": self.beta}


# ``OpenCVFisheye`` is KB4 under OpenCV's naming; kept as an alias so dataset
# code can say what the calibration file said.
OpenCVFisheye = KannalaBrandt


# ---------------------------------------------------------------------------
# Constructors from the calibrations this repo already has
# ---------------------------------------------------------------------------


def from_opencv_fisheye(K, D, width: int, height: int,
                        theta_max: Optional[float] = None) -> KannalaBrandt:
    """Build from an OpenCV ``fisheye`` calibration (``K`` 3x3, ``D`` length-4)."""
    K = torch.as_tensor(K, dtype=torch.float64).reshape(3, 3)
    D = torch.as_tensor(D, dtype=torch.float64).reshape(-1)[:4]
    cam = KannalaBrandt(
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        width=width, height=height, k=tuple(float(x) for x in D),
    )
    cam.theta_max = _default_theta_max(cam) if theta_max is None else theta_max
    return cam


def from_aria(height: int, width: int, rotated: bool = True) -> KannalaBrandt:
    """ADT / Aria 214-1 fisheye, reusing this repo's calibration of record.

    Delegates to ``finetune.eval.baselines.aria_fisheye`` so the KB4 coefficients
    and the usable-cone limit stay defined in exactly one place (see CONTEXT.md,
    "Imaged cone" -- ``theta_max`` is the min of the lens cone and the KB4
    turnover, not the turnover alone).
    """
    from finetune.eval.baselines.aria_fisheye import aria_intrinsics

    a = aria_intrinsics(height, width, rotated=rotated)
    return KannalaBrandt(
        fx=float(a.fx), fy=float(a.fy), cx=float(a.cx), cy=float(a.cy),
        width=width, height=height, k=tuple(float(x) for x in a.k),
        theta_max=float(a.usable_theta_max()),
    )


def from_scannetpp(intrinsics: dict, width: int, height: int) -> KannalaBrandt:
    """ScanNet++ DSLR fisheye entry (``OPENCV_FISHEYE`` camera in ``nerfstudio``
    /``colmap`` export: params ``[fx, fy, cx, cy, k1, k2, k3, k4]``)."""
    p = intrinsics["params"] if "params" in intrinsics else intrinsics
    fx, fy, cx, cy, k1, k2, k3, k4 = [float(x) for x in list(p)[:8]]
    cam = KannalaBrandt(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height,
                        k=(k1, k2, k3, k4))
    cam.theta_max = _default_theta_max(cam)
    return cam


def _default_theta_max(cam: KannalaBrandt) -> float:
    """Largest theta that is both below the KB4 turnover and inside the frame.

    Mirrors ``aria_fisheye.usable_max_incidence``: the forward polynomial is only
    monotonic up to its turnover, and past it rays alias onto in-cone pixels.

    The frame bound is the *corner* radius, not the nearest-edge radius. This
    matters for full-frame fisheyes -- ScanNet++'s DSLR being the case in hand,
    where the whole rectangle carries real image content and the KB4 polynomial
    is monotone all the way out to it. Taking ``min(rx, ry)`` would give the
    largest circle *inscribed* in the sensor and silently drop every pixel
    outside it: on a 504x336 ScanNet++ frame that is 47% of the image, excluded
    from ``Omega`` in every loss (Eq. 8, 10) and every metric (Eq. 16-18). The
    paper's ``Omega`` is "the set of pixels inside the valid fisheye disc",
    which for a lens whose image circle covers the sensor is the whole frame.
    Circular fisheyes are unaffected: their real cone limit is smaller than the
    corner and comes from the lens spec (see :func:`from_aria`) or the turnover.
    """
    th = torch.linspace(0.0, math.pi / 2, 4096, dtype=torch.float64)
    r = cam.r_of_theta(th)
    turn = torch.nonzero(r[1:] <= r[:-1])
    theta_turn = float(th[int(turn[0]) ]) if turn.numel() else math.pi / 2
    # Largest image radius any pixel in the rectangle actually attains.
    r_far = math.hypot(
        max(cam.cx, cam.width - 1 - cam.cx) / cam.fx,
        max(cam.cy, cam.height - 1 - cam.cy) / cam.fy,
    )
    mono = th[: (int(turn[0]) + 1 if turn.numel() else len(th))]
    r_mono = cam.r_of_theta(mono)
    inside = torch.nonzero(r_mono <= r_far)
    theta_frame = float(mono[int(inside[-1])]) if inside.numel() else theta_turn
    return min(theta_turn, theta_frame)


def convert_depth(depth: Tensor, camera: "Camera", *, src: str, dst: str,
                  min_cos: float = 1e-6) -> Tensor:
    """Convert a depth map between planar z and euclidean range.

    ``range = z / cos(theta)``, where ``cos(theta)`` is the z component of the
    unit ray through each pixel -- so this is a *per-pixel, radially varying*
    rescaling, not a global one. That is precisely why it cannot be left to the
    scale alignment in Eq. 16-18 to absorb: those solve for a single scalar.

    Everything downstream of a backbone (Eq. 7's ``X = D kappa^-1``, ``d_reproj``,
    ``AbsRel``) is only correct when the normalisation of ``kappa^-1`` matches the
    definition of ``D``. This function is what makes the two agree; see
    :class:`~raytun3r.backbones.Prediction` for which convention each producer is
    tagged with.

    ``min_cos`` floors the divisor, and matches ``backproject``'s own clamp so
    that the two stay exact inverses. Do not raise it as a "safety" measure: a
    larger floor silently stops this being the inverse of Eq. 7's ``z`` reading
    precisely at the grazing angles where they differ most. Mask to
    ``camera.valid_mask`` instead -- past ~85 deg the planar-z reading is
    physically degenerate anyway (``1/cos`` exceeds 10 and diverges at 90).
    """
    if src not in DEPTH_CONVENTIONS or dst not in DEPTH_CONVENTIONS:
        raise ValueError(f"depth convention must be one of {DEPTH_CONVENTIONS}; "
                         f"got src={src!r}, dst={dst!r}")
    if src == dst:
        return depth
    h, w = depth.shape[-2:]
    cos = camera.ray_grid(h, w, device=depth.device)[..., 2].to(depth.dtype)
    if src == "z":
        return depth / cos.clamp_min(min_cos)
    return depth * cos
