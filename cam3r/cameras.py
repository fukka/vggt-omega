# Copyright (c) 2026.
"""Camera models that emit per-pixel ground-truth ray fields.

CAM3R supervises the Ray Module with rays "computed per pixel using the exact
sensor calibration unprojection routines, which account for radial distortions"
(paper Sec. D.3.1).  This module is that routine, for the three lens families
the paper trains on:

* :class:`Pinhole`   -- MegaDepth, and the pinhole half of ADT's paired renders.
* :class:`KB4`       -- Kannala-Brandt fisheye; ADT's Aria 214-1 RGB sensor.
* :class:`Equirect`  -- full 360x180 panoramas; 2D3DS and 360Loc.

All models share one contract: ``ray_field(H, W)`` returns ``(H, W, 3)`` unit
rays in the repo's camera frame (x right, y down, z forward) plus an ``(H, W)``
bool validity mask, and ``project(rays)`` is its inverse back to pixel
coordinates.  A camera is serializable to/from a plain dict via :meth:`spec` and
:func:`build_camera`, so a dataset can record which lens produced a frame.

The Aria KB4 constants and the turnover-masking rule are taken from
``VGGT-360-fisheye/utils/fisheye_cam.py`` (validated to <0.22 deg against
projectaria_tools); ``cam3r/tests/test_cameras.py`` asserts agreement with that
implementation so the two cannot silently diverge.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

# Aria 214-1 RGB calibration at native 1408x1408.
_ARIA_F_NATIVE = 610.94
_ARIA_CX_NATIVE = 715.11
_ARIA_CY_NATIVE = 716.71
_ARIA_NATIVE = 1408.0
_ARIA_KB4 = (0.3852, -0.4442, 0.5591, -0.3254)


def _pixel_grid(H: int, W: int, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integer pixel centres, matching ``fisheye_ray_lut``'s ``arange`` grid."""
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=dtype), torch.arange(W, dtype=dtype), indexing="ij"
    )
    return xs, ys


class Camera:
    """Base class: a lens that maps pixels to unit rays and back."""

    kind: str = "camera"

    def ray_field(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def project(self, rays: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def spec(self) -> Dict:
        raise NotImplementedError


class Pinhole(Camera):
    """Standard perspective camera."""

    kind = "pinhole"

    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)

    @classmethod
    def from_fov(cls, H: int, W: int, hfov_deg: float) -> "Pinhole":
        f = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(f, f, (W - 1) / 2.0, (H - 1) / 2.0)

    def ray_field(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = _pixel_grid(H, W)
        dirs = torch.stack([(xs - self.cx) / self.fx, (ys - self.cy) / self.fy, torch.ones_like(xs)], -1)
        return torch.nn.functional.normalize(dirs, dim=-1), torch.ones(H, W, dtype=torch.bool)

    def project(self, rays: torch.Tensor) -> torch.Tensor:
        z = rays[..., 2].clamp(min=1e-8)
        return torch.stack([self.fx * rays[..., 0] / z + self.cx, self.fy * rays[..., 1] / z + self.cy], -1)

    def spec(self) -> Dict:
        return {"kind": "pinhole", "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}


class KB4(Camera):
    """Kannala-Brandt 4-coefficient fisheye (OPENCV_FISHEYE).

    ``theta_d = theta (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)``, ``t = theta``.

    Two different limits bound the usable cone, and CONTEXT.md is explicit that
    they must not be conflated:

    * the **fold-back turnover** (~62.33 deg for Aria 214-1), past which the
      polynomial is non-monotonic and rays alias onto wrong pixels; and
    * the **imaged cone** (54.83 deg for Aria 214-1), the half-FoV the lens
      actually illuminates, recovered from the inscribed image circle.

    ``theta_max`` returns the ``min`` of the two, matching the repo authority
    ``VGGT-360-fisheye/utils/fisheye_cam.py::FisheyeCam.theta_max``.  Using the
    turnover alone as the cone admits a ring of dead vignette pixels.
    """

    kind = "kb4"

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 k: Tuple[float, float, float, float], *, table_n: int = 8192,
                 valid_theta: Optional[float] = None) -> None:
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.k = tuple(float(v) for v in k)
        self._table_n = table_n
        self.valid_theta = valid_theta
        self._theta_tab: Optional[torch.Tensor] = None
        self._theta_d_tab: Optional[torch.Tensor] = None

    # -- forward polynomial and its tabulated inverse ----------------------- #

    def forward_theta(self, theta: torch.Tensor) -> torch.Tensor:
        k1, k2, k3, k4 = self.k
        t2 = theta * theta
        return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def _tables(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monotonic prefix of ``theta -> theta_d``, cached."""
        if self._theta_tab is None:
            tg = torch.linspace(0.0, math.pi / 2 + 0.2, self._table_n, dtype=torch.float64)
            td = self.forward_theta(tg)
            dec = (td[1:] - td[:-1]) <= 0
            if bool(dec.any()):
                turn = int(torch.argmax(dec.long())) + 1
                tg, td = tg[:turn], td[:turn]
            self._theta_tab, self._theta_d_tab = tg, td
        return self._theta_tab, self._theta_d_tab

    def turnover_theta(self) -> float:
        """Fold-back angle: where the forward polynomial stops being monotonic."""
        return float(self._tables()[0][-1])

    def inscribed_valid_theta(self, H: int, W: int) -> float:
        """Imaged half-FoV implied by the inscribed image circle.

        A fisheye sensor is sized so the image circle just fits the frame, so
        the smallest principal-point-to-border margin is the circle radius;
        unprojecting it gives the illuminated half-FoV without needing the
        vendor's valid-radius field.  54.83 deg for Aria 214-1.
        """
        theta_d = min(
            self.cx / self.fx, (W - 1.0 - self.cx) / self.fx,
            self.cy / self.fy, (H - 1.0 - self.cy) / self.fy,
        )
        return float(self.unproject_theta(torch.tensor(theta_d, dtype=torch.float64)))

    def theta_max(self, H: Optional[int] = None, W: Optional[int] = None) -> float:
        """Usable incidence limit: ``min(imaged cone, fold-back turnover)``.

        ``valid_theta`` overrides the inscribed-circle estimate when a
        calibrated valid radius is known.  Without it and without ``(H, W)``,
        only the turnover guard applies.
        """
        turnover = self.turnover_theta()
        valid = self.valid_theta
        if valid is None and H is not None and W is not None:
            valid = self.inscribed_valid_theta(H, W)
        return turnover if valid is None else min(float(valid), turnover)

    def unproject_theta(self, theta_d: torch.Tensor) -> torch.Tensor:
        """Invert the forward polynomial by interpolating its monotonic prefix.

        Table inversion rather than Newton: Newton diverges near the turnover,
        exactly where wide-FoV pixels live.
        """
        tg, td = self._tables()
        flat = theta_d.reshape(-1).to(torch.float64).clamp(0.0, float(td[-1]))
        idx = torch.searchsorted(td.contiguous(), flat.contiguous()).clamp(1, len(td) - 1)
        lo, hi = idx - 1, idx
        span = (td[hi] - td[lo]).clamp(min=1e-12)
        w = (flat - td[lo]) / span
        return (tg[lo] + w * (tg[hi] - tg[lo])).reshape(theta_d.shape).to(theta_d.dtype)

    # -- Camera contract ---------------------------------------------------- #

    def ray_field(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = _pixel_grid(H, W, dtype=torch.float64)
        x = (xs - self.cx) / self.fx
        y = (ys - self.cy) / self.fy
        theta_d = torch.sqrt(x * x + y * y)
        theta = self.unproject_theta(theta_d)
        sin_t = torch.sin(theta)
        inv = torch.where(theta_d > 1e-9, 1.0 / theta_d.clamp(min=1e-12), torch.zeros_like(theta_d))
        rays = torch.stack([sin_t * x * inv, sin_t * y * inv, torch.cos(theta)], -1)
        on_axis = theta_d <= 1e-9
        rays[on_axis] = torch.tensor([0.0, 0.0, 1.0], dtype=rays.dtype)
        valid = theta <= (self.theta_max(H, W) - 1e-6)
        return rays.to(torch.float32), valid

    def project(self, rays: torch.Tensor) -> torch.Tensor:
        rays = torch.nn.functional.normalize(rays, dim=-1)
        theta = torch.arccos(rays[..., 2].clamp(-1.0, 1.0))
        theta_d = self.forward_theta(theta)
        rxy = rays[..., :2].norm(dim=-1)
        scale = torch.where(rxy > 1e-9, theta_d / rxy.clamp(min=1e-12), torch.zeros_like(rxy))
        return torch.stack(
            [self.fx * rays[..., 0] * scale + self.cx, self.fy * rays[..., 1] * scale + self.cy], -1
        )

    def spec(self) -> Dict:
        return {"kind": "kb4", "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "k": list(self.k), "valid_theta": self.valid_theta}


class Equirect(Camera):
    """Full 360x180 equirectangular panorama.

    Column ``u`` maps to longitude in ``[-pi, pi)``, row ``v`` to latitude in
    ``[pi/2, -pi/2]``.  Longitude 0 is the image centre and looks along +z.

    Paper Sec. D.3 fixes the convention: *"Using the normalized pixel center
    convention ``(u, v) = ((u_idx + 0.5)/W, (v_idx + 0.5)/H)``, we map to
    longitude ``lambda = (u - 0.5) 2pi`` and latitude ``phi = (0.5 - v) pi``."*
    The half pixel matters because ERP has no intrinsic matrix to absorb it: it
    is a fixed ``180/W`` degrees of longitude between the image content, which
    *is* sampled at pixel centres, and the ray assigned to it -- 0.35 deg at the
    paper's 512 px.  Nothing downstream removes it, so it would sit in the GT
    ray field for every panoramic corpus as a constant bias.

    ``project`` is the exact inverse and returns *index-space* coordinates, the
    same convention as :class:`Pinhole` and :class:`KB4`, so a caller resampling
    with ``grid_sample`` has to add the usual half pixel itself.
    """

    kind = "erp"

    def ray_field(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = _pixel_grid(H, W, dtype=torch.float64)
        lon = ((xs + 0.5) / W - 0.5) * 2.0 * math.pi
        lat = (0.5 - (ys + 0.5) / H) * math.pi
        cos_lat = torch.cos(lat)
        rays = torch.stack([cos_lat * torch.sin(lon), -torch.sin(lat), cos_lat * torch.cos(lon)], -1)
        return rays.to(torch.float32), torch.ones(H, W, dtype=torch.bool)

    def project(self, rays: torch.Tensor, H: Optional[int] = None, W: Optional[int] = None) -> torch.Tensor:
        if H is None or W is None:
            raise ValueError("Equirect.project needs the target (H, W)")
        rays = torch.nn.functional.normalize(rays, dim=-1)
        lon = torch.atan2(rays[..., 0], rays[..., 2])
        lat = torch.arcsin((-rays[..., 1]).clamp(-1.0, 1.0))
        u = (lon / (2.0 * math.pi) + 0.5) * W - 0.5
        v = (0.5 - lat / math.pi) * H - 0.5
        return torch.stack([u, v], -1)

    def spec(self) -> Dict:
        return {"kind": "erp"}


def aria_valid_theta_max() -> float:
    """Aria 214-1 imaged half-FoV in radians (54.83 deg), derived once at native size.

    Deriving it from the *native* 1408x1408 geometry keeps it a property of the
    lens rather than of whatever grid a frame was resampled onto -- the
    ``(W - 1)`` term in the inscribed-circle margin otherwise shrinks the cone
    by ~1 deg at small resolutions.  Mirrors
    ``VGGT-360-fisheye/utils/fisheye_cam.py::aria_valid_theta_max``.
    """
    native = KB4(_ARIA_F_NATIVE, _ARIA_F_NATIVE, _ARIA_CX_NATIVE, _ARIA_CY_NATIVE, _ARIA_KB4)
    return native.inscribed_valid_theta(int(_ARIA_NATIVE), int(_ARIA_NATIVE))


def aria_214_1_kb4(H: int, W: int, rotated: bool = True) -> KB4:
    """Aria 214-1 RGB intrinsics scaled to an ``H x W`` frame.

    ``rotated=True`` accounts for the ADT convention of rotating stored frames
    270 deg CCW: for a square frame that swaps fx/fy and moves the principal
    point to ``(H-1-cy, cx)``.
    """
    sx, sy = W / _ARIA_NATIVE, H / _ARIA_NATIVE
    fx, fy = _ARIA_F_NATIVE * sx, _ARIA_F_NATIVE * sy
    cx, cy = _ARIA_CX_NATIVE * sx, _ARIA_CY_NATIVE * sy
    if rotated:
        fx, fy = fy, fx
        cx, cy = (H - 1) - cy, cx
    return KB4(fx, fy, cx, cy, _ARIA_KB4, valid_theta=aria_valid_theta_max())


def build_camera(spec: Dict) -> Camera:
    """Rebuild a camera from :meth:`Camera.spec` output."""
    kind = spec["kind"]
    if kind == "pinhole":
        return Pinhole(spec["fx"], spec["fy"], spec["cx"], spec["cy"])
    if kind == "kb4":
        return KB4(spec["fx"], spec["fy"], spec["cx"], spec["cy"], tuple(spec["k"]),
                   valid_theta=spec.get("valid_theta"))
    if kind == "erp":
        return Equirect()
    raise ValueError(f"unknown camera kind {kind!r}")
