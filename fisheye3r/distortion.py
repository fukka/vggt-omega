"""Kannala-Brandt fisheye synthesis and undistortion (paper Sec. 3.3 + supp Eq. 16-18).

The paper synthesizes fisheye training images from perspective ones with the
Kannala-Brandt (KB) model

    theta = arctan(sqrt(x^2 + y^2) / z)                      (Eq. 16)
    r(theta) = theta + k1 th^3 + k2 th^5 + k3 th^7 + k4 th^9 (Eq. 17)
    u = fx * x / sqrt(x^2+y^2) * r(theta) + cx               (Eq. 18)
    v = fy * y / sqrt(x^2+y^2) * r(theta) + cy

and randomizes (supp Sec. 6): fisheye focal = U[1, 1.2] x perspective focal,
principal point shifted by U[-10, 10] px, k1..k3 ~ U(-0.5, 0.5),
k4 ~ U(-0.05, 0.05).

Both cameras share the optical center and orientation, so a z-depth value is
attached to a viewing ray and is *identical* at corresponding pixels of the two
images. T (distort) and T^-1 (undistort) are therefore pure resampling warps
for images and for dense scalar predictions (depth / confidence).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class KannalaBrandtCamera:
    """Batched KB fisheye camera plus the source perspective pinhole.

    All tensors have shape (B,). The perspective camera (fx_p, fy_p, cx_p,
    cy_p) is the camera of the *undistorted* source image; the KB parameters
    (fx, fy, cx, cy, k1..k4) define the synthesized fisheye image of the same
    pixel size.
    """

    fx: torch.Tensor
    fy: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor
    k1: torch.Tensor
    k2: torch.Tensor
    k3: torch.Tensor
    k4: torch.Tensor
    fx_p: torch.Tensor
    fy_p: torch.Tensor
    cx_p: torch.Tensor
    cy_p: torch.Tensor
    height: int
    width: int

    def to(self, device: torch.device) -> "KannalaBrandtCamera":
        moved = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in self.__dict__.items()
        }
        return KannalaBrandtCamera(**moved)

    @property
    def batch_size(self) -> int:
        return self.fx.shape[0]


def sample_kb_cameras(
    batch_size: int,
    height: int,
    width: int,
    perspective_intrinsics: torch.Tensor | None = None,
    default_hfov_deg: float = 60.0,
    focal_range: tuple[float, float] = (1.0, 1.2),
    principal_shift_px: float = 10.0,
    k123_range: tuple[float, float] = (-0.5, 0.5),
    k4_range: tuple[float, float] = (-0.05, 0.05),
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> KannalaBrandtCamera:
    """Sample random KB cameras following the paper's supplementary ranges.

    perspective_intrinsics: optional (B, 3, 3) pinhole K of the source images.
    When absent, a pinhole with `default_hfov_deg` horizontal FoV is assumed.
    """

    def uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(batch_size, device=device).uniform_(lo, hi, generator=generator)

    if perspective_intrinsics is not None:
        fx_p = perspective_intrinsics[:, 0, 0].to(device)
        fy_p = perspective_intrinsics[:, 1, 1].to(device)
        cx_p = perspective_intrinsics[:, 0, 2].to(device)
        cy_p = perspective_intrinsics[:, 1, 2].to(device)
    else:
        import math

        f = 0.5 * width / math.tan(math.radians(default_hfov_deg) / 2.0)
        fx_p = torch.full((batch_size,), f, device=device)
        fy_p = fx_p.clone()
        cx_p = torch.full((batch_size,), width / 2.0, device=device)
        cy_p = torch.full((batch_size,), height / 2.0, device=device)

    focal_scale = uniform(*focal_range)
    return KannalaBrandtCamera(
        fx=fx_p * focal_scale,
        fy=fy_p * focal_scale,
        cx=cx_p + uniform(-principal_shift_px, principal_shift_px),
        cy=cy_p + uniform(-principal_shift_px, principal_shift_px),
        k1=uniform(*k123_range),
        k2=uniform(*k123_range),
        k3=uniform(*k123_range),
        k4=uniform(*k4_range),
        fx_p=fx_p,
        fy_p=fy_p,
        cx_p=cx_p,
        cy_p=cy_p,
        height=height,
        width=width,
    )


def _r_of_theta(theta: torch.Tensor, cam: KannalaBrandtCamera) -> torch.Tensor:
    t2 = theta * theta
    k1 = cam.k1.view(-1, 1, 1)
    k2 = cam.k2.view(-1, 1, 1)
    k3 = cam.k3.view(-1, 1, 1)
    k4 = cam.k4.view(-1, 1, 1)
    return theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))


def _dr_dtheta(theta: torch.Tensor, cam: KannalaBrandtCamera) -> torch.Tensor:
    t2 = theta * theta
    k1 = cam.k1.view(-1, 1, 1)
    k2 = cam.k2.view(-1, 1, 1)
    k3 = cam.k3.view(-1, 1, 1)
    k4 = cam.k4.view(-1, 1, 1)
    return 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))


def _theta_monotonic_max(cam: KannalaBrandtCamera, theta_lim: float = 1.55, steps: int = 512) -> torch.Tensor:
    """Largest theta up to which r(theta) stays strictly increasing, per camera.

    The randomized k's can make the KB polynomial fold back inside the FoV;
    beyond that point the projection is not invertible, so we mask it out.
    Returns shape (B,).
    """
    theta = torch.linspace(1e-4, theta_lim, steps, device=cam.fx.device)
    grid = theta.view(1, 1, steps).expand(cam.batch_size, 1, steps)
    deriv_ok = _dr_dtheta(grid, cam) > 1e-3
    # First violation along the grid; everything before it is monotonic.
    first_bad = (~deriv_ok).float().argmax(dim=-1).view(-1)
    all_ok = deriv_ok.all(dim=-1).view(-1)
    idx = torch.where(all_ok, torch.full_like(first_bad, steps - 1), (first_bad - 1).clamp(min=0))
    return theta[idx]


def _invert_r(r_target: torch.Tensor, cam: KannalaBrandtCamera, theta_max: torch.Tensor, iters: int = 12) -> torch.Tensor:
    """Newton inversion of r(theta) = r_target. r_target: (B, H, W) >= 0."""
    theta = r_target.clamp(min=0.0, max=float(theta_max.max()))
    tmax = theta_max.view(-1, 1, 1)
    for _ in range(iters):
        resid = _r_of_theta(theta, cam) - r_target
        theta = theta - resid / _dr_dtheta(theta, cam).clamp(min=1e-3)
        theta = theta.clamp(min=0.0).minimum(tmax)
    return theta


def _normalize_grid(px: torch.Tensor, py: torch.Tensor, height: int, width: int) -> torch.Tensor:
    gx = 2.0 * px / max(width - 1, 1) - 1.0
    gy = 2.0 * py / max(height - 1, 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


def _pixel_grid(batch: int, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return gx.expand(batch, height, width), gy.expand(batch, height, width)


def distortion_grids(cam: KannalaBrandtCamera) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute the two resampling grids for T and T^-1.

    Returns (distort_grid, distort_valid, undistort_grid, undistort_valid):
      distort_grid   (B, H, W, 2): for each *fisheye* pixel, the normalized
                     source location in the perspective image (used by T).
      undistort_grid (B, H, W, 2): for each *perspective* pixel, the normalized
                     source location in the fisheye image (used by T^-1).
      *_valid        (B, H, W) bool masks of pixels whose mapping is defined.
    """
    B, H, W = cam.batch_size, cam.height, cam.width
    device = cam.fx.device
    theta_max = _theta_monotonic_max(cam)

    # ---- T: fisheye pixel -> perspective pixel ------------------------------
    gx, gy = _pixel_grid(B, H, W, device)
    rho_x = (gx - cam.cx.view(-1, 1, 1)) / cam.fx.view(-1, 1, 1)
    rho_y = (gy - cam.cy.view(-1, 1, 1)) / cam.fy.view(-1, 1, 1)
    r = torch.sqrt(rho_x * rho_x + rho_y * rho_y).clamp(min=1e-9)
    theta = _invert_r(r, cam, theta_max)
    # Ray direction: polar angle theta, azimuth (rho_x, rho_y)/r.
    tan_theta = torch.tan(theta.clamp(max=1.53))  # keep < ~87.7 deg for the pinhole plane
    px = cam.fx_p.view(-1, 1, 1) * tan_theta * (rho_x / r) + cam.cx_p.view(-1, 1, 1)
    py = cam.fy_p.view(-1, 1, 1) * tan_theta * (rho_y / r) + cam.cy_p.view(-1, 1, 1)
    resid = (_r_of_theta(theta, cam) - r).abs()
    distort_valid = (
        (theta < theta_max.view(-1, 1, 1) - 1e-4)
        & (resid < 1e-3)
        & (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
    )
    distort_grid = _normalize_grid(px, py, H, W)

    # ---- T^-1: perspective pixel -> fisheye pixel ---------------------------
    gx, gy = _pixel_grid(B, H, W, device)
    X = (gx - cam.cx_p.view(-1, 1, 1)) / cam.fx_p.view(-1, 1, 1)
    Y = (gy - cam.cy_p.view(-1, 1, 1)) / cam.fy_p.view(-1, 1, 1)
    rho = torch.sqrt(X * X + Y * Y).clamp(min=1e-9)
    theta_p = torch.atan(rho)
    r_p = _r_of_theta(theta_p, cam)
    fu = cam.fx.view(-1, 1, 1) * r_p * (X / rho) + cam.cx.view(-1, 1, 1)
    fv = cam.fy.view(-1, 1, 1) * r_p * (Y / rho) + cam.cy.view(-1, 1, 1)
    undistort_valid = (
        (theta_p < theta_max.view(-1, 1, 1) - 1e-4)
        & (fu >= 0) & (fu <= W - 1) & (fv >= 0) & (fv <= H - 1)
    )
    undistort_grid = _normalize_grid(fu, fv, H, W)

    return distort_grid, distort_valid, undistort_grid, undistort_valid


def _warp(x: torch.Tensor, grid: torch.Tensor, valid: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    out = F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=True)
    return out * valid.unsqueeze(1).to(out.dtype)


def distort_images(images: torch.Tensor, cam: KannalaBrandtCamera) -> tuple[torch.Tensor, torch.Tensor]:
    """T: synthesize fisheye images from perspective ones.

    images: (B, 3, H, W). Returns (fisheye_images, fisheye_valid (B, H, W)).
    """
    grid, valid, _, _ = distortion_grids(cam)
    return _warp(images, grid, valid), valid


def undistort_dense(dense: torch.Tensor, cam: KannalaBrandtCamera) -> tuple[torch.Tensor, torch.Tensor]:
    """T^-1: resample dense fisheye-domain predictions onto the perspective grid.

    dense: (B, C, H, W) scalar fields (depth / confidence / features). Values
    are ray-attached, so resampling is the correct transformation for z-depth
    shared between the two cameras. Returns (undistorted, valid (B, H, W)).
    """
    grids = distortion_grids(cam)
    _, distort_valid, grid, valid = grids
    # A perspective pixel is only supervisable if its fisheye source pixel was
    # itself rendered from valid perspective content: chain the two masks.
    chained = _warp(distort_valid.unsqueeze(1).float(), grid, valid, mode="nearest").squeeze(1) > 0.5
    return _warp(dense, grid, valid), chained & valid
