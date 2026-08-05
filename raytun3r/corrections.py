"""Parameter-free camera-model corrections.

Paper Sec. 4.2, paragraphs "Prediction-grid coordinates" and "Patch
tokenization". Nothing here is trainable -- these remove residual pinhole
assumptions that sit outside the positional encoding:

1. Patch tokenization: each patch is resampled with the local linearisation of
   the fisheye->pinhole map at its centre, so a tokenized patch looks more like a
   pinhole crop from the same viewing direction (following Qin and Li [43]).
2. Border tokens: patches outside the valid lens circle are replaced by the mean
   valid token, so the black vignette does not enter attention as content.
3. Prediction-grid coordinates: the DPT head's regular 2D grid is replaced by
   undistorted, camera-aware coordinates.

Two places need an interpretation the paper does not pin down; both are exposed
as options and both default to the numerically stable choice:

* **Scale of the patch linearisation.** A raw linearisation magnifies rim patches
  by a large factor on a 185-200 deg lens, so the resampler would read a few
  source pixels and upsample them. ``preserve_scale=True`` (default) divides the
  Jacobian by ``sqrt(|det|)``, correcting patch *shape* (anisotropy and shear)
  while keeping its footprint -- the part a pinhole-trained tokenizer actually
  cares about. Set ``preserve_scale=False`` for the literal linearisation.
* **Radial coordinate of the prediction grid.** The literal fisheye->pinhole map
  is ``tan(theta)``, which diverges at 90 deg and so is unusable at 185-200 deg
  FOV. ``grid_mode="auto"`` (default) picks ``tan`` for lenses inside
  ``tan_clamp_deg`` of half-angle and the angular radius ``theta`` beyond it.
  ``grid_mode="tan"`` forces the clamped ``tan``, rescaled to the span the head
  was trained on; ``grid_mode="angular"`` forces ``theta``, which is monotone
  over the full cone.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .cameras import Camera, pixel_grid

__all__ = [
    "patch_valid_mask",
    "local_undistort_jacobian",
    "patch_undistort_grid",
    "undistort_patches",
    "fill_border_tokens",
    "camera_aware_uv_grid",
]


def patch_valid_mask(camera: Camera, height: int, width: int, patch: int,
                     *, coverage: float = 0.5, device=None) -> Tensor:
    """``(grid_h, grid_w)`` bool: does this patch see enough of the imaged disc?

    A patch counts as valid when at least ``coverage`` of its pixels are inside
    the lens cone. Using coverage rather than the patch centre alone keeps the
    partially-vignetted ring of patches at the boundary, which carries real image
    content on wide lenses.
    """
    gh, gw = height // patch, width // patch
    valid = camera.valid_mask(height, width, device=device).to(torch.float32)
    frac = valid[: gh * patch, : gw * patch].reshape(gh, patch, gw, patch).mean(dim=(1, 3))
    return frac >= coverage


def local_undistort_jacobian(camera: Camera, height: int, width: int, patch: int,
                             *, device=None, dtype=torch.float32) -> Tensor:
    """``(grid_h, grid_w, 2, 2)`` Jacobian of the local fisheye->pinhole map.

    The map is taken in a *local* tangent (gnomonic) frame about each patch's own
    viewing direction rather than one global pinhole. A single global pinhole is
    undefined at 90 deg incidence and therefore cannot cover the 185-200 deg
    lenses in the paper's benchmark; the local frame is well conditioned
    everywhere inside the cone and agrees with the global map near the axis.

    Central differences in float64, not autograd: ``Camera.unproject`` divides by
    the image radius, so its gradient is NaN at the principal point -- exactly the
    patch where the correction has to come out as the identity.
    """
    gh, gw = height // patch, width // patch
    cam = camera.resized(width, height) if (height, width) != (camera.height, camera.width) else camera

    py = (torch.arange(gh, device=device, dtype=torch.float64) + 0.5) * patch - 0.5
    px = (torch.arange(gw, device=device, dtype=torch.float64) + 0.5) * patch - 0.5
    vv, uu = torch.meshgrid(py, px, indexing="ij")
    centres = torch.stack((uu, vv), dim=-1).reshape(-1, 2)

    with torch.no_grad():
        d = cam.unproject(centres)                                    # (N, 3) unit
        # Tangent basis from the shortest-arc rotation taking +z onto d. This is
        # the frame of a tangent view about the ray: it is exactly the image axes
        # on-axis (so the correction is the identity at the principal point) and
        # varies continuously out to theta -> 180 deg, unlike a cross-product
        # basis built from a fixed helper axis, which flips near the poles.
        dx, dy, dz = d.unbind(-1)
        denom = (1.0 + dz).clamp_min(1e-6)
        e1 = F.normalize(torch.stack(
            (1.0 - dx * dx / denom, -dx * dy / denom, -dx), dim=-1), dim=-1)
        e2 = F.normalize(torch.stack(
            (-dx * dy / denom, 1.0 - dy * dy / denom, -dy), dim=-1), dim=-1)

        # Local gnomonic coordinates, scaled by the on-axis focal so the map is
        # the identity to first order at the image centre.
        f = 0.5 * (cam.fx + cam.fy)

        def local(p: Tensor) -> Tensor:
            r = cam.unproject(p)
            z = (r * d).sum(-1).clamp_min(1e-6)
            return torch.stack((f * (r * e1).sum(-1) / z,
                                f * (r * e2).sum(-1) / z), dim=-1)

        step = 1e-4
        cols = []
        for axis in range(2):
            off = torch.zeros(2, device=centres.device, dtype=centres.dtype)
            off[axis] = step
            cols.append((local(centres + off) - local(centres - off)) / (2 * step))
        jac = torch.stack(cols, dim=-1)                               # (N, 2, 2)

    return jac.reshape(gh, gw, 2, 2).to(dtype)


def patch_undistort_grid(camera: Camera, height: int, width: int, patch: int,
                         *, preserve_scale: bool = True, device=None,
                         dtype=torch.float32) -> Tensor:
    """``(height, width, 2)`` ``grid_sample`` field that locally undistorts patches.

    Output pixel ``(patch i, offset o)`` reads the source image at
    ``centre_i + A_i^{-1} o``, where ``A_i`` is the local fisheye->pinhole
    Jacobian. Returned in normalised ``align_corners=False`` coordinates.
    """
    gh, gw = height // patch, width // patch
    jac = local_undistort_jacobian(camera, height, width, patch, device=device, dtype=dtype)

    if preserve_scale:
        det = jac.det().abs().clamp_min(1e-8).sqrt()
        jac = jac / det[..., None, None]

    # Invert the 2x2 blocks; fall back to identity where the block is singular.
    det = jac.det()
    ok = det.abs() > 1e-8
    inv = torch.zeros_like(jac)
    a, b = jac[..., 0, 0], jac[..., 0, 1]
    c, d = jac[..., 1, 0], jac[..., 1, 1]
    safe_det = torch.where(ok, det, torch.ones_like(det))
    inv[..., 0, 0] = d / safe_det
    inv[..., 0, 1] = -b / safe_det
    inv[..., 1, 0] = -c / safe_det
    inv[..., 1, 1] = a / safe_det
    eye = torch.eye(2, device=jac.device, dtype=jac.dtype).expand_as(inv)
    inv = torch.where(ok[..., None, None], inv, eye)

    # Intra-patch offsets, centred on the patch.
    off = torch.arange(patch, device=device, dtype=dtype) - (patch - 1) / 2.0
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    offs = torch.stack((ox, oy), dim=-1)                              # (p, p, 2)

    # (gh, gw, p, p, 2) = A^-1 @ offset, broadcast over the token grid.
    src = torch.einsum("hwij,pqj->hwpqi", inv, offs)

    py = (torch.arange(gh, device=device, dtype=dtype) + 0.5) * patch - 0.5
    px = (torch.arange(gw, device=device, dtype=dtype) + 0.5) * patch - 0.5
    vv, uu = torch.meshgrid(py, px, indexing="ij")
    src = src + torch.stack((uu, vv), dim=-1)[:, :, None, None, :]

    src = src.permute(0, 2, 1, 3, 4).reshape(gh * patch, gw * patch, 2)

    # Pixel centres -> normalised coords for align_corners=False.
    norm = torch.empty_like(src)
    norm[..., 0] = 2.0 * (src[..., 0] + 0.5) / width - 1.0
    norm[..., 1] = 2.0 * (src[..., 1] + 0.5) / height - 1.0
    return norm


def undistort_patches(images: Tensor, grid: Tensor, *, padding_mode: str = "border") -> Tensor:
    """Apply a :func:`patch_undistort_grid` field to ``(B, C, H, W)`` images."""
    b = images.shape[0]
    g = grid.to(images.dtype).to(images.device).unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(images, g, mode="bilinear", padding_mode=padding_mode,
                         align_corners=False)


def fill_border_tokens(tokens: Tensor, valid: Tensor) -> Tensor:
    """Replace invalid patch tokens by the mean valid token.

    ``tokens`` is ``(B, N, C)`` over the patch grid (no prefix tokens), ``valid``
    is ``(N,)`` or ``(B, N)`` bool. If every patch is valid this is a no-op, and
    if none are the tokens pass through untouched rather than becoming NaN.
    """
    if valid.dim() == 1:
        valid = valid.unsqueeze(0).expand(tokens.shape[0], -1)
    if bool(valid.all()) or not bool(valid.any()):
        return tokens
    w = valid.to(tokens.dtype).unsqueeze(-1)
    mean = (tokens * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.where(valid.unsqueeze(-1), tokens, mean.expand_as(tokens))


def camera_aware_uv_grid(camera: Camera, grid_w: int, grid_h: int, *,
                         aspect_ratio: Optional[float] = None,
                         mode: str = "auto", tan_clamp_deg: float = 80.0,
                         device=None, dtype=torch.float32) -> Tensor:
    """Camera-aware replacement for VGGT/DA3's ``create_uv_grid``.

    Returns ``(grid_h, grid_w, 2)`` -- the same layout and normalisation as the
    original (unit half-diagonal), so it is a drop-in for the DPT head's
    positional grid. The difference is that coordinates now follow the *rays* of
    the calibrated fisheye rather than a uniform pinhole grid.

    Note the shape: ``create_uv_grid``'s own docstring claims ``(width, height,
    2)``, but it builds the grid with ``meshgrid(x, y, indexing="xy")``, which
    returns ``(len(y), len(x)) = (height, width)``. The caller then does
    ``permute(2, 0, 1)`` and adds the result to an ``(B, C, patch_h, patch_w)``
    feature map, which only lines up under the ``(H, W, 2)`` reading. Following
    the docstring instead transposes the grid: a shape error on non-square input
    and, worse, a *silent* x/y swap when the token grid happens to be square.
    """
    img_h, img_w = camera.height, camera.width
    patch_h, patch_w = img_h / grid_h, img_w / grid_w

    py = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) * patch_h - 0.5
    px = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) * patch_w - 0.5
    vv, uu = torch.meshgrid(py, px, indexing="ij")
    uv = torch.stack((uu, vv), dim=-1)                                # (gh, gw, 2)

    with torch.no_grad():
        rays = camera.unproject(uv.reshape(-1, 2)).reshape(grid_h, grid_w, 3)
    theta = torch.acos(rays[..., 2].clamp(-1.0, 1.0))
    phi = torch.atan2(rays[..., 1], rays[..., 0])

    if mode == "auto":
        # tan() saturates once theta reaches the clamp: on a 180-200 deg lens the
        # outermost grid cells would all receive the same coordinate, collapsing
        # positional resolution exactly where the adapter needs it (the failure
        # Appendix C describes for naive PE remapping). Fall back to the angular
        # radius, which stays monotone over the whole cone.
        mode = "tan" if camera.theta_max <= math.radians(tan_clamp_deg) else "angular"
    if mode == "tan":
        radius = torch.tan(theta.clamp(max=math.radians(tan_clamp_deg)))
    elif mode == "angular":
        radius = theta
    else:
        raise ValueError(f"unknown grid mode {mode!r}; expected 'auto', 'tan' or 'angular'")

    xy = torch.stack((radius * torch.cos(phi), radius * torch.sin(phi)), dim=-1)

    # Rescale to the span the pretrained head expects, matching create_uv_grid.
    if aspect_ratio is None:
        aspect_ratio = float(grid_w) / float(grid_h)
    diag = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag * (grid_w - 1) / grid_w
    span_y = 1.0 / diag * (grid_h - 1) / grid_h

    # Normalise using only grid cells whose ray is inside the imaged cone, so a
    # handful of vignette corners cannot compress the whole grid.
    sel = theta <= camera.theta_max
    if not bool(sel.any()):
        sel = torch.ones_like(sel)

    mx = xy[..., 0][sel].abs().max().clamp_min(1e-6)
    my = xy[..., 1][sel].abs().max().clamp_min(1e-6)
    xy = torch.stack((xy[..., 0] / mx * span_x, xy[..., 1] / my * span_y), dim=-1)

    # (grid_h, grid_w, 2), matching what create_uv_grid actually returns.
    return xy.contiguous()
