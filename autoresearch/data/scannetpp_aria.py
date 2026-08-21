"""Resample a ScanNet++ DSLR frame into Aria's lens, keeping the void explicit.

THE PROBLEM THIS ENCODES
------------------------
ScanNet++'s DSLR is a ~169 deg-diagonal fisheye covering the whole rectangle.
Aria is a 109.66 deg cone inscribed as a disc in a square, with dark corners.
Resampling the former into the latter does not fill the latter: measured on
scene 00777c41d4, ScanNet++'s VERTICAL half-FOV is 51.53 deg against Aria's
54.83, so a crescent at the top and bottom of Aria's disc has NO source pixel.

    Aria disc                     75.51 % of the square
    of the disc, covered           98.06 %
    of the disc, VOID               1.94 %   at incidence 51.57-54.83 deg
    void as a share of the RIM band 3.21 %

1.94 % sounds ignorable. It is not, because all of it sits in the outermost
band -- the exact zone every experiment in this repo is about. The decision
(human, 2026-08-22) is to **keep the void and mask it explicitly, never fill
it**. So this module returns a validity mask alongside every array and there is
no code path that substitutes a value for a missing one. `fovbench` has been
bitten before by a sampler quietly averaging in padding; the guard here is that
the mask is a *return value*, not an option.

THREE THINGS THAT ARE EASY TO GET WRONG
---------------------------------------
1. **Depth resamples with NEAREST, never bilinear.** Interpolating depth across
   a silhouette invents a surface at the average of foreground and background,
   in a thin ribbon around every object. That ribbon is a smooth radial-looking
   error, i.e. shaped exactly like the effect being measured.
2. **Planar z is invariant here, so the VALUES are not converted.** This is a
   pure lens re-parameterisation: same optical centre, same optical axis, no
   rotation. A 3D point keeps its distance along that axis, so z resamples
   unchanged. (Under any rotation this stops being true and `convert_depth`
   would be required both ways -- see docs/research/dataset-scope-2026-08.md.)
   ScanNet++'s `render_depth` is planar z in millimetres and this project's ADT
   depth is planar z too (ticket 016), so the two agree by construction.
3. **Mesh holes bleed.** 97.84 % of pure-white pixels in `render_rgb` are
   zero-depth holes, but a further 0.312 % of the frame is white with NONZERO
   depth -- antialiased hole boundaries whose RGB is already contaminated. They
   are dropped by dilating the hole mask by one pixel before it is warped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["AriaRemap", "build_maps"]


def build_maps(src_cam, dst_cam, dst_hw: Tuple[int, int]
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(map_x, map_y, in_cone, void)`` taking ``dst`` pixels from ``src``.

    ``in_cone`` is Aria's imaged disc (everything outside is the black corner
    mask and was never imaged). ``void`` is the part of that disc for which the
    source lens has no pixel -- the thing this module exists to keep visible.
    """
    import torch
    h, w = dst_hw
    # Both cameras come from raytun3r.cameras (from_aria / from_scannetpp), so
    # the lens of record is defined in exactly one place -- ticket 018's rule.
    with torch.no_grad():
        rays = dst_cam.ray_grid(h, w, dtype=torch.float64)      # (h, w, 3)
        in_cone = dst_cam.valid_mask(h, w, dtype=torch.float64).numpy()
        uv = src_cam.project(rays)                              # (h, w, 2)
    u = uv[..., 0].numpy()
    v = uv[..., 1].numpy()
    # A ray can leave the SOURCE lens two ways: land off the sensor, or exceed
    # the source's own imaged cone. Both are "no pixel here"; neither may be
    # filled. project() still returns a number past the cone, so check theta
    # rather than trusting the coordinate.
    theta_src = torch.acos(rays[..., 2].clamp(-1.0, 1.0)).numpy()
    inside = (np.isfinite(u) & np.isfinite(v)
              & (u >= 0) & (u <= src_cam.width - 1)
              & (v >= 0) & (v <= src_cam.height - 1)
              & (theta_src <= float(src_cam.theta_max)))
    void = in_cone & ~inside
    mx = np.where(in_cone & inside, u, -1.0).astype(np.float32)
    my = np.where(in_cone & inside, v, -1.0).astype(np.float32)
    return mx, my, in_cone.astype(bool), void.astype(bool)


@dataclass
class AriaRemap:
    """Precomputed remap from one ScanNet++ camera into one Aria camera."""
    map_x: np.ndarray
    map_y: np.ndarray
    in_cone: np.ndarray
    void: np.ndarray

    @property
    def covered(self) -> np.ndarray:
        """Disc pixels that DO have a source pixel."""
        return self.in_cone & ~self.void

    def stats(self) -> dict:
        disc = int(self.in_cone.sum())
        return {"disc_frac_of_square": float(self.in_cone.mean()),
                "covered_frac_of_disc": float(self.covered.sum() / max(disc, 1)),
                "void_frac_of_disc": float(self.void.sum() / max(disc, 1))}

    def image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """RGB -> (warped, valid). Void and corners come back as 0 AND masked."""
        import cv2
        out = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[~self.covered] = 0
        return out, self.covered.copy()

    def depth(self, depth_mm: np.ndarray,
              hole_dilate: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Planar-z millimetres -> (warped mm, valid).

        NEAREST on purpose (see module docstring). The source's own holes
        (`depth == 0`) are dilated before warping so antialiased boundary
        pixels do not survive as thin ribbons of invented surface.
        """
        import cv2
        hole = (depth_mm == 0).astype(np.uint8)
        if hole_dilate > 0:
            hole = cv2.dilate(hole, np.ones((2 * hole_dilate + 1,) * 2, np.uint8))
        d = cv2.remap(depth_mm.astype(np.float32), self.map_x, self.map_y,
                      cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=0)
        hw = cv2.remap(hole, self.map_x, self.map_y, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=1)
        valid = self.covered & (hw == 0) & (d > 0)
        d[~valid] = 0
        return d, valid

    @classmethod
    def build(cls, src_cam, dst_cam, dst_hw: Tuple[int, int]) -> "AriaRemap":
        return cls(*build_maps(src_cam, dst_cam, dst_hw))
