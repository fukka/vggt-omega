"""Transport between an Aria fisheye frame and a CO-AXIAL virtual pinhole.

WHY THIS EXISTS
---------------
Ticket 024A measured the one fact this experiment is built on: after the
GT-depth (distance) control, the rim/centre AbsRel ratio on RAW fisheye is
1.25-1.81x depending on backbone, and on RECTIFIED input it collapses to
~1.0 -- "the rect rim penalty is mostly the furniture". Same pixels, same
weights, same scene. The information the model needs is evidently present at
the rim; what it cannot do is read it out of a fisheye image.

Four rim-targeted interventions have since lost to their own controls (H5, H6,
H7+MoE, H12). Every intervention that has helped is a GLOBAL lens operation.
So instead of designing a fifth mechanism, this transports the answer the
backbone ALREADY GIVES in pinhole space back into the fisheye domain and
distils it -- no depth labels anywhere in the loop.

WHAT IS AND IS NOT CONVERTED
----------------------------
The virtual pinhole shares the fisheye's optical centre AND its axis. Under a
pure lens re-parameterisation with no rotation, a pixel in either camera that
sees the same world point sees it along the SAME RAY. Both planar z and
euclidean range are functions of (ray, point) alone -- z = range * cos(theta)
with the same theta on both sides -- so BOTH are invariant and the transport is
a resampling with NO radiometric or geometric conversion.

That is a licence to do nothing, and doing nothing is exactly the step this
repo has got wrong before: `raytun3r_row.py` divided an already-range GT by
cos(theta) a second time and inflated rim GT by up to 1.73x, flipping a
published conclusion (#38 v1, quarantined). So co-axiality is ASSERTED here
rather than assumed (`assert_shared_axis`), and
`test_rect_teacher.py::test_range_and_z_are_both_invariant_under_the_transport`
checks the invariance numerically instead of taking this docstring's word.

Pure torch on the geometry only -- no backbone import -- so the whole module is
testable on a CPU with no weights and no data.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from raytun3r.cameras import Camera, Pinhole, pixel_grid  # noqa: E402

__all__ = ["virtual_pinhole", "assert_shared_axis", "grid_fisheye_to_pinhole",
           "grid_pinhole_to_fisheye", "warp", "coverage", "OUT_OF_RANGE"]

#: Normalised grid value that lands outside [-1, 1] in both axes, so
#: ``grid_sample(padding_mode="zeros")`` returns 0 and the companion mask says
#: so. Anything inside the range would be a fabricated observation.
OUT_OF_RANGE = -2.0


def virtual_pinhole(fisheye: Camera, fov_deg: float, size: int,
                    patch: int = 14) -> Pinhole:
    """A co-axial square pinhole that images the WHOLE fisheye cone.

    ``fov_deg`` is the horizontal field of view, so the imaged disc
    ``theta <= theta_max`` fits inside the pinhole frame iff
    ``fov_deg >= 2 * theta_max``. Aria's usable cone is 54.83 deg, hence the
    110 deg default at the call sites: 109.66 deg is the exact requirement and
    110 clears it.

    ``size`` is a free parameter with a real cost on both sides, so it is
    chosen rather than defaulted at the call site:

      * a 110 deg pinhole has focal ``(size/2)/tan(55 deg) = 0.3501 * size``,
        against the fisheye's measured ``fx = 218.69`` at 504 px, so rendering
        at the SAME size gives 176.45 px and DOWNSAMPLES the centre by 0.807x.
        The centre is the region every method in this project has had to
        protect; blurring it in the teacher is the obvious way to poison the
        student there.
      * ``size = 630`` restores centre parity (0.3501*630 = 220.6, ratio
        1.009) and oversamples the rim, which is free accuracy: upsampling
        loses nothing. 630 = 45*14, so it is also patch-aligned.

    An earlier draft of this docstring said 0.72x and 700 px, from a guessed
    ``0.487*size`` focal. The real one is ``610.94 * size/1408`` (cam3r's
    calibration of record); the test below pins the corrected numbers against
    the camera itself rather than against arithmetic done in a comment.

    Size must stay a multiple of the ViT patch, or `Backbone.install` refuses
    it -- checked here so the failure lands at construction rather than a
    hundred lines into a cache run.
    """
    if size % patch:
        raise ValueError(
            f"pinhole size {size} is not a multiple of patch {patch}; "
            f"Backbone.install would reject it after the data is loaded")
    need = 2.0 * math.degrees(fisheye.theta_max)
    if fov_deg < need - 1e-6:
        raise ValueError(
            f"fov_deg={fov_deg} does not image the whole cone: theta_max is "
            f"{math.degrees(fisheye.theta_max):.2f} deg, so at least "
            f"{need:.2f} deg is required. A narrower pinhole silently drops "
            f"the rim -- which is the region this experiment exists to fix.")
    return fisheye.to_pinhole(fov_deg=fov_deg, width=size, height=size)


def assert_shared_axis(fisheye: Camera, pin: Pinhole, tol_deg: float = 1e-3
                       ) -> None:
    """Refuse to transport depth between cameras that do not share an axis.

    The no-conversion transport in this module is valid ONLY for a pure lens
    re-parameterisation: same optical centre, same optical AXIS, no rotation.
    The condition is NOT "the principal point sits at the frame centre" --
    Aria's is 4.5 px off centre at 504 px, and an earlier draft of this file
    rejected the real camera for it. What matters is that each camera sends its
    own principal point to +z, i.e. both are central cameras expressed in the
    same frame.

    A future tilted view (raytun3r's `VirtualView` with `R_vc != I`) would need
    `raytun3r.cameras.convert_depth` on both sides, and would be a smooth
    radial error if it did not -- the exact signature of "the model is bad at
    the rim", which is the conclusion this project is trying to measure.
    """
    for cam, name in ((fisheye, "fisheye"), (pin, "pinhole")):
        pp = torch.tensor([[cam.cx, cam.cy]], dtype=torch.float64)
        axis = cam.unproject(pp)[0]
        off = math.degrees(math.acos(min(1.0, max(-1.0, float(axis[2])))))
        if off > tol_deg:
            raise ValueError(
                f"{name} camera does not send its principal point to +z "
                f"(off by {off:.4f} deg); the two cameras are then not "
                f"co-axial and depth may not be transported without "
                f"conversion.")


def _normalise(uv: Tensor, width: int, height: int) -> Tensor:
    """Pixel addresses -> ``grid_sample`` coordinates (align_corners=False)."""
    return torch.stack((2.0 * (uv[..., 0] + 0.5) / width - 1.0,
                        2.0 * (uv[..., 1] + 0.5) / height - 1.0), dim=-1)


def grid_fisheye_to_pinhole(fisheye: Camera, pin: Pinhole
                            ) -> Tuple[Tensor, Tensor]:
    """``(grid, valid)`` pulling a FISHEYE map into the PINHOLE frame.

    ``grid`` is ``(Hp, Wp, 2)`` for ``grid_sample`` on a fisheye-sized tensor;
    ``valid`` is ``(Hp, Wp)`` bool -- pinhole pixels whose ray is inside the
    fisheye's imaged cone. Pixels outside it are pushed out of range so the
    sampler returns 0 rather than a border-replicated pixel: a replicated pixel
    is a fabricated observation and the model would be scored on its response
    to one.
    """
    assert_shared_axis(fisheye, pin)
    uv = pixel_grid(pin.height, pin.width, dtype=torch.float32)
    rays = pin.unproject(uv)
    theta = torch.acos(rays[..., 2].clamp(-1.0, 1.0))
    src = fisheye.project(rays)
    valid = (theta <= fisheye.theta_max) & torch.isfinite(src).all(-1)
    valid &= ((src[..., 0] >= 0) & (src[..., 0] <= fisheye.width - 1)
              & (src[..., 1] >= 0) & (src[..., 1] <= fisheye.height - 1))
    g = _normalise(src, fisheye.width, fisheye.height)
    g = torch.where(valid[..., None], g, torch.full_like(g, OUT_OF_RANGE))
    return g, valid


def grid_pinhole_to_fisheye(fisheye: Camera, pin: Pinhole
                            ) -> Tuple[Tensor, Tensor]:
    """``(grid, valid)`` pulling a PINHOLE map back onto the FISHEYE grid.

    ``valid`` is the mask this experiment calls ``covered``: fisheye pixels
    that are inside the imaged cone AND land inside the pinhole frame. With a
    cone-covering ``fov_deg`` it is the whole disc, which is the point --
    Center-PH's measured failure on ADT was 49.6% near-rim coverage, and a
    teacher that cannot see the rim cannot teach it.
    """
    assert_shared_axis(fisheye, pin)
    rays = fisheye.ray_grid(fisheye.height, fisheye.width)
    theta = torch.acos(rays[..., 2].clamp(-1.0, 1.0))
    uv = pin.project(rays)
    valid = ((theta <= fisheye.theta_max) & (rays[..., 2] > 1e-6)
             & torch.isfinite(uv).all(-1))
    valid &= ((uv[..., 0] >= 0) & (uv[..., 0] <= pin.width - 1)
              & (uv[..., 1] >= 0) & (uv[..., 1] <= pin.height - 1))
    g = _normalise(uv, pin.width, pin.height)
    g = torch.where(valid[..., None], g, torch.full_like(g, OUT_OF_RANGE))
    return g, valid


def warp(src: Tensor, grid: Tensor, mode: str = "bilinear") -> Tensor:
    """Resample ``src`` (..., H, W) with a ``(Ho, Wo, 2)`` grid.

    Accepts (H, W), (C, H, W) or (N, C, H, W) and returns the same rank. RGB is
    bilinear and depth should be bilinear too HERE (unlike a GT depth map,
    where nearest is mandatory): the teacher's depth is a smooth prediction,
    not a mesh render with holes, so there is no discontinuity to protect and
    nearest would just add quantisation noise to the target.
    """
    squeeze = []
    x = src
    if x.dim() == 2:
        x = x[None, None]
        squeeze = [0, 0]
    elif x.dim() == 3:
        x = x[None]
        squeeze = [0]
    g = grid.to(x.device, x.dtype)[None].expand(x.shape[0], -1, -1, -1)
    out = F.grid_sample(x, g, mode=mode, padding_mode="zeros",
                        align_corners=False)
    for _ in squeeze:
        out = out[0]
    return out


def coverage(fisheye: Camera, pin: Pinhole) -> float:
    """Fraction of the fisheye's imaged cone the pinhole can answer for."""
    _, valid = grid_pinhole_to_fisheye(fisheye, pin)
    cone = fisheye.valid_mask(fisheye.height, fisheye.width)
    return float((valid & cone).sum()) / float(cone.sum())
