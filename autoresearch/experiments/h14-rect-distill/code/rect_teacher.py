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
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from raytun3r.cameras import Camera, Pinhole, pixel_grid  # noqa: E402

__all__ = ["virtual_pinhole", "assert_shared_axis", "grid_fisheye_to_pinhole",
           "grid_pinhole_to_fisheye", "warp", "coverage", "Rig", "ViewSpec", "OUT_OF_RANGE"]

#: Normalised grid value that lands outside [-1, 1] in both axes, so
#: ``grid_sample(padding_mode="zeros")`` returns 0 and the companion mask says
#: so. Anything inside the range would be a fabricated observation.
OUT_OF_RANGE = -2.0


def virtual_pinhole(fisheye: Camera, fov_deg: float, size: int,
                    patch: int = 14, require_full_cone: bool = True) -> Pinhole:
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
    if require_full_cone and fov_deg < need - 1e-6:
        raise ValueError(
            f"fov_deg={fov_deg} does not image the whole cone: theta_max is "
            f"{math.degrees(fisheye.theta_max):.2f} deg, so at least "
            f"{need:.2f} deg is required. A narrower pinhole silently drops "
            f"the rim -- which is the region this experiment exists to fix. "
            f"Pass require_full_cone=False only for a diagnostic sweep.")
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


# ---------------------------------------------------------------------------
# The view rig: what the teacher is actually shown
# ---------------------------------------------------------------------------
#
# The FOV sweep on seq131 (60 frames, teacher vs raw on identical pixels):
#
#   fov   cone cov   frame real   near_rim    near_center   center     far
#    85     0.666      100.0%     -41.07%      +50.86%      -1.55%   -15.12%
#    89     0.735      100.0%     -37.10%      +49.39%      -4.30%   -15.14%
#    95     0.837       98.7%     -35.33%      +34.36%      -5.44%   -13.64%
#   110     1.000       77.5%     +15.27%      +33.48%     +13.07%   +13.28%
#
# 024A reproduces -- the teacher is 35-41% better at the near rim -- as long as
# the pinhole frame is real content, and inverts in EVERY zone exactly when the
# frame goes 22.5% black. A large black vignette is an image statistic the
# backbone has never seen, and it costs more than the rim coverage buys.
#
# THE HIDDEN CONSTRAINT THAT COST A DESIGN
# ----------------------------------------
# The first reading of that sweep was "covering the cone and keeping the frame
# filled are not simultaneously achievable", from
#
#     tilt + atan(sqrt(2) * tan(fov/2)) <= theta_max
#
# and the sqrt(2) in it is the diagonal OF A SQUARE. The frame being square was
# never a requirement -- a ViT wants a patch-aligned rectangle, not a square --
# and the whole impossibility was an artefact of assuming one. A view stretched
# TANGENTIALLY and squeezed RADIALLY sweeps along the annulus instead of
# reaching across it, and stays inside the cone while doing it:
#
#   layout        views  frame                 cone coverage   mean fill
#   single 95      1     630 x 630 square      0.84            0.99
#   square 5x90    5     630 x 630 square      1.00            0.66   <- worse
#   ring           1+6   630^2 + 280 x 154     1.00            ~1.00
#
# `Rig.ring` is that layout: one 89 deg centre view for the middle of the cone,
# and six tangentially-elongated views tilted to 36 deg that tile the annulus.
#
# Views are run independently, so their depths are only defined up to their own
# scale. `Rig` aligns each to the centre view on their overlap before fusing,
# and unaligned fusion would stitch a step discontinuity into the target along
# every view boundary that the student would then learn.


@dataclass
class ViewSpec:
    """One virtual pinhole: where it points and what shape its frame is.

    ``fov_x_deg`` fixes the focal length together with ``width``; ``height``
    then fixes the vertical field, so the frame's aspect ratio IS the radial /
    tangential trade-off. ``tilt_deg`` is the polar angle of the view axis and
    ``azimuth_deg`` where it points around the cone. The view's own +x axis is
    laid along the TANGENTIAL direction, so widening ``fov_x`` sweeps around
    the annulus and does not reach further out of the cone.
    """

    fov_x_deg: float
    width: int
    height: int
    tilt_deg: float = 0.0
    azimuth_deg: float = 0.0

    def focal(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.fov_x_deg) / 2.0)

    def fov_y_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.height / 2.0) / self.focal()))

    def check_patch(self, patch: int = 14) -> None:
        if self.width % patch or self.height % patch:
            raise ValueError(
                f"view {self.width}x{self.height} is not patch({patch})-aligned; "
                f"Backbone.install would reject it after the data is loaded")

    def rotation(self) -> Tensor:
        """``R_vc``: view frame -> fisheye frame, columns (x_v, y_v, z_v)."""
        t = math.radians(self.tilt_deg)
        ph = math.radians(self.azimuth_deg)
        if abs(t) < 1e-9:
            return torch.eye(3)
        st, ct, sp, cp = math.sin(t), math.cos(t), math.sin(ph), math.cos(ph)
        z_v = torch.tensor([st * cp, st * sp, ct])            # the view axis
        radial = torch.tensor([ct * cp, ct * sp, -st])        # increasing theta
        tangential = torch.tensor([-sp, cp, 0.0])             # around the cone
        # x = tangential (the long axis), y = INWARD radial, z = axis.
        # The inward sign is what makes it right-handed: tangential x radial
        # = -axis, so the image's +v must point toward the cone's centre.
        # det(R) = -1 otherwise, which is a mirror, and every view would be
        # sampled flipped -- checked by test_rotation_is_a_rotation.
        return torch.stack([tangential, -radial, z_v], dim=1)


@dataclass
class _View:
    spec: ViewSpec
    pin: Pinhole
    R_vc: Tensor                 # rotates a ray from view frame -> fisheye frame
    grid_in: Tensor              # (h, w, 2) sample the fisheye to build this view
    fill: Tensor                 # (h, w) bool: view pixels with real content
    addr: Tensor                 # (Hf, Wf, 2) where each fisheye ray lands here
    cover: Tensor                # (Hf, Wf) bool: fisheye pixels this view sees
    cos_local: Tensor            # (Hf, Wf) cos of the angle to THIS view's axis


class Rig:
    """A set of co-centred virtual pinholes covering a fisheye cone.

    The depth bookkeeping, once, explicitly. The backbone is installed with
    ``depth_convention="z"``, its native one, so nothing is converted inside
    it. What comes back is planar z **in the view's frame**. For a fisheye ray
    ``r``, ``local = r @ R_vc`` is that ray in the view's frame, so
    ``range = z_view / local_z`` -- and because both cameras share an optical
    centre, that range is the range along the fisheye ray. Exactly one
    division, against the axis the value is defined by. Installing the fisheye
    and letting `_finalize` divide by the FISHEYE's cos would use the wrong
    axis for every tilted view, and a wrong cos is a smooth radial error that
    scale alignment cannot absorb.
    """

    def __init__(self, fisheye: Camera, views: Sequence[ViewSpec],
                 patch: int = 14):
        if not views:
            raise ValueError("a rig needs at least one view")
        self.fisheye = fisheye
        self.specs = list(views)
        hf, wf = fisheye.height, fisheye.width
        rays = fisheye.ray_grid(hf, wf)
        theta_f = torch.acos(rays[..., 2].clamp(-1.0, 1.0))
        in_cone = theta_f <= fisheye.theta_max

        self.views: List[_View] = []
        for spec in self.specs:
            spec.check_patch(patch)
            f = spec.focal()
            pin = Pinhole(fx=f, fy=f, cx=(spec.width - 1) / 2.0,
                          cy=(spec.height - 1) / 2.0,
                          width=spec.width, height=spec.height,
                          theta_max=math.radians(89.0))
            R = spec.rotation()
            uv = pixel_grid(spec.height, spec.width, dtype=torch.float32)
            vr = pin.unproject(uv) @ R.transpose(0, 1)        # view -> fisheye
            th = torch.acos(vr[..., 2].clamp(-1.0, 1.0))
            src = fisheye.project(vr)
            fill = (th <= fisheye.theta_max) & torch.isfinite(src).all(-1)
            fill &= ((src[..., 0] >= 0) & (src[..., 0] <= wf - 1)
                     & (src[..., 1] >= 0) & (src[..., 1] <= hf - 1))
            gin = _normalise(src, wf, hf)
            gin = torch.where(fill[..., None], gin, torch.full_like(gin, OUT_OF_RANGE))

            local = rays @ R                                  # fisheye -> view
            at = pin.project(local)
            cov = (in_cone & (local[..., 2] > 1e-6) & torch.isfinite(at).all(-1))
            cov &= ((at[..., 0] >= 0) & (at[..., 0] <= spec.width - 1)
                    & (at[..., 1] >= 0) & (at[..., 1] <= spec.height - 1))
            addr = _normalise(at, spec.width, spec.height)
            addr = torch.where(cov[..., None], addr, torch.full_like(addr, OUT_OF_RANGE))
            self.views.append(_View(spec, pin, R, gin, fill, addr, cov,
                                    local[..., 2].clamp_min(1e-6)))

        self.in_cone = in_cone
        self.covered = torch.zeros_like(in_cone)
        for v in self.views:
            self.covered |= v.cover
        self.covered &= in_cone

    # -- constructors ------------------------------------------------------
    @classmethod
    def single(cls, fisheye: Camera, fov_deg: float = 95.0, size: int = 630,
               **kw) -> "Rig":
        """One co-axial square view -- the H14 configuration."""
        return cls(fisheye, [ViewSpec(fov_x_deg=fov_deg, width=size, height=size)], **kw)

    @classmethod
    def square_multi(cls, fisheye: Camera, fov_deg: float = 90.0, size: int = 630,
                     n: int = 5, tilt_deg: float = 40.0, **kw) -> "Rig":
        """The square tilted rig. Kept because its FAILURE is the measurement
        that motivated the ring: at fov 90 / tilt 40 it fills only 66% of its
        frames, worse than the 110 deg single view that already inverted."""
        specs = [ViewSpec(fov_x_deg=fov_deg, width=size, height=size)]
        for k in range(n - 1):
            specs.append(ViewSpec(fov_x_deg=fov_deg, width=size, height=size,
                                  tilt_deg=tilt_deg, azimuth_deg=90.0 * k))
        return cls(fisheye, specs, **kw)

    @classmethod
    def ring(cls, fisheye: Camera, *, centre_fov_deg: float = 89.0,
             centre_size: int = 630, n_ring: int = 6, tilt_deg: float = 36.0,
             ring_fov_x_deg: float = 50.0, ring_width: int = 280,
             ring_height: int = 154, **kw) -> "Rig":
        """One centre view plus a ring of TANGENTIALLY ELONGATED views.

        The defaults are derived, not tuned: 89 deg is the widest co-axial
        square that still fills its frame on Aria (the sweep measured 100.0% at
        89 and 98.7% at 95); the ring's tilt and radial half-field are the
        largest that keep the outward corner inside the 54.7 deg cone; and six
        views is what it takes to tile 360 deg of azimuth at that tangential
        width with overlap to align on. `test_the_ring_fills_its_frames_and_
        covers_the_cone` is what actually checks all of that.
        """
        specs = [ViewSpec(fov_x_deg=centre_fov_deg, width=centre_size,
                          height=centre_size)]
        for k in range(n_ring):
            specs.append(ViewSpec(fov_x_deg=ring_fov_x_deg, width=ring_width,
                                  height=ring_height, tilt_deg=tilt_deg,
                                  azimuth_deg=360.0 * k / n_ring))
        return cls(fisheye, specs, **kw)

    # -- reporting ---------------------------------------------------------
    @property
    def coverage(self) -> float:
        return float(self.covered.sum()) / float(self.in_cone.sum())

    @property
    def fill_fraction(self) -> float:
        """Mean fraction of each view's frame that is real content."""
        return float(torch.stack([v.fill.float().mean() for v in self.views]).mean())

    def zone_coverage(self, theta: Tensor, lo_deg: float, hi_deg: float) -> float:
        band = self.in_cone & (theta >= math.radians(lo_deg)) & (theta <= math.radians(hi_deg))
        return float((self.covered & band).sum()) / max(float(band.sum()), 1.0)

    @property
    def sizes(self) -> List[Tuple[int, int]]:
        """Distinct (h, w) frames, so a caller can install once per size."""
        out = []
        for v in self.views:
            hw = (v.spec.height, v.spec.width)
            if hw not in out:
                out.append(hw)
        return out

    # -- the two pipelines -------------------------------------------------
    def _fuse(self, per_view: List[Tensor], align: bool) -> Tuple[Tensor, Dict]:
        """Pick the view whose axis is closest to each ray, after aligning scales.

        Alignment is a single log-scale per view fitted on its overlap with the
        centre view, by MEDIAN: a least-squares fit over an overlap that
        includes a depth discontinuity would be dragged by it, and the overlap
        band is exactly where discontinuities are most likely (it is the edge of
        somebody's field of view).
        """
        dev = per_view[0].device
        ref = per_view[0]
        info = {"log_scale": [0.0]}
        out = [ref]
        for k, d in enumerate(per_view[1:], start=1):
            ov = (self.views[0].cover & self.views[k].cover).to(dev)
            ov = ov & (ref > 1e-6) & (d > 1e-6)
            if align and int(ov.sum()) > 1000:
                s = float(torch.median(torch.log(ref[ov]) - torch.log(d[ov])))
            else:
                s = 0.0
            info["log_scale"].append(s)
            out.append(d * math.exp(s))

        best = torch.full_like(ref, -2.0)
        fused = torch.zeros_like(ref)
        for k, d in enumerate(out):
            cov = self.views[k].cover.to(dev)
            score = torch.where(cov, self.views[k].cos_local.to(dev),
                                torch.full_like(ref, -1.0))
            take = cov & (score > best)
            fused = torch.where(take, d, fused)
            best = torch.where(take, score, best)
        return fused, info

    def teach(self, forward_z, image: Tensor, align: bool = True
              ) -> Tuple[Tensor, Dict]:
        """Run the backbone in every view and fuse onto the fisheye grid.

        ``forward_z(img, view)`` must return planar z on that view's own grid.
        The view is passed because the rig's frames differ in size, and the
        caller has to re-install the backbone when the size changes.
        """
        per_view = []
        for v in self.views:
            warped = warp(image, v.grid_in.to(image.device))
            z_view = forward_z(warped, v)
            samp = warp(z_view, v.addr.to(z_view.device), mode="bilinear")
            per_view.append(samp / v.cos_local.to(samp.device))
        return self._fuse(per_view, align)

    def roundtrip(self, depth_fisheye: Tensor, align: bool = True
                  ) -> Tuple[Tensor, Dict]:
        """The control: the same resampling, without the change of projection.

        A fisheye-space depth map is pushed out to every view and pulled back
        through the identical addresses and the identical fusion, so it carries
        the same interpolation budget and the same seams. What it does not
        carry is a backbone that ever saw a pinhole frame.
        """
        per_view = []
        for v in self.views:
            to_view = warp(depth_fisheye, v.grid_in.to(depth_fisheye.device))
            back = warp(to_view, v.addr.to(to_view.device), mode="bilinear")
            per_view.append(back)
        return self._fuse(per_view, align)
