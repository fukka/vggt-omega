# Copyright (c) 2026.
"""View construction and ground-truth warping for the ADT-FOV test.

The benchmark asks one question — *does depth accuracy depend on where in the
field of view the content sits?* — and answers it two ways:

**Window sweep.** A fixed-size angular window is rendered at a sweep of
eccentricities (``tilt``) and handed to a model on its own. Two constructions
of the same window are scored side by side:

  ``rect``     a gnomonic (rectified perspective) crop — straight lines stay
               straight, the periphery is radially stretched;
  ``fisheye``  the *raw* pixels covering the same directions — the axis-aligned
               square in the fisheye frame that contains the ``rect`` window's
               own source footprint, centred on where the window axis lands.
               Same content, same resolution budget, no undistortion.

Pairing the two this way is what makes "rectified vs raw" a controlled
comparison rather than two unrelated runs: by construction the raw window sees
everything the rectified one saw.

**Radial profile.** The model sees the *whole* frame once (rectified pinhole,
or raw fisheye), and per-pixel errors are binned by incidence angle. The
rectified and raw arms share the camera's optical axis, so a bin means the same
physical direction in both — and the bins' pixel counts double as the coverage
story (a ~85 deg rectified pinhole simply has no pixels past ~42 deg off-axis
except in its corners).

Depth conventions
-----------------
ADT ``depth_npy`` is **planar z about the camera's optical axis** (CONTEXT.md;
measured by ``VGGT-360-fisheye/checks/check_gt_depth_domain.py``). Every depth
head here emits planar z about *its own* view axis. For a window at ``tilt``
those two axes differ, so GT is converted once, at the warp:

    range  = gt_z / cos(theta_camera)        # z about the camera axis -> ray range
    gt_view = range * cos(theta_window)      # ray range -> z about the window axis

Both factors are per-pixel and radial, so no scale-and-shift alignment can
absorb a mistake here: on the Aria cone ``1/cos(theta)`` runs to 2.15x at the
rim, against benchmark effects in the tens of percent. ``tests/test_geometry.py``
pins the conversion against analytic scenes.

Everything is numpy + cv2 — no torch, no weights, no GPU.
"""
from __future__ import annotations

import math
import os
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from fovbench import _REPO  # noqa: F401  (import registers sys.path)

from utils.fisheye_cam import (FisheyeCam, aria_intrinsics,  # noqa: E402
                               fisheye_ray_lut, kb4_forward_theta)
from utils.fisheye_views import (fisheye_to_persp, view_center_dir,  # noqa: E402
                                 view_rotation)

from finetune.eval.metrics import align_depth, depth_metrics  # noqa: E402

#: Incidence-angle bin edges (degrees). Bins are half-open ``[lo, hi)`` so they
#: partition exactly; the top edge is 55, just past the Aria usable cone
#: (54.83 deg), so the rim band is not clipped by a rounding.
THETA_EDGES: Tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0)

#: Metric names carried through the whole pipeline — ``depth_metrics`` output
#: plus this module's alignment-free ``raw_scale_ratio`` and the bin's own GT
#: depth. Named once so the driver's frame-averaging and the report's CSV cannot
#: drift apart.
#:
#: ``gt_median``/``gt_std`` are not scores: they are what the bin was *looking
#: at*, and they are measured on the real GT of the frames actually scored.
#: Every relative metric here grows with depth, so a rise in AbsRel toward the
#: rim is a statement about field position only once the bins are known to sit
#: at comparable depths. Carrying the depth per bin is what lets that be checked
#: against the data rather than argued about.
METRIC_KEYS: Tuple[str, ...] = ("AbsRel", "SqRel", "RMSE", "RMSElog", "log10",
                                "delta1", "delta2", "delta3",
                                "scale_ratio", "raw_scale_ratio",
                                "anchored_ratio",
                                "gt_median", "gt_std", "gt_spread")

#: Distance-from-the-optical-centre bin edges, in units of the frame's HALF
#: WIDTH: 1.0 is the middle of a frame edge, and a corner sits at sqrt(2).
#: This is the image-plane reading of "where in the field of view"; THETA_EDGES
#: is the ray-geometry reading. They are not interchangeable — see
#: ``radius_map_*``, and the caveat that the two views use different image
#: planes, so a given radius is a different direction in each.
RADIUS_EDGES: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.45)

#: Fine edges for the CONTINUOUS profiles — the same two coordinates read at
#: the resolution of the effect rather than of a table. 1 deg of incidence
#: angle and 0.025 half-widths, which on a 518 px frame is ~6 px of radius, so
#: a curve drawn on these is limited by the data and not by the bin width.
#:
#: These are not a finer version of the tables and must not be read as one. The
#: coarse bins are averaged **per frame and then across frames**, so every frame
#: counts equally; a fine bin can hold a handful of pixels in one frame and
#: thousands in another, so the profiles are **pooled over all frames** and are
#: therefore pixel-weighted. The two agree only when the pixel counts do.
PROFILE_THETA_EDGES: Tuple[float, ...] = tuple(float(x) for x in range(0, 56))
PROFILE_RADIUS_EDGES: Tuple[float, ...] = tuple(
    round(0.025 * i, 4) for i in range(59))

#: Depth strata for ``standardise_by_depth``, which the ADT-FOV experiment does
#: not use; ``datasets_egosynth`` does. Four is the last value that standardises
#: every bin on a scene whose whole radial penalty is depth — the share of that
#: penalty still standing runs 100% at 1 stratum, 44% at 2, 25% at 4, and beyond
#: that the bins stop overlapping in depth and nothing can be said at all.
DEPTH_STRATA: int = 4

#: A (bin x depth stratum) cell below this many pixels is not a measurement.
MIN_STRATUM_PX: int = 64

#: Minimum GT interquartile-range-over-median for a band to anchor an affine
#: fit. Real ADT bands measure 0.71-0.88; a single flat wall measures 0.00.
#:
#: The ADT-FOV report no longer has a ``drift`` column and does not use this;
#: ``datasets_egosynth`` does, for its own anchored pooling, and imports the
#: constant rather than restating it so the two cannot diverge.
MIN_ANCHOR_SPREAD = 0.05

#: ``finetune/data/rectify.py`` renders the pinhole at ``0.55 * max(H, W)``.
RECTIFIER_FOCAL_FRAC = 0.55

_RAY_CACHE: dict = {}

#: Per-pixel maps that depend only on the render geometry, never on the frame.
#: A 300-frame run rebuilt each of these 1200 times per model; they are shared
#: read-only instead. Entries are marked non-writeable so a caller that mutated
#: one would raise here rather than silently corrupt every later frame — and so
#: the worker threads in :mod:`fovbench.run` can share them without a lock.
_MAP_CACHE: dict = {}
_MAP_LOCK = threading.Lock()


def _cached_map(key: tuple, build) -> np.ndarray:
    hit = _MAP_CACHE.get(key)
    if hit is None:
        hit = build()
        hit.setflags(write=False)
        _MAP_CACHE[key] = hit
    return hit


def rectifier(H: int, W: int, preset: str = "aria-214-1"):
    """The shared :class:`FisheyeRectifier`, with its remap grids already built.

    The grids cost an ``initUndistortRectifyMap`` over the full 1408x1408 frame
    and depend only on the preset and the size, but the rectifier was
    constructed *inside* the per-frame view builder, so its own cache was cold
    every single frame and every frame paid for the grids again.

    One instance per preset, warmed for ``(H, W)`` under a lock before any
    worker can reach it — ``FisheyeRectifier._get_maps`` is a check-then-set
    that two threads would otherwise race on (harmlessly, but twice).
    """
    from finetune.data.rectify import FisheyeRectifier
    with _MAP_LOCK:
        rec = _MAP_CACHE.get(("rectifier", preset))
        if rec is None:
            rec = _MAP_CACHE[("rectifier", preset)] = FisheyeRectifier(preset)
        rec._get_maps(H, W)
    return rec


# --------------------------------------------------------------------------- #
# Cameras and ray fields
# --------------------------------------------------------------------------- #

def aria_cam(H: int, W: int, rotated: bool = True) -> FisheyeCam:
    """Aria 214-1 KB4 intrinsics for an ``H x W`` frame (the ADT convention)."""
    return aria_intrinsics(H, W, rotated=rotated)


def scaled_cam(cam: FisheyeCam, out_size: int) -> FisheyeCam:
    """``cam`` re-expressed for a frame that ``cv2.resize`` shrank to ``out_size``.

    Not the same as ``aria_cam(out_size, out_size)``. ``cv2.resize`` maps output
    pixel centre ``j + 0.5`` to source ``(j + 0.5) * W_src / W_out``, i.e.
    ``c' = (c + 0.5) * s - 0.5``, whereas scaling the calibration directly gives
    ``c' = c * s``. The two differ by ``0.5 * (1 - s)`` px — under a pixel, but a
    third of a degree of incidence angle at a small working size, which is
    enough to smear a ground-truth depth against the theta it is binned by.
    """
    s = float(out_size) / float(cam.W)
    sy = float(out_size) / float(cam.H)
    return FisheyeCam(H=out_size, W=out_size,
                      fx=cam.fx * s, fy=cam.fy * sy,
                      cx=(cam.cx + 0.5) * s - 0.5,
                      cy=(cam.cy + 0.5) * sy - 0.5,
                      k=cam.k, valid_theta=cam.valid_theta)


def fisheye_rays(cam: FisheyeCam) -> Tuple[np.ndarray, np.ndarray]:
    """``(rays[H,W,3], inside_cone[H,W])`` for a fisheye frame, memoised.

    The LUT inverts the KB4 polynomial on every pixel, which is the single most
    expensive thing in a frame's geometry; the answer depends only on the
    intrinsics, so a sweep over hundreds of frames pays for it once.
    """
    key = (cam.H, cam.W, cam.fx, cam.fy, cam.cx, cam.cy, cam.k, cam.valid_theta)
    if key not in _RAY_CACHE:
        _RAY_CACHE[key] = fisheye_ray_lut(cam)
    return _RAY_CACHE[key]


def theta_map_fisheye(cam: FisheyeCam) -> np.ndarray:
    """Per-pixel incidence angle (degrees) of a raw fisheye frame. Read-only."""
    key = ("theta_fisheye", cam.H, cam.W, cam.fx, cam.fy, cam.cx, cam.cy,
           cam.k, cam.valid_theta)

    def build():
        rays, _ = fisheye_rays(cam)
        return np.degrees(np.arccos(np.clip(rays[..., 2], -1.0, 1.0))).astype(np.float32)
    return _cached_map(key, build)


def theta_map_pinhole(H: int, W: int,
                      focal_frac: float = RECTIFIER_FOCAL_FRAC,
                      focal_px: Optional[float] = None,
                      cx: Optional[float] = None,
                      cy: Optional[float] = None) -> np.ndarray:
    """Per-pixel incidence angle (degrees) of the rectified pinhole frame.

    The rectifier keeps the fisheye's optical centre *and* axis (identity
    rotation), so this is the same physical angle ``theta_map_fisheye``
    measures — which is what lets the rectified and raw arms be binned on one
    axis and compared bin for bin.

    ``focal_px``/``cx``/``cy`` override the defaults when the frame has been
    resized after rectification: ``FisheyeRectifier`` puts the principal point
    at ``W_src / 2`` (the *centre of* that pixel, not the frame's midline), and
    carrying that convention through a resize is the only way the theta label
    lines up with the depth it labels.
    """
    f = focal_px if focal_px is not None else focal_frac * max(H, W)
    cx = (W / 2.0 - 0.5) if cx is None else cx
    cy = (H / 2.0 - 0.5) if cy is None else cy

    def build():
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        x = (xs - cx) / f
        y = (ys - cy) / f
        return np.degrees(np.arctan(np.sqrt(x * x + y * y))).astype(np.float32)
    return _cached_map(("theta_pinhole", H, W, f, cx, cy), build)


#: A ring counts as whole above this fraction of its azimuth being imaged.
#: Below 1.0 because the outermost imaged pixel of a full ring can miss an
#: azimuth cell to rounding; the real collapse is not marginal (100% to 71%
#: in one degree on the rectified view), so the threshold never has to judge.
FULL_RING_FRAC = 0.995

#: Azimuth cells used to ask "is this direction imaged at this angle".
_RING_AZ = 720


def view_maps(out_size: int, kind: str, src: int = 1408):
    """``(theta, azimuth, imaged)`` for a view, without any frame behind it.

    The same construction :func:`full_frame_view` uses, minus the pixels — so
    coverage questions can be answered from a ``results.json`` alone, with no
    data, no GPU and no re-scoring.
    """
    cam = aria_cam(src, src)
    if kind == "fisheye":
        small = scaled_cam(cam, out_size)
        _, cone = fisheye_rays(small)
        theta, imaged, cx, cy = theta_map_fisheye(small), cone, small.cx, small.cy
    elif kind == "rect":
        s = float(out_size) / float(cam.W)
        cx = (cam.W / 2.0 + 0.5) * s - 0.5
        cy = (cam.H / 2.0 + 0.5) * s - 0.5
        theta = theta_map_pinhole(
            out_size, out_size,
            focal_px=RECTIFIER_FOCAL_FRAC * max(cam.H, cam.W) * s, cx=cx, cy=cy)
        imaged = theta <= math.degrees(cam.theta_max())
    else:
        raise ValueError(f"unknown frame kind {kind!r} (choose 'rect' or 'fisheye')")

    def build():
        ys, xs = np.mgrid[0:out_size, 0:out_size]
        return (np.degrees(np.arctan2(ys - cy, xs - cx)) % 360.0).astype(np.float32)
    return theta, _cached_map(("azimuth", out_size, cx, cy), build), imaged


def reach_by_azimuth(out_size: int, kind: str) -> np.ndarray:
    """Per azimuth direction, the largest incidence angle the view images.

    Asked along *rays* rather than within rings on purpose. A ring at a small
    angle is only a few hundred pixels wide, so counting how many azimuth cells
    it happens to land in measures the sampling, not the coverage — an early
    version of this did exactly that and reported the fisheye ring "incomplete"
    at 13 deg, where it is a whole circle. A ray has many pixels at every
    azimuth, so a maximum along it is exact.

    Flat at the cone limit for the raw fisheye (every direction reaches 54.8
    deg); on the rectified pinhole it runs from the edge midpoint on the axes to
    the corner on the diagonals, which is the whole of the story below.
    """
    theta, az, imaged = view_maps(out_size, kind)
    cell = np.clip((az / 360.0 * _RING_AZ).astype(int), 0, _RING_AZ - 1)
    reach = np.zeros(_RING_AZ, np.float64)
    np.maximum.at(reach, cell[imaged], theta[imaged])
    return reach


def ring_fraction(out_size: int, kind: str, theta_deg) -> np.ndarray:
    """Fraction of the 360-degree ring this view images, at each given angle."""
    reach = reach_by_azimuth(out_size, kind)
    t = np.atleast_1d(np.asarray(theta_deg, np.float64))
    return (reach[None, :] >= t[:, None]).mean(axis=1)


def ring_coverage(out_size: int, kind: str, edges: Sequence[float]) -> List[dict]:
    """How much of each incidence-angle **ring** the view actually images.

    The pixel counts already in the report say a bin is thin. They do not say
    *why*, and the difference matters: a bin on the raw fisheye is a whole
    360-degree annulus, while the same bin on the rectified pinhole is whatever
    part of that annulus survived a square crop. Past the inscribed circle the
    rectified arm stops being a ring at all and becomes four corner wedges — so
    its outer bins sample a **biased set of directions**, not merely fewer of
    them, and its curve is cut short at an angle no table currently prints.

    Returns per bin: ``complete_frac`` (fraction of azimuth imaged),
    ``mean_theta`` (the angle the bin *actually* averages, which is not the bin
    midpoint once the ring is partial), and ``n_px``.

    Pure geometry — no frame, no GT, no model.
    """
    theta, _, imaged = view_maps(out_size, kind)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (theta >= lo) & (theta < hi) & imaged
        inside = np.linspace(float(lo), float(hi), 21)[:-1]
        out.append(dict(bin_lo=float(lo), bin_hi=float(hi),
                        complete_frac=float(ring_fraction(out_size, kind, inside).mean()),
                        mean_theta=float(theta[band].mean()) if band.any() else float("nan"),
                        n_px=int(band.sum())))
    return out


def full_ring_limit(out_size: int, kind: str) -> float:
    """Largest incidence angle at which this view still images the WHOLE ring.

    Beyond it a theta bin is a set of corner wedges, so two views read past
    their own limits are not compared on the same directions — which is the
    confound this whole pair of functions exists to expose. Measured rather than
    derived, so it carries the off-centre principal point.
    """
    return float(reach_by_azimuth(out_size, kind).min())


def radius_map(H: int, W: int, cx: Optional[float] = None,
               cy: Optional[float] = None) -> np.ndarray:
    """Per-pixel distance from the optical centre, in HALF-WIDTHS.

    ``1.0`` is the middle of a frame edge; a corner is ``sqrt(2)``. Normalising
    by the half width rather than leaving pixels makes the number independent of
    the render size (518 vs 512) while staying a plain image-plane distance.

    Read against ``theta_map_*`` with care: on the raw fisheye, radius is nearly
    proportional to the incidence angle (KB4 is close to equidistant), whereas
    the rectified pinhole puts radius at ``f*tan(theta)``, which runs away near
    the edge. The same radius is therefore a *different direction* in the two
    views, and a radius-binned comparison between them is not like-for-like.
    ``theta`` is the axis on which the two views mean the same thing.
    """
    cx = (W / 2.0 - 0.5) if cx is None else cx
    cy = (H / 2.0 - 0.5) if cy is None else cy

    def build():
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        return (np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
                / (0.5 * W)).astype(np.float32)
    return _cached_map(("radius", H, W, cx, cy), build)


def project_dirs(cam: FisheyeCam, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """KB4-project unit directions ``(..., 3)`` to fisheye pixel ``(u, v)``.

    A second copy of the projection that ``fisheye_to_persp`` performs inside
    its remap; needed here to place the raw-fisheye window's crop box, which the
    remap does not expose. ``test_project_dirs_agrees_with_the_rendered_maps``
    holds the two together to well under a pixel.
    """
    z = np.clip(dirs[..., 2], -1.0, 1.0)
    theta = np.arccos(z)
    theta_d = kb4_forward_theta(theta, cam.k)
    rxy = np.sqrt(dirs[..., 0] ** 2 + dirs[..., 1] ** 2)
    inv = np.zeros_like(rxy)
    np.divide(1.0, rxy, out=inv, where=rxy > 1e-12)
    u = cam.cx + cam.fx * theta_d * dirs[..., 0] * inv
    v = cam.cy + cam.fy * theta_d * dirs[..., 1] * inv
    return u, v


def tangent_rays(azimuth: float, tilt: float, fov_deg: float, out_size: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Ray grid of a tangent window: ``(dirs_in_camera_frame, cos_from_window_axis)``.

    Identical grid and rotation to ``fisheye_to_persp``, so the directions line
    up with the pixels that function renders. ``cos_view`` is read off *before*
    the rotation because the window axis is the view frame's ``+z``; rotating
    cannot change an angle measured from the axis that rotates with it.
    """
    t = math.tan(math.radians(fov_deg) / 2.0)
    xs = np.linspace(-t, t, out_size, dtype=np.float64)
    ys = np.linspace(-t, t, out_size, dtype=np.float64)
    xv, yv = np.meshgrid(xs, ys)
    v = np.stack([xv, yv, np.ones_like(xv)], axis=-1)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    cos_view = v[..., 2].copy()
    d_cam = v @ view_rotation(azimuth, tilt).T
    return d_cam, cos_view.astype(np.float32)


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #

@dataclass
class Window:
    """One angular window, ready to score.

    ``gt_z`` is planar z **about the window axis** — the convention every depth
    head emits — so a model's raw output is comparable to it without further
    conversion. ``theta`` stays measured from the *camera's* optical axis: it is
    the distortion coordinate, and a window-relative angle would make every
    window look alike.
    """

    kind: str                 # "rect" | "fisheye"
    azimuth: float
    tilt: float
    fov: float
    axis: np.ndarray          # (3,) window optical axis in camera coords
    rgb: np.ndarray           # (N, N, 3) uint8 — exactly what the model sees
    gt_z: np.ndarray          # (N, N) float32 metres, planar z about `axis`
    valid: np.ndarray         # (N, N) bool — in-cone AND GT present
    cos_view: np.ndarray      # (N, N) float32 cos(angle from `axis`)
    theta: np.ndarray         # (N, N) float32 incidence from the camera axis, deg
    in_cone_frac: float       # analytic: share of the window the lens images
    src_px_per_out_px: float  # sampling density — see below

    # ``src_px_per_out_px`` is the mean number of *raw fisheye* pixels behind one
    # window pixel. It is the window arm's own confound, made visible: a fisheye
    # compresses the periphery, so a window of fixed angular width is built from
    # progressively fewer source pixels as it is aimed off-axis, and is therefore
    # progressively softer. A rising AbsRel across aims is then partly resolution
    # and partly geometry, and nothing else in the table separates them. Below 1
    # the window is upsampled — invented detail, and the error attributed to
    # "the periphery" is at least in part the blur.

    @property
    def tag(self) -> str:
        return f"{self.kind}_t{int(round(self.tilt))}_a{int(round(self.azimuth))}"


@dataclass
class FrameView:
    """A whole frame in one construction, for the radial profile."""

    kind: str                 # "rect" | "fisheye"
    rgb: np.ndarray           # (N, N, 3) uint8
    gt_z: np.ndarray          # (N, N) float32 metres, planar z about the camera axis
    valid: np.ndarray         # (N, N) bool
    theta: np.ndarray         # (N, N) float32 incidence angle, deg
    radius: np.ndarray        # (N, N) float32 distance from the optical centre,
                              #        in half-widths (see ``radius_map``)
    in_cone_frac: float


def _to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _resize_nearest(arr: np.ndarray, out_size: int) -> np.ndarray:
    """Nearest resample on the **pixel-centre** convention, ``src = (j+0.5)/s - 0.5``.

    Not ``cv2.INTER_NEAREST``, which floors ``j / s`` and so shifts the image by
    up to half an output pixel. That shift is invisible on RGB and matters here:
    ``scaled_cam`` and ``theta_map_*`` place their pixels on the centre
    convention, so a floored GT resample labels each depth with the incidence
    angle of a *neighbouring* ray. Measured on the 3 m analytic sphere at a
    176 -> 64 downsample (``tests/test_geometry.py``), cv2's floored resample
    puts the GT 0.045 m away from the theta it is binned by; this one holds it
    to 0.014 m, the residual of the downsample itself. Whether 0.045 m would
    have mattered against real model-to-model differences is unknown — no
    network has been run against this code.
    """
    H, W = arr.shape[:2]
    j = np.clip(np.rint((np.arange(out_size) + 0.5) * (W / out_size) - 0.5)
                .astype(np.int64), 0, W - 1)
    i = np.clip(np.rint((np.arange(out_size) + 0.5) * (H / out_size) - 0.5)
                .astype(np.int64), 0, H - 1)
    return arr[i[:, None], j[None, :]]


def _square_crop(arr: np.ndarray, cx: float, cy: float, side: float,
                 out_size: int, nearest: bool) -> np.ndarray:
    """Zero-padded square crop centred on ``(cx, cy)``, resized to ``out_size``.

    Zero padding rather than cv2's border replication: a replicated depth or
    validity value outside the frame is a fabricated measurement, and it would
    be scored.
    """
    s = max(2, int(math.ceil(side)))
    x0 = int(math.floor(cx - s / 2.0))
    y0 = int(math.floor(cy - s / 2.0))
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r = max(0, x0 + s - arr.shape[1])
    pad_b = max(0, y0 + s - arr.shape[0])
    if pad_l or pad_t or pad_r or pad_b:
        pads = [(pad_t, pad_b), (pad_l, pad_r)] + [(0, 0)] * (arr.ndim - 2)
        arr = np.pad(arr, pads, mode="constant")
        x0 += pad_l
        y0 += pad_t
    patch = arr[y0:y0 + s, x0:x0 + s]
    if nearest:
        return _resize_nearest(patch, out_size)
    return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)


def _rect_window(rgb, gt_z, gt_valid, cam, azimuth, tilt, fov, out_size,
                 supersample):
    """Gnomonic window + GT sampled through the very same map."""
    d_cam, cos_view = tangent_rays(azimuth, tilt, fov, out_size)
    # Maps at output resolution (return_maps forces supersample=1) drive the GT
    # sampling; the RGB is re-rendered with anti-aliasing, since a point-sampled
    # crop from a 1408 frame is what the model would otherwise be judged on.
    _, cone_f, mapx, mapy = fisheye_to_persp(gt_z, cam, azimuth, tilt, fov,
                                             height=out_size, width=out_size,
                                             return_maps=True)
    rgb_win, _ = fisheye_to_persp(rgb, cam, azimuth, tilt, fov,
                                  height=out_size, width=out_size,
                                  supersample=supersample)
    gt_win = cv2.remap(gt_z.astype(np.float32), mapx, mapy, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    gtv_win = cv2.remap(gt_valid.astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    cos_fish = d_cam[..., 2].astype(np.float32)
    in_cone = cone_f > 0.5
    # |Jacobian| of (output pixel) -> (source pixel): source px per output px.
    dudi, dudj = np.gradient(mapx.astype(np.float64))
    dvdi, dvdj = np.gradient(mapy.astype(np.float64))
    jac = np.abs(dudj * dvdi - dudi * dvdj)
    dens = float(np.mean(jac[in_cone])) if in_cone.any() else float("nan")
    return (_to_u8(rgb_win), gt_win, gtv_win, cos_fish, cos_view, in_cone, dens)


def _fisheye_window(rgb, gt_z, gt_valid, cam, azimuth, tilt, fov, out_size):
    """Raw pixels over the same directions: the square containing the gnomonic
    window's source footprint, centred on the window axis's image point."""
    d_cam, _ = tangent_rays(azimuth, tilt, fov, out_size)
    axis = view_center_dir(azimuth, tilt)
    cu, cv_ = project_dirs(cam, axis[None, :])
    cu, cv_ = float(cu[0]), float(cv_[0])

    u, v = project_dirs(cam, d_cam)
    inside = np.arccos(np.clip(d_cam[..., 2], -1, 1)) <= cam.theta_max()
    if not inside.any():                      # window entirely outside the lens
        inside = np.ones_like(inside)
    side = 2.0 * max(np.abs(u[inside] - cu).max(), np.abs(v[inside] - cv_).max())

    rays, cone = fisheye_rays(cam)
    frame_ok = np.ones(gt_z.shape[:2], np.uint8)
    d_src = _square_crop(rays, cu, cv_, side, out_size, nearest=True)
    d_src /= np.maximum(np.linalg.norm(d_src, axis=-1, keepdims=True), 1e-9)
    rgb_win = _square_crop(rgb.astype(np.float32), cu, cv_, side, out_size, False)
    gt_win = _square_crop(gt_z.astype(np.float32), cu, cv_, side, out_size, True)
    gtv_win = _square_crop(gt_valid.astype(np.uint8), cu, cv_, side, out_size, True) > 0
    cone_win = _square_crop(cone.astype(np.uint8), cu, cv_, side, out_size, True) > 0
    in_frame = _square_crop(frame_ok, cu, cv_, side, out_size, True) > 0

    cos_fish = d_src[..., 2].astype(np.float32)
    cos_view = (d_src @ axis).astype(np.float32)
    return (_to_u8(rgb_win), gt_win, gtv_win, cos_fish, cos_view,
            cone_win & in_frame, (side / float(out_size)) ** 2)


def render_window(rgb: np.ndarray, gt_z: np.ndarray, gt_valid: np.ndarray,
                  cam: FisheyeCam, azimuth: float, tilt: float, fov: float,
                  out_size: int, kind: str, supersample: int = 3) -> Window:
    """Render one angular window and warp GT into its depth convention.

    Parameters
    ----------
    rgb, gt_z, gt_valid : the raw fisheye frame, its planar-z GT (metres) and
        GT validity, all at ``cam``'s resolution.
    azimuth, tilt : where the window points — ``tilt`` is the eccentricity the
        benchmark sweeps, ``azimuth`` the direction of that tilt in the image.
    fov : window field of view (degrees), held FIXED across the sweep. Varying
        it alongside tilt is the confound that made an earlier sweep in this
        repo unreadable: wider windows overhang the imaged cone, so width and
        dead area moved together and the score was attributed to width alone.
    kind : ``"rect"`` (gnomonic) or ``"fisheye"`` (raw pixels, same directions).
    """
    if kind == "rect":
        parts = _rect_window(rgb, gt_z, gt_valid, cam, azimuth, tilt, fov,
                             out_size, supersample)
    elif kind == "fisheye":
        parts = _fisheye_window(rgb, gt_z, gt_valid, cam, azimuth, tilt, fov,
                                out_size)
    else:
        raise ValueError(f"unknown window kind {kind!r} (choose 'rect' or 'fisheye')")
    rgb_win, gt_win, gtv_win, cos_fish, cos_view, in_cone, density = parts

    # z about the camera axis -> ray range -> z about the window axis.
    gt_view = (gt_win * cos_view / np.clip(cos_fish, 1e-3, None)).astype(np.float32)
    valid = in_cone & gtv_win & (gt_win > 0)
    theta = np.degrees(np.arccos(np.clip(cos_fish, -1.0, 1.0))).astype(np.float32)
    return Window(kind=kind, azimuth=float(azimuth), tilt=float(tilt),
                  fov=float(fov), axis=view_center_dir(azimuth, tilt),
                  rgb=rgb_win, gt_z=gt_view * valid, valid=valid,
                  cos_view=cos_view, theta=theta,
                  in_cone_frac=float(in_cone.mean()),
                  src_px_per_out_px=float(density))


def view_rgb(rgb: np.ndarray, cam: FisheyeCam, out_size: int,
             kind: str) -> np.ndarray:
    """The RGB half of :func:`full_frame_view`, for frames that are not scored.

    A multi-frame pass hands the model context frames that have no ground truth
    and are never measured — but they must arrive in exactly the *same view* as
    the target, or cross-view attention is being asked to match a rectified
    frame against a raw one. This shares the target's code path for the pixels
    and skips the GT work, which on the rectified arm is a second full remap.
    """
    if kind == "fisheye":
        return _to_u8(cv2.resize(rgb.astype(np.float32), (out_size, out_size),
                                 interpolation=cv2.INTER_AREA))
    if kind == "rect":
        rec = rectifier(*rgb.shape[:2])
        out = rec(rgb.astype(np.float32) / 255.0) * 255.0
        return _to_u8(cv2.resize(out, (out_size, out_size),
                                 interpolation=cv2.INTER_AREA))
    raise ValueError(f"unknown frame kind {kind!r} (choose 'rect' or 'fisheye')")


def full_frame_view(rgb: np.ndarray, gt_z: np.ndarray, gt_valid: np.ndarray,
                    cam: FisheyeCam, out_size: int, kind: str) -> FrameView:
    """The whole frame in one construction, GT kept in camera-axis planar z.

    ``kind="rect"`` runs the repo's validated ``FisheyeRectifier`` (~85 deg
    pinhole) over both RGB and GT with the *same* maps; rectification shares the
    camera centre and axis, so planar z is unchanged and only the pixel grid
    moves. ``kind="fisheye"`` hands over the raw frame untouched.
    """
    if kind == "fisheye":
        small = scaled_cam(cam, out_size)
        _, cone = fisheye_rays(small)
        rgb_v = _to_u8(cv2.resize(rgb.astype(np.float32), (out_size, out_size),
                                  interpolation=cv2.INTER_AREA))
        gt_v = _resize_nearest(gt_z.astype(np.float32), out_size)
        gtv = _resize_nearest(gt_valid.astype(np.uint8), out_size) > 0
        theta = theta_map_fisheye(small)
        in_cone = cone
    elif kind == "rect":
        rec = rectifier(*rgb.shape[:2])
        rgb_r = rec(rgb.astype(np.float32) / 255.0) * 255.0
        gt_r = rec.rectify_depth(gt_z.astype(np.float32))
        gtv_r = rec.rectify_depth(gt_valid.astype(np.float32)) > 0.5
        rgb_v = _to_u8(cv2.resize(rgb_r, (out_size, out_size), interpolation=cv2.INTER_AREA))
        gt_v = _resize_nearest(gt_r, out_size)
        gtv = _resize_nearest(gtv_r.astype(np.uint8), out_size) > 0
        # Carry the rectifier's own principal point (W_src / 2) through the
        # resize, so theta labels the pixel it is actually attached to.
        s = float(out_size) / float(cam.W)
        theta = theta_map_pinhole(
            out_size, out_size,
            focal_px=RECTIFIER_FOCAL_FRAC * max(cam.H, cam.W) * s,
            cx=(cam.W / 2.0 + 0.5) * s - 0.5,
            cy=(cam.H / 2.0 + 0.5) * s - 0.5)
        in_cone = theta <= math.degrees(cam.theta_max())
    else:
        raise ValueError(f"unknown frame kind {kind!r} (choose 'rect' or 'fisheye')")

    valid = in_cone & gtv & (gt_v > 0)
    # The fisheye's principal point is off-centre by ~4 px at 1408; carry it, so
    # "distance from the optical centre" is from the optical centre and not from
    # the middle of the sensor.
    if kind == "fisheye":
        sc = scaled_cam(cam, out_size)
        radius = radius_map(out_size, out_size, cx=sc.cx, cy=sc.cy)
    else:
        radius = radius_map(out_size, out_size)
    return FrameView(kind=kind, rgb=rgb_v, gt_z=gt_v * valid, valid=valid,
                     theta=theta, radius=radius,
                     in_cone_frac=float(in_cone.mean()))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def radial_profile(pred_z: np.ndarray, gt_z: np.ndarray, mask: np.ndarray,
                   theta_deg: np.ndarray, edges: Sequence[float],
                   align_mode: str,
                   min_depth: float = 0.01,
                   max_depth: float = 100.0) -> dict:
    """Depth metrics per incidence-angle bin, under ONE alignment fit.

    The fit is least-squares over *every* valid pixel of the frame and is then
    frozen for all bins. That is the whole design: an up-to-scale model whose
    depth bends with eccentricity has, by definition, no single scale that fits
    every radius, and fitting each bin separately hands it one — which reports a
    flat, healthy curve for exactly the failure the benchmark exists to detect.
    ``tests/test_geometry.py::test_per_bin_alignment_would_erase_the_effect``
    measures the difference: a bias that reaches AbsRel 0.19 under the shared
    fit reads under 0.02 when each bin gets its own.

    ``raw_scale_ratio`` per bin is ``median(gt/pred)`` on the **unaligned**
    prediction: the one column with no alignment in the way at all, and so the
    only one that stays monotone under a radial bias. (``scale_ratio``, which
    ``depth_metrics`` computes, is measured on the *aligned* map and inherits
    the same distortion as AbsRel — they are not interchangeable.)

    Bins are half-open ``[lo, hi)`` and partition the pixels whose ``theta``
    lies in ``[edges[0], edges[-1])``.
    """
    edges = [float(e) for e in edges]
    in_range = mask & (theta_deg >= edges[0]) & (theta_deg < edges[-1])
    if not in_range.any():
        raise ValueError("no valid pixels inside the theta range "
                         f"[{edges[0]}, {edges[-1]}) — check the view's masks")

    # ONE fit, over every valid pixel of the frame — never per bin. Binning is a
    # masking step applied afterwards to an already-frozen prediction. See
    # ``bin_by`` for the multi-axis form, which shares this single fit.
    aligned = align_depth(pred_z, gt_z, mask, mode=align_mode)
    overall = depth_metrics(aligned, gt_z, in_range, min_depth, max_depth)

    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = in_range & (theta_deg >= lo) & (theta_deg < hi)
        met = depth_metrics(aligned, gt_z, m, min_depth, max_depth)
        # depth_metrics returns an all-NaN dict for an empty bin; a count must
        # stay a count, so the report can say "no pixels" rather than "no value".
        n = met.get("n_valid", 0)
        met["n_valid"] = 0 if not np.isfinite(n) else int(n)
        # ``n_bin`` counts the bin's valid pixels before depth_metrics' own
        # in-range filter; the two differ where a prediction lands out of range,
        # and only ``n_bin`` partitions the frame.
        met["n_bin"] = int(m.sum())
        met["raw_scale_ratio"] = raw_scale_ratio(pred_z, gt_z, m)
        met["theta_lo"], met["theta_hi"] = lo, hi
        bins.append(met)
    overall["raw_scale_ratio"] = raw_scale_ratio(pred_z, gt_z, in_range)
    return {"align": align_mode, "overall": overall, "bins": bins,
            "n_frame_valid": int(in_range.sum())}


def bin_by(pred_z: np.ndarray, gt_z: np.ndarray, mask: np.ndarray,
           align_mode: str, axes: dict,
           min_depth: float = 0.01, max_depth: float = 100.0,
           profile_edges: Optional[dict] = None) -> dict:
    """Score one prediction on several binning axes under **one** alignment fit.

    ``axes`` maps a name to ``(coordinate_map, edges)`` — e.g. incidence angle
    and distance from the optical centre. The scale (and shift) is fitted once,
    over every valid pixel of the frame, and then frozen; each axis only applies
    a different set of masks to that same frozen prediction. Nothing about the
    binning can reach back into the alignment, so the axes are two readings of
    one measurement rather than two measurements.

    Fitting per bin instead would hand an up-to-scale model a separate scale at
    every radius, which flattens exactly the effect the benchmark exists to find
    (``tests/test_geometry.py::test_per_bin_alignment_would_erase_the_effect``).

    ``profile_edges`` maps an axis name to a *fine* set of edges, and adds a
    continuous profile of that axis under ``profiles`` — off the same frozen
    prediction as everything else, so the curve and the table are two readings
    of one measurement and cannot disagree about the alignment.
    """
    aligned = align_depth(pred_z, gt_z, mask, mode=align_mode)
    out = {"align": align_mode, "n_frame_valid": int(mask.sum()),
           "overall": depth_metrics(aligned, gt_z, mask, min_depth, max_depth)}
    out["overall"]["raw_scale_ratio"] = raw_scale_ratio(pred_z, gt_z, mask)
    out["overall"].update(_gt_stats(gt_z, mask))
    for name, (coord, edges) in axes.items():
        edges = [float(e) for e in edges]
        anchored = anchored_ratios(pred_z, gt_z, mask, coord, edges, align_mode)
        bins = []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = mask & (coord >= lo) & (coord < hi)
            met = depth_metrics(aligned, gt_z, m, min_depth, max_depth)
            n = met.get("n_valid", 0)
            met["n_valid"] = 0 if not np.isfinite(n) else int(n)
            met["n_bin"] = int(m.sum())
            met["raw_scale_ratio"] = raw_scale_ratio(pred_z, gt_z, m)
            met["anchored_ratio"] = anchored[i]
            # What the bin was looking at, so "the rim is worse" can be told
            # apart from "the rim is nearer". Model-independent by construction.
            met.update(_gt_stats(gt_z, m))
            met["bin_lo"], met["bin_hi"] = lo, hi
            bins.append(met)
        out[name] = bins
        if profile_edges and name in profile_edges:
            out.setdefault("profiles", {})[name] = fine_profile(
                aligned, gt_z, mask, coord, profile_edges[name],
                min_depth, max_depth)
    return out


def fine_profile(aligned: np.ndarray, gt_z: np.ndarray, mask: np.ndarray,
                 coord: np.ndarray, edges: Sequence[float],
                 min_depth: float = 0.01, max_depth: float = 100.0) -> dict:
    """Error as a *continuous* function of one coordinate, as SUMS not means.

    The tables answer "is the rim worse than the centre" with six bins. This
    answers "what is the shape" — where the rise starts, whether it is linear or
    a knee, whether the models turn over in the same place — which six bins
    cannot show and which the bin edges themselves can invent.

    Sums, not means, because the caller pools these across frames. A fine bin
    holds a few pixels in one frame and thousands in another, so a mean of
    per-frame means would weight a frame that barely reached the rim as heavily
    as one that filled it. Adding the sums and dividing once at the end is the
    pixel-weighted answer, which is the one a continuous curve should give.
    That is a **different estimator from the coarse tables**, which are averaged
    per frame; they agree only when the per-frame pixel counts do, and neither
    is wrong — see ``PROFILE_THETA_EDGES``.

    Returns ``n`` (pixels per bin) plus the summed AbsRel, delta1 hit count and
    GT depth. The report divides. Everything is one ``bincount`` pass, so the
    whole profile costs less than a single extra ``depth_metrics`` call.
    """
    nb = len(edges) - 1
    lo, hi = float(edges[0]), float(edges[-1])
    ok = (mask & np.isfinite(gt_z) & (gt_z > min_depth) & (gt_z < max_depth)
          & np.isfinite(aligned) & (aligned > 0) & (aligned <= max_depth)
          & (coord >= lo) & (coord < hi))
    empty = {"edges": [float(e) for e in edges],
             "n": [0] * nb, "sum_absrel": [0.0] * nb,
             "sum_delta1": [0.0] * nb, "sum_gt": [0.0] * nb,
             "sum_gt2": [0.0] * nb}
    if not ok.any():
        return empty
    p = np.clip(aligned[ok].astype(np.float64), 1e-6, None)
    g = np.clip(gt_z[ok].astype(np.float64), 1e-6, None)
    # searchsorted against the real edges, so unequal edge spacing stays honest
    # (np.digitize on a uniform grid would be faster and would silently lie if
    # a caller ever passed non-uniform edges).
    idx = np.searchsorted(np.asarray(edges, np.float64), coord[ok], side="right") - 1
    idx = np.clip(idx, 0, nb - 1)
    d1 = (np.maximum(p / g, g / p) < 1.25).astype(np.float64)
    return {"edges": [float(e) for e in edges],
            "n": np.bincount(idx, minlength=nb).astype(int).tolist(),
            "sum_absrel": np.bincount(idx, weights=np.abs(p - g) / g,
                                      minlength=nb).tolist(),
            "sum_delta1": np.bincount(idx, weights=d1, minlength=nb).tolist(),
            "sum_gt": np.bincount(idx, weights=g, minlength=nb).tolist(),
            "sum_gt2": np.bincount(idx, weights=g * g, minlength=nb).tolist()}


def pool_profiles(profiles: Sequence[dict]) -> dict:
    """Add per-frame profiles into one, and divide. The pooling step."""
    live = [p for p in profiles if p and p.get("n")]
    if not live:
        return {}
    edges = live[0]["edges"]
    nb = len(edges) - 1
    n = np.zeros(nb, np.int64)
    acc = {k: np.zeros(nb) for k in ("sum_absrel", "sum_delta1", "sum_gt",
                                     "sum_gt2")}
    for p in live:
        if p["edges"] != edges:
            raise ValueError("pool_profiles: profiles have different edges; "
                             "they describe different axes and cannot be added")
        n += np.asarray(p["n"], np.int64)
        for k in acc:
            acc[k] += np.asarray(p[k], np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        div = np.where(n > 0, n, np.nan)
        return {"edges": edges, "n": n.tolist(),
                "centre": [0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])],
                "AbsRel": (acc["sum_absrel"] / div).tolist(),
                "delta1": (acc["sum_delta1"] / div).tolist(),
                "gt_mean": (acc["sum_gt"] / div).tolist(),
                "gt_std": np.sqrt(np.maximum(
                    acc["sum_gt2"] / div - (acc["sum_gt"] / div) ** 2,
                    0.0)).tolist(),
                "n_frames": len(live)}


def standardise_by_depth(aligned: np.ndarray, gt_z: np.ndarray,
                         mask: np.ndarray, coord: np.ndarray,
                         edges: Sequence[float], n_strata: int = DEPTH_STRATA,
                         min_depth: float = 0.01, max_depth: float = 100.0,
                         min_cell_px: int = MIN_STRATUM_PX) -> List[dict]:
    """What each bin would score **if it saw the whole frame's depth mix**.

    **The ADT-FOV experiment does not use this.** It reports AbsRel and delta1
    against position in the field, with the measured per-bin GT depth beside
    them, and leaves the reader to weigh the confound rather than correcting for
    it. The function stays because ``datasets_egosynth`` calls it directly for
    its own pooled path.

    Direct standardisation, the oldest trick for the job. Cut
    the frame's valid GT at its own depth quantiles, score every
    (bin x stratum) cell separately, and average a bin's cells with the weight
    each stratum has in the frame rather than the weight it has in that bin.
    Quantile cuts make those weights uniform, so the standardised value is the
    plain mean over strata. Any difference between a bin's raw and standardised
    score is what its depth distribution was contributing.

    **It reduces the confound, it does not remove it.** Within a stratum the
    bins' depths still differ, and the residual scales with how coarse the
    strata are — measured at ~25% of a purely-depth penalty left standing at
    four strata (see ``DEPTH_STRATA``). So a standardised curve that is still
    rising is evidence of a real effect; a standardised curve that goes flat has
    only been shown to be *mostly* depth. Quote both columns, never this one
    alone.

    It cannot conjure overlap that is not there. If a bin misses a stratum — the
    rim of an egocentric frame may hold no far pixels at all — then *what the rim
    would score at the centre's depth* is not in this data, and the cell is
    ``nan`` with ``ds_strata`` recording how many strata it did populate. That is
    the honest answer; dropping the missing stratum and averaging the rest would
    reintroduce exactly the bias being removed.

    Returns one dict per bin, keyed ``AbsRel_ds``, ``delta1_ds``, ``ds_strata``.
    """
    empty = {"AbsRel_ds": float("nan"), "delta1_ds": float("nan"),
             "ds_strata": 0}
    n_bins = len(edges) - 1
    if n_strata < 1:
        return [dict(empty) for _ in range(n_bins)]
    g = gt_z[mask]
    g = g[np.isfinite(g) & (g > min_depth) & (g < max_depth)]
    if g.size < min_cell_px * n_strata:
        return [dict(empty) for _ in range(n_bins)]
    cuts = np.quantile(g.astype(np.float64), np.linspace(0.0, 1.0, n_strata + 1))
    # A frame that is one flat wall has collapsed quantiles: there is no depth
    # axis to standardise along, so say so rather than emit strata that are not
    # distinct. Only the interior cuts matter; the ends are opened out.
    if np.unique(cuts[1:-1]).size < n_strata - 1:
        return [dict(empty) for _ in range(n_bins)]
    cuts[0], cuts[-1] = -np.inf, np.inf
    strata = [(mask & (gt_z >= lo) & (gt_z < hi))
              for lo, hi in zip(cuts[:-1], cuts[1:])]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = mask & (coord >= lo) & (coord < hi)
        cells, used = [], 0
        for s in strata:
            m = b & s
            if int(m.sum()) < min_cell_px:
                cells.append(None)
                continue
            used += 1
            cells.append(depth_metrics(aligned, gt_z, m, min_depth, max_depth))
        if used < n_strata:
            out.append({"AbsRel_ds": float("nan"), "delta1_ds": float("nan"),
                        "ds_strata": used})
            continue
        out.append({"AbsRel_ds": float(np.mean([c["AbsRel"] for c in cells])),
                    "delta1_ds": float(np.mean([c["delta1"] for c in cells])),
                    "ds_strata": used})
    return out


def _gt_stats(gt_z: np.ndarray, mask: np.ndarray) -> dict:
    """The bin's own GT depth: median, standard deviation, and IQR/median.

    ``gt_std`` is the plain spread in metres, so a depth curve can be drawn with
    a band rather than as a bare line: a bin whose depth ranges over metres and
    one that is a flat wall tell very different stories about the same median.
    """
    nan = float("nan")
    empty = {"gt_median": nan, "gt_std": nan, "gt_spread": nan}
    if not mask.any():
        return dict(empty)
    g = gt_z[mask].astype(np.float64)
    g = g[np.isfinite(g) & (g > 0)]
    if g.size < 4:
        return dict(empty)
    return {"gt_median": float(np.median(g)),
            "gt_std": float(np.std(g)),
            "gt_spread": _relative_spread(gt_z, mask)}


def anchored_ratios(pred_z: np.ndarray, gt_z: np.ndarray, mask: np.ndarray,
                    theta_deg: np.ndarray, edges: Sequence[float],
                    align_mode: str, min_anchor_px: int = 64) -> List[float]:
    """Per-bin ``median(gt/pred)`` after fitting the model's own global affine
    **on the innermost populated bin alone**.

    **The ADT-FOV report does not use this.** That experiment reports AbsRel and
    delta1 against position in the field and nothing else, and the ``drift``
    column this fed was removed. It stays because ``datasets_egosynth`` mirrors
    it for its pooled path and cross-checks that path against ``bin_by``'s
    output, so the number must keep existing in the JSON even though no ADT
    table prints it.

    Why the anchor, rather than fitting on the whole frame or not at all — the
    two obvious choices, both of which are wrong in opposite directions:

    * **No fit** (``raw_scale_ratio``): every one of these models has an additive
      degree of freedom, and an offset makes ``gt/pred`` vary with the *scene
      depth* of each bin. On a scene whose depth falls with eccentricity — which
      an enclosed egocentric frame is, for a reason that is geometric rather
      than about clutter: past ~35 deg the field leaves the wall it faces and
      starts catching floor, ceiling and side walls, surfaces the optical axis
      is *parallel* to, where planar z goes as the perpendicular distance over
      ``tan(theta)`` and collapses (``tests/test_geometry.py::
      test_but_planar_z_falls_by_a_factor_of_two_on_the_same_room`` measures
      x0.57 across an empty 5x4x2.6 m room, while euclidean range over the same
      room stays flat) — a model with **no radial error at all**
      reports 0.648 or 1.253 depending on the offset's sign, and an
      affine-invariant disparity model reports 1.143. Those are the size of the
      effect being looked for, so the measure has no specificity.
    * **Fit on the whole frame** (``scale_ratio``): least squares then spends its
      scale and shift partly on the radial trend itself. Correct 1.000 on the
      no-distortion cases, but a real ``+0.6 theta^2`` bias reads 0.965 — the
      effect is absorbed, so the measure has no sensitivity.

    Anchoring gets both. The anchor band spans ~10 deg about the optical axis, so
    it carries almost none of the radial variation the fit must not absorb, while
    still removing the offset. Measured on analytic scenes
    (``tests/test_geometry.py``): exactly 1.000 for pure-scale, for either sign
    of offset, and for a disparity model — no false positive — and 1.37 against a
    true 1.49 for a real bias. It under-reports by <10%, because the anchor band
    itself carries a little bias, and never invents.

    The anchor must be **conditioned**, not merely populated. The affine has two
    parameters, so fitting it needs depth *spread*: on a band that is one flat
    wall at a constant range, ``(s, t)`` is undetermined and the fit returns an
    arbitrary pair that then corrupts every other bin. That is not hypothetical —
    a synthetic box scene whose central 25 deg is a single wall drives this
    function to report drift 1.86 for a model with no radial error whatever.
    Real ADT is nowhere near that: measured on seq131, IQR/median of the GT
    inside each band runs 0.71 to 0.88, because the centre of an egocentric frame
    sees a table, objects and floor at many ranges. So the guard below almost
    never fires on real data, and when it does the answer is NaN rather than a
    fabricated one.

    The anchor is the innermost band that is both populated and conditioned;
    bands are merged inward-outward until one qualifies. Returns NaN per bin when
    none does.
    """
    edges = [float(e) for e in edges]
    bands = [mask & (theta_deg >= lo) & (theta_deg < hi)
             for lo, hi in zip(edges[:-1], edges[1:])]
    anchor = None
    for b in bands:
        if b.sum() < min_anchor_px:
            continue
        if _relative_spread(gt_z, b) >= MIN_ANCHOR_SPREAD:
            anchor = b
            break
    if anchor is None:
        return [float("nan")] * len(bands)
    fitted = align_depth(pred_z, gt_z, anchor, mode=align_mode)
    return [raw_scale_ratio(fitted, gt_z, b) for b in bands]


def _relative_spread(gt_z: np.ndarray, mask: np.ndarray) -> float:
    """Interquartile range of GT over ``mask``, as a fraction of its median.

    The conditioning number for an affine fit on that region: 0 means every
    pixel is at the same depth and ``(scale, shift)`` cannot be separated.
    """
    if not mask.any():
        return 0.0
    g = gt_z[mask].astype(np.float64)
    g = g[np.isfinite(g) & (g > 0)]
    if g.size < 4:
        return 0.0
    med = float(np.median(g))
    if med <= 1e-9:
        return 0.0
    return float(np.percentile(g, 75) - np.percentile(g, 25)) / med


def raw_scale_ratio(pred_z: np.ndarray, gt_z: np.ndarray,
                    mask: np.ndarray) -> float:
    """``median(gt / pred)`` on the **unaligned** prediction over ``mask``.

    A diagnostic, **not** a distortion measure — see ``anchored_ratios``, which
    is what the report's ``drift`` column uses. Compared across incidence bins
    this quantity moves whenever the model carries an additive offset, whether
    or not anything is bending: for ``pred = (gt - 1.5)/3`` on a scene whose
    depth falls with eccentricity, and with no radial error whatever, it reports
    0.648; for ``pred = (gt + 2)/3`` it reports 1.253; for an affine-invariant
    disparity model it reports 1.143. Those are the size of the effect this
    benchmark exists to detect. Kept because it is the raw number and someone
    will want it, but do not read a trend off it.
    """
    if not mask.any():
        return float("nan")
    p = pred_z[mask].astype(np.float64)
    g = gt_z[mask].astype(np.float64)
    ok = np.isfinite(p) & np.isfinite(g) & (p > 1e-8) & (g > 0)
    if not ok.any():
        return float("nan")
    return float(np.median(g[ok] / p[ok]))
