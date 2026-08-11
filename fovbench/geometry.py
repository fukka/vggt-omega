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
#: plus this module's alignment-free ``raw_scale_ratio``. Named once so the
#: driver's frame-averaging and the report's CSV cannot drift apart.
METRIC_KEYS: Tuple[str, ...] = ("AbsRel", "SqRel", "RMSE", "RMSElog", "log10",
                                "delta1", "delta2", "delta3",
                                "scale_ratio", "raw_scale_ratio",
                                "anchored_ratio")

#: ``finetune/data/rectify.py`` renders the pinhole at ``0.55 * max(H, W)``.
RECTIFIER_FOCAL_FRAC = 0.55

#: Minimum GT interquartile-range-over-median for a band to anchor an affine
#: fit. Real ADT bands measure 0.71-0.88; a single flat wall measures 0.00.
MIN_ANCHOR_SPREAD = 0.05

_RAY_CACHE: dict = {}


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
    """Per-pixel incidence angle (degrees) of a raw fisheye frame."""
    rays, _ = fisheye_rays(cam)
    return np.degrees(np.arccos(np.clip(rays[..., 2], -1.0, 1.0))).astype(np.float32)


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
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    x = (xs - cx) / f
    y = (ys - cy) / f
    return np.degrees(np.arctan(np.sqrt(x * x + y * y))).astype(np.float32)


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
        from finetune.data.rectify import FisheyeRectifier
        rec = FisheyeRectifier("aria-214-1")
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
    return FrameView(kind=kind, rgb=rgb_v, gt_z=gt_v * valid, valid=valid,
                     theta=theta, in_cone_frac=float(in_cone.mean()))


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

    aligned = align_depth(pred_z, gt_z, in_range, mode=align_mode)
    overall = depth_metrics(aligned, gt_z, in_range, min_depth, max_depth)

    anchored = anchored_ratios(pred_z, gt_z, in_range, theta_deg, edges,
                               align_mode)

    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = in_range & (theta_deg >= lo) & (theta_deg < hi)
        met = depth_metrics(aligned, gt_z, m, min_depth, max_depth)
        met["anchored_ratio"] = anchored[i]
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


def anchored_ratios(pred_z: np.ndarray, gt_z: np.ndarray, mask: np.ndarray,
                    theta_deg: np.ndarray, edges: Sequence[float],
                    align_mode: str, min_anchor_px: int = 64) -> List[float]:
    """Per-bin ``median(gt/pred)`` after fitting the model's own global affine
    **on the innermost populated bin alone**. This is the benchmark's distortion
    measure; the report's ``drift`` column is its first-over-last ratio.

    Why the anchor, rather than fitting on the whole frame or not at all — the
    two obvious choices, both of which are wrong in opposite directions:

    * **No fit** (``raw_scale_ratio``): every one of these models has an additive
      degree of freedom, and an offset makes ``gt/pred`` vary with the *scene
      depth* of each bin. On a scene whose depth falls with eccentricity — which
      an egocentric indoor frame is — a model with **no radial error at all**
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
