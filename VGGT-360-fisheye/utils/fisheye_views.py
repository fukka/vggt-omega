# Copyright (c) 2026.
"""Fisheye -> perspective view generation (VGGT-360 module 1, fisheye port).

Upstream VGGT-360 covers the full sphere with 6 horizontal views + 2 poles
(``utils/gen_views.py``).  The Aria RGB fisheye only images a cone of half-angle
``theta_max ~ 62.3 deg``, so the layout is replaced by a **center + 8-direction
ring** (the "top, bottom, left, right, diagonals + center" layout):

    * 1 center view   : optical axis, FOV 60 deg
                        (covers incidence 0..30 deg at the edge, 39 deg corner)
    * 8 ring views    : azimuth psi in {0, 45, ..., 315} deg, tilt alpha=32 deg,
                        FOV 60 deg -> each covers incidence 2..62 deg along its
                        azimuth; the outer edge lands exactly on theta_max
                        (design rule: ``alpha + FOV/2 ~= theta_max``).

Overlap (the precondition for VGGT's cross-view attention to unify scale):
adjacent ring-view centers are ``2*asin(sin 32 * sin 22.5) ~ 23.4 deg`` apart —
far less than the 60-deg FOV — and ring<->center separation is 32 deg < 60.
The cone is covered without gaps; ring-view *corners* poke slightly beyond
theta_max and are zeroed by the analytic valid mask (they carry no image).

View parameterisation
---------------------
A view is ``(azimuth_deg, tilt_deg, fov_deg)``:

    * ``tilt``    (alpha) — angle between the view's optical axis and the
                  camera's optical axis (+z).
    * ``azimuth`` (psi)   — direction of the tilt in the image plane, measured
                  from +x (image right) toward +y (image down).  psi=0 tilts
                  the view toward the right edge of the fisheye frame, psi=90
                  toward the bottom.

This replaces upstream's (yaw, pitch): a cone has no poles, so the pole
special-casing in upstream ``augment_for_top2`` disappears entirely.  The
rotation mapping view coords -> camera coords is the minimal-twist rotation
taking +z to the view center direction (Rodrigues about ``z x d``).

Uncertainty-guided adaptive augmentation (paper module 1, step 2) is kept
verbatim in spirit: Sobel-gradient confidence scoring of the base views
(functions vendored from upstream ``gen_views.py``), pick the two least
confident, add neighbor views around each — with the pole clamp replaced by a
single cone constraint ``tilt + fov/2 <= theta_max + margin``.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .fisheye_cam import FisheyeCam, kb4_forward_theta

# A view: (azimuth_deg, tilt_deg, fov_deg)
ViewParam = Tuple[float, float, float]

# Default layout (see module docstring for the geometry derivation).
DEFAULT_FOV_DEG = 60.0
DEFAULT_RING_TILT_DEG = 32.0
DEFAULT_N_RING = 8


# --------------------------------------------------------------------------- #
# Rotation view <-> camera
# --------------------------------------------------------------------------- #

def view_center_dir(azimuth_deg: float, tilt_deg: float) -> np.ndarray:
    """Unit direction of a view's optical axis in the camera frame."""
    psi, a = math.radians(azimuth_deg), math.radians(tilt_deg)
    return np.array([math.sin(a) * math.cos(psi),
                     math.sin(a) * math.sin(psi),
                     math.cos(a)], dtype=np.float64)


def view_rotation(azimuth_deg: float, tilt_deg: float) -> np.ndarray:
    """Rotation matrix R with ``v_cam = R @ v_view``.

    The minimal-twist rotation taking the camera +z axis onto the view center
    direction: Rodrigues rotation of angle ``tilt`` about the (unit) axis
    ``z x d = (-sin psi, cos psi, 0)``.  For tilt=0 this is the identity —
    the center view shares the camera frame exactly.
    """
    a = math.radians(tilt_deg)
    if abs(a) < 1e-12:
        return np.eye(3)
    psi = math.radians(azimuth_deg)
    kx, ky, kz = -math.sin(psi), math.cos(psi), 0.0
    K = np.array([[0.0, -kz, ky],
                  [kz, 0.0, -kx],
                  [-ky, kx, 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def angular_sep_deg(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Angle in degrees between two view parameter tuples' center directions."""
    d1 = view_center_dir(v1[0], v1[1])
    d2 = view_center_dir(v2[0], v2[1])
    return math.degrees(math.acos(float(np.clip(np.dot(d1, d2), -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# Fisheye -> perspective warp  (the ERP2Persp replacement)
# --------------------------------------------------------------------------- #

def fisheye_to_persp(img: np.ndarray,
                     cam: FisheyeCam,
                     azimuth_deg: float,
                     tilt_deg: float,
                     fov_deg: float,
                     height: int = 512,
                     width: int = 512,
                     theta_max: Optional[float] = None,
                     return_maps: bool = False,
                     supersample: int = 1):
    """Render one perspective (tangent) view from a KB4 fisheye frame.

    Mirrors upstream ``ERP2Persp`` structure — build the tangent-plane ray
    grid, rotate it into the source frame, compute source pixel coordinates,
    ``cv2.remap`` — with the final "lon/lat -> ERP pixel" step replaced by the
    **KB4 forward projection** (closed form, no iterative inversion needed;
    the tricky inverse direction only appears in fusion, via the ray LUT).

    Parameters
    ----------
    img         : ``(H, W, C)`` or ``(H, W)`` fisheye frame (any dtype cv2 accepts).
    cam         : KB4 intrinsics for exactly this frame size / orientation.
    azimuth_deg, tilt_deg, fov_deg : view parameters (see module docstring).
    height, width : output view size (512x512 -> VGGT preprocessing keeps the
                  FOV and lands on the 518x518 / 37x37-token grid).
    theta_max   : incidence cone cutoff; default = the lens turnover.
    return_maps : also return the ``(u, v)`` sampling maps (for checkers).
    supersample : anti-aliasing factor.  ``cv2.remap`` with INTER_LINEAR only
                  reads a 2x2 neighbourhood, so rendering a view DIRECTLY at
                  518 from a 1408 fisheye point-samples (aliases) — the
                  fisheye region feeding one output pixel spans several source
                  pixels that never get averaged.  With ``supersample=s`` the
                  view is rendered at ``s*height x s*width`` then INTER_AREA-
                  downsampled to ``height x width`` (proper box filter), giving
                  a clean anti-aliased crop.  ``return_maps`` forces s=1 (the
                  maps must be at output resolution).  s=2..3 is plenty.

    Returns
    -------
    persp       : the rendered view, invalid pixels zeroed.
    valid       : ``(height, width)`` float32 in {0,1} — **analytic** validity
                  (ray inside the imaged cone AND source pixel inside frame).
                  Upstream uses ``rgb.sum() > 0`` to find invalid ERP pixels;
                  on fisheye that is unreliable (dark scenes, vignette), so
                  validity is computed from geometry and threaded through the
                  whole pipeline (SA masks, VGGT rgb_mask attention bias,
                  fusion weights).
    (mapx, mapy): only if ``return_maps``.
    """
    if theta_max is None:
        theta_max = cam.theta_max()

    # Anti-aliasing: render at s x resolution, box-downsample at the end.
    s = 1 if return_maps else max(1, int(supersample))
    out_h, out_w = height, width
    height, width = height * s, width * s

    # Tangent-plane ray grid; y down to match image row order (row 0 = top).
    t = math.tan(math.radians(fov_deg) / 2.0)
    xs = np.linspace(-t, t, width, dtype=np.float64)
    ys = np.linspace(-t, t, height, dtype=np.float64)
    xv, yv = np.meshgrid(xs, ys)
    vec = np.stack([xv, yv, np.ones_like(xv)], axis=-1)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

    # View -> camera frame (row vectors: v_cam = v_view @ R^T).
    R = view_rotation(azimuth_deg, tilt_deg)
    vec = vec @ R.T

    # KB4 forward projection.
    z = np.clip(vec[..., 2], -1.0, 1.0)
    theta = np.arccos(z)
    theta_d = kb4_forward_theta(theta, cam.k)
    rxy = np.sqrt(vec[..., 0] ** 2 + vec[..., 1] ** 2)
    inv = np.where(rxy > 1e-12, 1.0 / rxy, 0.0)
    u = cam.cx + cam.fx * theta_d * vec[..., 0] * inv
    v = cam.cy + cam.fy * theta_d * vec[..., 1] * inv

    # Half-pixel tolerance on the frame bounds: the Aria principal point is
    # off-center, so the imaged cone can overhang the frame border by a
    # fraction of a pixel; a strict check would punch pinholes into the
    # coverage exactly on the border column/row (found by checks test C).
    valid = ((theta <= theta_max - 1e-6)
             & (u > -0.5) & (u < cam.W - 0.5)
             & (v > -0.5) & (v < cam.H - 0.5))

    mapx = u.astype(np.float32)
    mapy = v.astype(np.float32)
    persp = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_f = valid.astype(np.float32)
    if persp.ndim == 3:
        persp = persp * valid_f[..., None]
    else:
        persp = persp * valid_f
    if return_maps:
        return persp, valid_f, mapx, mapy
    if s > 1:
        # Box-filter downsample to the requested output size (anti-alias).
        persp = cv2.resize(persp, (out_w, out_h), interpolation=cv2.INTER_AREA)
        valid_f = cv2.resize(valid_f, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return persp, valid_f


# --------------------------------------------------------------------------- #
# Uncertainty scoring — vendored unchanged from upstream utils/gen_views.py
# (paper module 1: Sobel-gradient "geometric uncertainty" of each view).
# --------------------------------------------------------------------------- #

def perspective_confidence_map(rgb: np.ndarray, tau: float = 0.5,
                               valid_mask: Optional[np.ndarray] = None):
    """Per-pixel confidence from Sobel gradients (upstream, + analytic mask).

    Identical numerics to upstream ``perspective_confidence_map`` except the
    valid mask can be supplied analytically instead of the black-pixel test.
    """
    g = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 1.0)

    p1, p99 = np.percentile(mag, 1), np.percentile(mag, 99)
    T = np.clip((mag - p1) / (p99 - p1 + 1e-6), 0.01, 0.99)

    t0 = float(np.median(T))
    conf_map = 1.0 / (1.0 + np.exp(-(T - t0) / max(tau, 1e-6)))

    if valid_mask is None:
        valid_mask = rgb.sum(axis=2) > 0
    valid_mask = valid_mask.astype(bool)
    conf_map[~valid_mask] = 0.0
    return conf_map, valid_mask


def confidence_score_from_map(conf_map: np.ndarray, valid_mask: np.ndarray,
                              trim_ratio: float = 0.1,
                              min_valid_ratio: float = 0.1) -> float:
    """Trimmed-mean confidence over valid pixels (upstream, verbatim).

    Normalising by the valid-pixel count matters more here than upstream:
    ring views have cone-clipped corners, and without it those views would be
    scored down for geometry rather than content.
    """
    H, W = conf_map.shape
    valid = valid_mask & np.isfinite(conf_map)
    valid_ratio = valid.sum() / float(H * W + 1e-6)
    if valid_ratio < min_valid_ratio or valid.sum() == 0:
        return 1.0
    vals = np.sort(conf_map[valid].ravel())
    n = len(vals)
    a = int(n * trim_ratio)
    b = int(n * (1.0 - trim_ratio))
    if b <= a:
        return float(vals.mean())
    return float(vals[a:b].mean())


# --------------------------------------------------------------------------- #
# Base layout + adaptive neighbor augmentation
# --------------------------------------------------------------------------- #

def base_views(fov_deg: float = DEFAULT_FOV_DEG,
               ring_tilt_deg: float = DEFAULT_RING_TILT_DEG,
               n_ring: int = DEFAULT_N_RING) -> List[ViewParam]:
    """Center + N-direction ring covering the fisheye cone (default 1+8)."""
    views: List[ViewParam] = [(0.0, 0.0, fov_deg)]
    for i in range(n_ring):
        views.append((360.0 * i / n_ring, ring_tilt_deg, fov_deg))
    return views


def augment_least_confident(base: List[ViewParam],
                            worst_idx: Sequence[int],
                            theta_max_deg: float,
                            max_total: int = 13,
                            neighbor_fov: float = DEFAULT_FOV_DEG,
                            d_azimuth: float = 25.0,
                            d_tilt: float = 15.0,
                            min_sep_deg: float = 10.0,
                            cone_margin_deg: float = 5.0) -> List[ViewParam]:
    """Add neighbor views around the least-confident base views.

    Port of upstream ``augment_for_top2`` with (yaw, pitch) -> (azimuth, tilt)
    and the pole special-case replaced by one cone constraint:

        tilt_nbr + fov_nbr / 2  <=  theta_max + cone_margin

    (a neighbor whose *entire* footprint left the imaged cone is useless; a
    small overshoot is fine — the analytic valid mask zeroes the sliver).
    Candidates are azimuth twists and tilt offsets around the uncertain view;
    for the center view (tilt 0, azimuth undefined) two fixed diagonal
    candidates are used.  ``min_sep_deg`` keeps neighbors from duplicating an
    existing view, exactly like upstream.
    """
    views = list(base)
    extra = max_total - len(views)
    tilt_cap = theta_max_deg + cone_margin_deg - neighbor_fov / 2.0

    for idx in worst_idx:
        if extra <= 0:
            break
        psi0, a0, _ = base[idx]
        if a0 < 1e-6:  # center view: azimuth undefined -> fixed diagonals
            cands = [(45.0, d_tilt), (225.0, d_tilt)]
        else:
            cands = [((psi0 - d_azimuth) % 360.0, a0),
                     ((psi0 + d_azimuth) % 360.0, a0),
                     (psi0, max(0.0, a0 - d_tilt)),
                     (psi0, a0 + d_tilt)]
        added = 0
        for (cpsi, ca) in cands:
            if extra <= 0 or added >= 2:
                break
            ca = min(ca, tilt_cap)
            cand = (cpsi, ca, neighbor_fov)
            if min(angular_sep_deg(cand, vw) for vw in views) >= min_sep_deg:
                views.append(cand)
                extra -= 1
                added += 1
    return views


def view_generation_fisheye(fisheye_img: np.ndarray,
                            cam: FisheyeCam,
                            fov_deg: float = DEFAULT_FOV_DEG,
                            ring_tilt_deg: float = DEFAULT_RING_TILT_DEG,
                            n_ring: int = DEFAULT_N_RING,
                            view_hw: Tuple[int, int] = (512, 512),
                            adaptive: bool = True,
                            max_total: int = 13) -> List[ViewParam]:
    """Full module-1 pipeline: base layout + uncertainty-guided augmentation.

    Renders the base views once (cheap, cv2.remap), scores each with the
    upstream Sobel-confidence heuristic over *analytically* valid pixels,
    picks the two least confident, and adds up to 2 neighbors around each.
    Returns the final list of ``(azimuth, tilt, fov)`` — the caller re-renders
    (matching upstream main.py, which also renders twice).
    """
    views = base_views(fov_deg, ring_tilt_deg, n_ring)
    if not adaptive:
        return views

    theta_max_deg = math.degrees(cam.theta_max())
    scores = []
    for (psi, a, f) in views:
        persp, valid = fisheye_to_persp(fisheye_img, cam, psi, a, f,
                                        height=view_hw[0], width=view_hw[1])
        conf_map, vmask = perspective_confidence_map(persp, valid_mask=valid > 0.5)
        scores.append(confidence_score_from_map(conf_map, vmask))
    worst2 = list(np.argsort(np.asarray(scores))[:2])
    return augment_least_confident(views, worst2, theta_max_deg,
                                   max_total=max_total, neighbor_fov=fov_deg)
