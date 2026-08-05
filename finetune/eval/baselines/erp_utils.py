# Copyright (c) 2026.
"""Equirectangular (ERP) helpers wrapping Depth-Any-Camera's geometry.

DAC does not predict depth in the image plane — it warps the (fisheye) frame into
an ERP patch, predicts there, and (for visualisation) warps back. To evaluate DAC
and to compare UniK3D against it **in the same domain**, we reuse DAC's own
``cam_to_erp_patch_fast`` (validated locally to produce correct ERP unwrapping of
Aria fisheye) + ``resize_for_input`` so every tensor lands on the identical
``fwd_sz`` ERP grid the network sees.

The common eval domain is therefore "ERP patch resized to fwd_sz, euclidean
range" — DAC's native space. GT and UniK3D's planar-z prediction are converted to
range and warped through the same path; DAC predicts there directly.

All of this depends only on the DAC repo being importable (``--dac-root`` /
``$DAC_ROOT``); no model weights are needed for the warp itself.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

from .aria_fisheye import AriaFisheye, _kb4_unproject_theta


def add_dac_to_path(repo_root: Optional[str]) -> str:
    """Resolve + sys.path-insert the depth_any_camera repo root."""
    root = repo_root or os.environ.get("DAC_ROOT")
    if root is None:
        # default: <vggt-omega>/third_party/depth_any_camera (in-repo; via setup_baselines.sh)
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))  # -> vggt-omega
        root = os.path.join(repo, "third_party", "depth_any_camera")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"depth_any_camera repo not found at {root!r}. Run "
            f"`bash finetune/eval/baselines/setup_baselines.sh dac` to clone+install "
            f"it under the repo, or pass --dac-root / set $DAC_ROOT."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def crop_size(cano: int, fwd_sz: Tuple[int, int], crop_wFoV: float) -> Tuple[int, int]:
    """ERP-crop (height, width) for a given canonical size / fwd_sz / crop FOV.

    Mirrors ``demo_dac_single``: ``crop_w = cano · wFoV/180``,
    ``crop_h = crop_w · fwd_h/fwd_w``.
    """
    crop_w = int(cano * crop_wFoV / 180.0)
    crop_h = int(crop_w * fwd_sz[0] / fwd_sz[1])
    return crop_h, crop_w


# --------------------------------------------------------------------------- #
# Cached warp geometry  (the speed-up)
# --------------------------------------------------------------------------- #
# ``cam_to_erp_patch_fast`` splits cleanly into (a) building the per-pixel ERP→
# fisheye sampling grid + masks — pure trigonometry over the whole ~1.3 M-pixel
# crop that depends only on the camera and ERP params, NOT the frame content — and
# (b) sampling RGB/depth through that grid. For a fixed camera (every ADT/EgoExo
# frame uses the same intrinsics) the grid is identical across all frames *and*
# across the three warps we do per frame (GT, UniK3D-range, DAC-image). Rebuilding
# that trig each time dominated runtime, so we build the grid ONCE per
# (camera, ERP, augmentation) key and reuse it. The grid is *captured from DAC's
# own* ``cam_to_erp_patch_fast`` — we transiently wrap its ``grid_sample`` to record
# the grid it computes — so the projection geometry is byte-for-byte DAC's (no
# re-derivation of the fisheye math), and every subsequent frame is just three
# cheap ``grid_sample`` calls.

from typing import NamedTuple as _NamedTuple


class _ErpGeom(_NamedTuple):
    new_grid: object        # torch [1, crop_h, crop_w, 2] sampling grid (fisheye coords)
    active: np.ndarray      # [crop_h, crop_w] float32 = mask_active * lens-cone
    lat_range: object       # torch [2]
    long_range: object      # torch [2]
    depth_scale: float      # scale_fac applied to depth (1.0 if no scale aug)


_GEOM_CACHE: Dict[tuple, _ErpGeom] = {}


def _geom_key(cam_params, img_h, img_w, cano, fwd_sz, crop_wFoV,
              theta, phi, roll, scale_fac, max_incidence_deg) -> tuple:
    cam_items = tuple(sorted(
        (k, (float(v) if isinstance(v, (int, float)) else str(v)))
        for k, v in cam_params.items()))
    return (cam_items, int(img_h), int(img_w), int(cano), tuple(fwd_sz),
            round(float(crop_wFoV), 6), round(float(theta), 9), round(float(phi), 9),
            None if roll is None else round(float(roll), 9),
            None if scale_fac is None else round(float(scale_fac), 9),
            None if max_incidence_deg is None else round(float(max_incidence_deg), 6))


def _build_erp_geometry(cam_params, img_h, img_w, cano, fwd_sz, crop_wFoV,
                        theta, phi, roll, scale_fac, max_incidence_deg) -> _ErpGeom:
    """Build (and cache) the frame-invariant ERP sampling grid + masks.

    The grid is captured from one faithful call to DAC's ``cam_to_erp_patch_fast``
    (its ``grid_sample`` is wrapped just long enough to record the grid argument),
    so the projection math is unchanged — only the redundant per-frame recompute is
    removed. ``mask_active``/``lat``/``lon`` come straight from that same call; the
    KB4 lens-cone is derived exactly as the old per-call path did.
    """
    key = _geom_key(cam_params, img_h, img_w, cano, fwd_sz, crop_wFoV,
                    theta, phi, roll, scale_fac, max_incidence_deg)
    cached = _GEOM_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from dac.utils import erp_geometry as _erpmod
        from dac.utils.erp_geometry import cam_to_erp_patch_fast
    except ModuleNotFoundError:
        add_dac_to_path(None)
        from dac.utils import erp_geometry as _erpmod
        from dac.utils.erp_geometry import cam_to_erp_patch_fast
    import torch

    crop_h, crop_w = crop_size(cano, fwd_sz, crop_wFoV)
    dummy_img = np.zeros((int(img_h), int(img_w), 3), np.float32)
    dummy_dep = np.zeros((int(img_h), int(img_w), 1), np.float32)

    # Record the sampling grid DAC computes (identical for every frame) by capturing
    # the ``grid`` argument of its first grid_sample call; the sampled (zero) outputs
    # are discarded. mask_active/lat/lon are taken from the real return values.
    captured: Dict[str, object] = {}
    _orig_gs = _erpmod.F.grid_sample

    def _capturing_grid_sample(inp, grid, *a, **kw):
        captured.setdefault("grid", grid.detach().clone())
        return _orig_gs(inp, grid, *a, **kw)

    _erpmod.F.grid_sample = _capturing_grid_sample
    try:
        _, _, _, mask_active, lat, lon = cam_to_erp_patch_fast(
            dummy_img, dummy_dep, dummy_dep, float(theta), float(phi),
            crop_h, crop_w, cano, cano * 2, dict(cam_params),
            roll=roll, scale_fac=scale_fac)
    finally:
        _erpmod.F.grid_sample = _orig_gs   # always restore the global
    new_grid = captured["grid"]

    # Lens cone — the angle the lens actually IMAGES (~54.8° for Aria), not the
    # KB4 fold-back turnover (62.33°), which is a property of the polynomial fit
    # and sits ~7.5° past the imaged circle; masking to the turnover admits dead
    # vignette pixels into the eval. See ``aria_fisheye.usable_max_incidence``.
    # Precomputed once here; ``lat``/``lon`` are the ERP patch grids from the call.
    if max_incidence_deg is None and str(cam_params.get("camera_model")) == "OPENCV_FISHEYE":
        from .aria_fisheye import usable_max_incidence
        max_incidence_deg = float(np.degrees(usable_max_incidence(
            (float(cam_params["k1"]), float(cam_params["k2"]),
             float(cam_params["k3"]), float(cam_params["k4"])),
            float(cam_params["fl_x"]), float(cam_params["fl_y"]),
            float(cam_params["cx"]), float(cam_params["cy"]),
            float(img_h), float(img_w))))
    cone = np.ones_like(mask_active, dtype=np.float32)
    if max_incidence_deg is not None:
        cos_c = (np.sin(phi) * np.sin(lat)
                 + np.cos(phi) * np.cos(lat) * np.cos(lon - theta))
        incidence = np.degrees(np.arccos(np.clip(cos_c, -1.0, 1.0)))
        cone = (incidence <= max_incidence_deg).astype(np.float32)
    active = mask_active.astype(np.float32) * cone

    geom = _ErpGeom(
        new_grid=new_grid,
        active=active,
        lat_range=torch.tensor([float(np.min(lat)), float(np.max(lat))], dtype=torch.float32),
        long_range=torch.tensor([float(np.min(lon)), float(np.max(lon))], dtype=torch.float32),
        depth_scale=1.0 if scale_fac is None else float(scale_fac),
    )
    _GEOM_CACHE[key] = geom
    return geom


def fisheye_to_erp_fwd(
    img_hwc01: np.ndarray,
    depth_hw: np.ndarray,
    valid_hw: np.ndarray,
    cam_params: Dict[str, object],
    cano: int,
    fwd_sz: Tuple[int, int],
    crop_wFoV: float,
    max_incidence_deg: Optional[float] = None,
    theta: float = 0.0,
    phi: float = 0.0,
    roll: Optional[float] = None,
    scale_fac: Optional[float] = None,
) -> dict:
    """Warp a fisheye frame (+depth+mask) to the ERP ``fwd_sz`` grid DAC predicts on.

    Parameters
    ----------
    img_hwc01 : ``[H,W,3]`` float in [0,1] (fisheye RGB). Pass zeros if only depth
                is needed.
    depth_hw  : ``[H,W]`` float (euclidean range, metres). Pass zeros for an
                image-only warp (DAC input).
    valid_hw  : ``[H,W]`` float/bool valid-depth mask.
    cam_params: OPENCV_FISHEYE dict from ``AriaFisheye.opencv_fisheye_params()``.
    cano      : ERP canonical height the model trained on (model config ``cano_sz``).
    fwd_sz    : network input (h, w).
    crop_wFoV : crop field-of-view in degrees (demo uses 180).
    max_incidence_deg : cone cutoff. Rays whose incidence angle from the optical
                axis exceeds this are dropped from ``active`` (see below). Default
                ``None`` → auto-computed from the KB4 turnover for OPENCV_FISHEYE
                cameras (no cutoff for other models).
    theta, phi : ERP patch centre (longitude, latitude) in radians — DAC's
                viewpoint augmentation. ``phi`` is a pitch on the sphere (places the
                content at a different latitude), ``theta`` a yaw. Default ``0, 0``
                (centred on the optical axis, the eval setting).
    roll      : in-plane camera roll in radians (rotation about the optical axis),
                or ``None`` for no roll.
    scale_fac : FOV-align / zoom factor on the gnomonic coords (DAC's
                ``scale_fac``); also scales the warped depth so metric range stays
                consistent. ``None`` = 1.0.

    Returns a dict with everything on the ``fwd_sz`` grid::

        image_u8     [fh,fw,3] uint8     (ERP RGB, network input pre-normalisation)
        depth        [fh,fw]   float32   (ERP range GT, metres)
        valid        [fh,fw]   float32   (valid-depth mask)
        active       [fh,fw]   float32   (in-FOV region after ERP warp, cone-limited)
        lat_range    (2,) / long_range (2,)  torch tensors (IDiscERP inputs)
        pred_scale_factor  float         (multiply network depth output by this)

    Cone mask
    ---------
    ``crop_wFoV`` (180 in the demo) makes the ERP patch span ±90° longitude — far
    wider than a real fisheye cone (~±62° for Aria). ``cam_to_erp_patch_fast``
    applies the *raw* fisheye polynomial, which past the KB4 forward turnover folds
    back and samples a wrong, in-cone source pixel (ghosting). Neither ``active``
    nor ``valid`` catches that fold, so we additionally zero ``active`` wherever the
    ray's incidence angle exceeds ``max_incidence_deg``. Every consumer (GT,
    UniK3D, DAC) goes through this same path, so all three are scored only inside
    the physically imaged cone.
    """
    # Frame-invariant geometry (sampling grid + mask_active*cone) — built once per
    # camera/ERP key and reused; see ``_build_erp_geometry``. The cone mask still
    # excludes rays beyond the KB4 turnover (~62° for Aria) so the saved viz and the
    # DAC network input both show a black periphery instead of fold-back ghosting.
    geom = _build_erp_geometry(
        dict(cam_params), int(img_hwc01.shape[0]), int(img_hwc01.shape[1]),
        cano, tuple(fwd_sz), float(crop_wFoV),
        float(theta), float(phi), roll, scale_fac, max_incidence_deg)

    try:
        from dac.dataloders.dataset import resize_for_input
    except ModuleNotFoundError:
        add_dac_to_path(None)  # honour $DAC_ROOT / third_party default
        from dac.dataloders.dataset import resize_for_input
    import torch
    import torch.nn.functional as F

    # Cheap per-frame sampling through the cached grid — identical modes/padding to
    # ``cam_to_erp_patch_fast`` (bilinear+border for RGB, nearest+border for masks).
    ng = geom.new_grid
    img_t = torch.from_numpy(np.ascontiguousarray(
        img_hwc01.astype(np.float32).transpose(2, 0, 1)))[None]
    erp_img = F.grid_sample(img_t, ng, mode="bilinear", padding_mode="border",
                            align_corners=True)[0].permute(1, 2, 0).numpy()
    depth_t = torch.from_numpy(depth_hw.astype(np.float32))[None, None] * geom.depth_scale
    erp_depth = F.grid_sample(depth_t, ng, mode="nearest", padding_mode="border",
                              align_corners=True)[0, 0].numpy()
    valid_t = torch.from_numpy(valid_hw.astype(np.float32))[None, None]
    erp_valid = F.grid_sample(valid_t, ng, mode="nearest", padding_mode="border",
                              align_corners=True)[0, 0].numpy()

    # Apply mask_active*cone (matches the old path: cam_to_erp_patch_fast multiplied
    # by mask_active inside, then we zeroed the cone in RGB/depth/valid).
    active = geom.active
    erp_img = erp_img * active[..., None]
    erp_depth = erp_depth * active
    erp_valid = erp_valid * active

    ch, cw = active.shape
    # resize ERP crop -> fwd_sz (image+depth+active mask), exactly like the demo.
    img_u8, depth_fwd, _pad, psf, active_fwd = resize_for_input(
        (np.clip(erp_img, 0, 1) * 255.0).astype(np.uint8), erp_depth, list(fwd_sz),
        None, [ch, cw], 1.0, padding_rgb=[0, 0, 0], mask=active,
    )
    # valid mask needs the same spatial transform; resize via a second pass.
    _, valid_fwd, _, _, _ = resize_for_input(
        (np.clip(erp_img, 0, 1) * 255.0).astype(np.uint8), erp_valid, list(fwd_sz),
        None, [ch, cw], 1.0, padding_rgb=[0, 0, 0], mask=active,
    )
    return {
        "image_u8": img_u8,
        "depth": np.asarray(depth_fwd, np.float32),
        "valid": np.asarray(valid_fwd, np.float32),
        "active": np.asarray(active_fwd, np.float32),
        "lat_range": geom.lat_range,
        "long_range": geom.long_range,
        "pred_scale_factor": float(psf),
    }


def aria_fisheye_ray_lut(cam: AriaFisheye, max_fov_deg: float = 95.0) -> np.ndarray:
    """Per-pixel ray LUT ``[H,W,4]`` for ``erp_patch_to_cam_fast`` (DAC remap-back).

    ``[...,:3]`` = unit ray direction (x,y,z) for each fisheye pixel (KB4
    unprojection); ``[...,3]`` = isnan flag (1 = outside FOV, excluded). Used only
    for optional fisheye-frame visualisation of DAC output — not on the metrics
    path. Route it by setting ``cam_params={'dataset':'scannetpp'}``.
    """
    H, W = cam.H, cam.W
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    x = (us - cam.cx) / cam.fx
    y = (vs - cam.cy) / cam.fy
    theta_d = np.sqrt(x * x + y * y)
    theta = _kb4_unproject_theta(theta_d.astype(np.float64), cam.k)
    sin_t = np.sin(theta)
    inv = np.where(theta_d > 1e-9, 1.0 / theta_d, 0.0)
    rx = sin_t * x * inv
    ry = sin_t * y * inv
    rz = np.cos(theta)
    isnan = (theta > np.deg2rad(max_fov_deg)).astype(np.float32)
    lut = np.stack([rx, ry, rz, isnan], axis=-1).astype(np.float32)
    return lut
