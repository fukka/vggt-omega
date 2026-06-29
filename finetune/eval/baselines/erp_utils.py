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
    try:
        from dac.utils.erp_geometry import cam_to_erp_patch_fast
        from dac.dataloders.dataset import resize_for_input
    except ModuleNotFoundError:
        add_dac_to_path(None)  # honour $DAC_ROOT / third_party default
        from dac.utils.erp_geometry import cam_to_erp_patch_fast
        from dac.dataloders.dataset import resize_for_input
    import torch

    crop_h, crop_w = crop_size(cano, fwd_sz, crop_wFoV)
    depth3 = depth_hw.astype(np.float32)[..., None]
    valid3 = valid_hw.astype(np.float32)[..., None]

    erp_img, erp_depth, erp_valid, erp_active, lat, lon = cam_to_erp_patch_fast(
        img_hwc01.astype(np.float32), depth3, valid3,
        float(theta), float(phi), crop_h, crop_w, cano, cano * 2, dict(cam_params),
        roll=roll, scale_fac=scale_fac,
    )

    # Cone mask: exclude rays beyond the lens's valid FOV (fold-back ghosting).
    # Incidence angle is measured from the optical axis = the patch CENTRE
    # (theta, phi); cos of it is the gnomonic cosine cos_c. For an un-augmented
    # patch (theta=phi=0) this reduces to arccos(cos(lat)·cos(lon)).
    if max_incidence_deg is None and str(cam_params.get("camera_model")) == "OPENCV_FISHEYE":
        from .aria_fisheye import kb4_max_incidence
        max_incidence_deg = float(np.degrees(kb4_max_incidence(
            (float(cam_params["k1"]), float(cam_params["k2"]),
             float(cam_params["k3"]), float(cam_params["k4"])))))
    if max_incidence_deg is not None:
        cos_c = (np.sin(phi) * np.sin(lat)
                 + np.cos(phi) * np.cos(lat) * np.cos(lon - theta))
        incidence = np.degrees(np.arccos(np.clip(cos_c, -1.0, 1.0)))
        erp_active = (erp_active * (incidence <= max_incidence_deg)).astype(np.float32)

    lat_range = torch.tensor([float(np.min(lat)), float(np.max(lat))], dtype=torch.float32)
    long_range = torch.tensor([float(np.min(lon)), float(np.max(lon))], dtype=torch.float32)

    # resize ERP crop -> fwd_sz (image+depth+active mask), exactly like the demo.
    img_u8, depth_fwd, _pad, psf, active_fwd = resize_for_input(
        (np.clip(erp_img, 0, 1) * 255.0).astype(np.uint8), erp_depth, list(fwd_sz),
        None, [erp_img.shape[0], erp_img.shape[1]], 1.0,
        padding_rgb=[0, 0, 0], mask=erp_active,
    )
    # valid mask needs the same spatial transform; resize via a second pass.
    _, valid_fwd, _, _, _ = resize_for_input(
        (np.clip(erp_img, 0, 1) * 255.0).astype(np.uint8), erp_valid, list(fwd_sz),
        None, [erp_img.shape[0], erp_img.shape[1]], 1.0,
        padding_rgb=[0, 0, 0], mask=erp_active,
    )
    return {
        "image_u8": img_u8,
        "depth": np.asarray(depth_fwd, np.float32),
        "valid": np.asarray(valid_fwd, np.float32),
        "active": np.asarray(active_fwd, np.float32),
        "lat_range": lat_range,
        "long_range": long_range,
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
