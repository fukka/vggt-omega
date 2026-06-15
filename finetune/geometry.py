# Copyright (c) 2026.
# Self-supervised egocentric finetuning utilities for VGGT-Omega.
"""Differentiable multi-view geometry used by the self-supervised losses.

All conventions mirror the released VGGT-Omega checkpoints
(see ``vggt_omega/utils/pose_enc.py`` and ``vggt_omega/utils/rotation.py``):

* pose encoding is 9D: translation (3) + quaternion **XYZW, scalar-last** (4)
  + (fov_h, fov_w);
* extrinsics are **camera-from-world** in OpenCV coordinates;
* intrinsics are pinhole with the principal point at the image centre;
* depth is metric and strictly positive, shape ``[..., H, W]``.

``decode_pose_encoding`` is re-implemented locally (rather than importing
``vggt_omega.utils.pose_enc``) so this module is dependency-free and can run
under any Python/torch version, including the CPU smoke test.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-6


# --------------------------------------------------------------------------- #
# reductions
# --------------------------------------------------------------------------- #
def masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None, eps: float = EPS) -> torch.Tensor:
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(eps)


def masked_weighted_mean(
    x: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    m = mask.to(x.dtype) * weight
    return (x * m).sum() / m.sum().clamp_min(eps)


# --------------------------------------------------------------------------- #
# pose encoding -> camera matrices (mirrors vggt_omega.utils.pose_enc)
# --------------------------------------------------------------------------- #
def quat_to_mat(quat: torch.Tensor) -> torch.Tensor:
    """XYZW scalar-last quaternion -> 3x3 rotation matrix."""
    i, j, k, r = torch.unbind(quat, -1)
    two_s = 2.0 / (quat * quat).sum(-1).clamp_min(EPS)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quat.shape[:-1] + (3, 3))


def decode_pose_encoding(
    pose_enc: torch.Tensor, image_hw: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """9D encoding ``[...,9]`` -> (extrinsics ``[...,4,4]``, intrinsics ``[...,3,3]``)."""
    H, W = image_hw
    T = pose_enc[..., :3]
    quat = pose_enc[..., 3:7]
    fov_h = pose_enc[..., 7]
    fov_w = pose_enc[..., 8]

    R = quat_to_mat(quat)
    lead = pose_enc.shape[:-1]
    extr = torch.zeros(lead + (4, 4), device=pose_enc.device, dtype=pose_enc.dtype)
    extr[..., :3, :3] = R
    extr[..., :3, 3] = T
    extr[..., 3, 3] = 1.0

    fy = (H / 2.0) / torch.tan(fov_h / 2.0)
    fx = (W / 2.0) / torch.tan(fov_w / 2.0)
    K = torch.zeros(lead + (3, 3), device=pose_enc.device, dtype=pose_enc.dtype)
    K[..., 0, 0] = fx
    K[..., 1, 1] = fy
    K[..., 0, 2] = W / 2.0
    K[..., 1, 2] = H / 2.0
    K[..., 2, 2] = 1.0
    return extr, K


def invert_se3(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3:] = -Rt @ t
    out[..., 3, 3] = 1.0
    return out


def relative_pose(extr_ref: torch.Tensor, extr_src: torch.Tensor) -> torch.Tensor:
    """Transform mapping a point in camera ``ref`` to camera ``src``: ``E_src @ inv(E_ref)``."""
    return extr_src @ invert_se3(extr_ref)


# --------------------------------------------------------------------------- #
# projection / warping (batched over a flat leading dim N)
# --------------------------------------------------------------------------- #
def pixel_grid(H: int, W: int, device, dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs + 0.5, ys + 0.5], dim=-1)  # [H,W,2] pixel centres (u, v)


def backproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """``depth [N,H,W]``, ``K [N,3,3]`` -> camera-space points ``[N,H,W,3]``."""
    N, H, W = depth.shape
    grid = pixel_grid(H, W, depth.device, depth.dtype)
    ones = torch.ones(H, W, 1, device=depth.device, dtype=depth.dtype)
    pix_h = torch.cat([grid, ones], dim=-1)  # [H,W,3]
    Kinv = torch.inverse(K)
    dirs = torch.einsum("nij,hwj->nhwi", Kinv, pix_h)
    return dirs * depth.unsqueeze(-1)


def transform_points(pts: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    return torch.einsum("nij,nhwj->nhwi", R, pts) + t[:, None, None, :]


def project(pts_cam: torch.Tensor, K: torch.Tensor, eps: float = EPS):
    """Camera points ``[N,H,W,3]`` -> (pixel coords ``[N,H,W,2]``, depth ``[N,H,W]``)."""
    proj = torch.einsum("nij,nhwj->nhwi", K, pts_cam)
    z = proj[..., 2].clamp_min(eps)
    pix = proj[..., :2] / z.unsqueeze(-1)
    return pix, pts_cam[..., 2]


def sample(img: torch.Tensor, pix: torch.Tensor):
    """Bilinearly sample ``img [N,C,H,W]`` at pixel-centre coords ``pix [N,H,W,2]``.

    Returns ``(sampled [N,C,H,W], valid [N,H,W])``.
    """
    N, C, H, W = img.shape
    u, v = pix[..., 0], pix[..., 1]
    gx = 2.0 * u / W - 1.0
    gy = 2.0 * v / H - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    sampled = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    valid = (gx > -1) & (gx < 1) & (gy > -1) & (gy < 1)
    return sampled, valid


def warp(src_img, depth_ref, K_ref, K_src, T_src_ref):
    """Inverse-warp ``src_img`` into the reference frame using ref depth + relative pose.

    Returns ``(warped [N,C,H,W], proj_depth [N,H,W], pix [N,H,W,2], valid [N,H,W])``
    where ``proj_depth`` is the depth of the reference surface expressed in the
    source camera (used by the geometric-consistency residual).
    """
    pts_ref = backproject(depth_ref, K_ref)
    pts_src = transform_points(pts_ref, T_src_ref)
    pix, proj_depth = project(pts_src, K_src)
    warped, valid = sample(src_img, pix)
    return warped, proj_depth, pix, valid


def shifted_indices(S: int, offset: int, device) -> torch.Tensor:
    """Temporal neighbour indices, clamped at the sequence ends."""
    return torch.clamp(torch.arange(S, device=device) + offset, 0, S - 1)
