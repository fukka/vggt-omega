# Copyright (c) 2026.
"""Dynamic-scene handling helpers.

The primary dynamic signal is the *free* mask returned by
``compute_self_supervised_losses`` (geometric residual). These helpers add the
optional rigid-flow-vs-optical-flow motion segmentation and a utility to fuse
several masks (geometric residual, optical-flow residual, semantic hand/object
masks) into one static-pixel weight.
"""
from __future__ import annotations

from typing import Sequence

import torch

from ..geometry import (
    backproject,
    decode_pose_encoding,
    pixel_grid,
    project,
    relative_pose,
    shifted_indices,
    transform_points,
)


def rigid_flow(depth: torch.Tensor, pose_enc: torch.Tensor, offset: int) -> torch.Tensor:
    """Camera-induced (rigid) optical flow implied by depth + relative pose.

    Returns flow ``[B,S,H,W,2]`` (du, dv) mapping each ref pixel to where a
    *static* scene point would land in the neighbour ``ref + offset``.
    """
    B, S = depth.shape[:2]
    H, W = depth.shape[-2:]
    device = depth.device
    extr, K = decode_pose_encoding(pose_enc.float(), (H, W))
    idx = shifted_indices(S, offset, device)

    ref_d = depth.reshape(B * S, H, W)
    ref_E = extr.reshape(B * S, 4, 4)
    ref_K = K.reshape(B * S, 3, 3)
    src_E = extr[:, idx].reshape(B * S, 4, 4)
    src_K = K[:, idx].reshape(B * S, 3, 3)

    T = relative_pose(ref_E, src_E)
    pts = transform_points(backproject(ref_d, ref_K), T)
    pix, _ = project(pts, src_K)
    grid = pixel_grid(H, W, device, depth.dtype)[None]  # [1,H,W,2]
    flow = (pix - grid).reshape(B, S, H, W, 2)
    return flow


def flow_residual_mask(
    rigid: torch.Tensor, optical: torch.Tensor, threshold: float = 2.0
) -> torch.Tensor:
    """Static-pixel mask from ``||optical - rigid||`` (pixels). 1 = static.

    ``rigid, optical`` are ``[...,2]`` flow fields in pixels. Large residual =>
    independently moving surface (e.g. a hand) => 0.
    """
    residual = (optical - rigid).norm(dim=-1)
    return (residual <= threshold).to(rigid.dtype)


def combine_masks(masks: Sequence[torch.Tensor], mode: str = "min") -> torch.Tensor:
    """Fuse several static-pixel masks/weights in ``[0,1]`` into one.

    ``"min"`` is conservative (a pixel must look static under *every* cue);
    ``"prod"`` is softer.
    """
    out = masks[0]
    for m in masks[1:]:
        out = torch.minimum(out, m) if mode == "min" else out * m
    return out
