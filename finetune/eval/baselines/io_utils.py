# Copyright (c) 2026.
"""Saving helpers: input RGB, metric depth (.npy + colorized .png), point cloud (.ply).

Every saved sample writes the full triple so a prediction can be inspected as an
image *and* as 3D geometry::

    <tag>_input.png   the RGB frame fed to the model (fisheye, or ERP for DAC)
    <tag>_depth.npy   raw metric depth (float32, model's native scale — unaligned)
    <tag>_depth.png   viridis colorization over the valid depth range
    <tag>_pcd.ply     coloured point cloud (binary little-endian PLY)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

from ..adt_depth import _colorize_depth


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> int:
    """Write a coloured point cloud as binary little-endian PLY.

    ``xyz`` ``[...,3]`` float, ``rgb`` ``[...,3]`` (uint8 or [0,1] float). Points
    that are non-finite or exactly at the origin (DAC's masked/out-of-FOV pixels)
    are dropped. Returns the number of points written.
    """
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb).reshape(-1, 3)
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255.0 if float(rgb.max()) <= 1.001 else rgb).clip(0, 255).astype(np.uint8)
    keep = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 1e-9)
    xyz, rgb = xyz[keep], rgb[keep]
    n = int(xyz.shape[0])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rec = np.empty(n, dtype=np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())
    return n


def save_depth_png(path: str, depth: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
    vals = depth[mask] if mask is not None and mask.any() else depth[np.isfinite(depth)]
    vmin = float(np.percentile(vals, 2)) if vals.size else 0.0
    vmax = float(np.percentile(vals, 98)) if vals.size else 1.0
    col = _colorize_depth(depth, vmin, vmax)
    if mask is not None:
        col[~mask] = 24
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(col).save(path)


def save_sample(
    out_dir: str,
    tag: str,
    rgb_u8: np.ndarray,
    depth: np.ndarray,
    points: Optional[np.ndarray] = None,
    point_rgb: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> None:
    """Write input + depth(.npy/.png) + point cloud(.ply) for one prediction."""
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(rgb_u8).save(os.path.join(out_dir, f"{tag}_input.png"))
    np.save(os.path.join(out_dir, f"{tag}_depth.npy"), depth.astype(np.float32))
    save_depth_png(os.path.join(out_dir, f"{tag}_depth.png"), depth, mask)
    if points is not None:
        prgb = point_rgb if point_rgb is not None else rgb_u8
        n = write_ply(os.path.join(out_dir, f"{tag}_pcd.ply"), points, prgb)
        if n == 0:
            print(f"  [io] WARN {tag}: point cloud empty (all points masked/invalid)")
