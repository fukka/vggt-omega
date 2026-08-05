"""Sequence sources and the three-frame adaptation windows.

Paper Sec. 5, "Evaluation protocol": for each sequence, build a short adaptation
set of **30 three-frame windows**, dropping nearly-static windows whose average
optical-flow displacement is below **2 pixels**; fit the adapter on that set and
evaluate on the full sequence.

Two sources are implemented, matching the datasets chosen for this reproduction:

* :class:`ScanNetPPFisheye` -- the paper's ScanNet++ [50] setting (115 deg DSLR
  fisheye), with reference poses and rendered depth, so Tab. 1 and Tab. 3 (right
  half) are directly comparable.
* :class:`ADTSequence` -- this repo's ADT / Aria fisheye, reusing the calibration
  and trajectory helpers already in ``cam3r/adt.py`` and
  ``finetune/eval/baselines/aria_fisheye.py`` rather than restating them.

**ADT pose caveat.** ADT's ``groundtruth/aria_trajectory.csv`` gives
world-from-*device* poses. Camera poses need ``T_device_camera``, which lives in
the MPS calibration or the VRS. When it cannot be resolved,
``cam3r.adt.resolve_extrinsics`` falls back to the device frame and reports
``exact=False``; this module then leaves ``gt_R``/``gt_t`` unset rather than
emit poses that are wrong by the sensor lever arm and mounting rotation. Depth
metrics still work; pose and ``d_reproj`` are skipped and the evaluator says so.
Adaptation itself is unaffected -- it never reads ground-truth pose.
"""

from __future__ import annotations

import glob
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .cameras import Camera, KannalaBrandt, from_aria, from_scannetpp
from .matching import Matcher, Matches, mean_flow_magnitude, relative_pose_magsac

__all__ = ["Window", "SequenceSource", "ScanNetPPFisheye", "ADTSequence",
           "build_windows", "load_sequence"]


@dataclass
class Window:
    """One adaptation/evaluation window of ``S`` frames (``S = 3`` in the paper)."""

    images: Tensor                                  # (S, 3, H, W) in [0, 1]
    indices: List[int]
    camera: Camera
    matches: Dict[Tuple[int, int], Matches] = field(default_factory=dict)
    pose_targets: Dict[Tuple[int, int], Optional[Tuple[Tensor, Tensor]]] = field(default_factory=dict)
    gt_R: Optional[Tensor] = None                   # (S, 3, 3) camera-from-world
    gt_t: Optional[Tensor] = None                   # (S, 3)
    gt_depth: Optional[Tensor] = None               # (S, H, W) metres
    gt_valid: Optional[Tensor] = None               # (S, H, W) bool

    def as_batch(self) -> dict:
        return {"images": self.images, "matches": self.matches,
                "pose_targets": self.pose_targets}

    def gt_relative(self, i: int, j: int) -> Optional[Tuple[Tensor, Tensor]]:
        if self.gt_R is None:
            return None
        R = self.gt_R[j] @ self.gt_R[i].transpose(-1, -2)
        return R, self.gt_t[j] - R @ self.gt_t[i]


class SequenceSource:
    """A posed image sequence from one camera."""

    name: str = "sequence"
    camera: Camera

    def __len__(self) -> int:
        raise NotImplementedError

    def image(self, i: int) -> Tensor:
        """``(3, H, W)`` float in [0, 1] at the model's working resolution."""
        raise NotImplementedError

    def pose(self, i: int) -> Optional[Tuple[Tensor, Tensor]]:
        """Camera-from-world ``(R, t)``, or ``None`` if unavailable."""
        return None

    def depth(self, i: int) -> Optional[Tuple[Tensor, Tensor]]:
        """``(depth_metres, valid_mask)``, or ``None``."""
        return None


def _resize_to(img: Tensor, size: Tuple[int, int], mode: str = "bilinear") -> Tensor:
    if img.shape[-2:] == size:
        return img
    kw = {"antialias": True} if mode == "bilinear" else {}
    return F.interpolate(img[None].float(), size=size, mode=mode,
                         align_corners=False if mode == "bilinear" else None, **kw)[0]


def _patch_align(value: int, patch: int) -> int:
    return max(patch, (value // patch) * patch)


def working_size(height: int, width: int, *, max_size: int = 504, patch: int = 14
                 ) -> Tuple[int, int]:
    """Largest patch-aligned size within ``max_size`` preserving aspect ratio.

    The paper resizes "to a maximum patch-aligned resolution of 504x504".
    """
    scale = min(max_size / max(height, width), 1.0) if max(height, width) > max_size else \
        max_size / max(height, width)
    h = _patch_align(int(round(height * scale)), patch)
    w = _patch_align(int(round(width * scale)), patch)
    return h, w


# ---------------------------------------------------------------------------
# ScanNet++ DSLR fisheye
# ---------------------------------------------------------------------------


class ScanNetPPFisheye(SequenceSource):
    """One ScanNet++ scene's DSLR fisheye capture (~115 deg FOV).

    Expects the official layout::

        <scene>/dslr/nerfstudio/transforms.json      OPENCV_FISHEYE intrinsics + poses
        <scene>/dslr/resized_images/<file_path>      RGB
        <scene>/dslr/render_depth/<stem>.png         rendered depth, uint16 mm (optional)

    ``transforms.json`` stores camera-to-world in the nerfstudio/OpenGL frame
    (x right, y up, z backwards); it is converted to the OpenCV camera-from-world
    convention used everywhere else here.
    """

    def __init__(self, scene_dir: str, *, max_size: int = 504, patch: int = 14,
                 max_frames: Optional[int] = None, transforms: Optional[str] = None,
                 depth_scale: float = 1e-3):
        self.scene_dir = scene_dir
        self.name = os.path.basename(scene_dir.rstrip("/"))
        self.depth_scale = depth_scale

        tpath = transforms or os.path.join(scene_dir, "dslr", "nerfstudio", "transforms.json")
        if not os.path.exists(tpath):
            raise FileNotFoundError(f"no transforms.json at {tpath}")
        with open(tpath) as f:
            meta = json.load(f)

        model = meta.get("camera_model", "OPENCV_FISHEYE")
        if model != "OPENCV_FISHEYE":
            warnings.warn(f"{self.name}: camera_model is {model!r}, not OPENCV_FISHEYE; "
                          f"treating it as Kannala-Brandt anyway", RuntimeWarning)

        self.orig_w, self.orig_h = int(meta["w"]), int(meta["h"])
        full = from_scannetpp(
            [meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"],
             meta.get("k1", 0.0), meta.get("k2", 0.0),
             meta.get("k3", 0.0), meta.get("k4", 0.0)],
            self.orig_w, self.orig_h,
        )

        frames = meta["frames"]
        frames = sorted(frames, key=lambda fr: fr["file_path"])
        if max_frames:
            frames = frames[:max_frames]
        self.frames = frames

        self.h, self.w = working_size(self.orig_h, self.orig_w, max_size=max_size, patch=patch)
        self.camera = full.resized(self.w, self.h)
        self._image_root = os.path.join(scene_dir, "dslr", "resized_images")
        if not os.path.isdir(self._image_root):
            self._image_root = os.path.join(scene_dir, "dslr", "undistorted_images")
        self._depth_root = os.path.join(scene_dir, "dslr", "render_depth")

    def __len__(self) -> int:
        return len(self.frames)

    def _read_rgb(self, path: str) -> Tensor:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        arr = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        return _resize_to(arr, (self.h, self.w))

    def image(self, i: int) -> Tensor:
        return self._read_rgb(os.path.join(self._image_root, self.frames[i]["file_path"]))

    def pose(self, i: int) -> Optional[Tuple[Tensor, Tensor]]:
        m = self.frames[i].get("transform_matrix")
        if m is None:
            return None
        c2w = torch.tensor(m, dtype=torch.float64)
        # nerfstudio/OpenGL -> OpenCV: negate the y and z camera axes.
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))
        c2w = c2w @ flip
        w2c = torch.linalg.inv(c2w)
        return w2c[:3, :3].float(), w2c[:3, 3].float()

    def depth(self, i: int) -> Optional[Tuple[Tensor, Tensor]]:
        stem = os.path.splitext(os.path.basename(self.frames[i]["file_path"]))[0]
        path = os.path.join(self._depth_root, stem + ".png")
        if not os.path.exists(path):
            return None
        from PIL import Image

        d = torch.from_numpy(np.asarray(Image.open(path)).astype(np.float32)) * self.depth_scale
        d = _resize_to(d[None], (self.h, self.w), mode="nearest")[0]
        return d, d > 1e-6


# ---------------------------------------------------------------------------
# ADT / Aria fisheye
# ---------------------------------------------------------------------------


class ADTSequence(SequenceSource):
    """One ADT sequence's raw Aria RGB fisheye stream.

    Frames are stored rotated 90 deg CW on Aria; a 270 deg CCW rotation is applied
    to RGB and depth alike, matching this repo's ADT convention, and the
    intrinsics come from ``aria_fisheye.aria_intrinsics(rotated=True)``.

    ADT ``depth_npy`` is **planar z** (CONTEXT.md); it is converted to euclidean
    range on load so it matches the ``"range"`` depth convention the losses and
    metrics default to. Pass ``depth_convention="z"`` to keep it as stored.
    """

    def __init__(self, seq_dir: str, *, max_size: int = 504, patch: int = 14,
                 max_frames: Optional[int] = None, rgb_subdir: str = "videos_rgb",
                 depth_subdir: str = "depth_npy", depth_scale: float = 1e-3,
                 rotation: int = 270, extrinsics_json: Optional[str] = None,
                 depth_convention: str = "range"):
        from cam3r.adt import load_trajectory, resolve_extrinsics, _frame_timestamp_us

        self.seq_dir = seq_dir
        self.name = os.path.basename(seq_dir.rstrip("/"))
        self.depth_scale = depth_scale
        self.rotation = rotation
        self.depth_convention = depth_convention

        rgb_dir = os.path.join(seq_dir, rgb_subdir)
        self.rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg"))
                                + glob.glob(os.path.join(rgb_dir, "*.png")))
        if max_frames:
            self.rgb_paths = self.rgb_paths[:max_frames]
        if not self.rgb_paths:
            raise FileNotFoundError(f"no frames under {rgb_dir}")
        self.depth_dir = os.path.join(seq_dir, depth_subdir)

        from PIL import Image

        probe = np.asarray(Image.open(self.rgb_paths[0]))
        oh, ow = probe.shape[:2]
        if rotation in (90, 270):
            oh, ow = ow, oh
        self.h, self.w = working_size(oh, ow, max_size=max_size, patch=patch)
        self.camera = from_aria(oh, ow, rotated=True).resized(self.w, self.h)

        # Poses: only usable with a real device->camera extrinsic (see module doc).
        self.gt_available = False
        self._poses = None
        traj = os.path.join(seq_dir, "groundtruth", "aria_trajectory.csv")
        if os.path.exists(traj):
            T_dc, source, exact = resolve_extrinsics(seq_dir, extrinsics_json)
            if not exact:
                warnings.warn(
                    f"{self.name}: no device->camera extrinsic ({source}); ADT pose and "
                    f"d_reproj metrics will be skipped. Pass --extrinsics-json to enable "
                    f"them. Depth metrics and adaptation are unaffected.", RuntimeWarning)
            else:
                ts, world_from_device = load_trajectory(traj)
                frame_ts = torch.tensor(
                    [_frame_timestamp_us(p) or float("nan") for p in self.rgb_paths],
                    dtype=torch.float64)
                idx = torch.searchsorted(ts, frame_ts.clamp(float(ts[0]), float(ts[-1])))
                idx = idx.clamp(0, len(ts) - 1)
                w_from_d = world_from_device[idx]
                w_from_c = w_from_d @ T_dc[None]
                c_from_w = torch.linalg.inv(w_from_c)
                self._poses = (c_from_w[:, :3, :3].float(), c_from_w[:, :3, 3].float())
                self.gt_available = True

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def _rotate(self, x: Tensor) -> Tensor:
        k = {0: 0, 90: 3, 180: 2, 270: 1}[self.rotation % 360]
        return torch.rot90(x, k, dims=(-2, -1)) if k else x

    def image(self, i: int) -> Tensor:
        from PIL import Image

        arr = torch.from_numpy(
            np.asarray(Image.open(self.rgb_paths[i]).convert("RGB"))
        ).permute(2, 0, 1).float() / 255.0
        return _resize_to(self._rotate(arr), (self.h, self.w))

    def pose(self, i: int) -> Optional[Tuple[Tensor, Tensor]]:
        if self._poses is None:
            return None
        return self._poses[0][i], self._poses[1][i]

    def depth(self, i: int) -> Optional[Tuple[Tensor, Tensor]]:
        stem = os.path.splitext(os.path.basename(self.rgb_paths[i]))[0]
        path = os.path.join(self.depth_dir, stem + ".npy")
        if not os.path.exists(path):
            return None
        d = torch.from_numpy(np.array(np.load(path), dtype=np.float32)) * self.depth_scale
        d = _resize_to(self._rotate(d)[None], (self.h, self.w), mode="nearest")[0]

        # Restrict to the imaged cone *before* converting: outside it the ray is
        # near-grazing, so 1/cos(theta) explodes and would manufacture depths of
        # thousands of metres from ordinary indoor GT.
        cone = self.camera.valid_mask(self.h, self.w, device=d.device)
        valid = (d > 1e-6) & cone
        if self.depth_convention == "range":
            # ADT stores planar z; the losses/metrics default to euclidean range.
            cos = self.camera.ray_grid(self.h, self.w)[..., 2]
            d = torch.where(valid, d / cos.clamp_min(1e-3), torch.zeros_like(d))
        else:
            d = torch.where(valid, d, torch.zeros_like(d))
        return d, valid


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------


def build_windows(source: SequenceSource, matcher: Matcher, *,
                  n_windows: int = 30, seq_len: int = 3, stride: int = 1,
                  min_flow_px: float = 2.0, with_pose_targets: bool = True,
                  with_gt: bool = False, max_scan: Optional[int] = None,
                  seed: int = 0, device="cpu", verbose: bool = True) -> List[Window]:
    """The paper's adaptation set: ``n_windows`` non-static ``seq_len``-frame windows.

    Windows are scanned in order and kept when the mean confident flow across
    consecutive frames reaches ``min_flow_px``. Correspondences are computed
    once here and reused for every training step, which is what makes the fit
    cheap -- the matcher runs ``O(windows)`` times, not once per iteration.
    """
    n = len(source)
    valid = source.camera.valid_mask(source.h, source.w, device=device)

    starts = list(range(0, max(n - (seq_len - 1) * stride, 0)))
    if max_scan:
        starts = starts[:max_scan]
    rng = np.random.RandomState(seed)
    rng.shuffle(starts)

    out: List[Window] = []
    n_static = 0
    for s in starts:
        if len(out) >= n_windows:
            break
        idx = [s + k * stride for k in range(seq_len)]
        if idx[-1] >= n:
            continue
        imgs = torch.stack([source.image(i) for i in idx]).to(device)

        pairs = [(i, j) for i in range(seq_len) for j in range(seq_len) if i != j]
        matches = {}
        for (i, j) in pairs:
            matches[(i, j)] = matcher(imgs[i], imgs[j], valid=valid)

        flow = float(np.mean([mean_flow_magnitude(matches[(k, k + 1)])
                              for k in range(seq_len - 1)]))
        if flow < min_flow_px:
            n_static += 1
            continue

        win = Window(images=imgs, indices=idx, camera=source.camera, matches=matches)

        if with_pose_targets:
            for (i, j) in pairs:
                win.pose_targets[(i, j)] = relative_pose_magsac(matches[(i, j)], source.camera)

        if with_gt:
            poses = [source.pose(i) for i in idx]
            if all(p is not None for p in poses):
                win.gt_R = torch.stack([p[0] for p in poses]).to(device)
                win.gt_t = torch.stack([p[1] for p in poses]).to(device)
            depths = [source.depth(i) for i in idx]
            if all(d is not None for d in depths):
                win.gt_depth = torch.stack([d[0] for d in depths]).to(device)
                win.gt_valid = torch.stack([d[1] for d in depths]).to(device)

        out.append(win)

    if verbose:
        n_targets = sum(1 for w in out for v in w.pose_targets.values() if v is not None)
        print(f"[data] {source.name}: {len(out)} windows kept, {n_static} dropped as static "
              f"(<{min_flow_px} px), {n_targets} MAGSAC++ pose targets")
    if not out:
        raise RuntimeError(
            f"{source.name}: no window exceeded {min_flow_px} px of flow. The paper's "
            f"limitation (v): with degenerate motion the self-supervised constraints are "
            f"too weak to adapt. Try a larger --stride or a different segment.")
    return out


def load_sequence(kind: str, path: str, **kw) -> SequenceSource:
    kinds = {"scannetpp": ScanNetPPFisheye, "adt": ADTSequence}
    if kind not in kinds:
        raise ValueError(f"unknown dataset {kind!r}; choose from {sorted(kinds)}")
    return kinds[kind](path, **kw)
