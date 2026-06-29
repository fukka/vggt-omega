# Copyright (c) 2026.
"""EgoExo4D fisheye frames warped into DAC's ERP canonical space for unsupervised
DAv2 finetuning.

The DAC novelty is that depth is reasoned in an **equirectangular (ERP) canonical
space**: every camera (here the raw Aria-style fisheye ego stream) is unwrapped to
the same angular grid, so the network sees the fisheye content the way DAC's
network does. This dataset turns each EgoExo4D ego frame into that ERP patch via
``finetune.eval.baselines.erp_utils.fisheye_to_erp_fwd`` (the validated,
cone-masked warp). There is no depth GT — the finetuning is unsupervised
(equivariance consistency + self-anchor), so we only emit the ERP RGB and its
in-FOV ``active`` mask.

Camera model (verified against the EgoExo4D sample)
---------------------------------------------------
The ego stream (``ego_GT.mp4``, 448×448) is a genuine **fisheye** (circular
vignette + barrel distortion, centred, already gravity-upright — so **no** 270°
rotation, unlike raw ADT). The dataset ships only a *canonical pinhole placeholder*
intrinsic in ``meta.json`` (f=150 @ 512, no distortion), **not** the true per-take
Aria calibration. We therefore default to the Aria-214-1 KB4 model with a *centred*
principal point (:func:`aria_centered`) and expose ``focal_scale`` to tune the FOV
if a better calibration is known. Because the objective is unsupervised and
scale-shift invariant, the exact focal is far less critical than for metric eval:
the consistency target only needs the *same* fisheye model applied consistently.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..eval.baselines.aria_fisheye import AriaFisheye, aria_centered
from ..eval.baselines.egoexo_data import find_egoexo_clips


def _enumerate_frames(egoexo_root: str, stream: str, frames_per_clip: int
                      ) -> List[Tuple[str, str, int]]:
    """Return ``(clip_name, video_path, frame_idx)`` evenly sampled per clip."""
    import cv2

    clips = find_egoexo_clips(egoexo_root, stream=stream)
    out: List[Tuple[str, str, int]] = []
    for clip, vp in clips:
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()
        if frames_per_clip <= 0 or frames_per_clip >= total:
            idxs = list(range(total))
        else:
            idxs = list(np.linspace(0, total - 1, frames_per_clip).round().astype(int))
        out.extend((clip, vp, int(i)) for i in idxs)
    return out


def _read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Decode a single RGB frame ``[H,W,3]`` uint8 via OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx} of {video_path!r}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def egoexo_camera(H: int, W: int, focal_scale: float = 1.0) -> AriaFisheye:
    """Centred Aria-KB4 fisheye for an EgoExo ego frame, focal optionally rescaled.

    ``focal_scale`` < 1 widens the modelled FOV (smaller focal), > 1 narrows it —
    a single knob to fit the (unknown) true ego FOV if the default looks off.
    """
    cam = aria_centered(H, W)
    if focal_scale != 1.0:
        cam = AriaFisheye(H=cam.H, W=cam.W, fx=cam.fx * focal_scale,
                          fy=cam.fy * focal_scale, cx=cam.cx, cy=cam.cy, k=cam.k)
    return cam


class ErpEgoExoDataset(Dataset):
    """EgoExo4D ego fisheye frames → DAC ERP patches (RGB + active mask).

    Each item is one frame warped to a single **canonical** ERP patch (centred on
    the optical axis). The viewpoint/scale *augmentation* that drives the
    equivariance loss is applied on-GPU in the trainer (a 2-D similarity on the ERP
    patch), so the dataset stays cheap and deterministic. ``input_scale_jitter``
    optionally adds DAC ``scale_fac`` (FOV) jitter at warp time for extra input
    variety.
    """

    def __init__(
        self,
        egoexo_root: str,
        stream: str = "ego_GT",
        frames_per_clip: int = 16,
        erp_cano: int = 1400,
        erp_fwd_sz: Tuple[int, int] = (500, 750),
        crop_wfov: float = 180.0,
        focal_scale: float = 1.0,
        input_scale_jitter: float = 0.0,
        max_incidence_deg: Optional[float] = None,
        seed: int = 0,
    ) -> None:
        self.frames = _enumerate_frames(egoexo_root, stream, frames_per_clip)
        if not self.frames:
            raise FileNotFoundError(
                f"no {stream} frames under {egoexo_root!r} (expected videos/<clip>/{stream}.mp4)")
        self.erp_cano = int(erp_cano)
        self.erp_fwd_sz = (int(erp_fwd_sz[0]), int(erp_fwd_sz[1]))
        self.crop_wfov = float(crop_wfov)
        self.focal_scale = float(focal_scale)
        self.input_scale_jitter = float(input_scale_jitter)
        self.max_incidence_deg = max_incidence_deg
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        from ..eval.baselines.erp_utils import fisheye_to_erp_fwd

        clip, vp, fi = self.frames[idx]
        rgb = _read_frame(vp, fi)                       # [H,W,3] uint8
        H, W = rgb.shape[:2]
        cam = egoexo_camera(H, W, self.focal_scale)
        img01 = rgb.astype(np.float32) / 255.0
        zeros = np.zeros((H, W), np.float32)

        scale_fac = None
        if self.input_scale_jitter > 0:
            j = self.input_scale_jitter
            scale_fac = float(self._rng.uniform(1.0 - j, 1.0 + j))

        warp = fisheye_to_erp_fwd(
            img01, zeros, zeros, cam.opencv_fisheye_params(),
            self.erp_cano, self.erp_fwd_sz, self.crop_wfov,
            max_incidence_deg=self.max_incidence_deg, scale_fac=scale_fac,
        )
        erp = torch.from_numpy(warp["image_u8"]).permute(2, 0, 1).float() / 255.0  # [3,h,w]
        active = torch.from_numpy(np.asarray(warp["active"], np.float32))           # [h,w]
        return {"erp": erp, "active": active, "clip": clip, "frame": fi}


def collate_erp(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Stack ERP tensors into ``[B,1,3,H,W]`` / ``[B,1,H,W]`` (seq dim = 1)."""
    erp = torch.stack([b["erp"] for b in batch], 0).unsqueeze(1)       # [B,1,3,H,W]
    active = torch.stack([b["active"] for b in batch], 0).unsqueeze(1)  # [B,1,H,W]
    return {"images": erp, "active": active,
            "clip": [b["clip"] for b in batch], "frame": [b["frame"] for b in batch]}


class DummyErpDataset(Dataset):
    """Random ERP-shaped tensors for offline/CPU smoke runs (no DAC, no data)."""

    def __init__(self, n: int = 64, fwd_sz: Tuple[int, int] = (70, 98), seed: int = 0) -> None:
        self.n = n
        self.h, self.w = fwd_sz
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, object]:
        g = torch.Generator().manual_seed(self.seed + idx)
        erp = torch.rand(3, self.h, self.w, generator=g)
        active = (torch.rand(self.h, self.w, generator=g) > 0.2).float()
        return {"erp": erp, "active": active, "clip": "dummy", "frame": idx}
