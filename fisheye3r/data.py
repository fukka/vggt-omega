"""Sequence datasets for Fisheye3R training/evaluation.

Expected layout (one directory per scene):

    root/
      scene_x/
        rgb/    000000.jpg|png ...
        depth/  000000.npy            (optional; meters; <=0 or NaN = invalid)
        cameras.npz                   (optional; intrinsics (N,3,3),
                                       extrinsics (N,4,4) camera-from-world)

Simplifications vs. the paper (supp Sec. 6, documented in README):
  - frame selection uses a random window with random stride instead of the
    precomputed pairwise-covisibility matrix (>=25% covisible pixels);
  - photometric augmentations: color jitter / grayscale / gaussian blur
    (the paper additionally uses random resizing and cropping).
Training samples 2-24 frames per sequence; evaluation uses fixed lengths
{4, 8, 16, 32} (supp "Sequence sampling").
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _list_scene(scene: Path) -> list[Path]:
    rgb = scene / "rgb"
    frames = sorted(p for p in rgb.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not frames:
        raise FileNotFoundError(f"no images under {rgb}")
    return frames


def _load_image(path: Path) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _resize_to(img: torch.Tensor, size_hw: tuple[int, int], mode: str) -> torch.Tensor:
    return F.interpolate(img.unsqueeze(0), size=size_hw, mode=mode).squeeze(0)


class SequenceDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        resolution: int = 512,
        patch_size: int = 16,
        min_frames: int = 2,
        max_frames: int = 24,
        eval_length: int = 8,
        require_depth: bool = False,
        augment: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.resolution = resolution
        self.patch_size = patch_size
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.eval_length = eval_length
        self.require_depth = require_depth
        self.augment = augment and split == "train"

        self.scenes = sorted(p for p in self.root.iterdir() if (p / "rgb").is_dir())
        if not self.scenes:
            raise FileNotFoundError(f"no scenes with an rgb/ directory under {root}")
        self.frames = {s: _list_scene(s) for s in self.scenes}

        if split == "train":
            self._index = list(range(len(self.scenes) * 32))  # virtual epoch
        else:
            self._index = []
            for si, s in enumerate(self.scenes):
                n = len(self.frames[s])
                for start in range(0, max(n - eval_length + 1, 1), eval_length):
                    self._index.append((si, start))

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------- internals

    def _target_shape(self, h: int, w: int) -> tuple[int, int]:
        scale = self.resolution / max(h, w)
        th = max(int(round(h * scale / self.patch_size)) * self.patch_size, self.patch_size)
        tw = max(int(round(w * scale / self.patch_size)) * self.patch_size, self.patch_size)
        return th, tw

    def _pick_frames(self, scene: Path) -> list[int]:
        n = len(self.frames[scene])
        length = random.randint(self.min_frames, min(self.max_frames, n))
        stride = random.randint(1, max((n - 1) // max(length - 1, 1), 1))
        start = random.randint(0, max(n - (length - 1) * stride - 1, 0))
        return [min(start + i * stride, n - 1) for i in range(length)]

    def _photometric_aug(self, images: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.8:
            b, c, s = (1 + 0.3 * (random.random() - 0.5) for _ in range(3))
            mean = images.mean(dim=(-2, -1), keepdim=True)
            images = ((images - mean) * c + mean) * b
            gray = images.mean(dim=-3, keepdim=True)
            images = (images - gray) * s + gray
        if random.random() < 0.1:
            images = images.mean(dim=-3, keepdim=True).expand_as(images).contiguous()
        if random.random() < 0.2:
            k = 2 * random.randint(1, 3) + 1
            pad = k // 2
            kernel = torch.ones(1, 1, k, k) / (k * k)
            blurred = F.conv2d(
                images.reshape(-1, 1, *images.shape[-2:]), kernel, padding=pad
            ).reshape(images.shape)
            images = blurred
        return images.clamp(0, 1)

    # --------------------------------------------------------------- getitem

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.split == "train":
            scene = self.scenes[idx % len(self.scenes)]
            frame_ids = self._pick_frames(scene)
        else:
            si, start = self._index[idx]
            scene = self.scenes[si]
            n = len(self.frames[scene])
            frame_ids = [min(start + i, n - 1) for i in range(self.eval_length)]

        paths = [self.frames[scene][i] for i in frame_ids]
        images = [_load_image(p) for p in paths]
        h, w = images[0].shape[-2:]
        th, tw = self._target_shape(h, w)
        images = torch.stack([_resize_to(im, (th, tw), "bilinear") for im in images])
        if self.augment:
            images = self._photometric_aug(images)

        out: dict[str, torch.Tensor] = {
            "images": images,
            "frame_ids": torch.tensor(frame_ids),
            "scene": str(scene.name),
        }

        depth_dir = scene / "depth"
        if depth_dir.is_dir():
            depths = []
            for p in paths:
                d = np.load(depth_dir / (p.stem + ".npy"))
                d = torch.from_numpy(d.astype(np.float32)).unsqueeze(0)
                depths.append(_resize_to(d, (th, tw), "nearest").squeeze(0))
            out["depth"] = torch.stack(depths)
        elif self.require_depth:
            raise FileNotFoundError(f"{scene} has no depth/ but require_depth=True")

        cam_file = scene / "cameras.npz"
        if cam_file.exists():
            cams = np.load(cam_file)
            K = torch.from_numpy(cams["intrinsics"][frame_ids].astype(np.float32)).clone()
            K[:, 0] *= tw / w
            K[:, 1] *= th / h
            out["intrinsics"] = K
            out["extrinsics"] = torch.from_numpy(cams["extrinsics"][frame_ids].astype(np.float32))
        return out


def sequence_collate(batch: list[dict]) -> dict:
    """Sequences have variable length/size; train with batch_size=1 sequence
    per step (the paper packs 1-12 sequences / <=24 frames per GPU instead)."""
    assert len(batch) == 1, "use batch_size=1; increase --accum for larger effective batches"
    item = batch[0]
    return {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in item.items()}
