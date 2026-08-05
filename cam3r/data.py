# Copyright (c) 2026.
"""Dataset contract, two-phase curriculum, and panorama -> lens synthesis.

CAM3R trains on four corpora -- 2D3DS and 360Loc (panoramic), ADT (fisheye) and
MegaDepth (pinhole) -- under a curriculum that runs homogeneous pairs first and
heterogeneous (cross-lens) pairs second.  The paper's ablation makes that second
phase load-bearing: homogeneous-only training scores 65.4 RRA@15 on 2D3DS
against 97.7 for the full recipe.

Only ADT is present in this repo, so this module defines the *interface* the
other three plug into (:class:`TwoViewSource`) plus the machinery that is
dataset-independent:

* :class:`CurriculumSampler` -- phase 1 draws both views from one lens family,
  phase 2 deliberately mixes them.
* :func:`synthesize_view` -- renders a pinhole or fisheye view out of an
  equirectangular panorama, which is how the paper turns 2D3DS/360Loc into
  fisheye and perspective training data ("synthetic fisheye and perspective
  views ... via equidistant projection").

A sample is a plain dict, matching what :class:`cam3r.adt.ADTPairDataset`
returns; see :class:`TwoViewSource` for the required keys.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from cam3r.cameras import Camera, Equirect

SAMPLE_KEYS = ("images", "rays", "points", "valid", "R", "t")


@dataclass
class TwoViewSource:
    """A dataset plus the lens family it produces.

    ``dataset`` must be indexable and return dicts with:

    ==========  ===========================================================
    ``images``  list[2] of (3, H, W) float32 in [0, 1]
    ``rays``    list[2] of (3, H, W) unit GT ray directions
    ``points``  list[2] of (3, H, W) GT pointmaps (``radial_distance * ray``)
    ``valid``   list[2] of (H, W) bool
    ``R``       (3, 3) GT rotation mapping view 2 into view 1
    ``t``       (3,) GT translation, same direction convention as ``R``
    ==========  ===========================================================

    ``kind`` is the lens family (``"pinhole"``, ``"kb4"``, ``"erp"``); the
    curriculum groups by it.  ``rays`` or ``points`` may be ``None`` for a
    corpus that lacks that supervision -- :func:`cam3r.losses.cam3r_loss` drops
    the corresponding term.

    ``supports_heterogeneous`` declares that the dataset can render *the same
    scene* through two different lenses, i.e. that ``dataset[i, True]`` (or
    ``dataset.get(i, heterogeneous=True)``) returns a cross-lens pair.  This is
    what the paper does -- "simultaneous fisheye and pinhole renders per frame"
    -- and it is the only way a heterogeneous pair has a defined relative pose.
    Combining two *independent* datasets would not: two unrelated scenes share
    no geometry to regress.
    """

    dataset: object
    kind: str
    name: str = ""
    supports_heterogeneous: bool = False

    def __len__(self) -> int:
        return len(self.dataset)          # type: ignore[arg-type]


@dataclass(frozen=True)
class Draw:
    """One curriculum draw: which source, which item, and which pairing mode."""

    source: int
    index: int
    heterogeneous: bool = False


class CurriculumSampler:
    """Two-phase draw stream over several :class:`TwoViewSource` s.

    ``phase=1`` draws only homogeneous (single-lens) items.  ``phase=2`` asks
    for a cross-lens pair with probability ``hetero_ratio``, but only from
    sources that declare :attr:`TwoViewSource.supports_heterogeneous` -- a
    cross-lens pair has to come from one scene rendered two ways, not from two
    unrelated datasets.
    """

    def __init__(
        self,
        sources: Sequence[TwoViewSource],
        phase: int = 1,
        hetero_ratio: float = 0.5,
        seed: int = 0,
    ) -> None:
        if not sources:
            raise ValueError("need at least one source")
        if phase not in (1, 2):
            raise ValueError(f"phase must be 1 or 2 (got {phase})")
        self.sources = list(sources)
        self.phase = phase
        self.hetero_ratio = hetero_ratio
        self.rng = random.Random(seed)

        self.by_kind: Dict[str, List[int]] = {}
        for idx, s in enumerate(self.sources):
            self.by_kind.setdefault(s.kind, []).append(idx)
        self._hetero_capable = [i for i, s in enumerate(self.sources) if s.supports_heterogeneous]

    def can_do_heterogeneous(self) -> bool:
        return self.phase == 2 and bool(self._hetero_capable)

    def __len__(self) -> int:
        return sum(len(s) for s in self.sources)

    def sample(self) -> Draw:
        """One draw."""
        if self.can_do_heterogeneous() and self.rng.random() < self.hetero_ratio:
            si = self.rng.choice(self._hetero_capable)
            return Draw(si, self._index_in(si), heterogeneous=True)
        kind = self.rng.choice(list(self.by_kind))
        si = self.rng.choice(self.by_kind[kind])
        return Draw(si, self._index_in(si), heterogeneous=False)

    def _index_in(self, si: int) -> int:
        n = len(self.sources[si])
        if n == 0:
            raise RuntimeError(f"source {self.sources[si].name or si!r} is empty")
        return self.rng.randrange(n)

    def __iter__(self) -> Iterator[Draw]:
        for _ in range(len(self)):
            yield self.sample()


# --------------------------------------------------------------------------- #
# Panorama -> arbitrary lens
# --------------------------------------------------------------------------- #

def look_at_rotation(azimuth: float, elevation: float) -> torch.Tensor:
    """Camera-from-world rotation looking along ``(azimuth, elevation)`` radians."""
    ca, sa = math.cos(azimuth), math.sin(azimuth)
    ce, se = math.cos(elevation), math.sin(elevation)
    forward = torch.tensor([ce * sa, -se, ce * ca], dtype=torch.float64)
    forward = forward / forward.norm()
    world_up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)
    right = torch.cross(world_up, forward, dim=0)
    if float(right.norm()) < 1e-6:                # looking straight up/down
        right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    right = right / right.norm()
    down = torch.cross(forward, right, dim=0)
    return torch.stack([right, down, forward], dim=0)     # rows = camera axes


def synthesize_view(
    panorama: torch.Tensor,
    camera: Camera,
    out_hw: Tuple[int, int],
    rotation: Optional[torch.Tensor] = None,
    depth: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Render a ``camera`` view out of an equirectangular ``panorama``.

    ``panorama`` is (3, Hp, Wp) in [0, 1]; ``depth`` an optional (Hp, Wp) of
    euclidean range.  ``rotation`` (3, 3) is camera-from-panorama; identity
    looks along the panorama centre.

    Returns ``image``, ``rays`` (in the synthesized camera's own frame),
    ``valid``, and ``points``/``range`` when ``depth`` was supplied.  This is
    how a panoramic corpus becomes fisheye or perspective training data.
    """
    H, W = out_hw
    rays, cone = camera.ray_field(H, W)
    R = torch.eye(3, dtype=torch.float64) if rotation is None else rotation.double()

    # Camera ray -> panorama direction -> ERP pixel.
    world_dirs = rays.double().reshape(-1, 3) @ R
    Hp, Wp = panorama.shape[-2:]
    uv = Equirect().project(world_dirs, H=Hp, W=Wp).reshape(H, W, 2)

    grid = torch.stack([uv[..., 0] / Wp * 2 - 1, uv[..., 1] / Hp * 2 - 1], dim=-1)
    grid = grid.unsqueeze(0).to(panorama.dtype)
    image = F.grid_sample(
        panorama.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
    )[0]

    out: Dict[str, torch.Tensor] = {
        "image": image.clamp(0.0, 1.0),
        "rays": rays.permute(2, 0, 1).contiguous(),
        "valid": cone,
    }
    if depth is not None:
        rng = F.grid_sample(
            depth.view(1, 1, Hp, Wp), grid, mode="nearest", padding_mode="border", align_corners=False
        )[0, 0]
        out["range"] = rng
        out["points"] = out["rays"] * rng.unsqueeze(0)
        out["valid"] = cone & (rng > 0) & torch.isfinite(rng)
    return out


def random_camera_for(kind: str, H: int, W: int, rng: Optional[random.Random] = None) -> Camera:
    """A randomly parameterized camera of the requested family, for augmentation."""
    rng = rng or random.Random()
    if kind == "pinhole":
        return _random_pinhole(H, W, rng)
    if kind == "kb4":
        return _random_kb4(H, W, rng)
    if kind == "erp":
        return Equirect()
    raise ValueError(f"unknown camera kind {kind!r}")


def _random_pinhole(H: int, W: int, rng: random.Random) -> Camera:
    from cam3r.cameras import Pinhole

    return Pinhole.from_fov(H, W, rng.uniform(50.0, 100.0))


def _random_kb4(H: int, W: int, rng: random.Random) -> Camera:
    """Equidistant-ish fisheye with mild randomized distortion.

    Base model is equidistant (``theta_d = f * theta``, i.e. all ``k`` zero),
    which is the projection the paper names for synthesizing fisheye views; the
    small random ``k1``/``k2`` widen the distribution the Ray Module sees.
    """
    from cam3r.cameras import KB4

    fov = math.radians(rng.uniform(140.0, 200.0))
    f = (min(H, W) / 2.0) / (fov / 2.0)
    return KB4(f, f, (W - 1) / 2.0, (H - 1) / 2.0,
               (rng.uniform(-0.05, 0.05), rng.uniform(-0.02, 0.02), 0.0, 0.0))
