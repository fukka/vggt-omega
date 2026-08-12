# Copyright (c) 2026.
"""The frozen frame list — a reproducibility contract, not a query.

Two runs are comparable exactly when their digests match. The digest covers the
ordered frame keys *and* the protocol token, so a subsampled run can never be
compared against a fuller one by accident, and a run from the FOV experiment
cannot be mistaken for one of these.

Frames are spread across each take's clips rather than taken from one: a clip is
121 consecutive frames at 20 fps, about six seconds of one viewpoint, so any
single clip understates a take's variety.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from slambench import _REPO  # noqa: F401  (import registers sys.path)

from slambench.data import DATASETS, find_takes  # noqa: E402

#: Bumped whenever a change would make new numbers incomparable to old ones.
#: Distinct from anything the FOV experiment uses — different ground truth,
#: different protocol, different question.
PROTOCOL = "egosynth-slam-v1"


@dataclass
class Frame:
    """One scored frame: its point ground truth and the clip it comes from."""

    dataset: str
    take: str
    clip: str
    index: int
    npz: str
    video: str

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.take}/{self.clip}:{self.index}"


@dataclass
class Split:
    root: str
    frames: List[Frame]
    protocol: str = PROTOCOL
    takes_per_dataset: int = 0
    n_frames_per_take: int = 0

    @property
    def digest(self) -> str:
        h = hashlib.sha1(self.protocol.encode())
        for f in self.frames:
            h.update(f.key.encode())
        return h.hexdigest()[:12]

    @property
    def datasets(self) -> List[str]:
        out: List[str] = []
        for f in self.frames:
            if f.dataset not in out:
                out.append(f.dataset)
        return out

    @property
    def takes(self) -> List[str]:
        out: List[str] = []
        for f in self.frames:
            k = f"{f.dataset}/{f.take}"
            if k not in out:
                out.append(k)
        return out

    def by_clip(self) -> List[tuple]:
        """``(dataset, take, clip, npz, video, [frame indices])``, one per clip.

        Grouped so each mp4 is opened and decoded exactly once. This buys decode
        cost, not sampling — the split already chose the frames.
        """
        order: List[list] = []
        seen: Dict[tuple, int] = {}
        for f in self.frames:
            tag = (f.dataset, f.take, f.clip)
            if tag not in seen:
                seen[tag] = len(order)
                order.append([f.dataset, f.take, f.clip, f.npz, f.video, []])
            order[seen[tag]][5].append(f.index)
        return [(a, b, c, d, e, sorted(set(i))) for a, b, c, d, e, i in order]

    def to_dict(self) -> dict:
        return {"protocol": self.protocol, "root": self.root,
                "digest": self.digest, "n_frames": len(self.frames),
                "takes_per_dataset": self.takes_per_dataset,
                "n_frames_per_take": self.n_frames_per_take,
                "frames": [f.__dict__ for f in self.frames]}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Split":
        with open(path) as fh:
            d = json.load(fh)
        sp = cls(root=d["root"], frames=[Frame(**f) for f in d["frames"]],
                 protocol=d["protocol"],
                 takes_per_dataset=d.get("takes_per_dataset", 0),
                 n_frames_per_take=d.get("n_frames_per_take", 0))
        if sp.digest != d["digest"]:
            raise ValueError(
                f"{path}: digest {d['digest']} does not match the frame list "
                f"({sp.digest}). The manifest was edited by hand or written by a "
                f"different protocol version; results keyed to it are not "
                f"comparable.")
        return sp


def _evenly_spaced(items: Sequence, n: int) -> List:
    """``n`` items spread across ``items``, endpoints included, order kept."""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / float(n - 1)
    return [items[i] for i in sorted({int(round(i * step)) for i in range(n)})]


def build(egosynth_root: str, datasets: Optional[Sequence[str]] = None,
          n_frames: int = 25, takes_per_dataset: int = 8,
          verbose: bool = True) -> Split:
    """Freeze a frame list over the ego-synth 5B release.

    ``n_frames`` is per **take**, spread over all of that take's (clip, frame)
    pairs. ``takes_per_dataset`` caps each dataset's contribution — the release
    is 1 611 takes and 24 931 clips, which no single run should quietly attempt.
    Both caps enter the digest, so a capped run is never silently compared with
    an uncapped one.
    """
    datasets = tuple(datasets or DATASETS)
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"[slambench] unknown dataset(s) {unknown}; "
                         f"choose from {list(DATASETS)}")
    takes = find_takes(egosynth_root, datasets, limit=takes_per_dataset,
                       verbose=verbose)
    frames: List[Frame] = []
    for t in takes:
        pairs = [(c, i) for c in t.clips for i in range(t.frames_in(c))]
        for clip, i in _evenly_spaced(pairs, n_frames):
            frames.append(Frame(dataset=t.dataset, take=t.name, clip=clip,
                                index=i, npz=t.npz(clip), video=t.video(clip)))
    if not frames:
        raise SystemExit(f"[slambench] no clip under {egosynth_root!r} has a "
                         f"frame table; check meta.json's clips[]")
    sp = Split(root=os.path.abspath(egosynth_root), frames=frames,
               takes_per_dataset=takes_per_dataset, n_frames_per_take=n_frames)
    if verbose:
        print(f"  [slambench] split {sp.digest}: {len(frames)} frames over "
              f"{len(sp.takes)} take(s) of {len(sp.datasets)} dataset(s)")
    return sp
