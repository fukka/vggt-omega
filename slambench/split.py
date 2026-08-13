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

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from slambench import _REPO  # noqa: F401  (import registers sys.path)

from finetune.eval import manifest as _manifest  # noqa: E402
from slambench.data import DATASETS, find_takes  # noqa: E402

#: Bumped whenever a change would make new numbers incomparable to old ones.
#: Distinct from anything the FOV experiment uses — different ground truth,
#: different protocol, different question.
PROTOCOL = "egosynth-slam-v1"


@dataclass
class Frame:
    """One scored frame: its point ground truth and the clip it comes from.

    ``clip_frames`` is how many frames that clip's mp4 holds, read from
    ``meta.clips[].num_frames`` at build time. It is here for the multi-frame
    context (see :func:`context_window`), which needs to know where the clip ends
    before it can place a window inside it. A manifest written before context
    support carries 0, and the runner refuses a context run on one rather than
    guessing the length from the frames it happens to have — the split holds 25
    of a clip's 121 frames, so guessing would silently shrink every window.
    """

    dataset: str
    take: str
    clip: str
    index: int
    npz: str
    video: str
    clip_frames: int = 0

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
        """The comparability token. The rule is shared with the FOV experiment
        (``finetune/eval/manifest.py``); ``PROTOCOL`` is not, and that is what
        keeps a digest from one experiment out of the other's namespace."""
        return _manifest.digest(self.protocol, (f.key for f in self.frames))

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
        """``(dataset, take, clip, npz, video, [Frame])``, one entry per clip.

        Grouped so each mp4 is opened and decoded exactly once. This buys decode
        cost, not sampling — the split already chose the frames. The frames
        themselves rather than their indices, because a context run needs each
        one's clip length to place its window.
        """
        order: List[list] = []
        seen: Dict[tuple, int] = {}
        for f in self.frames:
            tag = (f.dataset, f.take, f.clip)
            if tag not in seen:
                seen[tag] = len(order)
                order.append([f.dataset, f.take, f.clip, f.npz, f.video, {}])
            order[seen[tag]][5].setdefault(f.index, f)
        return [(a, b, c, d, e, [fr[i] for i in sorted(fr)])
                for a, b, c, d, e, fr in order]

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


def context_window(clip_frames: int, index: int, n: int,
                   stride: int = 1) -> Tuple[List[int], int]:
    """The ``n`` frames to hand a multi-view model when scoring ``index``.

    Returns ``(indices, target_index)`` — the frames in temporal order, and where
    ``index`` sits among them. **Only ``index`` is ever scored**; the rest are
    company, so a 1-frame and a 10-frame run measure the identical points and the
    only thing that moves is the evidence the model had.

    The window **precedes** the target, ``index - (n-1)*stride ... index``, which
    is what a live camera would have: the frames before the one being asked
    about. Running off the start of the clip shifts the block forward rather than
    clamping, because clamping repeats a frame and a repeated frame is not
    evidence — it is the same view twice, which a multi-view model may even read
    as a stationary camera.

    The window stays **inside one clip**. A clip is its own mp4, so crossing the
    boundary would mean a second decode, and clips are not contiguous in the
    source recording anyway (their ids are start indices, sometimes far apart).

    Unlike the FOV experiment, this is computed at run time rather than baked
    into the split. There the context was one setting per run; here the sweep
    across 1/3/5/10 frames **is** the experiment, so every arm of it has to share
    one frozen frame list. That also means ``--context-frames`` cannot become a
    silent no-op when a manifest is reused: there is no stored window to override
    it.
    """
    n, stride = max(1, int(n)), max(1, int(stride))
    i, pool = int(index), int(clip_frames)
    if n <= 1 or pool <= 1:
        return ([i], 0)
    idx = [max(0, i - (n - 1 - k) * stride) for k in range(n)]
    if idx[0] < 0 or idx[0] != i - (n - 1) * stride:      # ran off the start
        idx = [k * stride for k in range(n)]              # ... shift the block
    if idx[-1] > pool - 1:                                # ... and off the end
        idx = [max(0, pool - 1 - (n - 1 - k) * stride) for k in range(n)]
    idx = sorted(set(i_ for i_ in idx if 0 <= i_ <= pool - 1))
    if i not in idx:              # the shift can walk past the target; it stays
        idx = sorted(set(idx[:-1] + [i]))
    return (idx, idx.index(i))


#: Shared with the FOV experiment — see ``finetune/eval/manifest.py``.
_evenly_spaced = _manifest.evenly_spaced


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
                                index=i, npz=t.npz(clip), video=t.video(clip),
                                clip_frames=t.frames_in(clip)))
    if not frames:
        raise SystemExit(f"[slambench] no clip under {egosynth_root!r} has a "
                         f"frame table; check meta.json's clips[]")
    sp = Split(root=os.path.abspath(egosynth_root), frames=frames,
               takes_per_dataset=takes_per_dataset, n_frames_per_take=n_frames)
    if verbose:
        print(f"  [slambench] split {sp.digest}: {len(frames)} frames over "
              f"{len(sp.takes)} take(s) of {len(sp.datasets)} dataset(s)")
    return sp
