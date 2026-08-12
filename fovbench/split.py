# Copyright (c) 2026.
"""The **ADT-FOV test** — a frozen frame list, not a query.

ADT ships two RGB streams per sequence beside one depth ground truth:

    ``videos_synthetic``   the *synthetic* input: re-rendered from the digital
                           twin, sharp and pixel-registered to ``depth_npy``.
    ``videos_rgb``         the *real* input: the actual Aria sensor frames, with
                           real motion blur, rolling shutter and photometry, and
                           only as well registered to the GT as the twin's pose.

The benchmark reports both, because "how does distortion hurt depth" has a
different answer when the pixels also carry sensor degradation — and because
quoting only the synthetic stream would overstate every model.

That comparison is only worth reading if the two streams are scored on the
**same frames**, so the split is defined once, up front:

1. keep sequences that have ``depth_npy`` *and both* RGB streams, non-empty
   (``find_adt_sequences(require_subdirs=...)``);
2. within a sequence keep the frame ids present in all three
   (``common_frame_ids``);
3. take ``n_frames`` of them spread evenly across the sequence, in frame-id
   order.

Step 3 is a stride, not a prefix: consecutive ADT frames are near-duplicates at
30 Hz, so the first N frames of a sequence are one instant of one viewpoint, and
their spread understates the sequence's.

The result is written next to the run as ``manifest.json`` with a ``digest``
over the frame keys. Two runs comparable only if their digests match — that is
what the digest is for, and the report prints it.

The split is deliberately *not* a train/test split: every model here is
evaluated vanilla, off-the-shelf, so there is nothing to hold out from. It is a
reproducibility contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

from fovbench import _REPO  # noqa: F401  (import registers sys.path)

from datasets.adt import (_pair_frames, common_frame_ids,  # noqa: E402
                          find_adt_sequences)

#: stream label -> ADT subdirectory. The labels are what the report prints.
STREAMS: Dict[str, str] = {"synthetic": "videos_synthetic", "real": "videos_rgb"}

#: Bumped whenever a change would make new numbers incomparable to old ones.
PROTOCOL = "adt-fov-v1"


@dataclass
class Frame:
    """One scored frame: a GT depth map and the same instant in both streams.

    ``context`` is the *unscored* company it keeps. A multi-view model can be
    handed several frames in one forward pass; only this frame is scored, and
    the others exist so cross-view attention has something to attend to. The
    list is in temporal order and includes this frame at ``target_index``, so a
    1-frame run is the same code path with a list of one.

    Context deliberately does **not** enter :attr:`Split.digest`. The digest
    exists to say "these runs scored the same pixels", and a 1-frame and a
    10-frame run do score the same pixels — that is the whole point of the
    comparison. Folding context in would give them different digests and make
    the harness refuse the only comparison worth making.
    """

    seq: str
    frame_id: str
    depth: str
    rgb: Dict[str, str]          # stream label -> image path
    context: Dict[str, List[str]] = field(default_factory=dict)
    target_index: int = 0

    @property
    def key(self) -> str:
        return f"{self.seq}/{self.frame_id}"

    def stack(self, stream: str) -> List[str]:
        """RGB paths to hand the model, in order. Falls back to this frame."""
        return self.context.get(stream) or [self.rgb[stream]]


@dataclass
class Split:
    root: str
    frames: List[Frame]
    streams: Dict[str, str] = field(default_factory=lambda: dict(STREAMS))
    protocol: str = PROTOCOL
    context_frames: int = 1
    context_stride: int = 1

    @property
    def digest(self) -> str:
        """Short hash over the ordered frame keys — the comparability token."""
        h = hashlib.sha1(self.protocol.encode())
        for f in self.frames:
            h.update(f.key.encode())
        return h.hexdigest()[:12]

    @property
    def sequences(self) -> List[str]:
        out: List[str] = []
        for f in self.frames:
            if f.seq not in out:
                out.append(f.seq)
        return out

    def to_dict(self) -> dict:
        return {"protocol": self.protocol, "root": self.root,
                "streams": self.streams, "digest": self.digest,
                "n_frames": len(self.frames),
                "context_frames": self.context_frames,
                "context_stride": self.context_stride,
                "frames": [{"seq": f.seq, "frame_id": f.frame_id,
                            "depth": f.depth, "rgb": f.rgb,
                            "context": f.context,
                            "target_index": f.target_index}
                           for f in self.frames]}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Split":
        with open(path) as fh:
            d = json.load(fh)
        sp = cls(root=d["root"],
                 frames=[Frame(seq=f["seq"], frame_id=f["frame_id"],
                               depth=f["depth"], rgb=f["rgb"],
                               context=f.get("context", {}),
                               target_index=int(f.get("target_index", 0)))
                         for f in d["frames"]],
                 streams=d["streams"], protocol=d["protocol"],
                 context_frames=int(d.get("context_frames", 1)),
                 context_stride=int(d.get("context_stride", 1)))
        if sp.digest != d["digest"]:
            raise ValueError(
                f"{path}: digest {d['digest']} does not match the frame list "
                f"({sp.digest}). The manifest has been edited by hand or written "
                f"by a different protocol version; results keyed to it are not "
                f"comparable.")
        return sp


def _evenly_spaced(items: List, n: int) -> List:
    """``n`` items spread across ``items``, endpoints included, order kept."""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / float(n - 1)
    idx = sorted({int(round(i * step)) for i in range(n)})
    return [items[i] for i in idx]


def _context_window(n_pool: int, i: int, n: int, stride: int) -> tuple:
    """Indices of the ``n`` context frames around pool position ``i``.

    The window **precedes** the target, ``i - (n-1)*stride ... i``, which is what
    a live camera would have: the frames before the one being asked about. If it
    would run off the start of the sequence it is shifted forward as a block
    rather than clamped, because clamping would feed the model the same frame
    several times and a repeated frame is not evidence.

    Returns ``(indices, target_index)`` — the target is last except where the
    shift moved it, and the caller must use the returned index rather than -1.
    """
    if n <= 1 or n_pool <= 1:
        return ([i], 0)
    span = (n - 1) * stride
    start = i - span
    if start < 0:                      # shift the block forward, do not repeat
        start = 0
    idx = [start + k * stride for k in range(n)]
    if idx[-1] >= n_pool:              # ... and back off the end the same way
        idx = [max(0, n_pool - 1 - (n - 1 - k) * stride) for k in range(n)]
    idx = sorted(set(idx))
    if i not in idx:                   # the shift can walk past the target
        idx = sorted(set(idx[:-1] + [i]))
    return (idx, idx.index(i))


def build_split(adt_root: str, n_frames: int = 25,
                streams: Dict[str, str] = None,
                context_frames: int = 1, context_stride: int = 1,
                verbose: bool = True) -> Split:
    """Assemble the ADT-FOV test split under ``adt_root``.

    ``n_frames`` is **per sequence**, not in total: sequences differ in length,
    and a global cap would silently weight the long ones. The report always
    prints the resulting per-sequence counts.
    """
    streams = dict(streams or STREAMS)
    subdirs = list(streams.values())
    seq_dirs = find_adt_sequences(adt_root, rgb_subdir=subdirs[0],
                                  require_subdirs=subdirs[1:], verbose=verbose)
    if not seq_dirs:
        raise SystemExit(
            f"[fovbench] no ADT sequence under {adt_root!r} has depth_npy plus "
            f"all of {subdirs}. The synthetic-vs-real comparison needs both "
            f"streams; point --adt-root at a full ADT export.")

    frames: List[Frame] = []
    for seq in seq_dirs:
        name = os.path.basename(seq)
        ids = common_frame_ids(seq, subdirs, "depth_npy")
        # _pair_frames is anchored on the GT depth files, so this order is the
        # depth order and is identical for every stream.
        pairs = [p for p in _pair_frames(os.path.join(seq, subdirs[0]),
                                         os.path.join(seq, "depth_npy"))[0]
                 if p[2] in ids]
        by_stream = {}
        for label, sub in streams.items():
            by_stream[label] = {p[2]: p[0] for p in
                                _pair_frames(os.path.join(seq, sub),
                                             os.path.join(seq, "depth_npy"))[0]}
        chosen = _evenly_spaced(pairs, n_frames)
        pos = {p[2]: k for k, p in enumerate(pairs)}      # frame id -> pool index
        for _rgb0, depth_path, fid in chosen:
            rgb = {label: by_stream[label][fid] for label in streams
                   if fid in by_stream[label]}
            if len(rgb) != len(streams):      # belt and braces; step 2 excludes these
                continue
            idx, tgt = _context_window(len(pairs), pos[fid],
                                       context_frames, context_stride)
            ctx = {}
            for label in streams:
                paths = [by_stream[label].get(pairs[k][2]) for k in idx]
                if any(p is None for p in paths):        # a gap in that stream
                    paths, tgt_l = [rgb[label]], 0
                else:
                    tgt_l = tgt
                ctx[label] = paths
            frames.append(Frame(seq=name, frame_id=fid, depth=depth_path,
                                rgb=rgb, context=ctx, target_index=tgt))
        if verbose:
            print(f"  [fovbench] {name}: {len(chosen)} of {len(pairs)} common frames")

    if not frames:
        raise SystemExit(f"[fovbench] no frame under {adt_root!r} is present in "
                         f"depth_npy and all of {subdirs}")
    sp = Split(root=os.path.abspath(adt_root), frames=frames, streams=streams,
               context_frames=context_frames, context_stride=context_stride)
    if verbose:
        ctx = (f", context {context_frames}x@stride {context_stride}"
               if context_frames > 1 else "")
        print(f"  [fovbench] ADT-FOV split {sp.digest}: {len(frames)} frames "
              f"across {len(sp.sequences)} sequence(s), "
              f"streams={list(streams)}{ctx}")
    return sp


# --------------------------------------------------------------------------- #
# ego-synth 5B
# --------------------------------------------------------------------------- #

#: Bumped whenever a change would make new egosynth numbers incomparable to old
#: ones. Separate from ``PROTOCOL``: the two datasets are scored by the same
#: rules but over different supports (a point list vs a dense map), so a digest
#: from one must never be mistaken for a digest from the other.
EGOSYNTH_PROTOCOL = "egosynth-fov-v1"


def build_egosynth_split(egosynth_root: str, datasets=None, n_frames: int = 25,
                         takes_per_dataset: int = 8, views=("rect", "fisheye"),
                         verbose: bool = True) -> Split:
    """Freeze a frame list over the ego-synth 5B sparse-depth release.

    Same contract as :func:`build_split` — a frozen list plus a digest, so two
    runs are comparable exactly when their digests match — over a differently
    shaped store. The mapping onto :class:`Frame` is:

        ``seq``       ``<dataset>/<take>``
        ``frame_id``  ``<clip>:<frame>``, the frame being 0..120 *within* the clip
        ``depth``     the clip's ``sparse_depth/<clip>.npz``
        ``rgb``       view label -> that view's ``<clip>.mp4``

    ``n_frames`` is per **take** and is spread evenly over all of that take's
    (clip, frame) pairs, not over one clip: a clip is 121 consecutive frames at
    20 fps, so any single clip is ~6 s of one viewpoint. ``takes_per_dataset``
    caps how many takes each dataset contributes — the full release is 1 611
    takes and 24 931 clips, which no single run should quietly attempt. Both
    caps land in the digest, so a subsampled run can never be compared against a
    fuller one by accident.
    """
    from fovbench.datasets_egosynth import DATASETS, VIEW_TO_VARIANT, find_takes

    datasets = tuple(datasets or DATASETS)
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"[fovbench] unknown egosynth dataset(s) {unknown}; "
                         f"choose from {list(DATASETS)}")
    bad_views = [v for v in views if v not in VIEW_TO_VARIANT]
    if bad_views:
        raise SystemExit(f"[fovbench] unknown view(s) {bad_views}; "
                         f"choose from {list(VIEW_TO_VARIANT)}")

    takes = find_takes(egosynth_root, datasets, limit=takes_per_dataset,
                       verbose=verbose)
    frames: List[Frame] = []
    for take in takes:
        pairs = [(clip, i) for clip in take.clips
                 for i in range(take.frames_in(clip))]
        for clip, i in _evenly_spaced(pairs, n_frames):
            frames.append(Frame(
                seq=take.key, frame_id=f"{clip}:{i}", depth=take.npz(clip),
                rgb={v: take.video(clip, v) for v in views}))
    if not frames:
        raise SystemExit(f"[fovbench] no clip under {egosynth_root!r} has a "
                         f"frame table; check meta.json's clips[]")

    sp = Split(root=os.path.abspath(egosynth_root), frames=frames,
               streams={v: VIEW_TO_VARIANT[v] for v in views},
               protocol=EGOSYNTH_PROTOCOL)
    if verbose:
        print(f"  [fovbench] egosynth split {sp.digest}: {len(frames)} frames "
              f"across {len(sp.sequences)} take(s) of {len(datasets)} dataset(s), "
              f"views={list(views)}")
    return sp
