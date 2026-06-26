# Copyright (c) 2026.
"""EgoExo4D egocentric fisheye frames for the qualitative baseline runs.

The EgoExo4D sample under ``EgoX/example/egoexo4D`` stores each clip's egocentric
view as an ``ego_GT.mp4`` — **raw Aria fisheye** (448×448, ~150° FOV, strong
barrel distortion + vignette), already upright (the EgoX renderer renders
gravity-aligned, so unlike raw ADT sensor frames these need **no** 270° rotation).

There is no metric depth GT for these ego renders (the ``depth_maps/*.npy`` are
relative-range priors), so EgoExo4D is used **qualitatively**: predicted depth is
just visualised next to the input. This loader yields evenly-spaced RGB frames per
clip via OpenCV (no ffmpeg dependency).
"""
from __future__ import annotations

import os
from typing import Iterator, List, Tuple

import numpy as np


def find_egoexo_clips(egoexo_root: str, stream: str = "ego_GT") -> List[Tuple[str, str]]:
    """Return ``(clip_name, video_path)`` for each clip's ego video under root.

    ``egoexo_root`` is the dir containing ``videos/<clip>/<stream>.mp4``
    (e.g. ``EgoX/example/egoexo4D``). ``stream`` is ``ego_GT`` (ground-truth ego
    render) or ``ego_Prior``.
    """
    vids_dir = os.path.join(egoexo_root, "videos")
    if not os.path.isdir(vids_dir):
        raise FileNotFoundError(f"no videos/ under {egoexo_root!r}")
    out = []
    for clip in sorted(os.listdir(vids_dir)):
        vp = os.path.join(vids_dir, clip, f"{stream}.mp4")
        if os.path.isfile(vp):
            out.append((clip, vp))
    if not out:
        raise FileNotFoundError(
            f"no {stream}.mp4 found under {vids_dir!r}/<clip>/")
    return out


def sample_frames(video_path: str, n: int = 4) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_index, rgb_uint8_hwc)`` for ``n`` evenly-spaced frames."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    if n >= total:
        idxs = list(range(total))
    else:
        idxs = list(np.linspace(0, total - 1, n).round().astype(int))
    want = set(idxs)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            yield i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
