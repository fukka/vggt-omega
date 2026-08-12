# Copyright (c) 2026.
"""Per-point depth scoring, under each model's own alignment protocol.

The metric definitions are not redefined here. ``finetune/eval/metrics.py`` owns
``align_depth`` and ``depth_metrics`` for the whole repository and both
experiments use them; this module is only the protocol wrapped around them —
what is fitted, over what, and how frames combine.

The protocol
------------
**The scale (and shift) is fitted once per frame, over every one of that frame's
valid points, and then frozen.** Per frame because each forward pass of an
up-to-scale model carries its own arbitrary scale, so a fit pooled across frames
would be fitting the spread of those constants. Over every point because a fit on
a subset is a fit chosen by whatever selected the subset.

Frames then combine as a **mean over frames**, not a mean over points. Point
counts vary by a factor of four across frames (1 300 to 5 300), and weighting by
them would let the densest frames — which are the ones with the most texture, so
not a random sample — set the score.

Each model's own alignment space
--------------------------------
Alignment mode travels with the model, from the registry, and getting it wrong is
worth more than any effect this benchmark measures:

    up-to-scale depth heads   VGGT, VGGT-Omega, DA3   affine in **depth**
    relative disparity        Depth-Anything V2       affine in **disparity**

Depth-Anything V2 predicts affine-invariant *disparity*; a depth-space affine
cannot undo a disparity-space shift and scores even a perfect prediction badly.
``models.py`` reads the mode from ``model_zoo``'s ``align_native`` rather than
hard-coding it, and :func:`check_protocol` is the assertion that it did.

A consequence worth stating plainly: **absolute AbsRel is not comparable across
models here**, because DAv2 is scored under a different alignment than the depth
heads. It is comparable across *baselines* of one model, which is the comparison
this benchmark exists for.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from slambench import _REPO  # noqa: F401  (import registers sys.path)

from finetune.eval.metrics import align_depth, depth_metrics  # noqa: E402

#: What a frame's score carries. ``depth_metrics``' own keys; named here so the
#: driver's averaging and the report's columns cannot drift apart.
METRIC_KEYS = ("AbsRel", "SqRel", "RMSE", "RMSElog", "log10",
               "delta1", "delta2", "delta3")

#: Ground-truth validity. The release's depths run 0.10 m to 114 m; the ceiling
#: is deliberately generous because Oxford is outdoors with a 23 m p99, and
#: cutting at an indoor range would silently make the four datasets different
#: experiments.
MIN_DEPTH_M = 0.01
MAX_DEPTH_M = 120.0


def score_frame(pred: np.ndarray, gt: np.ndarray, align_mode: str,
                min_depth: float = MIN_DEPTH_M,
                max_depth: float = MAX_DEPTH_M,
                min_points: int = 256) -> Optional[dict]:
    """One frame: fit the affine over all its points, freeze it, score.

    ``pred`` and ``gt`` are flat ``(N,)`` arrays — one entry per ground-truth
    point. Returns ``None`` when too few points survive to determine a
    two-parameter fit, which is a frame that was not measured rather than a
    frame that scored badly.
    """
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    ok = (np.isfinite(pred) & np.isfinite(gt) & (pred > 0)
          & (gt > min_depth) & (gt < max_depth))
    if int(ok.sum()) < min_points:
        return None
    aligned = align_depth(pred, gt, ok, mode=align_mode)
    met = depth_metrics(aligned, gt, ok, min_depth, max_depth)
    n = met.get("n_valid", 0)
    met["n_valid"] = 0 if not np.isfinite(n) else int(n)
    met["n_points"] = int(ok.sum())
    # What the frame was looking at. Not a score: every metric here is relative
    # and grows with depth, so a model that looks worse on Oxford than on AEA has
    # not yet been shown to be worse *at* anything.
    g = gt[ok]
    met["gt_median"] = float(np.median(g))
    return met


def aggregate(frames: Sequence[dict]) -> dict:
    """Mean over frames of the per-frame scores, skipping NaN.

    Deliberately unweighted — see the module docstring.
    """
    frames = [f for f in frames if f]
    out: Dict[str, float] = {}
    for k in METRIC_KEYS + ("gt_median",):
        vals = [f[k] for f in frames
                if isinstance(f.get(k), (int, float)) and np.isfinite(f[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_frames"] = len(frames)
    out["n_points_total"] = int(sum(f.get("n_points", 0) for f in frames))
    return out


def common_support(a: Sequence[Optional[np.ndarray]],
                   b: Sequence[Optional[np.ndarray]]) -> List[np.ndarray]:
    """Per frame, the points **both** baselines could predict.

    Two baselines do not automatically see the same points. A rectified pinhole
    cannot cover the whole fisheye cone, so ``rect_derect`` has no prediction for
    rim points while ``raw`` does. Comparing their aggregates over different
    point sets is not a comparison of baselines — it is partly a comparison of
    which points each happened to keep.

    So both are scored on the intersection, and how much each gave up to reach it
    is reported separately rather than folded into the score.
    """
    out = []
    for pa, pb in zip(a, b):
        if pa is None or pb is None:
            out.append(np.zeros(0, bool))
            continue
        out.append(np.isfinite(pa) & (pa > 0) & np.isfinite(pb) & (pb > 0))
    return out


def check_protocol(key: str, align_mode: str) -> None:
    """Refuse an alignment that is not the one this model was built for.

    A guard, not a lookup: ``models.py`` already reads the mode from the
    registry, and this is what catches a registry edit or a hand-passed override
    that would silently score a disparity model in depth space. The failure it
    prevents is worth more than every effect this benchmark measures, and it is
    invisible in the output — a mis-aligned model just looks bad.
    """
    disparity = align_mode == "disparity_scale_shift"
    looks_relative = key.startswith("dav2") or key.startswith("dpt")
    if looks_relative and not disparity:
        raise SystemExit(
            f"[slambench] {key} predicts affine-invariant disparity but would be "
            f"aligned with {align_mode!r}. A depth-space affine cannot undo a "
            f"disparity-space shift; the reference DAv2 eval fits in disparity. "
            f"Fix the registry's align_native rather than passing an override.")
    if disparity and not looks_relative:
        raise SystemExit(
            f"[slambench] {key} would be aligned in disparity space "
            f"({align_mode!r}), which is the protocol for a relative-disparity "
            f"model. If {key} really emits disparity, widen this check; if it "
            f"emits depth, fix the registry.")
