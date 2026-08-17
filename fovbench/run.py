# Copyright (c) 2026.
"""Run the ADT-FOV test: does depth accuracy depend on *where* in the field of
view the content sits, and does rectifying the pixels first change the answer?

    python -m fovbench.run --adt-root $ADT --models vggt_1b,vggt_omega,dav2_large,da3_large

The full grid is

    model x stream x view x protocol

    models     vggt_1b | vggt_omega | dav2_large | da3_large   (all vanilla)
    streams    synthetic (rendered, GT-registered) | real (Aria sensor)
    views      rect (rectified perspective) | fisheye (raw, undistorted-free)
    protocols  radial  — whole frame once, per-pixel error binned by incidence
                         angle: "where in this image"
               window  — a fixed 40 deg window re-aimed across the lens, scored
                         on its own: "where the camera was pointed"

Both protocols answer the same question from opposite ends, and they can
disagree in an informative way. ``radial`` asks how a model handles the
periphery *of an image it was given whole*; ``window`` asks how it handles a
region when that region is all it can see. A model that is merely bad at image
corners will show a ``radial`` gradient and a flat ``window`` curve; one whose
error is driven by the ray geometry will show both.

Loading order matters for cost, so the loops are nested model-outermost: each
network is loaded once and every frame, stream, view and window goes through it.

Outputs, all under ``--out``:

    manifest.json   the frozen frame list + digest (see :mod:`fovbench.split`)
    results.json    every metric, keyed by the digest
    results.csv     one flat row per (model, stream, view, protocol, bin/cell)
    report.txt      the printed tables
    figures/*.png   AbsRel vs eccentricity, if matplotlib is installed

What this does NOT do
---------------------
Nothing here corrects for distortion, and no model is given the lens. That is
the point: the numbers are a measurement of how four off-the-shelf networks
degrade on an uncorrected wide-FOV camera, not an attempt to make them better.
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from fovbench import _REPO  # noqa: F401  (import registers sys.path)

from fovbench import geometry as G          # noqa: E402
from fovbench import models as M            # noqa: E402
from fovbench import report as R            # noqa: E402
from fovbench.split import PROTOCOL, STREAMS, Split, build_split  # noqa: E402

from finetune.eval.metrics import align_depth, depth_metrics  # noqa: E402

#: Default window sweep. The FOV is held FIXED and only the aim moves — varying
#: both is what made an earlier sweep in this repo unreadable (wider windows
#: overhang the imaged cone, so width and dead area moved together).
DEFAULT_TILTS = (0.0, 10.0, 20.0, 30.0, 40.0)
DEFAULT_AZIMUTHS = (0.0, 90.0, 180.0, 270.0)
DEFAULT_WINDOW_FOV = 40.0

#: A window this far past the imaged cone is mostly black; scoring it measures
#: the vignette, not the model.
MIN_IN_CONE_FRAC = 0.5

#: Threads when ``--workers 0``. The per-frame work is decode, rectify, resample
#: and the binning arithmetic — numpy and cv2, which release the GIL — so
#: threads genuinely parallelise it, and being threads they share the memoised
#: ray/theta/radius maps instead of rebuilding them per process. Measured on a
#: 12-core machine the view stage scales 3.3x and the binning 2.2x before
#: saturating on memory bandwidth, so a large default mostly buys contention.
#: The box has 64 cores and often runs several configurations at once; raise it
#: with ``--workers`` after measuring rather than on principle.
DEFAULT_WORKERS = 8

#: Frames in flight beyond the one being consumed, per worker. Bounds memory:
#: an unbounded pool would decode all 300 frames of a split at once.
LOOKAHEAD_PER_WORKER = 2

_NULL_LOCK = contextlib.nullcontext()


def _ordered_map(fn, items, workers: int, lookahead: Optional[int] = None):
    """``map(fn, items)`` across threads, yielded strictly **in input order**.

    Order is not cosmetic here. The rows are pooled by summation and floating
    point addition is not associative, so accumulating them in *completion*
    order would move the last digits of every published number and make a
    parallel run non-comparable to a serial one. Yielding in input order keeps
    the arithmetic bit-for-bit identical to ``workers=1`` — which
    ``tests/test_end_to_end.py`` pins.

    Bounded: at most ``lookahead`` frames are in flight, so peak memory is a
    handful of decoded frames rather than the whole split.
    """
    if workers <= 1:
        for x in items:
            yield fn(x)
        return
    lookahead = lookahead or LOOKAHEAD_PER_WORKER * workers
    it = iter(items)
    with ThreadPoolExecutor(workers) as ex:
        q = deque(ex.submit(fn, x) for x in itertools.islice(it, lookahead))
        while q:
            yield q.popleft().result()
            for x in itertools.islice(it, 1):
                q.append(ex.submit(fn, x))


def _load_frame(frame, stream: str, depth_scale: float, depth_max_m: float,
                rotation_k: int = 3):
    """One ADT frame: upright fisheye RGB, planar-z GT in metres, validity, and
    the *context stack* the model will actually be handed.

    Mirrors ``datasets.adt.ADTFisheyeFrames.__getitem__`` — the same 270-deg-CCW
    rotation and the same mm->m scaling — but reads the paths the manifest
    froze, so both streams are guaranteed to be the same instant.
    """
    def _read(path):
        with Image.open(path) as im:
            a = np.array(im.convert("RGB"), dtype=np.uint8)
        return np.rot90(a, k=rotation_k).copy() if rotation_k else a

    stack = [_read(p) for p in frame.stack(stream)]
    rgb = stack[frame.target_index]
    d = np.load(frame.depth).astype(np.float32)
    if d.ndim == 3:
        d = d.squeeze(-1)
    d = np.where(np.isfinite(d), d, 0.0) * depth_scale
    if rotation_k:
        d = np.rot90(d, k=rotation_k).copy()
    return rgb, d, (d > 0) & (d <= depth_max_m), stack


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #

def _score_radial(model, view: "G.FrameView", edges, radius_edges,
                  max_depth, context=None, target=0, lock=None,
                  depth_edges=None) -> Optional[dict]:
    """Whole frame, ONE alignment fit, metrics per bin on two axes.

    The two axes — incidence angle and distance from the optical centre — are
    two readings of the same frozen prediction, not two measurements: the scale
    (and shift) is fitted once over every valid pixel and then never touched,
    and each axis only applies a different set of masks afterwards.

    ``lock`` serialises the forward pass when frames are scored on several
    threads: the surrounding work is numpy and cv2, which release the GIL, but
    one model on one device is not something to call re-entrantly.
    """
    if view.valid.sum() < 256:
        return None
    with (lock or _NULL_LOCK):
        if context and len(context) > 1:
            pred = model.predict_stack(context, target, gt_z=view.gt_z,
                                       theta_deg=view.theta)
        else:
            pred = model.predict(view.rgb, gt_z=view.gt_z, theta_deg=view.theta,
                                 view=view)
    prof = G.bin_by(pred, view.gt_z, view.valid, model.align_mode,
                    {"theta": (view.theta, edges),
                     "radius": (view.radius, radius_edges)},
                    max_depth=max_depth,
                    profile_edges={"theta": G.PROFILE_THETA_EDGES,
                                   "radius": G.PROFILE_RADIUS_EDGES},
                    joint_depth_edges={"theta": depth_edges or G.DEPTH_EDGES})
    prof["bins"], prof["radius_bins"] = prof.pop("theta"), prof.pop("radius")
    prof["in_cone_frac"] = view.in_cone_frac
    return prof


def _score_window(model, win: "G.Window", max_depth, lock=None) -> Optional[dict]:
    """One window scored on its own, as if it were the whole camera."""
    if win.valid.sum() < 256:
        return None
    with (lock or _NULL_LOCK):
        pred = model.predict(win.rgb, gt_z=win.gt_z, theta_deg=win.theta)
    aligned = align_depth(pred, win.gt_z, win.valid, mode=model.align_mode)
    met = depth_metrics(aligned, win.gt_z, win.valid, max_depth=max_depth)
    # Each window is aligned on its own, so a per-window scale is absorbed by
    # construction and only the UNALIGNED ratio can show drift between aims.
    met.update(tilt=win.tilt, azimuth=win.azimuth, fov=win.fov,
               in_cone_frac=win.in_cone_frac,
               src_px_per_out_px=win.src_px_per_out_px,
               raw_scale_ratio=G.raw_scale_ratio(pred, win.gt_z, win.valid),
               theta_mean=float(win.theta[win.valid].mean()))
    return met


def _edges(spec: str, default) -> tuple:
    """Bin edges from the CLI, or the Aria cone's own default when unset.

    The default top edge is 55 deg because the lens images to 54.83; a run that
    wants a different partition passes ``--theta-edges`` rather than editing the
    constant, so the edges land in ``results.json`` beside the numbers they cut.
    """
    return tuple(float(x) for x in spec.split(",")) if spec else tuple(default)


def _accumulate(store: Dict[str, list], key: str, value) -> None:
    if value is not None:
        store.setdefault(key, []).append(value)


#: Metrics that are a plain mean over pixels, so pooling them across frames is
#: exactly a weighted mean with the pixel count as the weight.
_PIXEL_MEAN = ("AbsRel", "SqRel", "log10", "delta1", "delta2", "delta3")

#: Metrics that are a mean over pixels *under a square root*. Pool the squares,
#: then take the root once.
_PIXEL_RMS = ("RMSE", "RMSElog")

#: Medians. A weighted mean of per-frame medians is NOT the pooled median, and
#: no summary can recover the pooled one, so these stay a frame mean and are
#: labelled as such wherever they are read. ``gt_median`` has a pooled twin in
#: the continuous profile (``gt_mean``), which is what the depth figure draws.
_FRAME_MEDIAN = ("scale_ratio", "raw_scale_ratio",
                 "gt_median", "gt_std", "gt_spread")


def _mean_metrics(rows: List[dict], keys) -> dict:
    """Pool a metric list across frames, **weighted by each frame's pixels**.

    Not a mean of per-frame means. Every metric here is an average over the
    pixels of one frame, so the quantity a reader wants across 200 frames is the
    average over all their pixels together — which is the weighted mean, with
    ``n_valid`` as the weight. An unweighted mean instead gives a frame that
    contributed a handful of pixels to a bin the same vote as one that filled
    it, and it makes the binned tables disagree with the continuous profiles,
    which are pooled (``geometry.fine_profile``). Measured on run
    ``fovbench-v3-24b38e1``, the two agreed to -0.0% on the three depth heads
    and differed by +20% in exactly one cell — DAv2's rectified 0-10 deg bin,
    where its disparity-space fit pushes some pixels out of the metric's depth
    range and the surviving count varies frame to frame. That disagreement is
    what this removes.

    NaN frames are skipped, as before: a metric that failed on one frame should
    not poison the rest.
    """
    out = {}
    for k in keys:
        vals, wts = [], []
        for r in rows:
            v = r.get(k)
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                continue
            vals.append(float(v))
            wts.append(float(r.get("n_valid", 0)))
        if not vals:
            out[k] = float("nan")
            continue
        v = np.asarray(vals)
        w = np.asarray(wts)
        if k in _FRAME_MEDIAN or w.sum() <= 0:
            out[k] = float(v.mean())
        elif k in _PIXEL_RMS:
            out[k] = float(np.sqrt(np.average(v ** 2, weights=w)))
        else:
            out[k] = float(np.average(v, weights=w))
    out["n_frames"] = len(rows)
    out["n_valid_total"] = int(sum(r.get("n_bin", r.get("n_valid", 0)) for r in rows))
    out["n_px_total"] = int(sum(r.get("n_valid", 0) for r in rows))
    return out


def _reduce_axis(runs: List[dict], key: str, edges) -> list:
    """Average one axis's per-bin metrics across frames, keeping the structure."""
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        rows = [r[key][i] for r in runs if r[key][i]["n_valid"] > 0]
        b = _mean_metrics(rows, G.METRIC_KEYS)
        b.update(bin_lo=lo, bin_hi=hi,
                 n_px_mean=float(np.mean([r[key][i]["n_bin"] for r in runs]))
                 if runs else 0.0)
        out.append(b)
    return out


def _reduce_radial(runs: List[dict], edges, radius_edges) -> dict:
    """Average per-bin metrics across frames, on both axes.

    The coarse bins are averaged per frame and then across frames; the fine
    profiles are **pooled** (summed, then divided once). A fine bin can hold a
    handful of pixels in one frame and thousands in another, so a mean of
    per-frame means would let a frame that barely reached the rim outvote one
    that filled it. See ``geometry.fine_profile``.
    """
    axes = sorted({k for r in runs for k in r.get("profiles", {})})
    profiles = {ax: G.pool_profiles([r.get("profiles", {}).get(ax) for r in runs])
                for ax in axes}
    jaxes = sorted({k for r in runs for k in r.get("joint", {})})
    joint = {ax: G.pool_joint([r.get("joint", {}).get(ax) for r in runs])
             for ax in jaxes}
    return {"overall": _mean_metrics([r["overall"] for r in runs], G.METRIC_KEYS),
            "bins": _reduce_axis(runs, "bins", edges),
            "radius_bins": _reduce_axis(runs, "radius_bins", radius_edges),
            "profiles": {k: v for k, v in profiles.items() if v},
            "joint": {k: v for k, v in joint.items() if v},
            "in_cone_frac": float(np.mean([r["in_cone_frac"] for r in runs]))
            if runs else float("nan")}


def _reduce_windows(runs: List[dict], tilts) -> dict:
    """Average across frames and azimuths, keeping the tilt structure."""
    cells = []
    for t in tilts:
        rows = [r for r in runs if abs(r["tilt"] - t) < 1e-6]
        c = _mean_metrics(rows, G.METRIC_KEYS
                          + ("in_cone_frac", "theta_mean", "src_px_per_out_px"))
        c["tilt"] = c["bin_lo"] = c["bin_hi"] = t
        cells.append(c)
    return {"overall": _mean_metrics(runs, G.METRIC_KEYS), "cells": cells}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run(a: argparse.Namespace) -> dict:
    os.makedirs(a.out, exist_ok=True)
    streams = [s.strip() for s in a.streams.split(",") if s.strip()]
    unknown = [s for s in streams if s not in STREAMS]
    if unknown:
        raise SystemExit(f"[fovbench] unknown stream(s) {unknown}; "
                         f"choose from {list(STREAMS)}")
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    protocols = [p.strip() for p in a.protocols.split(",") if p.strip()]
    keys = [k.strip() for k in a.models.split(",") if k.strip()]
    edges = _edges(a.theta_edges, G.THETA_EDGES)
    radius_edges = _edges(a.radius_edges, G.RADIUS_EDGES)
    depth_edges = _edges(a.depth_edges, G.DEPTH_EDGES)
    tilts = tuple(float(x) for x in a.tilts.split(","))
    azimuths = tuple(float(x) for x in a.azimuths.split(","))

    # A model that cannot answer on a requested view is refused before anything
    # is loaded or decoded. ``vggt360`` is the only such model: it consumes a
    # fisheye frame and a lens, so the rectified pinhole is not an input it has,
    # and the 40 deg ``window`` protocol hands it a crop a 55 deg nine-view
    # layout would tile with eight views of nothing. Either would still produce
    # a full column of plausible numbers, which is the failure mode worth an
    # explicit check.
    for k in keys:
        allowed = M.restricted_views(k)
        if allowed is None:
            continue
        bad = [v for v in views if v not in allowed]
        if bad:
            raise SystemExit(
                f"[fovbench] {k} cannot answer on view(s) {bad}: it is given "
                f"the lens and re-renders the frame into its own tangent views, "
                f"so it needs raw fisheye pixels. Run it with "
                f"--views {','.join(allowed)}, or drop it from --models.")
        if "window" in protocols:
            raise SystemExit(
                f"[fovbench] {k} cannot run the window protocol: a window is a "
                f"{a.window_fov:g} deg crop, and this model's layout tiles a "
                f"~55 deg cone with a centre view plus a ring — aimed at a crop, "
                f"most of that ring images nothing. Use --protocols radial.")

    # Model availability is checked BEFORE the data is touched: it is the
    # cheapest check and the most common failure (gated weights, a missing pip
    # package), so it should not be masked by an ADT path problem.
    ready, skipped = M.available(keys)
    for key, state, detail in skipped:
        print(f"[fovbench] {key}: {state} — {detail}")
    if skipped and not a.skip_unavailable:
        # Refuse rather than quietly produce a two-model report that looks like
        # a four-model one. VGGT-Omega's weights are gated and DA3 needs a pip
        # install, so "some models missing" is the normal first-run state, and
        # the resulting artefact would otherwise be indistinguishable from a
        # complete run except by reading the model list at the bottom.
        raise SystemExit(
            f"[fovbench] {len(skipped)} of {len(keys)} requested models cannot "
            f"run ({', '.join(k for k, _, _ in skipped)}). Fix them with the "
            f"instructions above, or pass --skip-unavailable to run the rest "
            f"anyway (the report and results.json will record what was left out).")
    if not ready:
        raise SystemExit("[fovbench] no model is runnable; see the lines above. "
                         "Use --models analytic for a weight-free harness run.")

    # The split requires exactly the streams that will be scored. Asking for one
    # stream must not exclude sequences for lacking the other — but the digest
    # then differs, so a one-stream run is never silently compared to a two.
    if a.context_frames > 1:
        cant = [k for k in ready if k not in M.CONTEXT_CAPABLE]
        if cant:
            raise SystemExit(
                f"[fovbench] --context-frames {a.context_frames} but "
                f"{', '.join(cant)} is monocular and cannot use a context. "
                f"Drop it from --models, or use --context-frames 1. A run that "
                f"silently scored the target alone would read as 'context does "
                f"not help'.")
        if "window" in protocols:
            raise SystemExit(
                "[fovbench] --context-frames applies to the radial protocol "
                "only: a window is a crop, and handing a model ten crops of "
                "ten different instants is not the experiment. Use "
                "--protocols radial.")
    split = (Split.load(a.manifest) if a.manifest
             else build_split(a.adt_root, n_frames=a.n_frames,
                              streams={s: STREAMS[s] for s in streams},
                              context_frames=a.context_frames,
                              context_stride=a.context_stride))
    # A manifest carries a *pre-baked* context list per frame, so --manifest wins
    # over --context-frames and the flags become a silent no-op: the run's own
    # `config` block still echoes them, and the artefact is indistinguishable
    # from a real context run except by diffing it against the baseline. That
    # produced four dead runs on ticket 010. Refuse the disagreement instead.
    if a.manifest and (a.context_frames != split.context_frames
                       or a.context_stride != split.context_stride):
        raise SystemExit(
            f"[fovbench] --manifest {a.manifest} was written with "
            f"--context-frames {split.context_frames} --context-stride "
            f"{split.context_stride}, but you asked for {a.context_frames}/"
            f"{a.context_stride}. The manifest's own context wins, so the run "
            f"would score {split.context_frames} frame(s) while reporting "
            f"{a.context_frames}. Pass the matching flags, or drop --manifest "
            f"— the digest excludes the context by design, so rebuilding the "
            f"split from --adt-root gives the same digest with a real context.")
    split.save(os.path.join(a.out, "manifest.json"))
    missing = [s for s in streams if s not in split.streams]
    if missing:
        raise SystemExit(f"[fovbench] --manifest {a.manifest} has no {missing} "
                         f"stream (it has {list(split.streams)})")

    device = None
    if ready != [M.ANALYTIC]:
        import torch
        device = torch.device(a.device)
        if device.type != "cuda":
            print("[fovbench] WARNING: running real weights on CPU — this is "
                  "minutes per frame. Use --device cuda on the GPU box.")

    workers = a.workers if a.workers > 0 else min(DEFAULT_WORKERS,
                                                  os.cpu_count() or 1)
    fwd_lock = threading.Lock() if workers > 1 else None
    if workers > 1:
        print(f"[fovbench] scoring frames on {workers} threads "
              f"(forward pass serialised; results pooled in split order)")

    runs: List[dict] = []
    for key in ready:
        print(f"\n[fovbench] ══ {key} ══")
        t0 = time.time()
        model = M.load_model(key, device, checkpoint=a.omega_checkpoint,
                             radial_bias=a.analytic_bias,
                             v360_fov=a.vggt360_fov,
                             v360_ring_tilt=a.vggt360_ring_tilt,
                             v360_n_ring=a.vggt360_n_ring,
                             v360_persp_size=a.vggt360_persp_size,
                             v360_source=a.vggt360_source,
                             v360_fuse=a.vggt360_fuse,
                             v360_head=a.vggt360_head,
                             v360_dtype=a.vggt360_dtype,
                             v360_adaptive=not a.vggt360_no_adaptive,
                             v360_sa_mask=not a.vggt360_no_sa_mask)
        n = model.input_size
        print(f"[fovbench]   {model.family} {model.size} | {model.params_m:.0f}M "
              f"params | align={model.align_mode} | views rendered at {n}px")

        # One frame's whole contribution, as an ordered list of (tag, row). The
        # frames are independent, so this runs on a worker thread; only the
        # forward pass is serialised, and _ordered_map hands the results back in
        # split order so the pooling arithmetic cannot notice the threads.
        def _frame_rows(frame, model=model, n=n):
            out = []
            for stream in streams:
                rgb, gt, gt_valid, stack = _load_frame(
                    frame, stream, a.depth_scale, a.depth_max_m)
                cam = G.aria_cam(*rgb.shape[:2])
                for kind in views:
                    if "radial" in protocols:
                        fv = G.full_frame_view(rgb, gt, gt_valid, cam, n, kind)
                        # Context frames go through the SAME view construction;
                        # the target's own view is reused rather than rendered
                        # twice, so a 1-frame run costs exactly what it did.
                        ctx = None
                        if len(stack) > 1:
                            ctx = [fv.rgb if k == frame.target_index
                                   else G.view_rgb(im, cam, n, kind)
                                   for k, im in enumerate(stack)]
                        out.append((f"radial|{stream}|{kind}",
                                    _score_radial(model, fv, edges, radius_edges,
                                                  a.metric_max_depth,
                                                  context=ctx,
                                                  target=frame.target_index,
                                                  lock=fwd_lock,
                                                  depth_edges=depth_edges)))
                    if "window" in protocols:
                        for tilt in tilts:
                            for az in (azimuths if tilt > 0 else (0.0,)):
                                w = G.render_window(rgb, gt, gt_valid, cam, az,
                                                    tilt, a.window_fov, n, kind,
                                                    supersample=a.supersample)
                                if w.in_cone_frac < MIN_IN_CONE_FRAC:
                                    continue
                                out.append((f"window|{stream}|{kind}",
                                            _score_window(model, w,
                                                          a.metric_max_depth,
                                                          lock=fwd_lock)))
            return out

        acc: Dict[str, list] = {}
        for i, rows in enumerate(_ordered_map(_frame_rows, split.frames, workers)):
            for tag, row in rows:
                _accumulate(acc, tag, row)
            if (i + 1) % max(1, a.log_every) == 0:
                print(f"[fovbench]   frame {i + 1}/{len(split.frames)} "
                      f"({time.time() - t0:.0f}s)")

        for tag, rows in sorted(acc.items()):
            protocol, stream, kind = tag.split("|")
            body = (_reduce_radial(rows, edges, radius_edges)
                    if protocol == "radial" else _reduce_windows(rows, tilts))
            runs.append(dict(model=key, family=model.family, size=model.size,
                             params_m=model.params_m, align=model.align_mode,
                             input_size=n, protocol=protocol, stream=stream,
                             view=kind, **body))
        del model
        if device is not None and device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
        print(f"[fovbench]   done in {time.time() - t0:.0f}s")

    payload = dict(
        protocol=PROTOCOL, digest=split.digest, adt_root=split.root,
        n_frames=len(split.frames), sequences=split.sequences,
        requested_models=keys,
        skipped_models=[dict(model=k, state=s, detail=d) for k, s, d in skipped],
        config=dict(streams=streams, views=views, protocols=protocols,
                    context_frames=a.context_frames,
                    context_stride=a.context_stride,
                    theta_edges=list(edges), radius_edges=list(radius_edges),
                    depth_edges=list(depth_edges),
                    tilts=list(tilts),
                    azimuths=list(azimuths), window_fov=a.window_fov,
                    depth_max_m=a.depth_max_m,
                    metric_max_depth=a.metric_max_depth,
                    min_in_cone_frac=MIN_IN_CONE_FRAC,
                    # The lens-aware model's own configuration, recorded
                    # whether or not it ran: a results.json that does not say
                    # which layout produced a vggt360 row is not readable, and
                    # one that says so only when the model was present makes
                    # the two cases look different for the wrong reason.
                    vggt360=dict(fov=a.vggt360_fov,
                                 ring_tilt=a.vggt360_ring_tilt,
                                 n_ring=a.vggt360_n_ring,
                                 persp_size=a.vggt360_persp_size,
                                 source=a.vggt360_source,
                                 fuse=a.vggt360_fuse, head=a.vggt360_head,
                                 dtype=a.vggt360_dtype,
                                 adaptive=not a.vggt360_no_adaptive,
                                 sa_mask=not a.vggt360_no_sa_mask),
                    # Recorded for provenance only: the worker count is pinned
                    # not to change any number (see _ordered_map), so two runs
                    # differing only here are still directly comparable.
                    workers=workers,
                    analytic_bias=a.analytic_bias),
        runs=runs)
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    R.write_all(payload, a.out)
    print(f"\n[fovbench] wrote {a.out}/results.json (+ csv, report.txt, figures/)")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adt-root", default=os.environ.get("ADT", ""),
                   help="ADT export root (sequences with depth_npy + both RGB streams)")
    p.add_argument("--manifest", default=None,
                   help="reuse a frozen split instead of rebuilding it")
    p.add_argument("--n-frames", type=int, default=25,
                   help="frames PER SEQUENCE, spread evenly (not a prefix)")
    p.add_argument("--models", default=",".join(M.DEFAULT_MODELS),
                   help=f"comma keys, or '{M.ANALYTIC}' for a weight-free run")
    p.add_argument("--context-frames", type=int, default=1,
                   help="frames handed to the model in ONE forward pass. Only "
                        "the split's own frame is scored; the rest are context "
                        "for cross-view attention. Multi-view models only")
    p.add_argument("--context-stride", type=int, default=1,
                   help="spacing of the context frames, in source frames. 1 is "
                        "truly consecutive (ADT is 30 Hz, so 10 frames span "
                        "0.30 s); a larger stride buys parallax")
    p.add_argument("--streams", default="synthetic,real")
    p.add_argument("--views", default="rect,fisheye",
                   help="rect = rectified perspective, fisheye = raw pixels")
    p.add_argument("--protocols", default="radial,window")
    p.add_argument("--theta-edges", default="",
                   help="incidence-angle bin edges (deg); default "
                        "0,10,20,30,40,50,55, the Aria cone stopping at 54.83")
    p.add_argument("--radius-edges", default="",
                   help="distance-from-optical-centre bin edges, in half-widths "
                        "(1.0 = middle of a frame edge, sqrt(2) = a corner); "
                        "default 0,0.2,0.4,0.6,0.8,1.0,1.45")
    p.add_argument("--depth-edges", default="",
                   help="GT-depth edges (m) for the joint incidence-angle x "
                        "depth table, which separates 'the rim is harder' from "
                        "'the rim is nearer'; default 0,1,2,3,5,10")
    p.add_argument("--tilts", default=",".join(str(t) for t in DEFAULT_TILTS),
                   help="window eccentricities (deg)")
    p.add_argument("--azimuths", default=",".join(str(t) for t in DEFAULT_AZIMUTHS),
                   help="window azimuths (deg); averaged over, to separate "
                        "eccentricity from any left/right asymmetry")
    p.add_argument("--window-fov", type=float, default=DEFAULT_WINDOW_FOV,
                   help="window field of view (deg), HELD FIXED across the sweep")
    p.add_argument("--supersample", type=int, default=3,
                   help="anti-aliasing factor for the rectified render")
    p.add_argument("--depth-scale", type=float, default=0.001, help="GT mm -> m")
    p.add_argument("--depth-max-m", type=float, default=10.0,
                   help="GT validity ceiling (which GT pixels count)")
    p.add_argument("--metric-max-depth", type=float, default=100.0,
                   help="excludes out-of-range PREDICTIONS from the metrics; "
                        "deliberately separate from --depth-max-m")
    p.add_argument("--omega-checkpoint", default=None,
                   help="local VGGT-Omega .pt (default: $VGGT_OMEGA_CKPT)")

    # -- the vggt360 model's own layout ------------------------------------- #
    # Defaults are main_adt.py's, so `--models vggt360` with no further flags
    # is the same model that driver reports, seeing this harness's 518 px view
    # instead of ADT's native 1408 (fovbench/models.py says why).
    g = p.add_argument_group(
        "vggt360", "VGGT-360-fisheye layout; ignored unless --models includes "
                   "vggt360. Defaults match VGGT-360-fisheye/main_adt.py.")
    g.add_argument("--vggt360-fov", type=float, default=60.0,
                   help="per-view FOV (deg) of the tangent views")
    g.add_argument("--vggt360-ring-tilt", type=float, default=26.0,
                   help="ring tilt off the optical axis (deg); the layout rule "
                        "is tilt + fov/2 >~ 54.8, the Aria usable cone")
    g.add_argument("--vggt360-n-ring", type=int, default=8,
                   help="ring view count (the centre view is extra)")
    g.add_argument("--vggt360-source", choices=["native", "view"],
                   default="native",
                   help="which pixels the nine tangent views are cut from. "
                        "'native' = ADT's own 1408 frame, the resolution the "
                        "port is designed for (a 60 deg view at 518 is then a "
                        "0.62x downsample). 'view' = the 518 frame the other "
                        "four models get, strictly resolution-matched but a "
                        "1.69x UPsample that starves the method. Either way the "
                        "answer is fused onto the 518 scoring grid, so the "
                        "metrics, masks and bins are identical")
    g.add_argument("--vggt360-persp-size", type=int, default=518,
                   help="side length each tangent view is rendered at. 518 is "
                        "the backbone's own token grid, so nothing is resampled "
                        "between the view and the network — the same rule the "
                        "four vanilla models are held to. main_adt.py uses 512, "
                        "which VGGT then bicubic-resizes up to 518 anyway")
    g.add_argument("--vggt360-fuse", choices=["attn", "mean"], default="attn",
                   help="correlation-weighted fusion (the paper) or uniform")
    g.add_argument("--vggt360-head", choices=["depth", "point"], default="depth",
                   help="range source: depth head z * secant, or the point "
                        "head's ||world_points||")
    g.add_argument("--vggt360-dtype", choices=["bf16", "fp16", "fp32"],
                   default="bf16", help="autocast dtype for the VGGT pass")
    g.add_argument("--vggt360-no-adaptive", action="store_true",
                   help="module-1 ablation: base views only, no "
                        "uncertainty-guided neighbours")
    g.add_argument("--vggt360-no-sa-mask", action="store_true",
                   help="module-2 ablation: vanilla attention, no "
                        "structure-saliency log-bias")
    p.add_argument("--analytic-bias", type=float, default=0.6,
                   help="radial bias injected by --models analytic, for "
                        "verifying that the harness reports what it is given")
    p.add_argument("--skip-unavailable", action="store_true",
                   help="run the models that are ready instead of refusing; the "
                        "report and results.json then name what was left out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=0,
                   help="threads scoring frames in parallel; 0 picks "
                        f"min({DEFAULT_WORKERS}, cpu_count). The forward pass "
                        "is serialised and results are pooled in split order, "
                        "so this changes the wall clock and nothing else — "
                        "1 reproduces the pre-threading path exactly.")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--out", default="eval_out/fovbench")
    return p


def main() -> None:
    a = build_parser().parse_args()
    if not a.adt_root and not a.manifest:
        raise SystemExit("[fovbench] pass --adt-root (or $ADT) or --manifest")
    run(a)


if __name__ == "__main__":
    main()
