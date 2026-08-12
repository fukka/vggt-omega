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
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from fovbench import _REPO  # noqa: F401  (import registers sys.path)

from fovbench import datasets_egosynth as EG  # noqa: E402
from fovbench import geometry as G          # noqa: E402
from fovbench import models as M            # noqa: E402
from fovbench import report as R            # noqa: E402
from fovbench.split import (EGOSYNTH_PROTOCOL, PROTOCOL, STREAMS,  # noqa: E402
                            Split, build_egosynth_split, build_split)

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
                  max_depth, context=None, target=0) -> Optional[dict]:
    """Whole frame, ONE alignment fit, metrics per bin on two axes.

    The two axes — incidence angle and distance from the optical centre — are
    two readings of the same frozen prediction, not two measurements: the scale
    (and shift) is fitted once over every valid pixel and then never touched,
    and each axis only applies a different set of masks afterwards.
    """
    if view.valid.sum() < 256:
        return None
    if context and len(context) > 1:
        pred = model.predict_stack(context, target, gt_z=view.gt_z,
                                   theta_deg=view.theta)
    else:
        pred = model.predict(view.rgb, gt_z=view.gt_z, theta_deg=view.theta)
    prof = G.bin_by(pred, view.gt_z, view.valid, model.align_mode,
                    {"theta": (view.theta, edges),
                     "radius": (view.radius, radius_edges)},
                    max_depth=max_depth,
                    profile_edges={"theta": G.PROFILE_THETA_EDGES,
                                   "radius": G.PROFILE_RADIUS_EDGES})
    prof["bins"], prof["radius_bins"] = prof.pop("theta"), prof.pop("radius")
    prof["in_cone_frac"] = view.in_cone_frac
    return prof


def _score_window(model, win: "G.Window", max_depth) -> Optional[dict]:
    """One window scored on its own, as if it were the whole camera."""
    if win.valid.sum() < 256:
        return None
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
    """Bin edges from the CLI, or the dataset's own default when unset.

    The two datasets image different cones and so cannot share one default: the
    Aria fisheye's usable cone stops at 54.83 deg, while ego-synth's rectified
    110 deg pinhole reaches 55 deg at the middle of an edge and 63.65 deg into a
    corner. An explicit ``--theta-edges`` still overrides both.
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
_FRAME_MEDIAN = ("scale_ratio", "raw_scale_ratio", "anchored_ratio",
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
    return {"overall": _mean_metrics([r["overall"] for r in runs], G.METRIC_KEYS),
            "bins": _reduce_axis(runs, "bins", edges),
            "radius_bins": _reduce_axis(runs, "radius_bins", radius_edges),
            "profiles": {k: v for k, v in profiles.items() if v},
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
    tilts = tuple(float(x) for x in a.tilts.split(","))
    azimuths = tuple(float(x) for x in a.azimuths.split(","))

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

    runs: List[dict] = []
    for key in ready:
        print(f"\n[fovbench] ══ {key} ══")
        t0 = time.time()
        model = M.load_model(key, device, checkpoint=a.omega_checkpoint,
                             radial_bias=a.analytic_bias)
        n = model.input_size
        print(f"[fovbench]   {model.family} {model.size} | {model.params_m:.0f}M "
              f"params | align={model.align_mode} | views rendered at {n}px")

        acc: Dict[str, list] = {}
        for i, frame in enumerate(split.frames):
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
                        _accumulate(acc, f"radial|{stream}|{kind}",
                                    _score_radial(model, fv, edges, radius_edges,
                                                  a.metric_max_depth,
                                                  context=ctx,
                                                  target=frame.target_index))
                    if "window" in protocols:
                        for tilt in tilts:
                            for az in (azimuths if tilt > 0 else (0.0,)):
                                w = G.render_window(rgb, gt, gt_valid, cam, az,
                                                    tilt, a.window_fov, n, kind,
                                                    supersample=a.supersample)
                                if w.in_cone_frac < MIN_IN_CONE_FRAC:
                                    continue
                                _accumulate(acc, f"window|{stream}|{kind}",
                                            _score_window(model, w,
                                                          a.metric_max_depth))
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
                    tilts=list(tilts),
                    azimuths=list(azimuths), window_fov=a.window_fov,
                    depth_max_m=a.depth_max_m,
                    metric_max_depth=a.metric_max_depth,
                    min_in_cone_frac=MIN_IN_CONE_FRAC,
                    analytic_bias=a.analytic_bias),
        runs=runs)
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    R.write_all(payload, a.out)
    print(f"\n[fovbench] wrote {a.out}/results.json (+ csv, report.txt, figures/)")
    return payload


# --------------------------------------------------------------------------- #
# ego-synth 5B driver
# --------------------------------------------------------------------------- #

def _egosynth_clips(split: Split) -> List[tuple]:
    """Group the split into ``(seq, clip, npz, rgb_paths, frame_indices)``.

    One entry per clip, so each mp4 is opened and decoded exactly once. Frames
    of a clip are near-duplicates at 20 fps and the split already strides across
    them; what this grouping buys is decode cost, not sampling.
    """
    order: List[tuple] = []
    seen: Dict[tuple, int] = {}
    for f in split.frames:
        clip, idx = f.frame_id.rsplit(":", 1)
        tag = (f.seq, clip)
        if tag not in seen:
            seen[tag] = len(order)
            order.append((f.seq, clip, f.depth, dict(f.rgb), []))
        order[seen[tag]][4].append(int(idx))
    return [(s, c, d, r, sorted(set(i))) for s, c, d, r, i in order]


def _egosynth_cells(bins: List[dict], n_pool: int) -> List[dict]:
    """Pooled bins in the shape the report reads.

    ``n_frames``/``n_px_mean`` are what ``report._populated`` gates on. In the
    ADT path they are "frames that had this bin" and "mean pixels per frame";
    here the pool is binned once, so they are the frames behind the pool and the
    bin's total point count. A bin with no points reports ``n_frames = 0`` and
    prints as ``—``: an empty bin is **missing**, not zero, and must never be
    filled in (a frame can populate no bin at all within 30 deg of the axis).

    """
    cells = []
    for b in bins:
        c = dict(b)
        c["n_frames"] = n_pool if b["n_bin"] > 0 else 0
        c["n_px_mean"] = float(b["n_bin"])
        cells.append(c)
    return cells


def run_egosynth(a: argparse.Namespace) -> dict:
    """Score the models against ego-synth 5B's sparse SLAM depth.

    Same question and same protocol as the ADT path — the scale and shift are
    fitted once per frame over every valid point and then frozen, and binning is
    a masking step afterwards — over a point-list ground truth instead of a
    dense map. The structural differences, and why each is forced, are in
    :mod:`fovbench.datasets_egosynth`.

    Only the ``radial`` protocol exists here. The ``window`` protocol re-renders
    an angular window out of the raw fisheye, which needs a fisheye camera
    model; this release ships none.
    """
    os.makedirs(a.out, exist_ok=True)
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    bad = [v for v in views if v not in EG.VIEW_TO_VARIANT]
    if bad:
        raise SystemExit(f"[fovbench] unknown view(s) {bad}; "
                         f"choose from {list(EG.VIEW_TO_VARIANT)}")
    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    keys = [k.strip() for k in a.models.split(",") if k.strip()]
    edges = _edges(a.theta_edges, EG.THETA_EDGES)
    radius_edges = _edges(a.radius_edges, G.RADIUS_EDGES)

    ready, skipped = M.available(keys)
    for key, state, detail in skipped:
        print(f"[fovbench] {key}: {state} — {detail}")
    if skipped and not a.skip_unavailable:
        raise SystemExit(
            f"[fovbench] {len(skipped)} of {len(keys)} requested models cannot "
            f"run ({', '.join(k for k, _, _ in skipped)}). Fix them with the "
            f"instructions above, or pass --skip-unavailable to run the rest "
            f"anyway (the report and results.json will record what was left out).")
    if not ready:
        raise SystemExit("[fovbench] no model is runnable; see the lines above. "
                         "Use --models analytic for a weight-free harness run.")

    split = (Split.load(a.manifest) if a.manifest
             else build_egosynth_split(a.egosynth_root, datasets=datasets,
                                       n_frames=a.n_frames,
                                       takes_per_dataset=a.egosynth_takes,
                                       views=views))
    if split.protocol != EGOSYNTH_PROTOCOL:
        raise SystemExit(f"[fovbench] --manifest {a.manifest} was written by "
                         f"{split.protocol!r}, not {EGOSYNTH_PROTOCOL!r}; an ADT "
                         f"manifest cannot be scored against ego-synth.")
    split.save(os.path.join(a.out, "manifest.json"))
    missing = [v for v in views if v not in split.streams]
    if missing:
        raise SystemExit(f"[fovbench] --manifest {a.manifest} has no {missing} "
                         f"view (it has {list(split.streams)})")

    device = None
    if ready != [M.ANALYTIC]:
        import torch
        device = torch.device(a.device)
        if device.type != "cuda":
            print("[fovbench] WARNING: running real weights on CPU — this is "
                  "minutes per frame. Use --device cuda on the GPU box.")

    clips = _egosynth_clips(split)
    runs: List[dict] = []
    gathers: set = set()
    for key in ready:
        print(f"\n[fovbench] ══ {key} ══")
        t0 = time.time()
        model = M.load_model(key, device, checkpoint=a.omega_checkpoint,
                             radial_bias=a.analytic_bias)
        n = model.input_size if a.egosynth_input_size <= 0 else a.egosynth_input_size
        print(f"[fovbench]   {model.family} {model.size} | {model.params_m:.0f}M "
              f"params | align={model.align_mode} | frames fed at {n}px")

        pools: Dict[tuple, EG.PointPool] = {}
        cone: Dict[tuple, List[float]] = {}
        thin = 0
        for ci, (seq, clip, npz, rgb_paths, idxs) in enumerate(clips):
            dataset = seq.split("/")[0]
            rect, mask = EG.context_for(npz)
            for view in views:
                # theta is computable on the rectified pinhole and on nothing
                # else in this release — see EG.ThetaUnavailable.
                axes = ("radius", "theta") if view == "rect" else ("radius",)
                frames = EG.decode_frames(rgb_paths[view], idxs)
                pool = pools.setdefault((dataset, view),
                                        EG.PointPool(model.align_mode))
                for i in idxs:
                    pts = EG.read_points(
                        npz, view, i, rect, sigma_max=a.egosynth_sigma_max,
                        valid_mask=mask if view == "rect" else None)
                    if len(pts) < EG.MIN_FRAME_POINTS:
                        thin += 1
                        continue
                    rgb = EG.resize_for_model(frames[i], n)
                    raw = model.predict(rgb, gt_z=pts.d, theta_deg=pts.theta)
                    pred = EG.prediction_at_points(raw, pts)
                    gathers.add("index" if np.ndim(raw) == 2
                                and np.shape(raw) == (EG.RES, EG.RES)
                                else ("per-point" if np.ndim(raw) == 1
                                      else "bilinear"))
                    if not pool.add_frame(pred, pts, axes,
                                          max_depth=a.metric_max_depth):
                        thin += 1
                cone.setdefault((dataset, view), []).append(
                    float(mask.mean()) if (view == "rect" and mask is not None)
                    else float("nan"))
            if (ci + 1) % max(1, a.log_every) == 0:
                print(f"[fovbench]   clip {ci + 1}/{len(clips)} "
                      f"({time.time() - t0:.0f}s)")

        for (dataset, view), pool in sorted(pools.items()):
            axes = {"radius": radius_edges}
            if view == "rect":
                axes["theta"] = edges
            prof = pool.profile(axes, max_depth=a.metric_max_depth)
            if prof is None:
                print(f"[fovbench]   {dataset}/{view}: no frame survived; skipped")
                continue
            body = {"overall": prof["overall"],
                    "radius_bins": _egosynth_cells(prof["radius"],
                                                   prof["n_pool_frames"]),
                    "n_pool_frames": prof["n_pool_frames"],
                    "anchor_bin": prof.get("theta_anchor_bin",
                                           prof.get("radius_anchor_bin", -1)),
                    "anchor_frames": prof.get("theta_anchor_frames",
                                              prof.get("radius_anchor_frames", 0)),
                    # The imaged fraction is the rectified render's valid mask.
                    # The raw fisheye arm has no such number — this release
                    # ships no fisheye camera model — so every entry is NaN and
                    # nanmean would warn on an empty slice rather than answer.
                    "in_cone_frac": float(np.mean(seen_cone))
                    if (seen_cone := [c for c in cone.get((dataset, view), ())
                                      if np.isfinite(c)]) else float("nan")}
            if "theta" in prof:
                body["bins"] = _egosynth_cells(prof["theta"],
                                               prof["n_pool_frames"])
            runs.append(dict(model=key, family=model.family, size=model.size,
                             params_m=model.params_m, align=model.align_mode,
                             input_size=n, protocol="radial", stream=dataset,
                             view=view, **body))
        if thin:
            print(f"[fovbench]   {thin} frame-views under "
                  f"{EG.MIN_FRAME_POINTS} points were not scored")
        del model
        if device is not None and device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
        print(f"[fovbench]   done in {time.time() - t0:.0f}s")

    payload = dict(
        protocol=EGOSYNTH_PROTOCOL, digest=split.digest,
        egosynth_root=split.root, n_frames=len(split.frames),
        sequences=split.sequences, requested_models=keys,
        skipped_models=[dict(model=k, state=s, detail=d) for k, s, d in skipped],
        config=dict(streams=datasets, views=views, protocols=["radial"],
                    datasets=datasets, theta_edges=list(edges),
                    radius_edges=list(radius_edges),
                    takes_per_dataset=a.egosynth_takes,
                    # The GT ships unfiltered on purpose, so the cut is part of
                    # the result and is written next to it, never left implicit.
                    sigma_max=a.egosynth_sigma_max,
                    sigma_column="inv_dist_std (1/m, scale-invariant)",
                    min_frame_points=EG.MIN_FRAME_POINTS,
                    gather=sorted(gathers),
                    # A bin's samples here are pooled SLAM points, not pixels of
                    # a dense map, so the figures' "too thin to read" ring needs
                    # its own floor — hundreds, not thousands.
                    thin_bin_px=1000.0,
                    depth_max_m=a.depth_max_m,
                    metric_max_depth=a.metric_max_depth,
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
    p.add_argument("--egosynth-root", default="",
                   help="ego-synth 5B root (aea/ nymeria/ egoexo4d/ oxford/ with "
                        "sparse_depth). Mutually exclusive with --adt-root; the "
                        "GT is a sparse SLAM point list, so only the radial "
                        "protocol runs. See docs/data/ego-synth-5b-sparse-depth.md")
    p.add_argument("--datasets", default=",".join(EG.DATASETS),
                   help="ego-synth datasets to score, as separate report rows: "
                        "scene scale differs by an order of magnitude across "
                        "them and every metric here is relative, so pooling "
                        "them into one row is the confound gt_median exists "
                        "to expose")
    p.add_argument("--egosynth-takes", type=int, default=8,
                   help="takes PER DATASET (0 = all). The release is 1 611 "
                        "takes / 24 931 clips; the cap is recorded in the "
                        "manifest, the report header and the split digest")
    p.add_argument("--egosynth-sigma-max", type=float,
                   default=EG.DEFAULT_SIGMA_MAX,
                   help="drop points whose MPS inv_dist_std (1/m) is at or above "
                        "this. The GT ships UNFILTERED by design; this cut is "
                        "written into results.json with every number it produced")
    p.add_argument("--egosynth-input-size", type=int, default=0,
                   help="feed frames at this size (0 = the model's own token "
                        "grid, matching the ADT path; 896 = the GT's own grid, "
                        "which makes the gather a literal pred[v, u])")
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
                   help="incidence-angle bin edges (deg); default is the "
                        "dataset's own cone — ADT 0,10,20,30,40,50,55 (the Aria "
                        "cone stops at 54.83) and ego-synth "
                        "0,10,20,30,40,50,58 (its valid mask admits no ray past "
                        "57.0; the render's corners are never imaged)")
    p.add_argument("--radius-edges", default="",
                   help="distance-from-optical-centre bin edges, in half-widths "
                        "(1.0 = middle of a frame edge, sqrt(2) = a corner); "
                        "default 0,0.2,0.4,0.6,0.8,1.0,1.45")
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
    p.add_argument("--analytic-bias", type=float, default=0.6,
                   help="radial bias injected by --models analytic, for "
                        "verifying that the harness reports what it is given")
    p.add_argument("--skip-unavailable", action="store_true",
                   help="run the models that are ready instead of refusing; the "
                        "report and results.json then name what was left out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--out", default="eval_out/fovbench")
    return p


def main() -> None:
    a = build_parser().parse_args()
    if a.egosynth_root and a.adt_root:
        raise SystemExit("[fovbench] pass --adt-root or --egosynth-root, not "
                         "both: they are different ground truths (a dense map "
                         "vs a sparse point list) and their digests are not "
                         "comparable.")
    if a.egosynth_root:
        run_egosynth(a)
        return
    if not a.adt_root and not a.manifest:
        raise SystemExit("[fovbench] pass --adt-root (or $ADT), "
                         "--egosynth-root, or --manifest")
    run(a)


if __name__ == "__main__":
    main()
