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


def _load_frame(frame, stream: str, depth_scale: float, depth_max_m: float,
                rotation_k: int = 3):
    """One ADT frame: upright fisheye RGB, planar-z GT in metres, validity.

    Mirrors ``datasets.adt.ADTFisheyeFrames.__getitem__`` — the same 270-deg-CCW
    rotation and the same mm->m scaling — but reads the paths the manifest
    froze, so both streams are guaranteed to be the same instant.
    """
    with Image.open(frame.rgb[stream]) as im:
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    d = np.load(frame.depth).astype(np.float32)
    if d.ndim == 3:
        d = d.squeeze(-1)
    d = np.where(np.isfinite(d), d, 0.0) * depth_scale
    if rotation_k:
        rgb = np.rot90(rgb, k=rotation_k).copy()
        d = np.rot90(d, k=rotation_k).copy()
    return rgb, d, (d > 0) & (d <= depth_max_m)


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #

def _score_radial(model, view: "G.FrameView", edges, max_depth) -> Optional[dict]:
    """Whole frame, one alignment fit, metrics per incidence-angle bin."""
    if view.valid.sum() < 256:
        return None
    pred = model.predict(view.rgb, gt_z=view.gt_z, theta_deg=view.theta)
    prof = G.radial_profile(pred, view.gt_z, view.valid, view.theta, edges,
                            model.align_mode, max_depth=max_depth)
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
               raw_scale_ratio=G.raw_scale_ratio(pred, win.gt_z, win.valid),
               theta_mean=float(win.theta[win.valid].mean()))
    return met


def _accumulate(store: Dict[str, list], key: str, value) -> None:
    if value is not None:
        store.setdefault(key, []).append(value)


def _mean_metrics(rows: List[dict], keys) -> dict:
    """Frame-mean of a metric list, skipping NaN — mirrors aggregate_metrics."""
    out = {}
    for k in keys:
        vals = [r[k] for r in rows
                if k in r and isinstance(r[k], (int, float)) and np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_frames"] = len(rows)
    out["n_valid_total"] = int(sum(r.get("n_bin", r.get("n_valid", 0)) for r in rows))
    return out


def _reduce_radial(runs: List[dict], edges) -> dict:
    """Average per-bin metrics across frames, keeping the bin structure."""
    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        rows = [r["bins"][i] for r in runs if r["bins"][i]["n_valid"] > 0]
        b = _mean_metrics(rows, G.METRIC_KEYS)
        b.update(theta_lo=lo, theta_hi=hi,
                 n_px_mean=float(np.mean([r["bins"][i]["n_bin"] for r in runs]))
                 if runs else 0.0)
        bins.append(b)
    return {"overall": _mean_metrics([r["overall"] for r in runs], G.METRIC_KEYS),
            "bins": bins,
            "in_cone_frac": float(np.mean([r["in_cone_frac"] for r in runs]))
            if runs else float("nan")}


def _reduce_windows(runs: List[dict], tilts) -> dict:
    """Average across frames and azimuths, keeping the tilt structure."""
    cells = []
    for t in tilts:
        rows = [r for r in runs if abs(r["tilt"] - t) < 1e-6]
        c = _mean_metrics(rows, G.METRIC_KEYS + ("in_cone_frac", "theta_mean"))
        c["tilt"] = t
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
    edges = tuple(float(x) for x in a.theta_edges.split(","))
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
    split = (Split.load(a.manifest) if a.manifest
             else build_split(a.adt_root, n_frames=a.n_frames,
                              streams={s: STREAMS[s] for s in streams}))
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
                rgb, gt, gt_valid = _load_frame(frame, stream, a.depth_scale,
                                                a.depth_max_m)
                cam = G.aria_cam(*rgb.shape[:2])
                for kind in views:
                    if "radial" in protocols:
                        fv = G.full_frame_view(rgb, gt, gt_valid, cam, n, kind)
                        _accumulate(acc, f"radial|{stream}|{kind}",
                                    _score_radial(model, fv, edges, a.metric_max_depth))
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
            body = (_reduce_radial(rows, edges) if protocol == "radial"
                    else _reduce_windows(rows, tilts))
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
                    theta_edges=list(edges), tilts=list(tilts),
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
    p.add_argument("--streams", default="synthetic,real")
    p.add_argument("--views", default="rect,fisheye",
                   help="rect = rectified perspective, fisheye = raw pixels")
    p.add_argument("--protocols", default="radial,window")
    p.add_argument("--theta-edges", default="0,10,20,30,40,50,55",
                   help="incidence-angle bin edges (deg); top edge = usable cone")
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
    if not a.adt_root and not a.manifest:
        raise SystemExit("[fovbench] pass --adt-root (or $ADT), or --manifest")
    run(a)


if __name__ == "__main__":
    main()
