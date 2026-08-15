# Copyright (c) 2026.
"""The FOV question, asked of ego-synth 5B's SLAM points.

    python -m slambench.run_fov --egosynth-root $EGOSYNTH --calib-root $CALIB \
        --models vggt_omega --baselines raw,rect_derect --protocols radial

The grid is

    model x baseline x context x dataset x protocol

and every arm of it is reduced to the same primitive: a **field position x
distance** table of pooled sums (:class:`slambench.fov.Table`). ``fov.py`` is
where the protocol and its one control are written down; this file is the loop
around it.

Why this is a second driver and not a flag on ``run``
-----------------------------------------------------
``run.py`` says of itself, and has said since it was written, that it has no
eccentricity axis and does no binning. Three published artefacts were produced
under that contract. Adding a binning mode to it would either break the contract
or bury the FOV question behind a flag nobody reads, and it would put the
already-published path at risk of an edit for a question it was not asked. So
this driver **imports** ``run``'s pieces — the split, the reader, the camera,
the baselines, the model registry — and shares every one of them, while owning
nothing but the binning.

What it needs that ``run`` does not
-----------------------------------
**A camera on both arms.** ``raw`` predicts without a camera model, but it
cannot be *asked* where in the field a point is without one, so ``--calib-root``
is required here even for a raw-only run. That is not a change of protocol: the
camera is used to describe the ground truth, never to alter a prediction.

Two things this run must be read with
--------------------------------------
**The pooled column is confounded and the standardised one is the answer.**
Distance falls by 3.6x from the centre of this data's field to its rim and every
metric here is relative, so the pooled ``AbsRel`` against eccentricity is partly
a distance curve. ``fov.py`` computes both and the report prints them side by
side; the gap between them is a result in its own right.

**Rectifying is also a resampling, and the resampling runs with eccentricity.**
``raw`` decimates the 896 frame by a flat 1.73x through ``cv2.INTER_AREA``,
which filters. ``rect_derect`` resamples it through ``cv2.remap`` /
``INTER_LINEAR``, which does not, by a factor running from 2.13x on axis to
0.78x at 55 deg — so the rectified image is *more* aliased than the raw one in
the middle and *sharper* at the rim, before any model has seen it. That is a
difference between the arms that is not the lens model, it is shaped like the
effect being measured, and no run in this package controls for it. It is written
here rather than in the results because it is a property of the harness.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from slambench import _REPO  # noqa: F401  (import registers sys.path)

from slambench import baselines as B      # noqa: E402
from slambench import camera as C         # noqa: E402
from slambench import data as D           # noqa: E402
from slambench import fov as F            # noqa: E402
from slambench import fov_report as FR    # noqa: E402
from slambench import metrics as MT       # noqa: E402
from slambench import models as M         # noqa: E402
from slambench import split as S          # noqa: E402

RADIAL = "radial"
WINDOW = "window"
PROTOCOLS = (RADIAL, WINDOW)


def _load_camera(calib_root: str, dataset: str, take: str, verify: bool
                 ) -> C.Fisheye624:
    path = C.calibration_path(calib_root, dataset, take)
    if not os.path.isfile(path):
        raise C.CalibrationUnavailable(
            f"no camera model at {path}. This run needs one on every arm — the "
            f"eccentricity of a point is a question about the lens — so unlike "
            f"`run --baselines raw` there is no camera-free path here. Fetch it "
            f"with tools/fetch_egosynth_calibration.py (ticket 012).")
    cam = C.load(path, dataset=dataset, take=take, out_size=D.RES)
    return C.require_verified(cam) if verify else cam


def _edges(spec: str, default) -> List[float]:
    out = [float(t) for t in str(spec).split(",") if t.strip()]
    if len(out) < 2 or any(b <= a for a, b in zip(out, out[1:])):
        raise SystemExit(f"[slambench] --theta-edges {spec!r}: expected two or "
                         f"more increasing degrees, e.g. "
                         f"{','.join(str(int(e)) for e in default)}")
    return out


def _floats(spec: str, flag: str) -> List[float]:
    out = sorted({float(t) for t in str(spec).split(",") if t.strip()})
    if not out:
        raise SystemExit(f"[slambench] {flag}: expected one or more numbers")
    return out


# --------------------------------------------------------------------------- #
# The distance strata, from ground truth alone
# --------------------------------------------------------------------------- #

def depth_strata(sp: "S.Split", clips, sigma_max: float, strata: int
                 ) -> Dict[str, List[float]]:
    """Per-dataset distance quantiles, in a pre-pass over the split's GT.

    Model-independent by construction — it opens the npz files and nothing else —
    so it leaks nothing into the scores it will stratify, and it is cheap:
    ``read_points`` is ~23 ms a frame against minutes a frame for a model.

    Per dataset rather than pooled because scene scale differs by an order of
    magnitude across the four, and a stratum that is "near" indoors is the whole
    of an outdoor frame.
    """
    by_ds: Dict[str, List[np.ndarray]] = {}
    for ds, _take, _clip, npz, _video, frs in clips:
        for f in frs:
            pts = D.read_points(npz, f.index, sigma_max=sigma_max)
            if len(pts) >= D.MIN_FRAME_POINTS:
                by_ds.setdefault(ds, []).append(pts.d)
    return {ds: F.depth_edges_from(v, strata) for ds, v in by_ds.items()}


# --------------------------------------------------------------------------- #
# The two protocols
# --------------------------------------------------------------------------- #

def _radial_frame(arm_objs, ctxs, windows, frames, pts, theta, model,
                  tables, ds, cfg) -> int:
    """One frame through every (arm, context), binned by incidence angle.

    The support intersection is ``run``'s, unchanged and for its reason: an arm
    that answered on fewer points would otherwise be compared against a
    different point set rather than a different treatment. Here it also fixes
    the *field* the comparison covers, which is why the default bins stop at
    55 deg — see ``fov.py``.
    """
    preds = {}
    for n in ctxs:
        idx, tgt = windows[n]
        stack = [frames[j] for j in idx]
        for arm in arm_objs:
            preds[(arm, n)] = arm_objs[arm].predict(stack, pts, tgt)
    support = np.ones(len(pts), bool)
    with np.errstate(invalid="ignore"):
        for p in preds.values():
            support &= np.isfinite(p) & (p > 0)
    thin = 0
    for (arm, n), p in preds.items():
        cells = F.frame_cells(np.where(support, p, np.nan), pts.d, theta,
                              model.align_mode, cfg["theta_edges"],
                              cfg["depth_edges"][ds],
                              max_depth=cfg["max_depth"],
                              min_points=cfg["min_points"])
        if cells is None:
            thin += 1
            continue
        tables[(ds, RADIAL, arm, n)].add(cells)
    return thin


def _window_frame(views, azimuths, ctxs, windows, frames, pts, model, tables,
                  ds, cfg) -> int:
    """One frame through every re-aimed window.

    Each window is its own little evaluation: the model sees that window and
    nothing else, the depth comes back about the *window's* axis and is
    converted to the camera's (``fov.Window.z_to_camera`` — the step a co-axial
    rectification does not need and this one cannot skip), and the score lands
    in the row for that window's tilt.

    Windows are scored independently rather than on a common support: they
    deliberately do not see the same points, because seeing a different part of
    the field is the whole of what a window is.
    """
    thin = 0
    for (tilt, az), view in views.items():
        for n in ctxs:
            idx, tgt = windows[n]
            stack = [view.render(frames[j]) for j in idx]
            raw = np.asarray(model.predict_stack(stack, tgt, gt=None))
            if raw.ndim != 2:
                raise SystemExit(
                    f"[slambench] {model.key!r} answers per point, so it cannot "
                    f"be run through the {WINDOW!r} protocol, which samples a "
                    f"rendered window.")
            p = view.sample(raw, pts)
            # The window's tilt IS the field-position coordinate here.
            coord = np.full(len(pts), float(tilt))
            cells = F.frame_cells(p, pts.d, coord, model.align_mode,
                                  cfg["tilt_edges"], cfg["depth_edges"][ds],
                                  max_depth=cfg["max_depth"],
                                  min_points=cfg["min_window_points"])
            if cells is None:
                thin += 1
                continue
            # A window at tilt 0 is the same window whichever azimuth it is
            # labelled with — there is nothing to rotate about — so it is
            # rendered once and entered into every azimuth's row. Without that,
            # each azimuth's sweep would start at a different tilt and their
            # ratios would not be against a common origin.
            for label in (azimuths if tilt == 0 else [az]):
                tables[(ds, WINDOW, f"az{int(label)}", n)].add(cells)
    return thin


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run(a: argparse.Namespace) -> dict:
    os.makedirs(a.out, exist_ok=True)
    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    arms = [b.strip() for b in a.baselines.split(",") if b.strip()]
    protos = [p.strip() for p in a.protocols.split(",") if p.strip()]
    for bad, good, flag in ((set(arms) - set(B.BASELINES), B.BASELINES, "--baselines"),
                            (set(protos) - set(PROTOCOLS), PROTOCOLS, "--protocols")):
        if bad:
            raise SystemExit(f"[slambench] unknown {flag} {sorted(bad)}; "
                             f"choose from {list(good)}")
    if not a.calib_root:
        raise SystemExit(
            "[slambench] run_fov needs --calib-root on every arm, including "
            "raw: binning by eccentricity is a question about the lens, and "
            "there is no nominal Aria calibration to fall back on. See "
            "tools/fetch_egosynth_calibration.py (ticket 012).")

    keys = [k.strip() for k in a.models.split(",") if k.strip()]
    ctxs = sorted({int(t) for t in str(a.context_frames).split(",") if t.strip()})
    if not ctxs or ctxs[0] < 1:
        raise SystemExit(f"[slambench] --context-frames {a.context_frames!r}")
    theta_edges = _edges(a.theta_edges, F.DEFAULT_THETA_EDGES)
    tilts = _floats(a.tilts, "--tilts")
    azimuths = _floats(a.azimuths, "--azimuths")
    # One bin per tilt, centred on it, so the window rows are the sweep itself.
    step = min((b - x for x, b in zip(tilts, tilts[1:])), default=10.0)
    tilt_edges = [tilts[0] - step / 2] + [x + step / 2 for x in tilts]

    ready, skipped = M.available(keys)
    for key, state, detail in skipped:
        print(f"[slambench] {key}: {state} — {detail}")
    if skipped and not a.skip_unavailable:
        raise SystemExit(f"[slambench] {len(skipped)} of {len(keys)} models "
                         f"cannot run; fix them or pass --skip-unavailable")
    if not ready:
        raise SystemExit("[slambench] no model is runnable; --models analytic "
                         "gives a weight-free harness run")
    if max(ctxs) > 1:
        mono = [k for k in ready if not M.takes_context(k)]
        if mono:
            raise SystemExit(
                f"[slambench] {mono} are monocular and cannot take a "
                f"{max(ctxs)}-frame context; run them at --context-frames 1")

    sp = (S.Split.load(a.manifest) if a.manifest
          else S.build(a.egosynth_root, datasets, a.n_frames, a.takes))
    if sp.protocol != S.PROTOCOL:
        raise SystemExit(f"[slambench] --manifest {a.manifest} was written by "
                         f"{sp.protocol!r}, not {S.PROTOCOL!r}")
    sp.save(os.path.join(a.out, "manifest.json"))
    clips = sp.by_clip()

    print("[slambench] distance strata, from ground truth only …")
    edges_by_ds = (dict.fromkeys(sp.datasets, _edges(a.depth_edges, (0.0, 1.0)))
                   if a.depth_edges else
                   depth_strata(sp, clips, a.sigma_max, a.depth_strata))
    for ds, e in sorted(edges_by_ds.items()):
        print(f"  {ds:10s} " + "  ".join(f"{x:.2f}" for x in e[1:-1]) + " m")

    device = None
    if [k for k in ready if k not in M.STANDINS]:
        import torch
        device = torch.device(a.device)
        if device.type != "cuda":
            print("[slambench] WARNING: real weights on CPU — minutes per frame")

    cfg = {"theta_edges": theta_edges, "tilt_edges": tilt_edges,
           "depth_edges": edges_by_ds, "max_depth": a.max_depth,
           "min_points": a.min_points, "min_window_points": a.min_window_points}

    tables: Dict[tuple, F.Table] = {}
    window_cover: Dict[tuple, List[float]] = {}
    runs: List[dict] = []
    for key in ready:
        print(f"\n[slambench] ══ {key} ══")
        t0 = time.time()
        model = M.load_model(key, device, checkpoint=a.omega_checkpoint,
                             bias=a.oracle_bias, noise=a.oracle_noise)
        MT.check_protocol(key, model.align_mode)
        print(f"[slambench]   {model.family} {model.size} | "
              f"align={model.align_mode} | frames at {model.input_size}px")
        thin = 0
        for ci, (ds, take, clip, npz, video, frs) in enumerate(clips):
            cam = _load_camera(a.calib_root, ds, take, not a.allow_unverified)
            arm_objs = {n: B.build(n, model, cam, a.rect_fov) for n in arms}
            views = {}
            if WINDOW in protos:
                for tilt in tilts:
                    for az in (azimuths if tilt else azimuths[:1]):
                        w = F.Window(model.input_size, a.window_fov, tilt, az)
                        v = F.WindowView(cam, w)
                        window_cover.setdefault((ds, tilt, az), []).append(
                            v.in_cone_frac)
                        if v.in_cone_frac >= F.MIN_IN_CONE_FRAC:
                            views[(tilt, az)] = v
            wins, want = {}, set()
            for f in frs:
                for n in ctxs:
                    w = S.context_window(f.clip_frames, f.index, n,
                                         a.context_stride)
                    wins[(f.index, n)] = w
                    want.update(w[0])
            frames = D.decode_frames(video, sorted(want))
            for f in frs:
                pts = D.read_points(npz, f.index, sigma_max=a.sigma_max)
                if len(pts) < D.MIN_FRAME_POINTS:
                    thin += 1
                    continue
                windows = {n: wins[(f.index, n)] for n in ctxs}
                if RADIAL in protos:
                    theta = F.theta_of(cam, pts)
                    for arm in arms:
                        for n in ctxs:
                            tables.setdefault(
                                (ds, RADIAL, arm, n),
                                F.Table(theta_edges, edges_by_ds[ds], "theta"))
                    thin += _radial_frame(arm_objs, ctxs, windows, frames, pts,
                                          theta, model, tables, ds, cfg)
                if views:
                    for az in azimuths:
                        for n in ctxs:
                            tables.setdefault(
                                (ds, WINDOW, f"az{int(az)}", n),
                                F.Table(tilt_edges, edges_by_ds[ds], "tilt"))
                    thin += _window_frame(views, azimuths, ctxs, windows,
                                          frames, pts, model, tables, ds, cfg)
            if (ci + 1) % max(1, a.log_every) == 0:
                print(f"[slambench]   clip {ci + 1}/{len(clips)} "
                      f"({time.time() - t0:.0f}s)")
        for (ds, proto, arm, n), tab in sorted(tables.items()):
            if tab.n_frames:
                runs.append(dict(model=key, family=model.family, size=model.size,
                                 align=model.align_mode,
                                 input_size=model.input_size, dataset=ds,
                                 protocol=proto, arm=arm, context=n,
                                 **tab.to_json()))
        tables.clear()
        if thin:
            print(f"[slambench]   {thin} frame/window(s) too thin to score")
        del model
        if device is not None and device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
        print(f"[slambench]   done in {time.time() - t0:.0f}s")

    payload = dict(
        protocol=S.PROTOCOL, digest=sp.digest, egosynth_root=sp.root,
        n_frames=len(sp.frames), datasets=sp.datasets, takes=sp.takes,
        requested_models=keys,
        skipped_models=[dict(model=k, state=s, detail=d) for k, s, d in skipped],
        config=dict(
            protocols=protos, baselines=arms, datasets=datasets,
            takes_per_dataset=sp.takes_per_dataset,
            n_frames_per_take=sp.n_frames_per_take,
            context_frames=ctxs, context_stride=int(a.context_stride),
            theta_edges=theta_edges, depth_edges=edges_by_ds,
            depth_strata=a.depth_strata, min_cell_points=F.MIN_CELL_POINTS,
            window_fov=a.window_fov, tilts=tilts, azimuths=azimuths,
            tilt_edges=tilt_edges, min_in_cone_frac=F.MIN_IN_CONE_FRAC,
            sigma_max=a.sigma_max,
            sigma_column="inv_dist_std (1/m, scale-invariant)",
            gt_variant=D.VARIANT, max_depth=a.max_depth,
            min_points=a.min_points, min_window_points=a.min_window_points,
            rect_fov=a.rect_fov, calib_root=a.calib_root,
            orientation_verified=list(C.VERIFIED_ROTATION)),
        window_coverage=[dict(dataset=ds, tilt=t, azimuth=az,
                              in_cone_frac=float(np.mean(v)),
                              scored=bool(np.mean(v) >= F.MIN_IN_CONE_FRAC))
                         for (ds, t, az), v in sorted(window_cover.items())],
        runs=runs)
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    FR.write_all(payload, a.out)
    print(f"\n[slambench] wrote {a.out}/results.json (+ csv, report.txt)")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--egosynth-root", default=os.environ.get("EGOSYNTH", ""))
    p.add_argument("--calib-root", default=os.environ.get("EGOSYNTH_CALIB", ""),
                   help="per-take camera models (ticket 012). REQUIRED here on "
                        "every arm, including raw: eccentricity is a question "
                        "about the lens")
    p.add_argument("--manifest", default=None,
                   help="reuse a frozen split — pass run's own manifest to bin "
                        "exactly the frames it scored")
    p.add_argument("--datasets", default=",".join(D.DATASETS))
    p.add_argument("--models", default=",".join(M.DEFAULT_MODELS))
    p.add_argument("--baselines", default=B.RAW,
                   help=f"{B.RAW} and/or {B.RECT_DERECT}. Adding "
                        f"{B.RECT_DERECT} truncates every arm's field at "
                        f"~55 deg, because the support is intersected")
    p.add_argument("--protocols", default=RADIAL,
                   help=f"{RADIAL} (bin one whole frame by incidence) and/or "
                        f"{WINDOW} (re-aim a fixed window). {WINDOW} costs one "
                        f"forward pass per window per frame")
    p.add_argument("--theta-edges",
                   default=",".join(str(int(e)) for e in F.DEFAULT_THETA_EDGES),
                   help="incidence bin edges in degrees. The default stops at "
                        "55 where the rectified arm does; a raw-only run may go "
                        "to 70, where the frame has only corners left")
    p.add_argument("--depth-strata", type=int, default=F.DEFAULT_DEPTH_STRATA,
                   help="equal-population distance strata, cut from the split's "
                        "own ground truth. This is the control: distance falls "
                        "3.6x from the centre of this field to its rim")
    p.add_argument("--depth-edges", default="",
                   help="fixed strata in metres instead of quantiles; the "
                        "standardisation weights then stop being uniform")
    p.add_argument("--window-fov", type=float, default=F.DEFAULT_WINDOW_FOV,
                   help="window width in degrees. HELD FIXED across the sweep "
                        "on purpose — varying width and aim together makes the "
                        "dead-pixel fraction move with the variable under test")
    p.add_argument("--tilts", default=",".join(str(int(t)) for t in F.DEFAULT_TILTS))
    p.add_argument("--azimuths",
                   default=",".join(str(int(t)) for t in F.DEFAULT_AZIMUTHS),
                   help="four azimuths at each tilt are a control on each "
                        "other: the lens is nominally radially symmetric, so "
                        "they should agree, and where they do not the thin "
                        "prism terms or the scene are talking")
    p.add_argument("--n-frames", type=int, default=25)
    p.add_argument("--takes", type=int, default=8)
    p.add_argument("--sigma-max", type=float, default=D.DEFAULT_SIGMA_MAX)
    p.add_argument("--max-depth", type=float, default=MT.MAX_DEPTH_M)
    p.add_argument("--min-points", type=int, default=D.MIN_FRAME_POINTS,
                   help="a frame with fewer usable points is not scored; the "
                        "affine is fitted over the whole frame")
    p.add_argument("--min-window-points", type=int, default=128,
                   help="the same floor for one window, which sees a fraction "
                        "of the frame and so cannot carry the frame's")
    p.add_argument("--rect-fov", type=float, default=B.DEFAULT_RECT_FOV_DEG)
    p.add_argument("--context-frames", default="1")
    p.add_argument("--context-stride", type=int, default=1)
    p.add_argument("--allow-unverified", action="store_true")
    p.add_argument("--oracle-bias", type=float, default=0.0,
                   help="multiplicative error injected by --models oracle. It "
                        "is the same at every eccentricity and every distance, "
                        "so BOTH curves must come back flat")
    p.add_argument("--oracle-noise", type=float, default=0.0,
                   help="error in METRES injected by --models oracle — the one "
                        "control worth running on this data. A fixed error in "
                        "metres is a relative error that grows as the ground "
                        "truth gets nearer, and the rim of this field is 3.6x "
                        "nearer than its centre, so the pooled curve MUST slope "
                        "and the distance-held-fixed curve MUST NOT. That is "
                        "the confound caught on the real data rather than in a "
                        "unit test")
    p.add_argument("--omega-checkpoint", default=None)
    p.add_argument("--skip-unavailable", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--out", default="eval_out/slambench-fov")
    return p


def main() -> None:
    a = build_parser().parse_args()
    if not a.egosynth_root and not a.manifest:
        raise SystemExit("[slambench] pass --egosynth-root (or $EGOSYNTH), "
                         "or --manifest")
    run(a)


if __name__ == "__main__":
    main()
