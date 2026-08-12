# Copyright (c) 2026.
"""Tables for the SLAM depth evaluation.

One table per baseline, one row per (model, dataset). No eccentricity axis, no
binning, no penalty ratio — those belong to the FOV experiment and appearing here
would invite the two to be read as one measurement.

Three things the tables say out loud, because each of them can turn a reading
upside down:

``coverage``   the share of ground-truth points that baseline could answer for,
               *before* the two arms were intersected. ``rect_derect`` loses the
               rim of the fisheye cone; a baseline that scores well on 60 % of
               the points has not beaten one that scored on all of them.

``gt_median``  what the rows were looking at. Every metric here is relative and
               grows with depth, and these datasets run from ~1.2 m indoors to
               ~5.3 m outdoors, so a worse number on Oxford is not yet a
               statement about the model.

**Absolute AbsRel is not comparable across models**, because Depth-Anything V2 is
scored under a disparity-space affine and the depth heads under a depth-space one
— each model's own protocol. It *is* comparable across baselines of one model,
which is the comparison this benchmark exists to make.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

import numpy as np

COLUMNS = ("AbsRel", "delta1", "delta2", "RMSE", "log10")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)


def _fmt(x, nd=3, width=9) -> str:
    if not _finite(x):
        return f"{'—':>{width}s}"
    return f"{x:>{width}.{nd}f}"


def _table(runs: List[dict], baseline: str) -> List[str]:
    sel = [r for r in runs if r["baseline"] == baseline]
    if not sel:
        return []
    head = (f"{'model':14s}{'dataset':11s}"
            + "".join(f"{c:>9s}" for c in COLUMNS)
            + f"{'cover':>8s}{'gt_med':>8s}{'frames':>8s}")
    lines = [f"  BASELINE · {baseline}", "  " + "-" * len(head),
             "  " + head, "  " + "-" * len(head)]
    for r in sorted(sel, key=lambda r: (r["model"], r["dataset"])):
        row = f"{r['model']:14s}{r['dataset']:11s}"
        for c in COLUMNS:
            row += _fmt(r.get(c))
        row += (_fmt(r.get("coverage"), 2, 8) + _fmt(r.get("gt_median"), 2, 8)
                + f"{r.get('n_frames', 0):>8d}")
        lines.append("  " + row)
    return lines + [""]


def _delta_table(runs: List[dict]) -> List[str]:
    """rect_derect against raw, per model and dataset — the actual question.

    A within-model ratio, so each model's own alignment protocol cancels and the
    column is comparable across models in a way absolute AbsRel is not.
    """
    by = {(r["model"], r["dataset"], r["baseline"]): r for r in runs}
    pairs = sorted({(m, d) for m, d, b in by})
    rows = [(m, d, by.get((m, d, "raw")), by.get((m, d, "rect_derect")))
            for m, d in pairs]
    rows = [t for t in rows if t[2] and t[3]]
    if not rows:
        return []
    head = (f"{'model':14s}{'dataset':11s}{'AbsRel raw':>12s}"
            f"{'rect_derect':>13s}{'ratio':>9s}{'cover raw':>11s}{'cover rd':>10s}")
    lines = ["  RECTIFY OR NOT · rect_derect / raw, within model "
             "(<1 means rectifying helped)",
             "  " + "-" * len(head), "  " + head, "  " + "-" * len(head)]
    for m, d, raw, rd in rows:
        ratio = (rd["AbsRel"] / raw["AbsRel"]
                 if _finite(raw.get("AbsRel")) and raw["AbsRel"] > 1e-9
                 and _finite(rd.get("AbsRel")) else float("nan"))
        lines.append("  " + f"{m:14s}{d:11s}" + _fmt(raw.get("AbsRel"), 4, 12)
                     + _fmt(rd.get("AbsRel"), 4, 13) + _fmt(ratio, 3, 9)
                     + _fmt(raw.get("coverage"), 2, 11)
                     + _fmt(rd.get("coverage"), 2, 10))
    return lines + [
        "  Both arms are scored on the points BOTH could answer for, so the",
        "  ratio is like-for-like; `cover` is what each gave up to get there and",
        "  is NOT in the ratio. A rect_derect that wins on 60% coverage has won",
        "  a different, smaller question.", ""]


def render(payload: dict) -> str:
    cfg = payload["config"]
    runs = payload["runs"]
    ds = payload["datasets"]
    out = [
        "=" * 92,
        f"  ego-synth SLAM depth · {payload['protocol']} · split {payload['digest']}",
        "=" * 92,
        f"  {payload['n_frames']} frames over {len(payload['takes'])} take(s), "
        f"datasets: {', '.join(ds)}",
        f"  GT: sparse MPS SLAM points, {cfg['gt_variant']} stream, native — the "
        f"harness rectifies nothing",
        f"  cut at {cfg['sigma_column']} < {cfg['sigma_max']} (the release ships "
        f"UNFILTERED, so this is part of every number below)",
        f"  baselines {cfg['baselines']} · {cfg['takes_per_dataset'] or 'all'} "
        f"take(s)/dataset · {cfg['n_frames_per_take']} frame(s)/take",
        "",
    ]
    for s in payload.get("skipped_models", []):
        out.append(f"  !! NOT RUN: {s['model']} ({s['state']}) — {s['detail']}")
    if payload.get("skipped_models"):
        out += [f"  !! {len(payload['skipped_models'])} of "
                f"{len(payload.get('requested_models', []))} requested models are "
                f"missing from every table below.", ""]
    if "rect_derect" in cfg.get("baselines", ()):
        verified = cfg.get("orientation_verified") or []
        unverified = [d for d in ds if d not in verified]
        if unverified:
            out += [f"  !! rect_derect ran on {unverified} whose sensor-to-upright "
                    f"rotation is UNVERIFIED. A quarter-turn error does not make",
                    f"  !! the score worse, it scores a different part of the "
                    f"image. Treat those rows as debugging output.", ""]
    out += [
        "  PROTOCOL: the scale (and shift) is fitted ONCE per frame over every one",
        "  of that frame's points, then frozen. Frames combine unweighted. Each",
        "  model keeps its own alignment space — disparity for Depth-Anything V2,",
        "  depth for the up-to-scale heads — so ABSOLUTE AbsRel IS NOT COMPARABLE",
        "  ACROSS MODELS. It is comparable across baselines of one model.",
        "",
    ]
    for b in cfg.get("baselines", ()):
        out += _table(runs, b)
    out += _delta_table(runs)

    models = sorted({r["model"] for r in runs})
    out += ["  MODELS", "  " + "-" * 60]
    for m in models:
        r = next(r for r in runs if r["model"] == m)
        out.append(f"  {m:14s}{r['family']} {r['size']} · {r['params_m']:.0f}M · "
                   f"align={r['align']} · frames at {r['input_size']}px")
    return "\n".join(out) + "\n"


def write_csv(payload: dict, path: str) -> str:
    cols = ["digest", "model", "family", "size", "params_m", "align",
            "input_size", "dataset", "baseline"] + list(COLUMNS) + [
        "coverage", "gt_median", "n_frames", "n_points_total"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in payload["runs"]:
            w.writerow([payload["digest"]] + [
                (f"{r[c]:.6f}" if _finite(r.get(c)) else
                 ("" if r.get(c) is None else r.get(c, "")))
                for c in cols[1:]])
    return path


def write_all(payload: dict, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    text = render(payload)
    print("\n" + text)
    txt = os.path.join(out_dir, "report.txt")
    with open(txt, "w") as fh:
        fh.write(text)
    return {"report": txt,
            "csv": write_csv(payload, os.path.join(out_dir, "results.csv"))}
