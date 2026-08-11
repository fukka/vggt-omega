# Copyright (c) 2026.
"""Tables and figures for the ADT-FOV test.

Two summary numbers carry the result, and they answer different questions:

``pen``   *eccentricity penalty* — AbsRel in the outermost populated bin divided
          by AbsRel in the innermost. How much worse is the periphery than the
          centre, in the metric a user reads.

``drift`` *radial scale drift* — ``anchored_ratio`` (median gt/pred after the
          model's own global affine is fitted **on the innermost bin alone**) in
          the innermost bin divided by the outermost. Above 1.0 the model
          over-predicts depth toward the rim relative to the centre; below 1.0 it
          under-predicts. ``radial`` protocol only — see ``summarise``.

Report both, because they can disagree and the disagreement is not noise. The
per-bin AbsRel is a *residual after a global fit*, and a least-squares affine is
free to choose the radius at which it is right — so a cleanly monotone radial
error comes out U-shaped, worst at both ends, and ``pen`` (an end-to-end ratio)
can read close to 1.0 while the model is in fact bending badly.

Measured, on the analytic stand-in given a known ``+0.6 theta^2`` bias: the
radial AbsRel curve comes out a bowl — ``0.175 0.153 0.112 0.047 0.081 0.172`` on
one small run — so ``pen`` reads 0.98, "the rim is as good as the centre", for a
model wrong by 50% at the rim. ``drift`` on the same input recovers the injected
bias; ``tests/test_end_to_end.py`` asserts exactly that, against a value derived
independently from the bin geometry, and needs no data to reproduce. ``pen`` says
how the periphery feels to a downstream user after they fit a scale; ``drift``
says what the model actually did.

Absolute AbsRel is **not** comparable across models here: Depth-Anything V2 is
scored under a disparity-space affine and the depth heads under a depth-space
one, because those are the protocols the models were built for. The columns that
*are* cross-comparable are ``pen`` and ``drift`` — both are within-model ratios,
so the alignment protocol cancels.
"""
from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fovbench import geometry as G

#: A bin thinner than this is a corner artefact, not a measurement.
MIN_BIN_FRAMES = 1
MIN_BIN_PX = 64.0

#: Below this many pixels a bin is corner slivers rather than a measurement, and
#: figures ring it. Set between the rectified rim bin (2874-2940 px on the first
#: real run — image corners only) and the innermost bin (4825-4945 px), which is
#: small for an honest geometric reason and IS a measurement.
THIN_BIN_PX = 3500.0


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)


def _populated(bins: Sequence[dict]) -> List[dict]:
    return [b for b in bins
            if b.get("n_frames", 0) >= MIN_BIN_FRAMES
            and b.get("n_px_mean", MIN_BIN_PX) >= MIN_BIN_PX
            and _finite(b.get("AbsRel"))]


def summarise(run: dict) -> dict:
    """``pen``/``drift`` plus the span they were measured over.

    Returns NaNs rather than guessing when fewer than two bins are populated —
    which is the normal state of the rectified arm's outer bins, and must read
    as "not measurable here" rather than as a good score.
    """
    cells = _populated(run["bins"] if run["protocol"] == "radial" else run["cells"])
    if len(cells) < 2:
        return dict(pen=float("nan"), drift=float("nan"),
                    lo=float("nan"), hi=float("nan"), n_cells=len(cells))
    a, b = cells[0], cells[-1]
    key = "theta_lo" if run["protocol"] == "radial" else "tilt"
    pen = (b["AbsRel"] / a["AbsRel"]) if a["AbsRel"] > 1e-9 else float("nan")
    drift = float("nan")
    # `drift` exists only for the radial protocol. Its bins come from ONE forward
    # pass, so the model's arbitrary global scale is shared and cancels in a
    # bin-to-bin ratio. Every window is a SEPARATE forward pass, and every model
    # here is up to scale, so a window-to-window ratio compares two arbitrary
    # constants: measured on the first real run, the same models' per-window raw
    # ratios sit at 2.87-3.14 with no anchoring possible between them. Reporting
    # a number there would be reporting the models' self-scaling as distortion.
    if run["protocol"] == "radial" \
            and _finite(a.get("anchored_ratio")) and _finite(b.get("anchored_ratio")) \
            and abs(b["anchored_ratio"]) > 1e-9:
        drift = a["anchored_ratio"] / b["anchored_ratio"]
    return dict(pen=pen, drift=drift, lo=a[key], hi=b[key], n_cells=len(cells))


# --------------------------------------------------------------------------- #
# Text tables
# --------------------------------------------------------------------------- #

def _axis(run: dict) -> Tuple[List[str], List[dict]]:
    if run["protocol"] == "radial":
        return ([f"{int(b['theta_lo'])}-{int(b['theta_hi'])}" for b in run["bins"]],
                run["bins"])
    return ([f"t{int(c['tilt'])}" for c in run["cells"]], run["cells"])


def _fmt(x, nd=3, width=7) -> str:
    if x is None or not _finite(x):
        return f"{'—':>{width}s}"
    return f"{x:>{width}.{nd}f}"


def _table(runs: List[dict], protocol: str, view: str, metric: str) -> List[str]:
    sel = [r for r in runs if r["protocol"] == protocol and r["view"] == view]
    if not sel:
        return []
    cols, _ = _axis(sel[0])
    unit = "deg off-axis" if protocol == "radial" else "window aim (deg off-axis)"
    head = (f"{'model':14s}{'stream':10s}" + "".join(f"{c:>8s}" for c in cols)
            + f"{'pen':>8s}{'drift':>8s}")
    lines = [f"  {protocol.upper()} · {view} · {metric}   ({unit})", "  " + "-" * len(head),
             "  " + head, "  " + "-" * len(head)]
    for r in sorted(sel, key=lambda r: (r["model"], r["stream"])):
        _, cells = _axis(r)
        s = summarise(r)
        row = f"{r['model']:14s}{r['stream']:10s}"
        for c in cells:
            row += _fmt(c.get(metric), 3, 8) if c.get("n_frames", 0) else f"{'—':>8s}"
        row += _fmt(s["pen"], 2, 8) + _fmt(s["drift"], 3, 8)
        lines.append("  " + row)
    return lines + [""]


def _coverage_note(runs: List[dict]) -> List[str]:
    """The rectified arm's thin outer bins are a finding, so say so with numbers."""
    rad = [r for r in runs if r["protocol"] == "radial"]
    if not rad:
        return []
    lines = ["  COVERAGE · mean valid pixels per incidence bin (radial protocol)",
             "  " + "-" * 82]
    cols, _ = _axis(rad[0])
    lines.append("  " + f"{'view':10s}{'stream':10s}{'px':>6s}"
                 + "".join(f"{c:>9s}" for c in cols))
    # Rows depend on (view, stream) AND the render size — views go to each model
    # at its own token grid (512 for VGGT-Omega, 518 elsewhere), so counts are
    # NOT shared across models. Collapsing on (view, stream) alone would print
    # whichever model happened to run first and label it as everyone's.
    seen = set()
    for r in sorted(rad, key=lambda r: (r["view"], r["stream"], r["input_size"])):
        tag = (r["view"], r["stream"], r["input_size"])
        if tag in seen:
            continue
        seen.add(tag)
        row = f"{r['view']:10s}{r['stream']:10s}{r['input_size']:>6d}"
        for b in r["bins"]:
            row += f"{int(b.get('n_px_mean', 0)):>9d}"
        lines.append("  " + row)
    return lines + [
        "  A rectified ~85 deg pinhole has no pixels past 42.3 deg off-axis except",
        "  in its corners (52.6 deg at most). Empty outer bins there are geometry,",
        "  not model failure — and are themselves the cost of rectifying.", ""]


def render_report(payload: dict) -> str:
    runs = payload["runs"]
    cfg = payload["config"]
    out = [
        "=" * 88,
        f"  ADT-FOV test · {payload['protocol']} · split {payload['digest']}",
        "=" * 88,
        f"  {payload['n_frames']} frames over {len(payload['sequences'])} sequence(s): "
        f"{', '.join(payload['sequences'])}",
        f"  streams {cfg['streams']} · views {cfg['views']} · protocols {cfg['protocols']}",
        f"  window FOV {cfg['window_fov']} deg held fixed; aims {cfg['tilts']} deg "
        f"x azimuths {cfg['azimuths']}",
        f"  GT valid <= {cfg['depth_max_m']} m; predictions beyond "
        f"{cfg['metric_max_depth']} m excluded from the metrics",
        "",
    ]
    # An incomplete line-up goes at the TOP, not in the footer: a two-model
    # table is otherwise indistinguishable from a four-model one.
    for s in payload.get("skipped_models", []):
        out.append(f"  !! NOT RUN: {s['model']} ({s['state']}) — {s['detail']}")
    if payload.get("skipped_models"):
        out += [f"  !! {len(payload['skipped_models'])} of "
                f"{len(payload.get('requested_models', []))} requested models are "
                f"missing from every table below.", ""]
    out += [
        "  pen   = AbsRel(outermost bin) / AbsRel(innermost)  — how much worse the periphery is",
        "  drift = anchored_ratio(innermost) / anchored_ratio(outermost), the model's",
        "          own affine fitted on the innermost bin alone.  > 1 = over-predicts",
        "          depth toward the rim.  Radial protocol ONLY: window cells are",
        "          separate forward passes of up-to-scale models, so a window-to-window",
        "          ratio compares two arbitrary constants and is left blank.",
        "  Absolute AbsRel is NOT comparable across models (each keeps its own",
        "  alignment protocol); pen and drift are, being within-model ratios.",
        "",
    ]
    for protocol in ("radial", "window"):
        for view in ("rect", "fisheye"):
            for metric in ("AbsRel", "delta1"):
                out += _table(runs, protocol, view, metric)
    out += _coverage_note(runs)

    models = sorted({r["model"] for r in runs})
    out += ["  MODELS", "  " + "-" * 60]
    for m in models:
        r = next(r for r in runs if r["model"] == m)
        out.append(f"  {m:14s}{r['family']} {r['size']} · {r['params_m']:.0f}M · "
                   f"align={r['align']} · views at {r['input_size']}px")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #

#: One definition, shared with the driver (see ``geometry.METRIC_KEYS``).
_CSV_METRICS = G.METRIC_KEYS


def write_csv(payload: dict, path: str) -> str:
    """One flat row per (model, stream, view, protocol, cell)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["digest", "model", "family", "size", "params_m", "align",
                    "protocol", "stream", "view", "cell", "theta_lo", "theta_hi",
                    "tilt", "n_frames", "n_px_mean"] + list(_CSV_METRICS)
                   + ["pen", "drift"])
        for r in payload["runs"]:
            s = summarise(r)
            _, cells = _axis(r)
            for c in cells:
                label = (f"{int(c['theta_lo'])}-{int(c['theta_hi'])}"
                         if r["protocol"] == "radial" else f"tilt{int(c['tilt'])}")
                w.writerow([payload["digest"], r["model"], r["family"], r["size"],
                            f"{r['params_m']:.2f}", r["align"], r["protocol"],
                            r["stream"], r["view"], label,
                            c.get("theta_lo", ""), c.get("theta_hi", ""),
                            c.get("tilt", ""), c.get("n_frames", 0),
                            f"{c.get('n_px_mean', 0):.0f}"]
                           + [f"{c[k]:.6f}" if _finite(c.get(k)) else ""
                              for k in _CSV_METRICS]
                           + [f"{s['pen']:.4f}" if _finite(s["pen"]) else "",
                              f"{s['drift']:.4f}" if _finite(s["drift"]) else ""])
    return path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def write_figures(payload: dict, out_dir: str) -> List[str]:
    """AbsRel and scale_ratio against eccentricity. Skipped without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    os.makedirs(out_dir, exist_ok=True)
    written = []
    styles = {"synthetic": "-", "real": "--"}
    for protocol in ("radial", "window"):
        # raw_scale_ratio, NOT scale_ratio: the latter is measured on the
        # aligned map and inherits the same bowl the alignment puts into
        # AbsRel, so plotting it would show the distorted column under the
        # figure that exists to show the undistorted one.
        for metric, ylab in (("AbsRel", "AbsRel  (lower is better)"),
                             ("delta1", r"$\delta_1$  (higher is better)"),
                             ("raw_scale_ratio", "median(gt/pred), unaligned")):
            sel = [r for r in payload["runs"] if r["protocol"] == protocol]
            if not sel:
                continue
            views = sorted({r["view"] for r in sel})
            fig, axes = plt.subplots(1, len(views), figsize=(5.6 * len(views), 4.4),
                                     squeeze=False, sharey=True)
            models = sorted({r["model"] for r in sel})
            cmap = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(models)}
            for ax, view in zip(axes[0], views):
                for r in sorted(sel, key=lambda r: (r["model"], r["stream"])):
                    if r["view"] != view:
                        continue
                    _, cells = _axis(r)
                    xs = [(0.5 * (c["theta_lo"] + c["theta_hi"])
                           if protocol == "radial" else c["tilt"]) for c in cells]
                    ys = [c.get(metric, float("nan")) for c in cells]
                    pts = [(x, y) for x, y in zip(xs, ys)
                           if _finite(y) and _finite(x)]
                    if len(pts) < 2:
                        continue
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            styles.get(r["stream"], ":"), marker="o", ms=4,
                            color=cmap[r["model"]],
                            label=f"{r['model']} · {r['stream']}")
                    # A bin held up by a few corner pixels is not a measurement;
                    # ring it so the rectified rim cannot be read as one.
                    thin = [(x, y, c) for (x, y), c in zip(zip(xs, ys), cells)
                            if _finite(y) and c.get("n_px_mean", 1e9) < THIN_BIN_PX]
                    if thin:
                        ax.plot([t[0] for t in thin], [t[1] for t in thin], "o",
                                ms=11, mfc="none", mec=cmap[r["model"]], mew=1.2)
                ax.set_title(f"{protocol} · {view}")
                ax.set_xlabel("incidence angle from the optical axis (deg)"
                              if protocol == "radial" else "window aim (deg off-axis)")
                ax.grid(alpha=0.3)
            axes[0][0].set_ylabel(ylab)
            if metric == "raw_scale_ratio":
                for ax in axes[0]:
                    ax.axhline(1.0, color="0.5", lw=0.8, zorder=0)
            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
                           frameon=False)
            fig.suptitle(f"ADT-FOV · {metric} vs eccentricity · split "
                         f"{payload['digest']}", fontsize=10)
            fig.tight_layout(rect=(0, 0.10, 1, 0.95))
            path = os.path.join(out_dir, f"{protocol}_{metric}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
    return written


def write_all(payload: dict, out_dir: str) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    text = render_report(payload)
    print("\n" + text)
    txt = os.path.join(out_dir, "report.txt")
    with open(txt, "w") as fh:
        fh.write(text)
    return {"report": txt,
            "csv": write_csv(payload, os.path.join(out_dir, "results.csv")),
            "figures": write_figures(payload, os.path.join(out_dir, "figures"))}
