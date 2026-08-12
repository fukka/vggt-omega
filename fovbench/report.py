# Copyright (c) 2026.
"""Tables and figures for the ADT-FOV test.

The experiment reports one thing: **how each baseline's depth accuracy varies
with where in the field of view the content sits.** That is two metrics —
``AbsRel`` and ``delta1`` — against two readings of "where":

  ``theta``   the incidence angle of the ray off the optical axis. The physical
              direction, and the only axis on which the rectified and raw arms
              mean the same thing.
  ``radius``  distance from the optical centre in the image plane, in half
              widths. Where in the *picture*. Each view measures it in its own
              image plane, so the same radius is a different direction in each.

Both are read at two resolutions off one frozen alignment fit: six coarse bins
for the tables, and a continuous profile (1 deg, 0.025 half-widths) for the
curves. **Both are pooled over frames, weighted by pixels** — the coarse bins by
``run._mean_metrics`` and the profile by ``geometry.pool_profiles`` — so the
dots and the line are the same estimator at two resolutions and any gap between
them is the bin width, not the arithmetic.

``pen`` summarises a curve as AbsRel(outermost bin) / AbsRel(innermost). It is a
within-model ratio, so it is comparable across models even though absolute
AbsRel is comparable only within an alignment protocol: VGGT-1B, VGGT-Omega and
DA3 share a depth-space affine and can be read against each other directly,
while Depth-Anything V2 is scored in disparity space because that is the
protocol it was built for.

``pen`` is not on its own a statement about field position. Every metric here is
relative and so grows with depth, so a bin that is nearer scores worse for that
reason alone. The per-bin GT depth is therefore reported beside every curve —
measured on the same frames, never modelled — and the reader is left to weigh
it. Nothing here attempts to correct for it.

Three figures come out, and only three: ``AbsRel``, ``delta1``, ``gt_depth``.
Each carries every model, both views, both streams and both axes.
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

#: A window cell imaged less completely than this does not set ``pen``. The
#: sweep's whole design is that only the aim moves; a cell with its corners
#: outside the lens differs from the on-axis cell in dead area too, and 40 deg
#: of aim on a 40 deg square window measures 0.84 on this lens.
MIN_CLEAN_CONE_FRAC = 0.98

#: A fine profile bin below this many pooled pixels is a spike, not a curve
#: point, and is not drawn. Small, because the profiles are pooled over every
#: frame: at 200 frames even a 1 deg bin on the axis carries thousands.
PROFILE_MIN_PX = 200


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)


def _populated(bins: Sequence[dict]) -> List[dict]:
    return [b for b in bins
            if b.get("n_frames", 0) >= MIN_BIN_FRAMES
            and b.get("n_px_mean", MIN_BIN_PX) >= MIN_BIN_PX
            and _finite(b.get("AbsRel"))]


def _fully_imaged(cells: Sequence[dict]) -> List[dict]:
    """Window cells the lens images completely.

    The window sweep holds the FOV fixed and moves only the aim, so that width
    and eccentricity cannot move together — that confound made an earlier sweep
    in this repo unreadable. But a fixed 40 deg *square* window has a 27.2 deg
    half-diagonal, so from an aim of 30 deg its corners already leave the Aria
    cone (54.83 deg) and at 40 deg it is 84% imaged: the remaining 16% is black
    wedge. `pen` across that step would again be comparing two windows that
    differ in dead area as well as in aim.

    So the ratio spans fully-imaged cells only. Clipped cells are still scored
    and still printed — they are the honest cost of aiming a fixed window that
    far off-axis — but they are marked, and they do not set `pen`.
    """
    return [c for c in cells
            if c.get("in_cone_frac", 1.0) >= MIN_CLEAN_CONE_FRAC]


def summarise(run: dict) -> dict:
    """``pen`` plus the span it was measured over.

    Returns NaNs rather than guessing when fewer than two bins are populated —
    which is the normal state of the rectified arm's outer bins, and must read
    as "not measurable here" rather than as a good score.
    """
    cells = _populated(_cells_of(run, run.get("_axis", "theta")))
    # A window sweep compares aims, so every cell in the ratio must differ ONLY
    # in aim. Cells the lens clips are dropped from the ratio (not from the
    # tables) — see ``_fully_imaged``.
    clipped = 0
    if run["protocol"] != "radial":
        clean = _fully_imaged(cells)
        clipped = len(cells) - len(clean)
        cells = clean
    if len(cells) < 2:
        return dict(pen=float("nan"), drift=float("nan"),
                    lo=float("nan"), hi=float("nan"),
                    n_cells=len(cells), clipped=clipped)
    a, b = cells[0], cells[-1]
    key = "bin_lo" if "bin_lo" in a else ("theta_lo" if "theta_lo" in a else "tilt")
    pen = (b["AbsRel"] / a["AbsRel"]) if a["AbsRel"] > 1e-9 else float("nan")
    # `drift` is no longer part of the ADT-FOV experiment and no ADT table
    # prints it: `bin_by` stopped emitting `anchored_ratio`, so it is NaN there
    # by construction rather than by a flag anyone could flip back. It survives
    # here only for `datasets_egosynth`, whose pooled path sets `anchored_ratio`
    # itself and reads this. Radial only — every window is a separate forward
    # pass of an up-to-scale model, so a window-to-window ratio would compare
    # two arbitrary constants.
    drift = float("nan")
    if run["protocol"] == "radial" \
            and _finite(a.get("anchored_ratio")) \
            and _finite(b.get("anchored_ratio")) \
            and abs(b["anchored_ratio"]) > 1e-9:
        drift = a["anchored_ratio"] / b["anchored_ratio"]
    return dict(pen=pen, drift=drift,
                lo=a[key], hi=b[key], n_cells=len(cells), clipped=clipped)


# --------------------------------------------------------------------------- #
# Text tables
# --------------------------------------------------------------------------- #

def _lo(c: dict) -> float:
    """Bin lower edge. ``bin_lo`` is the current key; ``theta_lo`` is what runs
    before the radius axis existed wrote, and those JSONs are still read."""
    return float(c.get("bin_lo", c.get("theta_lo", c.get("tilt", 0.0))))


def _hi(c: dict) -> float:
    return float(c.get("bin_hi", c.get("theta_hi", c.get("tilt", 0.0))))


def _cells_of(run: dict, axis: str = "theta") -> List[dict]:
    """The run's cells on one axis, or ``[]`` if it has none.

    A run can genuinely lack an axis. ego-synth's raw-fisheye arm ships no
    camera model, so an incidence angle is not computable for it at all and it
    is binned by radius only — an empty list here, and a skipped row in the
    theta tables, rather than a fabricated column.
    """
    if run["protocol"] != "radial":
        return run.get("cells", [])
    return run.get("radius_bins", []) if axis == "radius" else run.get("bins", [])


def _axis(run: dict, axis: str = "theta") -> Tuple[List[str], List[dict]]:
    cells = _cells_of(run, axis)
    if run["protocol"] != "radial":
        # A clipped aim is flagged in the column head, where the number is read.
        return ([f"t{int(c['tilt'])}"
                 + ("!" if c.get("in_cone_frac", 1.0) < MIN_CLEAN_CONE_FRAC else "")
                 for c in cells], cells)
    if axis == "radius":
        return ([f"{_lo(b):.1f}-{_hi(b):.1f}" for b in cells], cells)
    return ([f"{int(_lo(b))}-{int(_hi(b))}" for b in cells], cells)


def _fmt(x, nd=3, width=7) -> str:
    if x is None or not _finite(x):
        return f"{'—':>{width}s}"
    return f"{x:>{width}.{nd}f}"


def _table(runs: List[dict], protocol: str, view: str, metric: str,
           axis: str = "theta") -> List[str]:
    # Runs with no cells on this axis are dropped before the header is built,
    # not after: the column labels come from the first surviving run, and an
    # axis-less run would otherwise hand the table an empty column list.
    sel = [r for r in runs if r["protocol"] == protocol and r["view"] == view
           and _cells_of(r, axis)]
    if not sel:
        return []
    cols, _ = _axis(sel[0], axis)
    unit = ("half-widths from the optical centre" if axis == "radius" else
            ("deg off-axis" if protocol == "radial"
             else "window aim (deg off-axis)"))
    head = (f"{'model':14s}{'stream':10s}" + "".join(f"{c:>8s}" for c in cols)
            + f"{'pen':>8s}")
    lines = [f"  {protocol.upper()} · {view} · {metric}"
             + (" · by radius" if axis == "radius" else "")
             + f"   ({unit})", "  " + "-" * len(head),
             "  " + head, "  " + "-" * len(head)]
    for r in sorted(sel, key=lambda r: (r["model"], r["stream"])):
        _, cells = _axis(r, axis)
        s = summarise(dict(r, _axis=axis))
        row = f"{r['model']:14s}{r['stream']:10s}"
        for c in cells:
            row += _fmt(c.get(metric), 3, 8) if c.get("n_frames", 0) else f"{'—':>8s}"
        row += _fmt(s["pen"], 2, 8)
        lines.append("  " + row)
    return lines + [""]


def _coverage_note(runs: List[dict]) -> List[str]:
    """The rectified arm's thin outer bins are a finding, so say so with numbers."""
    rad = [r for r in runs if r["protocol"] == "radial"]
    if not rad:
        return []
    # Prefer the incidence-angle axis; fall back to radius for a run that has no
    # theta at all (ego-synth's raw fisheye ships no camera model). Mixing the
    # two in one table would put `tan theta` and `theta` in the same column.
    axis = "theta" if any(_cells_of(r, "theta") for r in rad) else "radius"
    rad = [r for r in rad if _cells_of(r, axis)]
    if not rad:
        return []
    label = "incidence bin" if axis == "theta" else "radius bin"
    lines = [f"  COVERAGE · mean valid samples per {label} (radial protocol)",
             "  " + "-" * 82]
    cols, _ = _axis(rad[0], axis)
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
        for b in _cells_of(r, axis):
            row += f"{int(b.get('n_px_mean', 0)):>9d}"
        lines.append("  " + row)
    return lines + [
        "  A rectified ~85 deg pinhole has no pixels past 42.3 deg off-axis except",
        "  in its corners (52.6 deg at most). Empty outer bins there are geometry,",
        "  not model failure — and are themselves the cost of rectifying.",
        ""] + _bin_depth_note(rad)


def _window_geometry_note(runs: List[dict]) -> List[str]:
    """What each window aim actually got: how much of it the lens images, and
    how many raw pixels sit behind one of its pixels.

    Both are pure geometry — identical for every model at a given render size —
    and both are confounds that move with the swept variable, so they belong
    beside the numbers rather than in the JSON only. ``in_cone`` below 1.0 is
    black wedge inside the window; ``src_px`` below 1.0 means the window is
    upsampled, i.e. the on-axis aims are the *soft* ones, not the rim.
    """
    win = [r for r in runs if r["protocol"] != "radial" and r.get("cells")]
    if not win:
        return []
    cols, _ = _axis(win[0])
    lines = ["  WINDOW GEOMETRY · what each aim was actually handed",
             "  " + "-" * 78,
             "  " + f"{'view':10s}{'px':>5s}{'':>9s}"
             + "".join(f"{c:>8s}" for c in cols)]
    seen = set()
    for r in sorted(win, key=lambda r: (r["view"], r["input_size"])):
        tag = (r["view"], r["input_size"])
        if tag in seen:
            continue
        seen.add(tag)
        for lab, key, nd in (("in_cone", "in_cone_frac", 3),
                             ("src_px", "src_px_per_out_px", 2)):
            row = f"{r['view']:10s}{r['input_size']:>5d}{lab:>9s}"
            for c in r["cells"]:
                row += _fmt(c.get(key), nd, 8)
            lines.append("  " + row)
    return lines + [
        "  A `!` on a column means that aim is NOT fully imaged, and it is left out",
        "  of `pen`: a 40 deg square window has a 27.2 deg half-diagonal, so from an",
        "  aim of 30 deg its corners leave the 54.83 deg Aria cone. Those cells are",
        "  still scored — they are the real cost of aiming that far off-axis — but a",
        "  ratio across them would compare two windows differing in dead area as",
        "  well as in aim, which is the confound this sweep exists to avoid.", ""]


def _bin_depth_note(rad: List[dict]) -> List[str]:
    """Median GT depth per bin — the confound, printed so it can be checked.

    Every metric in this report is relative, and a relative error grows with
    depth. So "the rim is worse" is a statement about field position only if
    the bins are at comparable depths; in an egocentric indoor frame they need
    not be. This table is model-independent (it is the GT), so it collapses on
    the GT itself: one row per distinct depth profile, labelled with the streams
    that share it. On ADT both streams read one depth map and collapse to a
    single row, as before. On ego-synth the ``stream`` column is the *dataset*,
    and their scene scales differ by an order of magnitude — ~1.2 m median
    indoors against ~5.3 m at Oxford with a 23 m p99 — so they do not collapse,
    and must not: pooling four scene scales into one row is exactly the confound
    this table exists to expose.
    """
    axis = "theta" if any(_cells_of(r, "theta") for r in rad) else "radius"
    have = [r for r in rad if any("gt_median" in b for b in _cells_of(r, axis))]
    if not have:
        return []
    cols, _ = _axis(have[0], axis)
    label = "incidence bin" if axis == "theta" else "radius bin"
    lines = [f"  BIN DEPTH · median GT depth per {label} (m) — a confound, not a score",
             "  " + "-" * 82,
             "  " + f"{'view':10s}{'stream':16s}{'px':>6s}"
             + "".join(f"{c:>9s}" for c in cols)]
    seen: Dict[tuple, List[str]] = {}
    for r in sorted(have, key=lambda r: (r["view"], r["input_size"], r["stream"])):
        prof = tuple(round(b["gt_median"], 3) if _finite(b.get("gt_median"))
                     else None for b in _cells_of(r, axis))
        tag = (r["view"], r["input_size"], prof)
        if r["stream"] not in seen.setdefault(tag, []):
            seen[tag].append(r["stream"])
    for (view, px, prof), streams in seen.items():
        row = f"{view:10s}{'+'.join(streams)[:15]:16s}{px:>6d}"
        for v in prof:
            row += (f"{v:>9.2f}" if v is not None else f"{'—':>9s}")
        lines.append("  " + row)
    return lines + [
        "  Read the AbsRel tables against this row: a bin that is both farther and",
        "  worse has not yet been shown to be worse *because of* where it sits.", ""]


def render_report(payload: dict) -> str:
    runs = payload["runs"]
    cfg = payload["config"]
    seqs = payload["sequences"]
    out = [
        "=" * 88,
        f"  {'ego-synth' if payload['protocol'].startswith('egosynth') else 'ADT'}"
        f"-FOV test · {payload['protocol']} · split {payload['digest']}",
        "=" * 88,
        f"  {payload['n_frames']} frames over {len(seqs)} sequence(s): "
        + (", ".join(seqs) if len(seqs) <= 6
           else f"{', '.join(seqs[:5])}, … (+{len(seqs) - 5} more)"),
        f"  streams {cfg['streams']} · views {cfg['views']} · protocols {cfg['protocols']}",
    ]
    # The window protocol re-renders an angular window out of the raw fisheye,
    # which needs a fisheye camera model. ego-synth ships none, so its runs are
    # radial-only and have no window line to print.
    if "window" in cfg.get("protocols", ()):
        out.append(f"  window FOV {cfg['window_fov']} deg held fixed; aims "
                   f"{cfg['tilts']} deg x azimuths {cfg['azimuths']}")
    if "sigma_max" in cfg:
        out += [f"  GT is a sparse SLAM point list, cut at "
                f"{cfg['sigma_column']} < {cfg['sigma_max']} — the release ships "
                f"UNFILTERED, so this cut is part of every number below",
                f"  {cfg.get('takes_per_dataset', 0) or 'all'} take(s) per dataset; "
                f"prediction read at the points by: {', '.join(cfg.get('gather', ())) or 'n/a'}"]
    out += [
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
        "  PROTOCOL: the scale (and shift) is fitted ONCE per frame, over every valid",
        "  pixel, and then frozen; binning is a masking step applied afterwards. Every",
        "  cell above — AbsRel, delta1, RMSE, and `pen` — obeys that.",
        "",
        "  pen    = AbsRel(outermost bin) / AbsRel(innermost)  — how much worse the",
        "           periphery is, in the metric a downstream user reads.",
        "  Absolute AbsRel is comparable only WITHIN an alignment protocol — see the",
        "  MODELS block: models sharing an `align=` may be read against each other,",
        "  one scored differently may not. pen is comparable across all of them,",
        "  being a within-model ratio.",
        "",
    ]
    for protocol in ("radial", "window"):
        for view in ("rect", "fisheye"):
            for metric in ("AbsRel", "delta1"):
                out += _table(runs, protocol, view, metric)
            # An arm with no incidence-angle axis at all would otherwise print
            # nothing: ego-synth's raw fisheye ships no camera model, so theta
            # is not computable for it and radius is the only axis it has. Give
            # it its own table, labelled by radius — never silently in the theta
            # tables, where the same number would mean a different direction.
            if protocol == "radial" and not any(
                    _cells_of(r, "theta") for r in runs
                    if r["protocol"] == protocol and r["view"] == view):
                for metric in ("AbsRel", "delta1"):
                    out += _table(runs, protocol, view, metric, axis="radius")
    out += _coverage_note(runs)
    out += _window_geometry_note(runs)

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
    """One flat row per (model, stream, view, protocol, cell).

    The ``axis`` column names what ``theta_lo``/``theta_hi`` measure on that
    row. It is incidence angle wherever incidence angle exists, and radius on a
    run that has no theta at all — ego-synth's raw fisheye, which ships no
    camera model. Without the column those two would be the same pair of
    numbers meaning different things, which is the confusion
    `fovbench/README.md` spends a paragraph warning about.
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["digest", "model", "family", "size", "params_m", "align",
                    "protocol", "stream", "view", "axis", "cell",
                    "theta_lo", "theta_hi",
                    "tilt", "n_frames", "n_px_mean"] + list(_CSV_METRICS)
                   + ["pen"])
        for r in payload["runs"]:
            axis = "theta" if _cells_of(r, "theta") else "radius"
            s = summarise(dict(r, _axis=axis))
            _, cells = _axis(r, axis)
            for c in cells:
                label = (f"{_lo(c):g}-{_hi(c):g}"
                         if r["protocol"] == "radial" else f"tilt{int(c['tilt'])}")
                w.writerow([payload["digest"], r["model"], r["family"], r["size"],
                            f"{r['params_m']:.2f}", r["align"], r["protocol"],
                            r["stream"], r["view"], axis, label,
                            _lo(c), _hi(c),
                            c.get("tilt", ""), c.get("n_frames", 0),
                            f"{c.get('n_px_mean', 0):.0f}"]
                           + [f"{c[k]:.6f}" if _finite(c.get(k)) else ""
                              for k in _CSV_METRICS]
                           + [f"{s['pen']:.4f}" if _finite(s["pen"]) else "",
                              ])
    return path


# --------------------------------------------------------------------------- #
# Figures — three, and only three
# --------------------------------------------------------------------------- #

_STREAM_ORDER = {"synthetic": 0, "real": 1}
_VIEW_ORDER = {"fisheye": 0, "rect": 1}

#: The four panels every figure carries, left to right: both views on the ray
#: axis, then both views on the image-plane axis. Reading across a row is the
#: whole experiment for one stream.
_PANELS = (("fisheye", "theta"), ("rect", "theta"),
           ("fisheye", "radius"), ("rect", "radius"))

_AXIS_LABEL = {"theta": "incidence angle (deg)",
               "radius": "distance from optical centre (half-widths)"}

#: Rendered per metric: the key in a bin/profile, the axis label, and whether
#: lower is better (used only for the caption).
_FIGURES = (
    ("AbsRel", "AbsRel   (lower is better)"),
    ("delta1", r"$\delta_1$   (higher is better)"),
)


def _profile_of(run: dict, axis: str) -> Optional[dict]:
    p = (run.get("profiles") or {}).get(axis)
    return p if p and p.get("n") else None


def _bin_points(run: dict, axis: str, key: str):
    """Coarse-bin centres and values.

    Nothing is flagged or hidden: how much a bin is holding up is a question the
    pixel-count panel of the depth figure answers directly, for every bin at
    once, instead of a ring that says "be careful" without saying how careful.
    """
    cells = _cells_of(run, axis)
    xs, ys = [], []
    for c in cells:
        if not _finite(c.get(key)) or not c.get("n_frames", 0):
            continue
        xs.append(0.5 * (_lo(c) + _hi(c)))
        ys.append(c[key])
    return xs, ys


def write_figures(payload: dict, out_dir: str) -> List[str]:
    """Three pictures: AbsRel, delta1, and the GT depth they were measured on.

    Each is one page carrying every model, both views, both streams and both
    axes — because the experiment's question ("how does accuracy vary with
    position in the field") is not answered by any one of those slices alone,
    and splitting it across a dozen files invites quoting one panel as the
    result.

    The line is the **continuous profile** where a run has one; the markers are
    the **coarse bins**. Showing both is deliberate: the curve has the shape and
    the bins are what the tables quote, and where they disagree the reader
    should see it rather than be told. Bins held up by corner slivers are drawn
    hollow, and a run with no profile falls back to the bins alone.

    Skipped entirely without matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    rad = [r for r in payload["runs"] if r["protocol"] == "radial"]
    if not rad:
        return []
    os.makedirs(out_dir, exist_ok=True)
    streams = sorted({r["stream"] for r in rad},
                     key=lambda s: _STREAM_ORDER.get(s, 9))
    models = sorted({r["model"] for r in rad})
    written = []

    for key, ylab in _FIGURES:
        fig, grid = plt.subplots(len(streams), len(_PANELS),
                                 figsize=(4.6 * len(_PANELS), 3.6 * len(streams)),
                                 squeeze=False, sharey=True)
        cmap = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(models)}
        for row, stream in enumerate(streams):
            for col, (view, axis) in enumerate(_PANELS):
                ax = grid[row][col]
                for r in sorted(rad, key=lambda r: r["model"]):
                    if r["view"] != view or r["stream"] != stream:
                        continue
                    colour = cmap[r["model"]]
                    prof = _profile_of(r, axis)
                    if prof:
                        x = np.asarray(prof["centre"], float)
                        y = np.asarray(prof.get(key, []), float)
                        n = np.asarray(prof["n"], float)
                        keep = np.isfinite(y) & (n >= PROFILE_MIN_PX)
                        if keep.sum() >= 2:
                            ax.plot(x[keep], y[keep], "-", lw=1.7, color=colour,
                                    label=r["model"], zorder=3)
                    bx, by = _bin_points(r, axis, key)
                    if bx:
                        # Without a profile the bins ARE the curve, so join them.
                        ax.plot(bx, by, marker="o", ms=5, color=colour,
                                ls="none" if prof else "--", lw=1.2, zorder=4,
                                label=None if prof else r["model"])
                ax.set_title(f"{view} · {axis}", fontsize=10)
                ax.grid(alpha=0.3)
                if row == len(streams) - 1:
                    ax.set_xlabel(_AXIS_LABEL[axis], fontsize=8.5)
            grid[row][0].set_ylabel(f"{stream}\n{ylab}", fontsize=9)
        handles, labels = grid[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=len(models),
                       fontsize=9, frameon=False)
        fig.suptitle(
            f"ADT-FOV · {key} vs position in the field · split "
            f"{payload['digest']} · {payload['n_frames']} frames\n"
            "line = 1-deg continuous profile, dots = the six binned values; both "
            "pooled over frames and pixel-weighted, so they differ only by bin "
            "width.  theta is comparable across views; radius is not.",
            fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 0.90))
        path = os.path.join(out_dir, f"{key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    written += _write_depth_figure(payload, rad, out_dir, streams, plt)
    return written


def _write_depth_figure(payload, rad, out_dir, streams, plt) -> List[str]:
    """The third picture: what the other two were divided by, and how many
    pixels said so.

    **Top row — the GT depth.** Not a score and not a model: the measured depth
    of the same frames, with a +/- 1 sigma band, because a bin whose depth
    ranges over metres and one that is a flat wall tell very different stories
    about the same median. It is here because AbsRel and delta1 are *relative*,
    so a bin that is nearer scores worse for that reason alone, and a rise in
    the first two figures is a statement about field position only insofar as
    this row is flat.

    **Bottom row — the pixel count, as a density.** How much each part of the
    curve is actually holding up. It replaces the ring the metric figures used
    to draw around thin bins: a ring says "be careful" without saying how
    careful, and it could only mark the coarse bins, whereas this shows the
    whole continuous axis. Read the two metric figures against it — where this
    row collapses, the curve above it is a handful of image corners.

    Density rather than a raw count, because the coarse bins are 10 deg wide and
    the profile's are 1 deg: plotting both counts on one axis would put a factor
    of ten between two things that agree exactly. The bars are annotated with
    their totals, which are the numbers the tables quote.

    GT is shared by every model and both streams, so one curve per view is the
    whole story; the columns are view x axis.
    """
    seen, picks = set(), []
    for r in sorted(rad, key=lambda r: (r["view"], r["input_size"])):
        if r["view"] in seen:
            continue
        seen.add(r["view"])
        picks.append(r)
    if not picks:
        return []
    panels = [(r, axis) for axis in ("theta", "radius") for r in picks]
    fig, grid = plt.subplots(2, len(panels), figsize=(4.6 * len(panels), 6.2),
                             squeeze=False, sharex="col")
    any_drawn = False
    for col, (r, axis) in enumerate(panels):
        top, bot = grid[0][col], grid[1][col]
        prof = _profile_of(r, axis)
        drew = False
        if prof and "gt_std" in prof:
            x = np.asarray(prof["centre"], float)
            m = np.asarray(prof["gt_mean"], float)
            sd = np.asarray(prof["gt_std"], float)
            n = np.asarray(prof["n"], float)
            k = np.isfinite(m) & (n >= PROFILE_MIN_PX)
            if k.sum() >= 2:
                top.fill_between(x[k], (m - sd)[k], (m + sd)[k], alpha=0.22,
                                 color="#1f77b4", lw=0)
                top.plot(x[k], m[k], "-", lw=2.0, color="#1f77b4",
                         label="pooled mean $\\pm$ 1 s.d.")
                # DENSITY, not a count: the coarse bins are 10 deg wide and
                # the profile's are 1 deg, so raw counts on one axis would put
                # a factor of ten between two things that agree. Per frame, and
                # per unit of x, both are the same quantity.
                nf = max(int(prof.get("n_frames", 1)), 1)
                ew = np.diff(np.asarray(prof["edges"], float))
                dens = n / nf / ew
                bot.fill_between(x[k], 0, dens[k], alpha=0.30,
                                 color="#1f77b4", lw=0)
                bot.plot(x[k], dens[k], "-", lw=1.6, color="#1f77b4",
                         label="continuous profile")
                drew = any_drawn = True
        bx, by = _bin_points(r, axis, "gt_median")
        if bx:
            any_drawn = True
            top.plot(bx, by, "o--" if not drew else "o", ms=5, lw=1.2,
                     color="#c1272d", label="binned median")
        cells = [c for c in _cells_of(r, axis) if c.get("n_frames", 0)]
        if cells:
            xs = [0.5 * (_lo(c) + _hi(c)) for c in cells]
            ns = [c.get("n_px_mean", 0.0) for c in cells]
            wd = [(_hi(c) - _lo(c)) for c in cells]
            dens = [n / max(w, 1e-9) for n, w in zip(ns, wd)]
            bot.bar(xs, dens, width=[0.9 * w for w in wd], color="#c1272d",
                    alpha=0.28, edgecolor="#c1272d", lw=0.8,
                    label="the six bins")
            # The annotation is the bin's TOTAL, which is the number the tables
            # quote; the bar's height is that total spread over the bin's width.
            for xx, dd, nn in zip(xs, dens, ns):
                bot.annotate(f"{nn:,.0f}".replace(",", " "), (xx, dd),
                             textcoords="offset points", xytext=(0, 3),
                             ha="center", fontsize=6.5, color="#8a1b20")
        top.set_title(f"{r['view']} · {axis}", fontsize=10)
        for ax in (top, bot):
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)
        bot.set_xlabel(_AXIS_LABEL[axis], fontsize=8.5)
    if not any_drawn:
        plt.close(fig)
        return []
    grid[0][0].set_ylabel("ground-truth depth (m)", fontsize=9)
    grid[1][0].set_ylabel("valid pixels per frame, per unit x\n"
                          "(bars annotated with the bin total)", fontsize=8.5)
    h, l = [], []
    for ax in (grid[0][0], grid[1][0]):
        hh, ll = ax.get_legend_handles_labels()
        h += hh
        l += ll
    if h:
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle(
        f"ADT-FOV · what the curves were divided by, and how many pixels said so"
        f" · split {payload['digest']} · {payload['n_frames']} frames\n"
        "measured on the scored frames, identical for every model and both "
        "streams. Top: AbsRel and $\\delta_1$ are relative, so a nearer bin "
        "scores worse for that reason alone. Bottom: where this collapses, the "
        "curve above is a few image corners.", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    path = os.path.join(out_dir, "gt_depth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


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
