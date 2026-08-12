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
one, because those are the protocols the models were built for. ``pen`` *is*
cross-comparable, being a within-model ratio, so the alignment protocol cancels.
``drift`` is a within-model ratio as well and so is comparable to other
``drift``s — but it is fitted differently from every other column, so it never
belongs in the same sentence as a ``pen``.
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

#: A window cell imaged less completely than this does not set ``pen``. The
#: sweep's whole design is that only the aim moves; a cell with its corners
#: outside the lens differs from the on-axis cell in dead area too, and 40 deg
#: of aim on a 40 deg square window measures 0.84 on this lens.
MIN_CLEAN_CONE_FRAC = 0.98

#: A bin standardisable in fewer than this share of frames gets no ``pen_ds``.
#: The frames that survive are the ones whose depth range happened to be wide
#: enough — a selected subset, not a sample — and a mean over them would read
#: like a full measurement. Half is a floor, not a blessing: check ``ds_frac``.
MIN_DS_FRAC = 0.5


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
    """``pen``/``drift`` plus the span they were measured over.

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
        return dict(pen=float("nan"), pen_ds=float("nan"), drift=float("nan"),
                    lo=float("nan"), hi=float("nan"), n_cells=len(cells),
                    clipped=clipped)
    a, b = cells[0], cells[-1]
    key = "bin_lo" if "bin_lo" in a else ("theta_lo" if "theta_lo" in a else "tilt")
    pen = (b["AbsRel"] / a["AbsRel"]) if a["AbsRel"] > 1e-9 else float("nan")
    # The same ratio after standardising both ends to the frame's depth mix.
    # nan whenever either end could not be standardised — a bin that missed a
    # depth stratum has no answer, and must not borrow the raw one.
    # ... and nan too when either end was standardisable in only a minority of
    # frames: the surviving frames are the ones whose depth range happened to be
    # wide enough, which is not a random subset of the run.
    pen_ds = float("nan")
    if _finite(a.get("AbsRel_ds")) and _finite(b.get("AbsRel_ds")) \
            and a["AbsRel_ds"] > 1e-9 \
            and min(a.get("ds_frac", 1.0), b.get("ds_frac", 1.0)) >= MIN_DS_FRAC:
        pen_ds = b["AbsRel_ds"] / a["AbsRel_ds"]
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
    return dict(pen=pen, pen_ds=pen_ds, drift=drift,
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
            + f"{'pen':>8s}{'pen_ds':>8s}{'drift*':>8s}")
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
        row += _fmt(s["pen"], 2, 8) + _fmt(s["pen_ds"], 2, 8) + _fmt(s["drift"], 3, 8)
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
        "  pen_ds = the same ratio after DEPTH STANDARDISATION: each bin re-scored on",
        "           the frame's own depth mix (quartile strata, direct standardisation)",
        "           so it is not rewarded or punished for how far away its content is.",
        "           AbsRel grows with depth and an egocentric frame's depth falls with",
        "           eccentricity, so pen carries some depth in it and pen_ds carries",
        "           less. It is a REDUCTION, not a removal — ~25% of a purely-depth",
        "           penalty still stands at four strata — so pen_ds well above 1.0 is",
        "           real, and pen_ds near 1.0 means 'mostly depth', not 'nothing here'.",
        "           `—` means a bin missed a depth stratum entirely, in at least half",
        "           the frames: what it would score at the other bins' depths is simply",
        "           not in this data. Averaging only the frames that did work would be",
        "           averaging the frames with the widest depth range — see ds_frac.",
        "  drift* = OUTSIDE THE PROTOCOL, and the only column that is. anchored_ratio",
        "           fits the model's own affine on the INNERMOST BIN ALONE, then takes",
        "           median(gt/pred) per bin; drift* is innermost / outermost, > 1 =",
        "           over-predicts depth toward the rim. It is kept because it separates",
        "           'the model bends depth with radius' from 'the model is just noisier",
        "           at the rim', which no whole-frame-fitted column can do — a global",
        "           affine spends itself on the radial trend and reads 0.965 for a real",
        "           +0.6*theta^2 bias. Read it as a diagnostic, not as a headline, and",
        "           never mix it with the columns above when quoting a protocol.",
        "           Radial only: window cells are separate forward passes of up-to-scale",
        "           models, so a window-to-window ratio compares two arbitrary constants.",
        "  Absolute AbsRel is NOT comparable across models (each keeps its own",
        "  alignment protocol); pen is, being a within-model ratio. drift* is a",
        "  within-model ratio too, but of a differently-fitted quantity — compare",
        "  it model to model only against other drift*, never against pen.",
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
                   + ["pen", "pen_ds", "drift"])
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
                              f"{s['pen_ds']:.4f}" if _finite(s["pen_ds"]) else "",
                              f"{s['drift']:.4f}" if _finite(s["drift"]) else ""])
    return path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

_STREAM_ORDER = {"synthetic": 0, "real": 1}
_VIEW_ORDER = {"fisheye": 0, "rect": 1}
_AXIS_NOTE = {
    "radius": ("each view measures radius in its OWN image plane: the same x is a "
               "different direction left vs right. Compare views on theta."),
    "window": "",
    "theta": ("theta is the ray direction, so left and right are like-for-like. "
              "Ringed = bin held up by a corner sliver."),
}

_WINDOW_NOTE = ("the window's FOV is held fixed and only its aim moves. "
                "Ringed = the lens clips that aim, so it is not a like-for-like "
                "point and does not set pen.")


def write_figures(payload: dict, out_dir: str) -> List[str]:
    """Per-bin curves against eccentricity, one panel per view x stream.

    A panel carries one curve per model and nothing else: eight curves in a
    shared panel could not be read, and the stream is the one axis where the
    levels genuinely differ (the sensor sets the level), so it gets its own row
    rather than a line style. Skipped without matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    os.makedirs(out_dir, exist_ok=True)
    # What counts as "too thin to be a measurement" is a count of samples, and
    # the two datasets count different things: ADT bins hold pixels of a dense
    # map (thousands), ego-synth bins hold pooled SLAM points (hundreds). One
    # constant cannot serve both, so the run states its own.
    thin_px = float(payload.get("config", {}).get("thin_bin_px", THIN_BIN_PX))
    written = []
    axes_spec = [("radial", "theta", "incidence angle from the optical axis (deg)"),
                 ("radial", "radius",
                  "distance from the optical centre (half-widths; 1.0 = edge midpoint)"),
                 ("window", "theta", "window aim (deg off-axis)")]
    for protocol, axis, xlab in axes_spec:
        # raw_scale_ratio, NOT scale_ratio: the latter is measured on the
        # aligned map and inherits the same bowl the alignment puts into
        # AbsRel, so plotting it would show the distorted column under the
        # figure that exists to show the undistorted one.
        for metric, ylab in (
                ("AbsRel", "AbsRel  (lower is better)"),
                ("delta1", r"$\delta_1$  (higher is better)"),
                ("AbsRel_ds", "AbsRel, depth-standardised  (lower is better)"),
                ("delta1_ds", r"$\delta_1$, depth-standardised"),
                ("raw_scale_ratio", "median(gt/pred), unaligned")):
            sel = [r for r in payload["runs"] if r["protocol"] == protocol]
            if axis == "radius":
                sel = [r for r in sel if "radius_bins" in r]
            if not sel:
                continue
            views = sorted({r["view"] for r in sel},
                           key=lambda v: _VIEW_ORDER.get(v, 9))
            streams = sorted({r["stream"] for r in sel},
                             key=lambda s: _STREAM_ORDER.get(s, 9))
            fig, grid = plt.subplots(len(streams), len(views),
                                     figsize=(5.4 * len(views), 3.6 * len(streams)),
                                     squeeze=False, sharey=True, sharex="col")
            models = sorted({r["model"] for r in sel})
            cmap = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(models)}
            for row, stream in enumerate(streams):
                for col, view in enumerate(views):
                    ax = grid[row][col]
                    for r in sorted(sel, key=lambda r: r["model"]):
                        if r["view"] != view or r["stream"] != stream:
                            continue
                        _, cells = _axis(r, axis)
                        xs = [(0.5 * (_lo(c) + _hi(c))
                               if protocol == "radial" else c["tilt"]) for c in cells]
                        ys = [c.get(metric, float("nan")) for c in cells]
                        pts = [(x, y) for x, y in zip(xs, ys)
                               if _finite(y) and _finite(x)]
                        if len(pts) < 2:
                            continue
                        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                                "-", marker="o", ms=4, lw=1.6,
                                color=cmap[r["model"]], label=r["model"])
                        # Ring the cells that are not clean measurements: a bin
                        # held up by a few corner pixels, and a window aim the
                        # lens clips. Both are still plotted — they happened —
                        # but neither may be read as a like-for-like point.
                        thin = [(x, y) for (x, y), c in zip(zip(xs, ys), cells)
                                if _finite(y)
                                and (c.get("n_px_mean", 1e9) < thin_px
                                     or c.get("in_cone_frac", 1.0)
                                     < MIN_CLEAN_CONE_FRAC)]
                        if thin:
                            ax.plot([t[0] for t in thin], [t[1] for t in thin], "o",
                                    ms=11, mfc="none", mec=cmap[r["model"]], mew=1.2)
                    ax.set_title(f"{view}  ·  {stream}", fontsize=10)
                    ax.grid(alpha=0.3)
                    if metric == "raw_scale_ratio":
                        ax.axhline(1.0, color="0.5", lw=0.8, zorder=0)
                    if row == len(streams) - 1:
                        ax.set_xlabel(xlab, fontsize=9)
                grid[row][0].set_ylabel(ylab, fontsize=9)
            # A run from before this metric existed draws no curves at all.
            # An empty PNG in the output directory reads as "measured, and
            # nothing there"; no PNG reads as "not measured". Say the latter.
            if not any(ax.lines for r in grid for ax in r):
                plt.close(fig)
                continue
            handles, labels = grid[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", ncol=len(models),
                           fontsize=8, frameon=False)
            note = (_WINDOW_NOTE if protocol != "radial"
                    else _AXIS_NOTE.get(axis, ""))
            fig.suptitle(f"ADT-FOV · {metric} vs {axis} · {protocol} protocol · "
                         f"split {payload['digest']}\n{note}", fontsize=10)
            fig.tight_layout(rect=(0, 0.06, 1, 0.93))
            name = f"{protocol}_{metric}" + ("_radius" if axis == "radius" else "")
            path = os.path.join(out_dir, f"{name}.png")
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
