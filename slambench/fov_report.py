# Copyright (c) 2026.
"""Tables for the FOV run.

Three of them, and the order is the argument:

    1. FIELD PROFILE   AbsRel against field position, pooled and standardised
                       side by side. The pooled column is the naive answer and
                       is confounded by distance; the standardised one is the
                       same points with distance held fixed. Printing them
                       together is the point — a reader who sees only one of
                       them cannot tell which they are looking at.
    2. BIN DISTANCE    median-ish GT depth per bin. Model-independent, so it is
                       printed once per (dataset, protocol) rather than per
                       model: it is the confound itself, laid out so the first
                       table can be checked against it by eye.
    3. THE FULL CELLS  position x distance, per model. Everything above is a
                       reduction of this; it is here so that a reduction can be
                       disagreed with.

Nothing here computes a metric. :func:`slambench.fov.standardise` is the one
implementation of the control and this module calls it rather than repeating
it — a second copy of a standardisation is a second place for it to drift from
the numbers it claims to summarise.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

from slambench import _REPO  # noqa: F401  (import registers sys.path)

from slambench import fov as F  # noqa: E402

W = 92


def _f(x, nd=3, dash="—"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    return dash if v != v else f"{v:.{nd}f}"


def _label(run: dict) -> str:
    """How a row names itself: the arm for radial, the azimuth for window."""
    return run["arm"]


def _pos_cols(run: dict) -> List[str]:
    e = run["pos_edges"]
    return [f"{int(a)}-{int(b)}" if run["pos_name"] == "theta" else f"{int(b - (b - a) / 2)}"
            for a, b in zip(e[:-1], e[1:])]


# --------------------------------------------------------------------------- #

def field_profile(runs: Sequence[dict], proto: str) -> List[str]:
    """AbsRel against field position: pooled, then with distance held fixed."""
    rows = [r for r in runs if r["protocol"] == proto]
    if not rows:
        return []
    axis = "incidence angle (deg)" if proto == "radial" else "window tilt (deg)"
    out = [f"  FIELD PROFILE · {proto} · AbsRel by {axis}  ↓ lower is better",
           "  " + "-" * W]
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        cols = _pos_cols(sub[0])
        out += ["", f"  {ds}",
                "  " + f"{'model':14s}{'arm':10s}{'ctx':>4s}{'':2s}"
                + "".join(f"{c:>9s}" for c in cols) + f"{'  spread':>9s}"]
        for r in sorted(sub, key=lambda r: (r["model"], r["arm"], r["context"])):
            ctl = F.controlled(r["pos_rows"], r["cells"], "AbsRel")
            n_s = len(ctl["strata"])
            for tag, series in (
                    ("pooled", [x["AbsRel"] for x in r["pos_rows"]]),
                    (f"at fixed distance ({n_s} of "
                     f"{len(r['depth_edges']) - 1} strata)", ctl["curve"]),
                    ("  ^ share of the bin that describes", ctl["share"])):
                live = [v for v in series if v == v]
                # Spread, not rim-over-centre. Under one frozen affine a
                # monotone radial bias comes back U-shaped — least squares puts
                # the scale in the middle of the bias range, so both ends rise —
                # and an endpoint ratio then understates the effect badly. The
                # window protocol does not have this problem (each window is
                # aligned on its own) but one statistic across both tables beats
                # two that have to be told apart.
                spread = (max(live) / min(live)) if len(live) > 1 and min(live) else float("nan")
                first = tag == "pooled"
                out.append("  " + f"{r['model'] if first else '':14s}"
                           + f"{_label(r) if first else '':10s}"
                           + f"{r['context'] if first else '':>4}" + "  "
                           + "".join(f"{_f(v):>9s}" for v in series)
                           + (f"{_f(spread, 2):>9s}" if "share" not in tag
                              else f"{'':>9s}") + f"   {tag}")
    out += ["",
            "  Two readings of the same points. `pooled` is every point in the",
            "  bin; `at fixed distance` averages the bin's score over the",
            "  distance strata every bin shares, so a bin cannot look good by",
            "  being nearer. Distance falls ~3.6x from this field's centre to",
            "  its rim and every metric here is relative, so the two rows",
            "  disagreeing is the expected case; the gap between them is the",
            "  size of the confound, and `spread` (the largest live bin over the",
            "  smallest) is the two answers to the actual question.",
            "",
            "  `spread` rather than rim-over-centre because one frozen affine",
            "  turns a MONOTONE radial bias into a U: least squares puts the",
            "  fitted scale in the middle of the bias range, so the centre is",
            "  over-corrected and the rim under-corrected and both ends rise.",
            "  An endpoint ratio reads ~1.2 on a bias that is 1.33 end to end.",
            "  (The window rows do not have this problem — each window is",
            "  aligned on its own — but one statistic beats two.)",
            "",
            "  The third row is the price of the control. Centre and rim barely",
            "  overlap in distance here, so the shared strata can be a small",
            "  slice of a bin; where the share is low the standardised number is",
            "  a sound comparison of that slice and not of the bin. Both rows",
            "  are printed because neither is the whole answer.",
            "",
            "  AbsRel = mean over points of |aligned − GT| / GT, under ONE",
            "  affine fitted per frame over all that frame's points and frozen",
            "  before any binning. Fitting per bin would hand an up-to-scale",
            "  model a new scale at every radius and flatten exactly the effect",
            "  this table exists to find.", ""]
    return out


def bin_distance(runs: Sequence[dict], proto: str) -> List[str]:
    """The confound itself: what each bin was looking at. Not a score."""
    rows = [r for r in runs if r["protocol"] == proto]
    if not rows:
        return []
    out = [f"  BIN DISTANCE · {proto} · mean GT depth per bin (m) — a confound, "
           f"not a score", "  " + "-" * W]
    seen: Dict[tuple, List[str]] = {}
    for r in rows:
        prof = tuple(round(x["gt_mean"], 2) if x["gt_mean"] == x["gt_mean"]
                     else None for x in r["pos_rows"])
        seen.setdefault((r["dataset"], prof), []).append(r["model"])
    cols = _pos_cols(rows[0])
    out.append("  " + f"{'dataset':14s}{'':12s}" + "".join(f"{c:>9s}" for c in cols))
    # Sorted on the dataset alone: a profile is a tuple that may hold None for
    # an empty bin, and None does not order against a float.
    for (ds, prof), models in sorted(seen.items(), key=lambda kv: kv[0][0]):
        out.append("  " + f"{ds:14s}{'':12s}"
                   + "".join(f"{v:>9.2f}" if v is not None else f"{'—':>9s}"
                             for v in prof))
    out += ["",
            "  Model-independent — it is the ground truth — so rows that share a",
            "  profile collapse. Read the field profile against this: a bin that",
            "  is both further out and nearer has two reasons to score the way",
            "  it does, and only the standardised row separates them.", ""]
    return out


def cells(runs: Sequence[dict], proto: str) -> List[str]:
    """position x distance, in full. Every reduction above comes from here."""
    rows = [r for r in runs if r["protocol"] == proto]
    if not rows:
        return []
    out = [f"  CELLS · {proto} · AbsRel at (field position, distance stratum)",
           "  " + "-" * W]
    for r in sorted(rows, key=lambda r: (r["dataset"], r["model"], r["arm"],
                                         r["context"])):
        cols = _pos_cols(r)
        de = r["depth_edges"]
        used = set(r["standardised_strata"])
        out += ["", f"  {r['dataset']} · {r['model']} · {_label(r)} · "
                    f"ctx {r['context']} · {r['n_frames']} frames",
                "  " + f"{'distance (m)':22s}" + "".join(f"{c:>9s}" for c in cols)
                + f"{'   n':>10s}"]
        by = {}
        for c in r["cells"]:
            by[(int(c["pos_bin"]), int(c["depth_stratum"]))] = c
        for s in range(len(de) - 1):
            lo = f"{de[s]:.2f}" if de[s] > 0 else "0"
            hi = f"{de[s + 1]:.2f}" if de[s + 1] != float("inf") else "∞"
            mark = " *" if s in used else "  "
            n = sum(by.get((t, s), {}).get("n", 0.0) for t in range(len(cols)))
            out.append("  " + f"{lo + '–' + hi + mark:22s}"
                       + "".join(f"{_f(by.get((t, s), {}).get('AbsRel')):>9s}"
                                 for t in range(len(cols)))
                       + f"{int(n):>10d}")
    out += ["",
            f"  * = a stratum populated above {F.MIN_CELL_POINTS} points in every",
            "  bin, so it is one of the ones the standardised row averages over.",
            "  A stratum without a star is present but not in every bin, and",
            "  including it would compare bins at different distances again.", ""]
    return out


def window_coverage(payload: dict) -> List[str]:
    """How much of each window the lens actually filled."""
    cov = payload.get("window_coverage") or []
    if not cov:
        return []
    out = ["  WINDOW COVERAGE · share of the window backed by real pixels",
           "  " + "-" * W,
           "  " + f"{'dataset':12s}{'tilt':>6s}{'azimuth':>9s}"
           f"{'in cone':>10s}{'scored':>9s}"]
    for c in cov:
        out.append("  " + f"{c['dataset']:12s}{c['tilt']:>6.0f}"
                   f"{c['azimuth']:>9.0f}{100 * c['in_cone_frac']:>9.1f}%"
                   f"{'yes' if c['scored'] else 'NO':>9s}")
    out += ["",
            f"  A window below {100 * F.MIN_IN_CONE_FRAC:.0f}% is mostly black and is not scored:",
            "  the model's response to padding is not a depth measurement. The",
            "  window WIDTH is fixed across the sweep, so this fraction moves",
            "  only with the aim — varying both is what made an earlier sweep in",
            "  this repository a measurement of dead pixels.", ""]
    return out


# --------------------------------------------------------------------------- #

def render(payload: dict) -> str:
    runs = payload.get("runs") or []
    cfg = payload.get("config", {})
    out = ["", "=" * W,
           "  FOV ON SLAM GROUND TRUTH · where in the field does depth degrade",
           "=" * W, "",
           f"  split digest   {payload.get('digest', '?')}",
           f"  frames         {payload.get('n_frames', 0)} over "
           f"{', '.join(payload.get('datasets', []))}",
           f"  points kept    inv_dist_std < {cfg.get('sigma_max')} 1/m "
           f"(the release ships unfiltered)",
           f"  protocols      {', '.join(cfg.get('protocols', []))}",
           f"  strata         {cfg.get('depth_strata')} equal-population "
           f"distance quantiles of this split's own GT", ""]
    for proto in cfg.get("protocols", []):
        out += field_profile(runs, proto)
        out += bin_distance(runs, proto)
        out += cells(runs, proto)
    out += window_coverage(payload)
    skipped = payload.get("skipped_models") or []
    if skipped:
        out += ["  NOT RUN", "  " + "-" * W]
        out += [f"  {s['model']:14s}{s['state']:12s}{s['detail']}" for s in skipped]
        out += [""]
    return "\n".join(out)


def write_all(payload: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    text = render(payload)
    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)

    fields = ["model", "dataset", "protocol", "arm", "context", "pos_name",
              "pos_lo", "pos_hi", "depth_lo", "depth_hi", "n", "AbsRel",
              "SqRel", "delta1", "gt_mean", "gt_std", "scale_ratio"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in payload.get("runs", []):
            head = {k: r[k] for k in ("model", "dataset", "protocol", "arm",
                                      "context", "pos_name")}
            for c in r["cells"]:
                w.writerow(dict(head, **c))
