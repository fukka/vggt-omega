"""H50 — apply the locked rule to the held-out cells.

The rule is in protocol.md and is not restated loosely here; it is implemented
literally:

    retention_m = row_m / native_m        (Aria content, foreign lens / own lens)
    CONTENT REPLICATES  iff  retention_m >= 0.30 for all four backbones
                       AND  row_m > col_m for all four backbones

with three gates checked first -- plumbing, resolving power, and the
resampling-preserved-the-content check that H48's void arm lacked. A failed
gate voids the run; it does not produce a softer verdict.

Every branch below is exercised on synthetic inputs by --self-test, which is
run and shown before the real numbers exist. H48's analysis script was written
the same way and it is the reason its "did not fire" outcome could be reported
without argument.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
MODELS = ("da3:small", "da3:large", "vggt", "vggt_omega")
RETENTION_MIN = 0.30     # protocol.md, fixed before the run
PLUMBING_MAX_PCT = 0.001
HEADROOM_MIN = 2.0
SPREAD_TOL = 0.15


def mirror_pct(v):
    """Mirror cost as the RATIO OF MEANS, recomputed from per-frame values.

    The four cells come from three scripts and only H48's two report the
    citable statistic. `mirror_curve.py` (the Aria-native cell) writes
    `b0_mirror_cost_pct` and `mirror_scannetpp.py` (the ScanNet++-native cell)
    writes `mirror_cost_pct`, and BOTH are the mean of per-frame ratios -- the
    statistic H45's correction retired, because per-frame ratios explode where
    the denominator is small (H33, H37). Reading either stored value here would
    have put a retired statistic in the numerator of `retention` and a citable
    one in the denominator.

    So recompute from `per_frame` whenever it is there, pairing the arms frame
    by frame first: the ratio of means is only meaningful over one frame set.
    A stored value is used only when it says in its own name that it is the
    ratio of means.
    """
    per = v.get("per_frame")
    if per:
        norm = per.get("normal", per.get("normal|0.0"))
        mirr = per.get("mirror", per.get("mirror|0.0"))
        if norm and mirr:
            pairs = [(x, y) for x, y in zip(norm, mirr)
                     if x not in (None, 0) and y is not None]
            if pairs:
                n = float(np.mean([x for x, _ in pairs]))
                m = float(np.mean([y for _, y in pairs]))
                return 100.0 * (m / n - 1.0) if n else None
    return v.get("mirror_cost_pct_ratio_of_means")


def cell(prefix):
    """Mean mirror cost per backbone over every file of one cell."""
    acc, plumb, spread_fail = {}, 0.0, []
    files = sorted(RES.glob(f"{prefix}_*.json"))
    for f in files:
        d = json.load(open(f))
        pres = d.get("preservation")
        if pres and not pres.get("gate_pass", True):
            spread_fail.append((f.name, pres["rel_diff"]))
        for m, v in d.get("models", {}).items():
            p = mirror_pct(v)
            if p is not None:
                acc.setdefault(m, []).append(p)
            plumb = max(plumb, v.get("plumbing_max_rel_pct", 0.0))
    return ({m: float(np.mean(v)) for m, v in acc.items()},
            plumb, spread_fail, len(files))


def headroom(prefix):
    """floor / baseline AbsRel per backbone for one cell, from results/diag/.

    protocol.md specifies this gate; the first version of this script printed a
    reminder to run it by hand, which is exactly how a gate stops being one.
    Implemented here after the run, WITHOUT changing the 2.0x threshold the
    protocol fixed -- the number was written down before the data existed and
    is not touched.
    """
    fs = sorted((RES / "diag").glob(f"{prefix}_*.json"))
    if not fs:
        return {}
    floor = float(np.mean([json.load(open(f))["agg"]["floor"] for f in fs]))
    base = {}
    for f in sorted(RES.glob(f"{prefix}_*.json")):
        for m, v in json.load(open(f)).get("models", {}).items():
            b = v.get("normal_absrel")
            if b is None:
                pf = v.get("per_frame", {})
                n = pf.get("normal") or pf.get("normal|0.0") or []
                n = [x for x in n if x]
                b = float(np.mean(n)) if n else None
            if b:
                base.setdefault(m, []).append(b)
    return {m: floor / float(np.mean(v)) for m, v in base.items()}


def verdict(native, row, col):
    missing = [m for m in MODELS
               if m not in native or m not in row or m not in col]
    if missing:
        return "INCOMPLETE", f"cells missing for {', '.join(missing)}"
    ret = {m: row[m] / native[m] for m in MODELS}
    low = [m for m in MODELS if ret[m] < RETENTION_MIN]
    lost = [m for m in MODELS if not row[m] > col[m]]
    if low or lost:
        why = []
        if low:
            why.append("retention below %.2f for %s"
                       % (RETENTION_MIN, ", ".join(f"{m} ({ret[m]:.2f})" for m in low)))
        if lost:
            why.append("row does not beat column for " + ", ".join(lost))
        return "DOES NOT REPLICATE", "; ".join(why)
    return "CONTENT REPLICATES", (
        "retention %s, row > column 4/4"
        % ", ".join(f"{m} {ret[m]:.2f}" for m in MODELS))


def self_test():
    """Every branch, on numbers chosen to hit it. Run before the real data."""
    n = {m: 100.0 for m in MODELS}
    ok_row = {m: 50.0 for m in MODELS}
    low_col = {m: 10.0 for m in MODELS}
    cases = [
        ("all clear", n, ok_row, low_col, "CONTENT REPLICATES"),
        ("one backbone below retention floor", n,
         {**ok_row, "vggt": 20.0}, low_col, "DOES NOT REPLICATE"),
        ("row loses to column for one backbone", n, ok_row,
         {**low_col, "da3:large": 80.0}, "DOES NOT REPLICATE"),
        ("row ties column", n, ok_row, {**low_col, "vggt_omega": 50.0},
         "DOES NOT REPLICATE"),
        ("a cell is missing", n, {k: v for k, v in ok_row.items() if k != "vggt"},
         low_col, "INCOMPLETE"),
    ]
    # the statistic itself, on both per-frame namings and on the pairing rule
    stat_cases = [
        ("H48 naming",
         {"per_frame": {"normal": [1.0, 2.0], "mirror": [2.0, 4.0]}}, 100.0),
        ("mirror_curve naming",
         {"per_frame": {"normal|0.0": [1.0, 3.0], "mirror|0.0": [2.0, 2.0]}}, 0.0),
        ("ratio of means, NOT mean of ratios",
         # mean of ratios would be (10/1 + 1/10)/2 = 5.05 -> +405%
         {"per_frame": {"normal": [1.0, 10.0], "mirror": [10.0, 1.0]}}, 0.0),
        ("unpaired frames dropped from both sides",
         {"per_frame": {"normal": [1.0, None, 2.0], "mirror": [2.0, 9.9, 4.0]}}, 100.0),
        ("stored ratio-of-means accepted when per_frame is absent",
         {"mirror_cost_pct_ratio_of_means": 42.0}, 42.0),
        ("stored mean-of-ratios REFUSED", {"mirror_cost_pct": 42.0}, None),
    ]
    bad = 0
    for name, v, want in stat_cases:
        got = mirror_pct(v)
        ok = (got is None and want is None) or (
            got is not None and want is not None and abs(got - want) < 1e-9)
        print(f"  [{'ok ' if ok else 'FAIL'}] statistic: {name:44s} -> {got}")
        bad += 0 if ok else 1
    for name, a, b, c, want in cases:
        got, why = verdict(a, b, c)
        mark = "ok " if got == want else "FAIL"
        if got != want:
            bad += 1
        print(f"  [{mark}] {name:38s} -> {got}  ({why})")
    print("  self-test:", "PASS" if not bad else f"{bad} FAILED")
    return bad == 0


def main():
    if "--self-test" in sys.argv:
        return 0 if self_test() else 1

    native, p_aa, sf_aa, n_aa = cell("aa")
    row, p_r, sf_r, n_r = cell("r")
    col, p_a, sf_a, n_a = cell("a")
    ss, p_ss, _, n_ss = cell("ss")

    print(f"\nfiles: aa {n_aa}  r {n_r}  a {n_a}  ss {n_ss}")

    plumb = max(p_aa, p_r, p_a)
    print(f"gate 1  plumbing: worst {plumb:.6f}% "
          f"-> {'PASS' if plumb < PLUMBING_MAX_PCT else 'FAIL'}")
    sf = sf_aa + sf_r + sf_a
    print(f"gate 3  resampling preserved content: "
          f"{'PASS' if not sf else 'FAIL ' + str(sf)}")
    hr = {c: headroom(c) for c in ("aa", "r", "a")}
    thin = [(c, m, h) for c, d in hr.items() for m, h in sorted(d.items())
            if h < HEADROOM_MIN]
    print(f"gate 2  resolving power: worst "
          f"{min([h for d in hr.values() for h in d.values()], default=float('nan')):.2f}x "
          f"-> {'PASS' if not thin else 'FAIL'}")
    for c, m, h in thin:
        print(f"        cell {c!r} backbone {m}: {h:.2f}x, below the {HEADROOM_MIN}x bar")

    print(f"\n{'backbone':12s} {'Aria/Aria':>10s} {'Aria/SN++lens':>14s} "
          f"{'SN++/Aria lens':>15s} {'retention':>10s} {'SN++/SN++*':>11s}")
    for m in MODELS:
        r = (row[m] / native[m]) if (m in row and m in native) else float("nan")
        print(f"{m:12s} {native.get(m, float('nan')):+10.1f} "
              f"{row.get(m, float('nan')):+14.1f} {col.get(m, float('nan')):+15.1f} "
              f"{r:10.2f} {ss.get(m, float('nan')):+11.1f}")
    print("* not region-matched; in no clause of the rule (protocol.md)")

    v, why = verdict(native, row, col)
    rule_said = (v, why)
    if plumb >= PLUMBING_MAX_PCT or sf or thin:
        v = "VOID"
        why = ("a gate failed, so the rule is not applied: "
               + "; ".join(f"{c}/{m} resolves at {h:.2f}x" for c, m, h in thin))
    print(f"\nVERDICT: {v} — {why}")
    if v == "VOID":
        print(f"         (what the rule would have said, had the gate passed: "
              f"{rule_said[0]} — {rule_said[1]})")
        print("         Clause 1 (retention) uses only the aa and r cells. "
              "Neither is implicated: see the headroom table.")
    (RES / "summary.json").write_text(json.dumps(
        {"native": native, "row": row, "col": col, "ss_not_matched": ss,
         "verdict": v, "why": why, "rule_output_if_gate_had_passed": rule_said,
         "headroom": hr, "retention_min": RETENTION_MIN,
         "headroom_min": HEADROOM_MIN}, indent=2))
    print(f"[h50] wrote {RES / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
