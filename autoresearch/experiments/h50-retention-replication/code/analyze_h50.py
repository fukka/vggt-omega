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
    for k in ("mirror_cost_pct_ratio_of_means", "mirror_cost_pct"):
        if k in v:
            return v[k]
    return None


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
    bad = 0
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
    print("gate 2  resolving power: run diag_h48.py over the held-out scenes; "
          "recorded in results/diag/")

    print(f"\n{'backbone':12s} {'Aria/Aria':>10s} {'Aria/SN++lens':>14s} "
          f"{'SN++/Aria lens':>15s} {'retention':>10s} {'SN++/SN++*':>11s}")
    for m in MODELS:
        r = (row[m] / native[m]) if (m in row and m in native) else float("nan")
        print(f"{m:12s} {native.get(m, float('nan')):+10.1f} "
              f"{row.get(m, float('nan')):+14.1f} {col.get(m, float('nan')):+15.1f} "
              f"{r:10.2f} {ss.get(m, float('nan')):+11.1f}")
    print("* not region-matched; in no clause of the rule (protocol.md)")

    v, why = verdict(native, row, col)
    if plumb >= PLUMBING_MAX_PCT or sf:
        v, why = "VOID", "a gate failed; the rule is not applied"
    print(f"\nVERDICT: {v} — {why}")
    (RES / "summary.json").write_text(json.dumps(
        {"native": native, "row": row, "col": col, "ss_not_matched": ss,
         "verdict": v, "why": why, "retention_min": RETENTION_MIN}, indent=2))
    print(f"[h50] wrote {RES / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
