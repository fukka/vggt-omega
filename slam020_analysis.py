# Copyright (c) 2026.
"""Ticket 020 -- what the three merged runs say together, and where they cannot.

`slambench.report` already renders each run's own tables (raw vs rect_derect
with coverage, the context sweep, gt_median). This adds only the statements that
need more than one run to make:

  * the three runs are one split -- same digest, same root, same config apart
    from the context axis;
  * **the two strides do not share a support.** `--context-stride` is a single
    int, so stride 1 and stride 10 are separate runs and each intersects its own
    support. Their `context=1` arm is the *identical computation* on the
    *identical frames*, so any difference between those two rows is purely the
    support moving underneath -- and that difference is the honest scale bar for
    reading a stride-1 number against a stride-10 one;
  * step 2's raw column against the published `slambench-raw-b1659a0`, which
    #020 predicts will differ because adding `rect_derect` removes the rim from
    both arms.

Usage:
    python slam020_analysis.py <dir-with-step2 step3-s1 step3-s10> [--raw OLD]
"""
from __future__ import annotations

import argparse
import json
import os

KEY = ("model", "dataset", "baseline", "context")
METRICS = ("AbsRel", "delta1", "RMSE", "log10")


def load(d, name):
    p = os.path.join(d, name, "results.json")
    with open(p) as fh:
        return json.load(fh)


def index(p):
    # The published raw run predates the context axis and its rows carry no
    # `context` key at all. Defaulting to 1 is the only reading that is not a
    # guess: that run was a single frame per forward pass by construction.
    return {(r["model"], r["dataset"], r["baseline"], int(r.get("context", 1))): r
            for r in p["runs"]}


def rule(t):
    print(f"\n{t}\n{'=' * len(t)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--raw", default=None, help="published slambench-raw run")
    a = ap.parse_args()

    runs = {n: load(a.root, n) for n in ("step2", "step3-s1", "step3-s10")}

    rule("1. ONE SPLIT?  digest, root and config across the three runs")
    ref = runs["step2"]
    ok = True
    for n, p in runs.items():
        same_d = p["digest"] == ref["digest"]
        same_r = p["egosynth_root"] == ref["egosynth_root"]
        ok &= same_d and same_r
        print(f"  {n:10s} digest {p['digest']}  {'OK' if same_d else '** DIFFERS **'}"
              f"   frames {p['n_frames']}   takes {len(p['takes'])}"
              f"   sharded_by {p.get('sharded_by', '-')}")
    # everything in config except the context axis, which is meant to differ
    drift = {}
    for n, p in runs.items():
        for k, v in p["config"].items():
            if k in ("context_frames", "context_stride"):
                continue
            if ref["config"].get(k) != v:
                drift.setdefault(k, {})[n] = v
    print(f"\n  config identical apart from the context axis: "
          f"{'YES' if not drift else '** NO ** ' + json.dumps(drift)}")
    print(f"  context axis: " + ", ".join(
        f"{n} = {p['config']['context_frames']} @ stride "
        f"{p['config']['context_stride']}" for n, p in runs.items()))
    print(f"\n  verdict: {'one split, three runs' if ok and not drift else 'NOT comparable'}")

    rule("2. THE STRIDE GAP  context=1 is the same computation in both stride runs")
    print("  Identical frames, identical window (a 1-frame window has no stride),")
    print("  so every difference below is the support intersection moving, not the")
    print("  model. Read it as the error bar on any stride-1 vs stride-10 comparison.")
    i1, i10 = index(runs["step3-s1"]), index(runs["step3-s10"])
    keys = sorted(k for k in i1 if k[3] == 1 and k in i10)
    print(f"\n  {'model':13s}{'baseline':13s}{'dataset':10s}"
          + "".join(f"{m:>12s}" for m in ("AbsRel s1", "AbsRel s10", "rel diff",
                                          "cover s1", "cover s10")))
    print("  " + "-" * 106)
    worst = 0.0
    for k in keys:
        r1, r10 = i1[k], i10[k]
        d = abs(r1["AbsRel"] - r10["AbsRel"]) / max(abs(r1["AbsRel"]), 1e-12)
        worst = max(worst, d)
        print(f"  {k[0]:13s}{k[2]:13s}{k[1]:10s}"
              f"{r1['AbsRel']:12.4f}{r10['AbsRel']:12.4f}{d:11.2%} "
              f"{r1['coverage']:12.3f}{r10['coverage']:12.3f}")
    print(f"\n  largest context=1 disagreement between the two runs: {worst:.2%}")
    print("  A stride effect smaller than this is not resolvable across the runs.")

    rule("3. STEP 2's RAW COLUMN vs the published raw-only run")
    if not a.raw:
        print("  (--raw not given)")
    else:
        with open(os.path.join(a.raw, "results.json")) as fh:
            old = json.load(fh)
        oi, ni = index(old), index(runs["step2"])
        print(f"  published digest {old['digest']}   this run {ref['digest']}"
              f"   {'same split' if old['digest'] == ref['digest'] else 'DIFFERENT split'}")
        print(f"  published datasets {old['config']['datasets']}"
              f"   this run {ref['config']['datasets']}")
        print("\n  #020 predicts these differ: a 110 deg pinhole cannot cover the")
        print("  fisheye cone, so adding rect_derect strips the rim from BOTH arms.")
        print(f"\n  {'model':13s}{'dataset':10s}{'AbsRel pub':>12s}{'AbsRel now':>12s}"
              f"{'change':>10s}{'cover now':>11s}")
        print("  " + "-" * 68)
        for k in sorted(ni):
            if k[2] != "raw" or k[3] != 1 or k not in oi:
                continue
            o, n = oi[k], ni[k]
            print(f"  {k[0]:13s}{k[1]:10s}{o['AbsRel']:12.4f}{n['AbsRel']:12.4f}"
                  f"{(n['AbsRel'] - o['AbsRel']) / o['AbsRel']:9.1%}"
                  f"{n['coverage']:11.3f}")

    rule("4. gt_median PER DATASET  (every metric here is relative)")
    seen = {}
    for n, p in runs.items():
        for r in p["runs"]:
            seen.setdefault(r["dataset"], set()).add(round(r["gt_median"], 4))
    for ds, v in sorted(seen.items()):
        print(f"  {ds:10s} gt_median {sorted(v)}  m")

    rule("5. WHAT WAS LEFT OUT")
    for n, p in runs.items():
        sk = p.get("skipped_models") or []
        got = sorted({r["model"] for r in p["runs"]})
        print(f"  {n:10s} ran {got}")
        for d in sk:
            print(f"             skipped {d['model']}: {d['state']} -- {d['detail']}")
        for s in p.get("shards", []):
            print(f"             shard {s['path']}: {s['models']} ({s['rows']} rows)")


if __name__ == "__main__":
    main()
