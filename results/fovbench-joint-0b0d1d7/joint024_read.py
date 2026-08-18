#!/usr/bin/env python3
"""Ticket 024 part A -- read the joint grid as the control it was built to be.

The 1-D theta table conflates two things. The rim of this lens looks at nearer
surfaces (GT median 2.95 m on axis, 1.64 m at 50-55 deg on this split), and
every metric here is relative, so "AbsRel rises with theta" is part geometry and
part a change of subject. The joint grid scores the same frozen prediction on an
(incidence x GT depth) grid, so a ROW holds the depth band fixed and varies only
where in the field the pixels sit.

Reported per model/stream/view:

  pen_1d      AbsRel(50-55) / AbsRel(0-10) from the 1-D table -- what the deck
              currently plots.
  pen_ctl     the same ratio computed INSIDE each depth band and then pooled
              across bands, weighted by min(n_centre, n_rim) so a band that is
              thin at either end cannot dominate. This is the depth-controlled
              rim penalty.
  survives    log(pen_ctl) / log(pen_1d) -- the fraction of the 1-D penalty that
              is not a depth-composition artefact. 1.0 means depth explains
              none of it; 0.0 means depth explains all of it.

Cells thinner than MIN_JOINT_CELL_PX (500 px) are dropped, matching what the
report draws.
"""
import json, sys, math
import numpy as np

MIN_CELL = 500

def load(path):
    return json.load(open(path))

def rows(res):
    for r in res["runs"]:
        yield r

def pen_1d(r, metric="AbsRel"):
    b = [x for x in r["bins"] if x.get("n_px_total", 0) > 0]
    if len(b) < 2: return float("nan"), float("nan"), float("nan")
    return b[-1][metric] / b[0][metric], b[0][metric], b[-1][metric]

def joint_table(r, metric="AbsRel"):
    t = r["joint"]["theta"]
    a = np.array(t[metric], dtype=float)
    n = np.array(t["n"], dtype=float)
    g = np.array(t["gt_mean"], dtype=float)
    a[n < MIN_CELL] = np.nan
    return t["coord_edges"], t["depth_edges"], a, n, g

def controlled(r, metric="AbsRel"):
    """Rim/centre inside each depth band, pooled over bands."""
    _, dedges, a, n, _ = joint_table(r, metric)
    c, rim = a[0], a[-1]
    wn = np.minimum(n[0], n[-1])
    ok = np.isfinite(c) & np.isfinite(rim) & (wn >= MIN_CELL) & (c > 0)
    if not ok.any(): return float("nan"), [], 0.0
    ratios = rim[ok] / c[ok]
    w = wn[ok]
    per_band = [(f"{dedges[i]:g}-{dedges[i+1]:g}", float(rim[i] / c[i]), float(wn[i]))
                for i in range(len(c)) if ok[i]]
    return float(np.exp(np.average(np.log(ratios), weights=w))), per_band, float(w.sum())

def depth_only_null(r):
    """The ticket's own null: a model whose error is a fixed metric offset reads
    a theta gradient only through the depth composition, and AbsRel x gt_mean
    comes back constant. Report the spread of that product across the row."""
    _, _, a, n, g = joint_table(r)
    prod = a * g
    out = []
    for i in range(a.shape[1]):
        col = prod[:, i][np.isfinite(prod[:, i])]
        if col.size >= 2:
            out.append(float(col.max() / col.min()))
    return out

def main(path, out=None):
    res = load(path)
    lines = []
    P = lines.append
    P("ADT-FOV ticket 024 part A -- the depth-controlled rim penalty")
    P(f"digest {res['digest']}  n_frames {res['n_frames']}  "
      f"sequences {len(res['sequences'])}  protocol {res['protocol']}")
    P("")
    P("pen_1d   = AbsRel(50-55 deg) / AbsRel(0-10 deg), the 1-D table")
    P("pen_ctl  = the same ratio inside each GT-depth band, pooled (n-weighted)")
    P("survives = log(pen_ctl)/log(pen_1d): share of the 1-D penalty that is NOT")
    P("           a change of subject.  1.0 = none of it was depth composition.")
    P("")
    P(f"{'model':<11}{'stream':<11}{'view':<9}{'centre':>8}{'rim':>8}"
      f"{'pen_1d':>9}{'pen_ctl':>9}{'survives':>10}")
    per_cfg = {}
    for r in rows(res):
        p1, c, rim = pen_1d(r)
        pc, bands, w = controlled(r)
        # log(pen_ctl)/log(pen_1d) blows up when pen_1d is itself ~1: the
        # denominator goes to zero and the ratio stops meaning anything. Those
        # rows are the interesting ones for a different reason (a flat 1-D curve
        # hiding a real controlled penalty), so mark them rather than print a 24.
        surv = (math.log(pc) / math.log(p1)) if (p1 > 0 and pc > 0 and abs(math.log(p1)) > 0.05) else float("nan")
        key = (r["model"], r["stream"], r["view"])
        per_cfg[key] = (p1, pc, surv, bands)
        sv = f"{surv:>10.2f}" if math.isfinite(surv) else f"{'flat 1-D':>10}"
        P(f"{r['model']:<11}{r['stream']:<11}{r['view']:<9}{c:>8.4f}{rim:>8.4f}"
          f"{p1:>9.3f}{pc:>9.3f}" + sv)
    P("")
    P("=== the rim penalty band by band (rim/centre AbsRel at fixed GT depth) ===")
    P(f"{'model':<11}{'stream':<11}{'view':<9}" + "".join(f"{b:>9}" for b in
      ("0-1", "1-2", "2-3", "3-5", "5-10")))
    for (m, s, v), (p1, pc, surv, bands) in per_cfg.items():
        d = {b: r for b, r, _ in bands}
        P(f"{m:<11}{s:<11}{v:<9}" + "".join(
            f"{d.get(b, float('nan')):>9.3f}" for b in ("0-1", "1-2", "2-3", "3-5", "5-10")))
    P("")
    P("=== is the residual a depth-only error in disguise? ===")
    P("max/min of AbsRel x gt_mean down each depth column. A model whose error is")
    P("a fixed metric offset holds this at ~1.0 (the ticket's closed-form null")
    P("comes back to 0.12 m within 5%); a theta-driven error does not.")
    P(f"{'model':<11}{'stream':<11}{'view':<9}{'worst column':>14}")
    for r in rows(res):
        sp = depth_only_null(r)
        P(f"{r['model']:<11}{r['stream']:<11}{r['view']:<9}{max(sp):>14.2f}")
    P("")
    P("=== what the rim is actually looking at (GT mean, m, by theta bin) ===")
    r0 = next(rows(res))
    _, _, _, n0, g0 = joint_table(r0)
    ce = r0["joint"]["theta"]["coord_edges"]
    tot = n0.sum(axis=1)
    gm = (g0 * n0).sum(axis=1) / np.where(tot > 0, tot, 1)
    P("theta bin   gt_mean   px share")
    for i in range(len(gm)):
        P(f"{ce[i]:>4g}-{ce[i+1]:<5g} {gm[i]:>8.2f}   {100*tot[i]/tot.sum():>6.1f}%")
    txt = "\n".join(lines)
    print(txt)
    if out:
        open(out, "w").write(txt + "\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
