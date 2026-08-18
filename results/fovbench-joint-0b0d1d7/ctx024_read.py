#!/usr/bin/env python3
"""Ticket 024 part B -- the context arms on the six-sequence split, plus the
cross-box control that has to come with them.

Two tables:

1. CONTEXT. Overall AbsRel per model/stream/view at N = 1, 3, 5, 10 frames at
   stride 10, all four measured on the SAME box (space-container, 4x A100), so
   the curve the deck plots never steps across hardware inside itself.

2. BOX CONTROL. The pod's N=1 arm against lambda's part A run -- same commit,
   same split digest, same 900 files (sha256-verified), different GPU. Judged
   against #021's PER-METRIC bounds, which is the mistake #023 made once and
   should not be repeated: #021 measured AbsRel 0.51%, delta1 0.20%, RMSE 0.14%
   and set no SqRel bound at all.

Usage: ctx024_read.py <partB_dir> <partA_results.json> [out.txt]
  partB_dir holds partB_6seq_1 / _3s / _5s / _10s.
"""
import json, os, sys

BOUNDS = {"AbsRel": 0.51, "delta1": 0.20, "RMSE": 0.14}
ARMS = [("1", "partB_6seq_1"), ("3s", "partB_6seq_3s"),
        ("5s", "partB_6seq_5s"), ("10s", "partB_6seq_10s")]
CTX_MODELS = ("vggt_1b", "vggt_omega", "da3_large", "da3_small")


def cells(path):
    d = json.load(open(path))
    out = {}
    for r in d["runs"]:
        out[(r["model"], r["stream"], r["view"])] = r
    return d, out


def pen(r):
    b = [x for x in r["bins"] if x.get("n_px_total", 0) > 0]
    return b[-1]["AbsRel"] / b[0]["AbsRel"] if len(b) >= 2 else float("nan")


def main(bdir, apath, out=None):
    L = []
    P = L.append
    arms, digests = {}, {}
    for tag, sub in ARMS:
        p = os.path.join(bdir, sub, "results.json")
        if not os.path.exists(p):
            P(f"!! missing arm {tag}: {p}")
            continue
        d, c = cells(p)
        arms[tag] = c
        digests[tag] = d["digest"]
    dA, cA = cells(apath)

    P("ADT-FOV ticket 024 part B -- context arms on the six-sequence split")
    P("digests: " + ", ".join(f"{k}={v}" for k, v in digests.items())
      + f", partA(lambda)={dA['digest']}")
    P("")
    P("=== overall AbsRel by context frames (stride 10; N=1 is stride 1) ===")
    P(f"{'model':<11}{'stream':<11}{'view':<9}" +
      "".join(f"{'N=' + t:>9}" for t, _ in ARMS) + f"{'10s/N1':>9}")
    for m in CTX_MODELS:
        for st in ("real", "synthetic"):
            for vw in ("fisheye", "rect"):
                k = (m, st, vw)
                if not all(k in arms.get(t, {}) for t, _ in ARMS):
                    continue
                vals = [arms[t][k]["overall"]["AbsRel"] for t, _ in ARMS]
                P(f"{m:<11}{st:<11}{vw:<9}" + "".join(f"{v:>9.4f}" for v in vals)
                  + f"{vals[-1] / vals[0]:>9.3f}")
    P("")
    P("=== pen (rim/centre AbsRel) by context frames -- does context change the shape? ===")
    P(f"{'model':<11}{'stream':<11}{'view':<9}" + "".join(f"{'N=' + t:>9}" for t, _ in ARMS))
    for m in CTX_MODELS:
        for st in ("real", "synthetic"):
            for vw in ("fisheye", "rect"):
                k = (m, st, vw)
                if not all(k in arms.get(t, {}) for t, _ in ARMS):
                    continue
                P(f"{m:<11}{st:<11}{vw:<9}" +
                  "".join(f"{pen(arms[t][k]):>9.3f}" for t, _ in ARMS))
    P("")
    P("=== BOX CONTROL: pod N=1 vs lambda part A, same commit and same 300 frames ===")
    P("bounds from #021, per metric: " +
      ", ".join(f"{k} {v}%" for k, v in BOUNDS.items()) + ". SqRel is unbounded there and is reported unjudged.")
    P(f"{'model':<11}{'stream':<11}{'view':<9}" +
      "".join(f"{m:>12}" for m in ("AbsRel", "delta1", "RMSE", "SqRel")) + "  verdict")
    worst = {k: 0.0 for k in BOUNDS}
    bad = 0
    n = 0
    for m in CTX_MODELS:
        for st in ("real", "synthetic"):
            for vw in ("fisheye", "rect"):
                k = (m, st, vw)
                if k not in arms.get("1", {}) or k not in cA:
                    continue
                n += 1
                row, ok = [], True
                for met in ("AbsRel", "delta1", "RMSE", "SqRel"):
                    a = cA[k]["overall"][met]
                    b = arms["1"][k]["overall"][met]
                    pc = 100.0 * (b - a) / a if a else float("nan")
                    row.append(pc)
                    if met in BOUNDS:
                        worst[met] = max(worst[met], abs(pc))
                        if abs(pc) > BOUNDS[met]:
                            ok = False
                if not ok:
                    bad += 1
                P(f"{m:<11}{st:<11}{vw:<9}" + "".join(f"{v:>+11.4f}%" for v in row)
                  + ("  ok" if ok else "  OVER BOUND"))
    P("")
    P(f"cells compared: {n}   over bound: {bad}")
    P("worst |delta| per metric: " + ", ".join(f"{k} {v:.4f}% (bound {BOUNDS[k]}%)"
                                               for k, v in worst.items()))
    P("CONTROL: " + ("PASS -- the A100 arm is the same measurement as the Ada one "
                     "to within #021's published box tolerance, so the four-point "
                     "curve above is not carrying a hardware step."
                     if bad == 0 else
                     "FAIL -- the box difference exceeds #021's bound. The context "
                     "curve is still internally consistent (one box, four arms), but "
                     "it may NOT be plotted against lambda numbers."))
    txt = "\n".join(L)
    print(txt)
    if out:
        open(out, "w").write(txt + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
