# Copyright (c) 2026.
"""Does the precision change move the SCORE, not just the depth map?

The depth map moves by ~0.2% median under bf16, but every metric here is fitted
through a per-frame affine first, so a change that is uniform in scale cancels
and a change that is spatially structured does not. Only the scored metric
answers the question, so this diffs three real runs of the same split.

    python slam020_prec_diff.py eval_out/prec-fp32 eval_out/prec-tf32 eval_out/prec-bf16
"""
from __future__ import annotations

import json
import os
import sys

KEY = ("model", "dataset", "baseline", "context")
SHOW = ("AbsRel", "delta1", "RMSE")


def load(d):
    with open(os.path.join(d, "results.json")) as fh:
        return json.load(fh)


def index(p):
    return {tuple(r[k] for k in KEY): r for r in p["runs"]}


def main():
    dirs = sys.argv[1:]
    if len(dirs) < 2:
        raise SystemExit(__doc__)
    ps = [(os.path.basename(d.rstrip("/")), load(d)) for d in dirs]
    base_name, base = ps[0]
    bi = index(base)

    digests = {n: p["digest"] for n, p in ps}
    if len(set(digests.values())) != 1:
        raise SystemExit(f"[prec] different splits, nothing to compare: {digests}")
    print(f"split digest {base['digest']}  ({base['n_frames']} frames, "
          f"{len(base['takes'])} takes)\n")

    for name, p in ps[1:]:
        oi = index(p)
        keys = sorted(k for k in bi if k in oi)
        print(f"== {name}  vs  {base_name} ==")
        print(f"  {'model':11s}{'baseline':13s}{'dataset':9s}{'ctx':>4s}"
              + "".join(f"{m:>11s}{'rel':>9s}" for m in SHOW))
        print("  " + "-" * 96)
        worst = {m: 0.0 for m in SHOW}
        for k in keys:
            b, o = bi[k], oi[k]
            cells = ""
            for m in SHOW:
                rel = (o[m] - b[m]) / b[m] if b[m] else float("nan")
                worst[m] = max(worst[m], abs(rel))
                cells += f"{o[m]:11.4f}{rel:8.2%} "
            print(f"  {k[0]:11s}{k[2]:13s}{k[1]:9s}{k[3]:4d}{cells}")
        print(f"\n  largest |change|: "
              + ",  ".join(f"{m} {worst[m]:.2%}" for m in SHOW))
        # The comparison that actually decides it: does any ORDERING flip?
        flips = []
        for arm in sorted({(k[1], k[2]) for k in keys}):
            for ctx in sorted({k[3] for k in keys}):
                sel = [k for k in keys if (k[1], k[2]) == arm and k[3] == ctx]
                rb = sorted(sel, key=lambda k: bi[k]["AbsRel"])
                ro = sorted(sel, key=lambda k: oi[k]["AbsRel"])
                if rb != ro:
                    flips.append((arm, ctx))
        print(f"  model ordering by AbsRel: "
              f"{'UNCHANGED in every arm' if not flips else f'FLIPPED in {flips}'}\n")


if __name__ == "__main__":
    main()
