"""H43b — the same arithmetic with the grid extended to +-30, so nothing is extrapolated.

H43's device-origin column needed g past +-22, the edge of the measured grid, so
it was computed twice and B1/B3 were evaluated on guesses. The wide-angle run
adds deltas 0, +-26, +-30. At a = 22 the device-origin arms then sit at
mu - 22 (down to about -30) and mu + 22 (up to about +23), both INSIDE the
extended grid for every recording. No extrapolation anywhere.

The three bars are re-evaluated exactly as H43 locked them.
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
H41 = HERE.parents[0] / "h41-residual-curve" / "results"
H42 = HERE.parents[0] / "h42-curve-four-backbones" / "results"
MODELS = ["da3:small", "da3:large", "vggt", "vggt_omega"]
A = 22.0


def gather():
    out: dict[str, dict] = {}
    for pat, keep in ((str(H41 / "*_seq*.json"), {"da3:small"}),
                      (str(H42 / "*_seq*.json"), None)):
        for f in sorted(glob.glob(pat)):
            d = json.loads(Path(f).read_text())
            s = "seq" + d["seq"].split("_seq")[1].split("_")[0]
            r = out.setdefault(s, {"psi": []})
            r["psi"].extend(d["roll_deg"])
            for m, v in d["models"].items():
                if keep and m not in keep:
                    continue
                for k, (mean, sd, n) in v["g"].items():
                    r.setdefault(m, {})[-float(k)] = mean
    for s, r in out.items():
        r["mu"] = float(np.median(r["psi"]))
    return out


data = gather()
seqs = sorted(s for s in data if all(m in data[s] for m in MODELS))
# The span must be checked PER MODEL: the wide-angle extension was run on
# three backbones, and a span read off one of them says nothing about another.
# The first version checked da3:small's span for every model and reported
# 0/13 coverage, which was true of da3:small and false of the other three.
need = {s: (data[s]["mu"] - A, data[s]["mu"] + A) for s in seqs}


def covered(s, m):
    xs = data[s][m]
    return min(xs) <= need[s][0] and need[s][1] <= max(xs)


inside = [s for s in seqs if all(covered(s, m) for m in MODELS)]
print(f"[h43b] {len(seqs)} recordings; grid covers what is needed on "
      f"{len(inside)}/{len(seqs)}")
if len(inside) < len(seqs):
    for s in seqs:
        miss = [m for m in MODELS if not covered(s, m)]
        if miss:
            print(f"   {s}: need [{need[s][0]:+.1f},{need[s][1]:+.1f}]; "
                  f"short on {', '.join(miss)}")
for m in MODELS:
    lo = min(min(data[s][m]) for s in seqs)
    hi = max(max(data[s][m]) for s in seqs)
    print(f"[h43b] {m:<12} grid spans {lo:+.0f} .. {hi:+.0f} degrees")
print()

rows, out = [], {}
print(f"{'model':<12}{'Delta_grav':>12}{'Delta_dev':>11}{'inflated':>10}"
      f"{'dev>0':>8}")
for m in MODELS:
    dg, dd = [], []
    for s in inside:
        xs = np.array(sorted(data[s][m]))
        ys = np.array([data[s][m][x] for x in xs])
        g = lambda r: float(np.interp(r, xs, ys))
        mu = data[s]["mu"]
        dg.append(100 * (g(+A) - g(-A)))
        dd.append(100 * (g(mu - A) - g(mu + A)))
        rows.append({"seq": s, "model": m, "mu": mu,
                     "d_grav": dg[-1], "d_dev": dd[-1]})
    out[m] = {"d_grav": st.mean(dg), "d_dev": st.mean(dd),
              "inflated_on": sum(1 for a, b in zip(dg, dd) if b > a),
              "dev_positive_on": sum(1 for b in dd if b > 0), "n": len(dg)}
    o = out[m]
    print(f"{m:<12}{o['d_grav']:>+11.2f}%{o['d_dev']:>+10.2f}%"
          f"{o['inflated_on']:>7}/{o['n']}{o['dev_positive_on']:>5}/{o['n']}")

tot = sum(out[m]["n"] for m in MODELS)
inf = sum(out[m]["inflated_on"] for m in MODELS)
b1 = inf > tot / 2
b2 = out["vggt"]["d_grav"] < 0 and out["vggt"]["dev_positive_on"] > out["vggt"]["n"] / 2
rho = float(spearmanr([r["mu"] for r in rows],
                      [r["d_dev"] - r["d_grav"] for r in rows]).correlation)
b3 = rho <= -0.4
print(f"\nB1  inflated on {inf}/{tot} -> {'PASS' if b1 else 'FAIL'}")
print(f"B2  vggt Delta_grav {out['vggt']['d_grav']:+.2f}%, dev positive on "
      f"{out['vggt']['dev_positive_on']}/{out['vggt']['n']} -> "
      f"{'PASS' if b2 else 'FAIL'}")
print(f"B3  Spearman(mu, inflation) = {rho:+.3f} -> {'PASS' if b3 else 'FAIL'}")

(HERE / "results" / "noextrap.json").write_text(json.dumps(
    {"recordings": inside, "per_model": out, "spearman": rho,
     "bars": {"b1": b1, "b2": b2, "b3": b3}}, indent=2))
print(f"\n[h43b] wrote {HERE / 'results' / 'noextrap.json'}")
