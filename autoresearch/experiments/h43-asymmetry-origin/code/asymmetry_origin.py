"""H43 — does the device-aligned origin manufacture the published roll asymmetry?

A device-origin sweep at +-a measures g(psi - a) - g(psi + a). With a negative
signed median psi that is positive even for a perfectly symmetric g. H17.1
measured the signed median as negative, so this asks whether H32's "+30 costs
2.3x what -30 costs" is that identity rather than a fact about models.

CPU only. Reads H41's and H42's committed curves and their per-frame roll.
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
H41 = HERE.parents[0] / "h41-residual-curve" / "results"
H42 = HERE.parents[0] / "h42-curve-four-backbones" / "results"
MODELS = ["da3:small", "da3:large", "vggt", "vggt_omega"]
A = 22.0


def gather():
    """{seq: {'mu': signed median psi, model: {residual: g}}}"""
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


def make_g(pts: dict[float, float], mode: str):
    xs = np.array(sorted(pts))
    ys = np.array([pts[x] for x in xs])

    def g(r):
        r = float(r)
        if xs[0] <= r <= xs[-1]:
            return float(np.interp(r, xs, ys))
        side = slice(0, 3) if r < xs[0] else slice(-3, None)
        sx, sy = xs[side], ys[side]
        if mode == "linear":
            sx, sy = (sx[:2], sy[:2]) if r < xs[0] else (sx[-2:], sy[-2:])
            k = (sy[1] - sy[0]) / (sx[1] - sx[0])
            x0, y0 = (sx[0], sy[0])
            return float(y0 + k * (r - x0))
        c = np.polyfit(sx, sy, 2)
        return float(np.polyval(c, r))
    return g


data = gather()
seqs = sorted(s for s in data if all(m in data[s] for m in MODELS))
print(f"[h43] {len(seqs)} recordings with all four backbones\n")
print(f"{'seq':<9}{'median psi':>12}")
for s in seqs:
    print(f"{s:<9}{data[s]['mu']:>+11.2f}°")
print(f"\npooled signed median psi: "
      f"{np.median([x for s in seqs for x in data[s]['psi']]):+.2f}°"
      f"   (H17.1 reported -2.70° over all frames)\n")

rows, out = [], {}
for mode in ("linear", "quadratic"):
    print(f"--- extrapolation: {mode} ---")
    print(f"{'model':<12}{'Delta_grav':>12}{'Delta_dev':>11}{'inflated':>10}")
    for m in MODELS:
        dg, dd = [], []
        for s in seqs:
            g = make_g(data[s][m], mode)
            mu = data[s]["mu"]
            dg.append(100 * (g(+A) - g(-A)))
            dd.append(100 * (g(mu - A) - g(mu + A)))
            rows.append({"seq": s, "model": m, "mode": mode, "mu": mu,
                         "d_grav": dg[-1], "d_dev": dd[-1]})
        n_inf = sum(1 for a, b in zip(dg, dd) if b > a)
        out.setdefault(mode, {})[m] = {
            "d_grav": st.mean(dg), "d_dev": st.mean(dd),
            "inflated_on": n_inf, "n": len(dg),
            "dev_positive_on": sum(1 for b in dd if b > 0)}
        print(f"{m:<12}{st.mean(dg):>+11.2f}%{st.mean(dd):>+10.2f}%"
              f"{n_inf:>7}/{len(dg)}")
    print()

cells = {mode: sum(out[mode][m]["inflated_on"] for m in MODELS) for mode in out}
tot = len(MODELS) * len(seqs)
b1 = all(v > tot / 2 for v in cells.values())
print(f"B1  Delta_dev > Delta_grav on "
      + ", ".join(f"{k}: {v}/{tot}" for k, v in cells.items())
      + f" -> {'PASS' if b1 else 'FAIL'}")

b2 = all(out[mode]["vggt"]["d_grav"] < 0
         and out[mode]["vggt"]["dev_positive_on"] > len(seqs) / 2
         for mode in out)
for mode in out:
    v = out[mode]["vggt"]
    print(f"B2  vggt {mode}: Delta_grav {v['d_grav']:+.2f}%, "
          f"Delta_dev positive on {v['dev_positive_on']}/{len(seqs)}")
print(f"B2  -> {'PASS' if b2 else 'FAIL'}")

from scipy.stats import spearmanr  # noqa: E402
b3_r = {}
for mode in out:
    sel = [r for r in rows if r["mode"] == mode]
    rho = spearmanr([r["mu"] for r in sel],
                    [r["d_dev"] - r["d_grav"] for r in sel]).correlation
    b3_r[mode] = float(rho)
    print(f"B3  {mode}: Spearman(mu, inflation) = {rho:+.3f}")
b3 = all(v <= -0.4 for v in b3_r.values())
print(f"B3  -> {'PASS' if b3 else 'FAIL'}")

(HERE / "results" / "summary.json").write_text(json.dumps(
    {"recordings": seqs, "mu": {s: data[s]["mu"] for s in seqs},
     "per_mode": out, "spearman": b3_r,
     "bars": {"b1": b1, "b2": b2, "b3": b3}}, indent=2))
print(f"\n[h43] wrote {HERE / 'results' / 'summary.json'}")
