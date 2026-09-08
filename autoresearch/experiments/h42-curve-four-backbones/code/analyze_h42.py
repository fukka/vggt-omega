"""H42 — the residual-roll curve for four backbones, and the three locked bars."""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
H41 = HERE.parents[0] / "h41-residual-curve" / "results"
MODELS = ["da3:small", "da3:large", "vggt", "vggt_omega"]
# H39b's measured values for vggt_omega, pooled over 13 recordings.
OMEGA_MEASURED = {"A": 0.56, "S": -0.48}


def load(pattern, keep=None):
    out: dict[str, dict[float, list[float]]] = {}
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        for m, v in d["models"].items():
            if keep and m not in keep:
                continue
            for k, (mean, sd, n) in v["g"].items():
                out.setdefault(m, {}).setdefault(-float(k), []).append(mean)
    return out


curves = load(str(H41 / "*_seq*.json"), keep={"da3:small"})
for m, c in load(str(RES / "*_seq*.json")).items():
    curves[m] = c

G, XS = {}, None
for m in MODELS:
    if m not in curves:
        print(f"[h42] {m}: missing"); continue
    xs = sorted(curves[m])
    XS = xs if XS is None else XS
    G[m] = {x: st.mean(curves[m][x]) for x in xs}

print(f"[h42] {len(G)} backbones, {len(XS)} residual angles\n")
hdr = f"{'residual':>9}" + "".join(f"{m:>13}" for m in MODELS if m in G)
print(hdr)
for x in XS:
    print(f"{x:>+8.1f}°" + "".join(f"{G[m][x]:>13.4f}" for m in MODELS if m in G))


def g_at(m, r):
    xs = sorted(G[m])
    return float(np.interp(r, xs, [G[m][x] for x in xs]))


print()
b1_ok = True
ends = {m: (G[m][-22.0], G[m][22.0]) for m in G}
for m, (lo, hi) in ends.items():
    print(f"  {m:<12} g(-22°) {lo:.4f}   g(+22°) {hi:.4f}")
for v in ("vggt", "vggt_omega"):
    for d in ("da3:small", "da3:large"):
        if v in ends and d in ends:
            if not (ends[v][0] < ends[d][0] and ends[v][1] < ends[d][1]):
                b1_ok = False
print(f"B1  both VGGT variants below both DA3 variants at ±22° -> "
      f"{'PASS' if b1_ok else 'FAIL'}")

psi = 11.0
if "vggt_omega" in G:
    g0, gp, g2 = g_at("vggt_omega", 0), g_at("vggt_omega", psi), g_at("vggt_omega", 2 * psi)
    A_pred = 100 * (g2 - g0) / (2 * gp)
    S_pred = 100 * ((g0 + g2) / (2 * gp) - 1)
    b2 = abs(A_pred) < 5.0 and abs(S_pred) < 3.0
    print(f"B2  vggt_omega from its own curve: A_pred {A_pred:+.2f}% "
          f"(measured {OMEGA_MEASURED['A']:+.2f}%), S_pred {S_pred:+.2f}% "
          f"(measured {OMEGA_MEASURED['S']:+.2f}%) -> {'PASS' if b2 else 'FAIL'}")
else:
    b2 = False; A_pred = S_pred = float("nan")

b3_ok, mins = True, {}
for m in G:
    lo = min(G[m], key=lambda x: G[m][x])
    mins[m] = lo
    if abs(lo) > 2.0:
        b3_ok = False
print(f"B3  minimum at residual 0 (±2° grid) for every backbone: "
      + ", ".join(f"{m} @ {mins[m]:+.0f}°" for m in mins)
      + f" -> {'PASS' if b3_ok else 'FAIL'}")

(RES / "summary.json").write_text(json.dumps(
    {"g": {m: {str(k): v for k, v in G[m].items()} for m in G},
     "ends": {m: list(ends[m]) for m in ends}, "minima": mins,
     "omega_pred": {"A": A_pred, "S": S_pred},
     "bars": {"b1": b1_ok, "b2": b2, "b3": b3_ok}}, indent=2))
print(f"\n[h42] wrote {RES / 'summary.json'}")
