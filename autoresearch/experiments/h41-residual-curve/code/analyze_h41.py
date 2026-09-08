"""H41 — the residual-roll curve, and the three locked bars against H39b."""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
M = "da3:small"
# H39b's measured values on frames with |psi| >= 8 deg (median ~11 deg).
H39B = {"A_hi": 21.52, "S_hi": 11.51, "A_lo": 0.26, "S_lo": 0.95}

# residual = -delta, pooled across recordings, each recording already normalised
# per frame by its own d = 0 render.
curve: dict[float, list[float]] = {}
seqs = set()
for f in sorted(glob.glob(str(RES / "*_seq*.json"))):
    d = json.loads(Path(f).read_text())
    if M not in d["models"]:
        continue
    seqs.add("seq" + d["seq"].split("_seq")[1].split("_")[0])
    g = d["models"][M]["g"]
    for k, (mean, sd, n) in g.items():
        curve.setdefault(-float(k), []).append(mean)

xs = sorted(curve)
print(f"[h41] {len(seqs)} recordings, {len(xs)} residual angles\n")
print(f"{'residual':>10}  {'g':>8}  {'sd':>6}  n")
G = {}
for x in xs:
    v = curve[x]
    G[x] = st.mean(v)
    print(f"{x:>+9.1f}°  {st.mean(v):>8.4f}  {st.stdev(v) if len(v)>1 else 0:>6.4f}  {len(v)}")


def g_at(r):
    """g at an arbitrary residual, linear between measured points."""
    return float(np.interp(r, xs, [G[x] for x in xs]))


print()
psi = 11.0
jensen = 0.5 * (g_at(0.0) + g_at(2 * psi)) - g_at(psi)
A_pred = 100 * (g_at(2 * psi) - g_at(0.0)) / (2 * g_at(psi))
S_pred = 100 * ((g_at(0.0) + g_at(2 * psi)) / (2 * g_at(psi)) - 1)
print(f"at psi = {psi:.0f}°:  g(0) {g_at(0):.4f}   g(psi) {g_at(psi):.4f}   "
      f"g(2psi) {g_at(2*psi):.4f}")
print(f"  Jensen gap {jensen:+.4f}")
print(f"  A_pred {A_pred:+.2f}%   (H39b measured {H39B['A_hi']:+.2f}%)")
print(f"  S_pred {S_pred:+.2f}%   (H39b measured {H39B['S_hi']:+.2f}%)")

psi_lo = 1.5
A_lo = 100 * (g_at(2 * psi_lo) - g_at(0.0)) / (2 * g_at(psi_lo))
S_lo = 100 * ((g_at(0.0) + g_at(2 * psi_lo)) / (2 * g_at(psi_lo)) - 1)
print(f"at psi = {psi_lo:.1f}°:  A_pred {A_lo:+.2f}% (measured {H39B['A_lo']:+.2f}%)"
      f"   S_pred {S_lo:+.2f}% (measured {H39B['S_lo']:+.2f}%)")

b1 = jensen > 0
b2 = H39B["A_hi"] / 2 <= A_pred <= H39B["A_hi"] * 2
b3 = H39B["S_hi"] / 2 <= S_pred <= H39B["S_hi"] * 2
print(f"\nB1  Jensen gap at psi=11 is positive: {jensen:+.4f} -> "
      f"{'PASS' if b1 else 'FAIL'}")
print(f"B2  A_pred {A_pred:.2f}% in [{H39B['A_hi']/2:.1f}, {H39B['A_hi']*2:.1f}] -> "
      f"{'PASS' if b2 else 'FAIL'}")
print(f"B3  S_pred {S_pred:.2f}% in [{H39B['S_hi']/2:.1f}, {H39B['S_hi']*2:.1f}] -> "
      f"{'PASS' if b3 else 'FAIL'}"
      + ("   (below -> the shortfall is a real operation price)"
         if S_pred < H39B["S_hi"] / 2 else ""))

# curvature and the near-zero shape, both reported whatever the bars do
sec = {}
for i in range(1, len(xs) - 1):
    h1, h2 = xs[i] - xs[i - 1], xs[i + 1] - xs[i]
    sec[xs[i]] = 2 * (G[xs[i - 1]] / (h1 * (h1 + h2))
                      - G[xs[i]] / (h1 * h2)
                      + G[xs[i + 1]] / (h2 * (h1 + h2)))
print("\nsecond difference (positive = convex):")
print("  " + "  ".join(f"{x:+.0f}°:{v:+.5f}" for x, v in sorted(sec.items())))
flat = [x for x in xs if abs(x) <= 6]
print(f"g within ±6°: " + ", ".join(f"{x:+.0f}°={G[x]:.4f}" for x in flat))

(RES / "summary.json").write_text(json.dumps(
    {"recordings": sorted(seqs), "g": {str(k): G[k] for k in xs},
     "second_difference": {str(k): v for k, v in sec.items()},
     "psi11": {"A_pred": A_pred, "S_pred": S_pred, "jensen": jensen},
     "psi1_5": {"A_pred": A_lo, "S_pred": S_lo},
     "bars": {"b1": b1, "b2": b2, "b3": b3}}, indent=2))
print(f"\n[h41] wrote {RES / 'summary.json'}")
