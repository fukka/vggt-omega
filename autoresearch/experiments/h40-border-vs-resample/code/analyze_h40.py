"""H40 — border x resampling, 2x2, at a matched 30 degrees."""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
MODELS = ["da3:small", "vggt_omega"]
ARMS = ["rt", "masked", "rt_masked"]
H39B_S = {"da3:small": 3.03, "vggt_omega": -0.48}   # real-roll price, H39b

rows: dict[str, dict] = {}
for f in sorted(glob.glob(str(RES / "*_seq*.json"))):
    d = json.loads(Path(f).read_text())
    s = "seq" + d["seq"].split("_seq")[1].split("_")[0]
    rows.setdefault(s, {}).update(d["models"])

have = sorted(s for s in rows if all(m in rows[s] for m in MODELS))
void = [(s, m) for s in have for m in MODELS
        if rows[s][m]["black_rotated"] > 0.002]
print(f"[h40] {len(have)} recordings: {', '.join(have)}")
print(f"[h40] construction check: "
      + ("OK, no black in any crop" if not void else f"VOID -> {void[:4]}") + "\n")

out = {}
for m in MODELS:
    o = {}
    for arm in ARMS + ["multiplicative"]:
        key = "multiplicative_pct" if arm == "multiplicative" else None
        v = [rows[s][m]["cost_pct"][arm] if key is None else rows[s][m][key]
             for s in have]
        o[arm] = (st.mean(v), st.stdev(v))
    o["ratio_border_over_resample"] = (
        o["masked"][0] / o["rt"][0] if abs(o["rt"][0]) > 1e-9 else float("nan"))
    o["rt_beats_zero_on"] = sum(1 for s in have if rows[s][m]["cost_pct"]["rt"] > 0)
    out[m] = o
    print(f"--- {m} ---")
    print(f"  resampling alone   {o['rt'][0]:+7.2f}% ± {o['rt'][1]:5.2f}"
          f"   (positive on {o['rt_beats_zero_on']}/{len(have)})")
    print(f"  border alone       {o['masked'][0]:+7.2f}% ± {o['masked'][1]:5.2f}")
    print(f"  both               {o['rt_masked'][0]:+7.2f}% ± {o['rt_masked'][1]:5.2f}"
          f"   (multiplicative prediction {o['multiplicative'][0]:+.2f}%)")

print()
# B1 — border at least 4x resampling, at a matched 30 deg.
b1 = []
for m in MODELS:
    r, b = out[m]["rt"][0], out[m]["masked"][0]
    ok = b > 4 * abs(r) if r != 0 else b > 0
    b1.append(ok)
    print(f"B1  {m:<12} border {b:+.2f}% vs 4x|resampling| {4*abs(r):.2f}%"
          f" -> {'PASS' if ok else 'FAIL'}")
# B2 — is the 30-degree resampling price above H39b's real-roll price?
for m in MODELS:
    print(f"B2  {m:<12} resampling at 30° {out[m]['rt'][0]:+.2f}% vs H39b's "
          f"real-roll price {H39B_S[m]:+.2f}% -> "
          f"{'above' if out[m]['rt'][0] > H39B_S[m] else 'BELOW'}")
# B3 — do the two terms compose?
b3 = []
for m in MODELS:
    got, pred = out[m]["rt_masked"][0], out[m]["multiplicative"][0]
    rel = abs(got - pred) / max(abs(pred), 1e-9)
    b3.append(rel <= 0.30)
    print(f"B3  {m:<12} both {got:+.2f}% vs multiplicative {pred:+.2f}% "
          f"(rel {rel:.2f}) -> {'PASS' if rel <= 0.30 else 'FAIL'}")

(RES / "summary.json").write_text(json.dumps(
    {"recordings": have, "void_cells": void, "per_model": out,
     "bars": {"b1": all(b1), "b3": all(b3)}}, indent=2))
print(f"\n[h40] wrote {RES / 'summary.json'}")
