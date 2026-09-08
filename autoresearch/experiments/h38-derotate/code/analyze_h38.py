"""H38 — is de-rotation a net win once the border it creates is paid for?"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
SEQS = ["seq136", "seq137", "seq138", "seq140", "seq141", "seq142"]
MODELS = ["vggt_omega", "vggt", "da3:small", "da3:large"]
ANG = ["30.0", "-30.0"]

rows: dict[str, dict] = {}
for f in sorted(glob.glob(str(RES / "*_seq*.json"))):
    d = json.loads(Path(f).read_text())
    s = "seq" + d["seq"].split("_seq")[1].split("_")[0]
    rows.setdefault(s, {}).update(d["summary"])

have = [s for s in SEQS if len(rows.get(s, {})) == 4]
print(f"[h38] {len(have)} recordings with all four backbones: {', '.join(have)}\n")

out = {}
for m in MODELS:
    out[m] = {}
    print(f"--- {m} ---")
    for k in ANG:
        rc = [rows[s][m]["angles"][k]["roll_cost_pct"] for s in have]
        dv = [rows[s][m]["angles"][k]["derot_vs_raw_pct"] for s in have]
        nv = [rows[s][m]["angles"][k]["null_vs_raw_pct"] for s in have]
        d0 = [rows[s][m]["angles"][k]["derot_vs_raw0_pct"] for s in have]
        out[m][k] = {
            "roll_cost": [st.mean(rc), st.stdev(rc)],
            "derot_vs_raw": [st.mean(dv), st.stdev(dv)],
            "null_vs_raw": [st.mean(nv), st.stdev(nv)],
            "derot_vs_raw0": [st.mean(d0), st.stdev(d0)],
            "derot_helps_n": sum(1 for x in dv if x < 0),
        }
        e = out[m][k]
        print(f"  {k:>6}°  roll cost {st.mean(rc):+6.1f}% ± {st.stdev(rc):4.1f}"
              f" | de-rotating vs raw {st.mean(dv):+6.1f}% ± {st.stdev(dv):4.1f}"
              f" (helps {e['derot_helps_n']}/{len(have)})"
              f" | price alone {st.mean(nv):+6.1f}%"
              f" | vs level camera {st.mean(d0):+6.1f}%")

maj = len(have) // 2 + 1
b1 = all(out["da3:small"][k]["derot_helps_n"] >= maj for k in ANG)
b2_gap = {k: out["da3:large"][k]["derot_vs_raw0"][0] for k in ANG}
b2 = all(v > 5.0 for v in b2_gap.values())
price = {m: st.mean([out[m][k]["null_vs_raw"][0] for k in ANG]) for m in MODELS}
b3 = max(price, key=price.get) == "da3:large"

print(f"\nB1  de-rotation reduces error for da3:small at both ±30 on a majority "
      f"({', '.join(str(out['da3:small'][k]['derot_helps_n']) + '/' + str(len(have)) for k in ANG)})"
      f" -> {'PASS' if b1 else 'FAIL'}")
print(f"B2  da3:large de-rotated is still above its level-camera reference by "
      f"{b2_gap['30.0']:+.1f}% / {b2_gap['-30.0']:+.1f}% (falsified under +5%)"
      f" -> {'PASS' if b2 else 'FAIL'}")
print(f"B3  pure price of the operation, most damaged = "
      f"{max(price, key=price.get)}  ({ {k: round(v,1) for k,v in price.items()} })"
      f" -> {'PASS' if b3 else 'FAIL'}")

(RES / "summary.json").write_text(json.dumps(
    dict(recordings=have, per_model=out, price=price,
         bars=dict(b1=b1, b2=b2, b2_gap=b2_gap, b3=b3)), indent=2))
print(f"\n[h38] wrote {RES / 'summary.json'}")
