"""H39 — does folding the real gravity vector into the warp pay?"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
MODELS = ["da3:small", "vggt_omega"]
# H35's independently computed expected cost of ignoring roll, from integrating
# a synthetic roll curve against the measured distribution.
H35 = {"da3:small": 1.80, "vggt_omega": 0.46}
ARMS = ("device", "grav_p", "grav_m")

rows: dict[str, dict] = {}
for f in sorted(glob.glob(str(RES / "*_seq*.json"))):
    d = json.loads(Path(f).read_text())
    s = "seq" + d["seq"].split("_seq")[1].split("_")[0]
    r = rows.setdefault(s, {"roll": np.abs(np.array(d["roll_deg"]))})
    for m, v in d["models"].items():
        r[m] = v
    r.setdefault("_pf", {}).update(
        {m: v["per_frame"] for m, v in d["models"].items()})

have = sorted(s for s in rows if all(m in rows[s] for m in MODELS))
print(f"[h39] {len(have)} recordings: {', '.join(have)}\n")

out = {}
for m in MODELS:
    print(f"--- {m}  (H35 predicts {H35[m]:.2f}% available) ---")
    per = {"grav_p": [], "grav_m": []}
    for s in have:
        P = rows[s][m]["pooled"]
        gp = 100 * (P["grav_p"]["all"] / P["device"]["all"] - 1)
        gm = 100 * (P["grav_m"]["all"] / P["device"]["all"] - 1)
        per["grav_p"].append(gp); per["grav_m"].append(gm)
        print(f"  {s}  |roll| med {np.median(rows[s]['roll']):5.2f}°"
              f"   grav_p {gp:+6.2f}%   grav_m {gm:+6.2f}%")
    mp, mm = st.mean(per["grav_p"]), st.mean(per["grav_m"])
    better, worse = ("grav_p", "grav_m") if mp < mm else ("grav_m", "grav_p")
    gain = -st.mean(per[better])          # positive = an improvement
    penalty = st.mean(per[worse])
    out[m] = {"per_seq": per, "better": better,
              "gain_pct": gain, "sd": st.stdev(per[better]),
              "wrong_sign_pct": penalty,
              "gain_over_n": sum(1 for x in per[better] if x < 0)}
    print(f"  mean: {better} {-gain:+.2f}% ± {out[m]['sd']:.2f} "
          f"(improves on {out[m]['gain_over_n']}/{len(have)}), "
          f"{worse} {penalty:+.2f}%")

    # B3 — does the gain scale with the roll? Per-frame, pooled over recordings.
    lo, hi = [], []
    for s in have:
        pf = rows[s][m]["per_frame"]
        roll = rows[s]["roll"]
        for k in range(len(roll)):
            dv, bv = pf["device"][k], pf[better][k]
            if dv is None or bv is None or dv <= 0:
                continue
            imp = 100 * (1 - bv / dv)
            (hi if roll[k] >= 8 else (lo if roll[k] < 3 else [])).append(imp)
    out[m]["frames_lo"], out[m]["frames_hi"] = len(lo), len(hi)
    out[m]["imp_lo"] = st.mean(lo) if lo else float("nan")
    out[m]["imp_hi"] = st.mean(hi) if hi else float("nan")
    print(f"  per-frame improvement: |roll|<3° {out[m]['imp_lo']:+.2f}% (n={len(lo)})"
          f"   |roll|>=8° {out[m]['imp_hi']:+.2f}% (n={len(hi)})")

print()
for m in MODELS:
    o = out[m]
    b1 = (o["gain_pct"] > 0) and (H35[m] / 3 <= o["gain_pct"] <= H35[m] * 3)
    b2 = o["wrong_sign_pct"] >= 3 * abs(o["gain_pct"])
    b3 = o["imp_hi"] > o["imp_lo"]
    o["b1"], o["b2"], o["b3"] = b1, b2, b3
    print(f"{m:<12} B1 {o['gain_pct']:+.2f}% vs H35's {H35[m]:.2f}% "
          f"[{H35[m]/3:.2f}, {H35[m]*3:.2f}] -> {'PASS' if b1 else 'FAIL'}"
          f" | B2 wrong sign {o['wrong_sign_pct']:+.2f}% vs 3x "
          f"{3*abs(o['gain_pct']):.2f} -> {'PASS' if b2 else 'FAIL'}"
          f" | B3 {o['imp_hi']:+.2f} > {o['imp_lo']:+.2f} -> {'PASS' if b3 else 'FAIL'}")

(RES / "summary.json").write_text(json.dumps(
    {"recordings": have, "per_model": {m: {k: v for k, v in out[m].items()}
                                       for m in MODELS}, "h35": H35}, indent=2))
print(f"\n[h39] wrote {RES / 'summary.json'}")
