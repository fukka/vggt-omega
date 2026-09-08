"""H39b — split H39's two gravity arms into resampling (S) and roll (A).

    grav_p = S - A      grav_m = S + A
    S = (grav_p + grav_m)/2   the price: same for both rotation signs
    A = (grav_m - grav_p)/2   the prize: the roll actually removed

The operation pays exactly when A > S. Compare S against H38's price for the
same rotation WITH a border, which is what says how much of that price was the
border rather than the resampling.
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
MODELS = ["da3:small", "vggt_omega"]
# H38's `null` arm: the same rotation plus a border, +-30 deg, 6 recordings.
H38_PRICE = {"da3:small": 64.1, "vggt_omega": 7.4}

rows: dict[str, dict] = {}
for f in sorted(glob.glob(str(RES / "*_seq*.json"))):
    d = json.loads(Path(f).read_text())
    s = "seq" + d["seq"].split("_seq")[1].split("_")[0]
    r = rows.setdefault(s, {"roll": np.abs(np.array(d["roll_deg"]))})
    r.update(d["models"])

have = sorted(s for s in rows if all(m in rows[s] for m in MODELS))
print(f"[h39b] {len(have)} recordings: {', '.join(have)}\n")

out = {}
for m in MODELS:
    S_seq, A_seq = [], []
    for s in have:
        P = rows[s][m]["pooled"]
        gp = 100 * (P["grav_p"]["all"] / P["device"]["all"] - 1)
        gm = 100 * (P["grav_m"]["all"] / P["device"]["all"] - 1)
        S_seq.append((gp + gm) / 2)
        A_seq.append((gm - gp) / 2)
    S, A = st.mean(S_seq), st.mean(A_seq)

    lo_S, hi_S, lo_A, hi_A = [], [], [], []
    for s in have:
        pf = rows[s][m]["per_frame"]
        roll = rows[s]["roll"]
        for k in range(len(roll)):
            dv, pv, mv = pf["device"][k], pf["grav_p"][k], pf["grav_m"][k]
            if None in (dv, pv, mv) or dv <= 0:
                continue
            gp, gm = 100 * (pv / dv - 1), 100 * (mv / dv - 1)
            s_, a_ = (gp + gm) / 2, (gm - gp) / 2
            if roll[k] >= 8:
                hi_S.append(s_); hi_A.append(a_)
            elif roll[k] < 3:
                lo_S.append(s_); lo_A.append(a_)
    out[m] = {
        "S_pct": S, "S_sd": st.stdev(S_seq), "A_pct": A, "A_sd": st.stdev(A_seq),
        "net_pct": S - A, "pays": A > S,
        "h38_price_pct": H38_PRICE[m], "S_over_h38": S / H38_PRICE[m],
        "S_lo": st.mean(lo_S), "S_hi": st.mean(hi_S),
        "A_lo": st.mean(lo_A), "A_hi": st.mean(hi_A),
        "n_lo": len(lo_S), "n_hi": len(hi_S),
    }
    o = out[m]
    print(f"--- {m} ---")
    print(f"  price  S = {S:+6.2f}% ± {o['S_sd']:.2f}   "
          f"prize  A = {A:+6.2f}% ± {o['A_sd']:.2f}   "
          f"net {S - A:+6.2f}%  -> {'PAYS' if A > S else 'does not pay'}")
    # Do not print a ratio when S is not distinguishable from zero: dividing
    # H38's price by a near-zero S produced "7.4e9x cheaper" on the first run,
    # which is H33's divide-by-noise trap in a new costume.
    if S > o["S_sd"] / len(have) ** 0.5:
        print(f"  S against H38's price for the same rotation WITH a border "
              f"({H38_PRICE[m]:.1f}%): {S / H38_PRICE[m]:.3f} of it")
    else:
        print(f"  S ({S:+.2f}% ± {o['S_sd']:.2f}) is not distinguishable from "
              f"zero; no ratio against H38's {H38_PRICE[m]:.1f}% is quotable")
    o["S_quotable"] = bool(S > o["S_sd"] / len(have) ** 0.5)
    print(f"  per-frame  S: |roll|<3° {o['S_lo']:+5.2f}%  ->  >=8° {o['S_hi']:+5.2f}%"
          f"   (n {o['n_lo']}/{o['n_hi']})")
    print(f"  per-frame  A: |roll|<3° {o['A_lo']:+5.2f}%  ->  >=8° {o['A_hi']:+5.2f}%")

print()
# P1 only means something where S is distinguishable from zero.
p1_models = [m for m in MODELS if out[m]["S_quotable"]]
p1 = bool(p1_models) and all(out[m]["S_over_h38"] < 0.5 for m in p1_models)
p2 = all(out[m]["S_hi"] > out[m]["S_lo"] for m in MODELS)
p3 = all((out[m]["A_hi"] - out[m]["A_lo"]) > (out[m]["S_hi"] - out[m]["S_lo"])
         for m in MODELS)
undec = [m for m in MODELS if m not in p1_models]
print("P1  resampling alone is under half of H38's bordered price: "
      + ", ".join(f"{m} {out[m]['S_over_h38']:.3f}" for m in p1_models)
      + (f"   [undecided, S ~ 0: {', '.join(undec)}]" if undec else "")
      + f" -> {'PASS' if p1 else 'FAIL'}")
print(f"P2  S grows with |roll| on both backbones -> {'PASS' if p2 else 'FAIL'}")
print(f"P3  A's slope beats S's slope on both -> {'PASS' if p3 else 'FAIL'}")

(RES / "decomposition.json").write_text(json.dumps(
    {"recordings": have, "per_model": out,
     "predictions": {"P1": p1, "P2": p2, "P3": p3}}, indent=2))
print(f"\n[h39b] wrote {RES / 'decomposition.json'}")
