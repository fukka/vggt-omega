"""H46 — the mirror test on ScanNet++, raw frame and rectified view."""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
MODELS = ["da3:small", "da3:large", "vggt", "vggt_omega"]
# H45's Aria numbers, hard-coded so the comparison target cannot drift.
H45 = {"da3:small": 254.2, "da3:large": 350.1, "vggt": 67.0, "vggt_omega": 58.0}
H45_BASELINE = "0.05-0.15 AbsRel"


def load(prefixes):
    """Tags are <arm><gpu-half>: A/B for the raw arm, RA/RB for the rectified."""
    rows: dict[str, dict] = {}
    for pre in prefixes:
        for f in sorted(glob.glob(str(RES / f"{pre}_*.json"))):
            d = json.loads(Path(f).read_text())
            rows.setdefault(d["scene"], {}).update(d["models"])
    return rows


def report(rows, title):
    scenes = sorted(s for s in rows if all(m in rows[s] for m in MODELS))
    if not scenes:
        print(f"--- {title}: no complete scenes yet ---\n"); return None
    print(f"--- {title}: {len(scenes)} scenes ---")
    plumb = max(rows[s][m]["plumbing_max_rel_pct"] for s in scenes for m in MODELS)
    print(f"B1  worst flip-twice deviation: {plumb:.6f}% -> "
          f"{'PASS' if plumb < 0.01 else 'FAIL — nothing here is a measurement'}")
    out = {}
    print(f"{'backbone':<12}{'baseline':>10}{'mirror':>11}{'sd':>7}"
          f"{'range':>18}{'H45 (Aria)':>12}")
    for m in MODELS:
        v = [rows[s][m]["mirror_cost_pct"] for s in scenes]
        b = [rows[s][m]["normal_absrel"] for s in scenes]
        out[m] = {"mirror": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0,
                  "min": min(v), "max": max(v), "baseline": st.mean(b),
                  "n": len(v)}
        o = out[m]
        print(f"{m:<12}{o['baseline']:>10.3f}{o['mirror']:>+10.1f}%{o['sd']:>7.1f}"
              f"{o['min']:>+9.0f}..{o['max']:>+6.0f}{H45[m]:>+11.0f}%")
    da3 = [out[m]["mirror"] for m in ("da3:small", "da3:large")]
    vg = [out[m]["mirror"] for m in ("vggt", "vggt_omega")]
    b2 = min(da3) > 25.0
    b3 = min(da3) > max(vg)
    print(f"B2  both DA3 above +25%: min {min(da3):.1f}% -> {'PASS' if b2 else 'FAIL'}")
    print(f"B3  both DA3 above both VGGT: {min(da3):.1f}% vs {max(vg):.1f}% -> "
          f"{'PASS' if b3 else 'FAIL'}\n")
    return {"scenes": scenes, "per_model": out, "worst_plumbing_pct": plumb,
            "bars": {"b1": plumb < 0.01, "b2": b2, "b3": b3}}


p = argparse.ArgumentParser()
p.parse_args()
res = {}
for prefix, title in ((["A", "B"], "raw captured frame"),
                      (["RA", "RB"], "rectified 89° view")):
    r = report(load(prefix), title)
    if r:
        res[title] = r
print("H45 on Aria, for reference: baseline " + H45_BASELINE + ", mirror "
      + ", ".join(f"{m} {H45[m]:+.0f}%" for m in MODELS))
(RES / "summary.json").write_text(json.dumps(res, indent=2))
print(f"\n[h46] wrote {RES / 'summary.json'}")
