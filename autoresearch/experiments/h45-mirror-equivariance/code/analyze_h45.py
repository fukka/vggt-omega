"""H45 — how far from mirror-equivariant are these models? Three locked bars."""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
MODELS = ["da3:small", "da3:large", "vggt", "vggt_omega"]

rows: dict[str, dict] = {}
for f in sorted(glob.glob(str(RES / "*_seq*.json"))):
    d = json.loads(Path(f).read_text())
    s = "seq" + d["seq"].split("_seq")[1].split("_")[0]
    for m, v in d["models"].items():
        # seq136 is run by both halves; keep one.
        rows.setdefault(s, {}).setdefault(m, v)

seqs = sorted(s for s in rows if all(m in rows[s] for m in MODELS))
print(f"[h45] {len(seqs)} recordings: {', '.join(seqs)}\n")

plumb = max(rows[s][m]["plumbing_max_rel_pct"] for s in seqs for m in MODELS)
b3 = plumb < 0.01
print(f"B3  worst flip-twice deviation anywhere: {plumb:.6f}% -> "
      f"{'PASS' if b3 else 'FAIL — nothing here is a measurement'}\n")

def ratio_of_means(v):
    """The citable statistic. `b0_mirror_cost_pct` in the JSONs is the MEAN OF
    PER-FRAME RATIOS, which H33 and H37 both recorded as the wrong one; on Aria
    it inflates by 10-60 points. See ../correction.md."""
    pf = v["per_frame"]
    n = [x for x, y in zip(pf["normal|0.0"], pf["mirror|0.0"])
         if x not in (None, 0) and y is not None]
    mm = [y for x, y in zip(pf["normal|0.0"], pf["mirror|0.0"])
          if x not in (None, 0) and y is not None]
    return 100 * (sum(mm) / sum(n) - 1)


out = {}
print(f"{'backbone':<12}{'mirror cost':>14}{'sd':>8}{'min':>9}{'max':>9}"
      f"{'as-reported':>13}")
for m in MODELS:
    v = [ratio_of_means(rows[s][m]) for s in seqs]
    old_v = [rows[s][m]["b0_mirror_cost_pct"] for s in seqs]
    out[m] = {"mean": st.mean(v), "sd": st.stdev(v), "min": min(v), "max": max(v),
              "n": len(v), "mean_of_ratios_as_reported": st.mean(old_v),
              "per_seq": {s: r for s, r in zip(seqs, v)}}
    o = out[m]
    print(f"{m:<12}{o['mean']:>+13.1f}%{o['sd']:>8.1f}{o['min']:>+8.1f}%"
          f"{o['max']:>+8.1f}%{o['mean_of_ratios_as_reported']:>+12.1f}%")

da3 = [out[m]["mean"] for m in ("da3:small", "da3:large")]
vg = [out[m]["mean"] for m in ("vggt", "vggt_omega")]
b1 = min(da3) > max(vg)
b2 = out["da3:small"]["mean"] > 100.0
print(f"\nB1  both DA3 above both VGGT: min(DA3) {min(da3):.1f}% vs "
      f"max(VGGT) {max(vg):.1f}% -> {'PASS' if b1 else 'FAIL'}")
print(f"B2  da3:small mean {out['da3:small']['mean']:.1f}% > 100% -> "
      f"{'PASS' if b2 else 'FAIL'}")

(RES / "summary.json").write_text(json.dumps(
    {"recordings": seqs, "per_model": out, "worst_plumbing_pct": plumb,
     "bars": {"b1": b1, "b2": b2, "b3": b3}}, indent=2))
print(f"\n[h45] wrote {RES / 'summary.json'}")
