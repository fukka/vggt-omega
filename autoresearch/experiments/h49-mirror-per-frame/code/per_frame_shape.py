"""H49 - is the Aria mirror effect carried by every frame, or by a subset?

EXPLORATORY. Post-hoc analysis of committed per-frame data from H45, H46 and
H47; no bars were locked before looking, and it is labelled that way.

The question matters because H46's four candidate differences include "content".
If particular scene configurations carried the effect - a hand in view, a
particular layout - the per-frame distribution would be tail-driven or bimodal.
If instead nearly every frame degrades, that rules out the SUBSET form of the
content explanation, though not a pervasive property of egocentric imagery.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
EXP = HERE.parents[0]
MODELS = ["da3:small", "da3:large", "vggt", "vggt_omega"]
SETS = [
    ("Aria 60° border-free (H47)", str(EXP / "h47-mirror-border/results/*_seq*.json"),
     "normal|0.0", "mirror|0.0"),
    ("Aria 89° (H45)", str(EXP / "h45-mirror-equivariance/results/*_seq*.json"),
     "normal|0.0", "mirror|0.0"),
    ("ScanNet++ rectified 89° (H46)", str(EXP / "h46-mirror-external/results/R*_*.json"),
     "normal", "mirror"),
]


def collect(pat, kn, km):
    per = {m: {"n": [], "m": []} for m in MODELS}
    for f in sorted(glob.glob(pat)):
        d = json.loads(Path(f).read_text())
        for m, v in d["models"].items():
            if m not in per:
                continue
            pf = v["per_frame"]
            a = pf[kn] if kn in pf else pf["normal"]
            b = pf[km] if km in pf else pf["mirror"]
            for x, y in zip(a, b):
                if x in (None, 0) or y is None:
                    continue
                per[m]["n"].append(x); per[m]["m"].append(y)
    return per


out = {}
for label, pat, kn, km in SETS:
    per = collect(pat, kn, km)
    print(f"--- {label} ---")
    print(f"{'backbone':<12}{'frames':>7}{'worse':>8}{'>1.5x':>8}"
          f"{'median':>9}{'p10':>7}{'p90':>7}{'corr(base,r)':>14}")
    out[label] = {}
    for m in MODELS:
        n = np.array(per[m]["n"]); mm = np.array(per[m]["m"])
        if len(n) < 5:
            continue
        r = mm / n
        rec = {"frames": int(len(n)), "frac_worse": float(np.mean(r > 1)),
               "frac_over_1p5": float(np.mean(r > 1.5)),
               "median_ratio": float(np.median(r)),
               "p10": float(np.percentile(r, 10)),
               "p90": float(np.percentile(r, 90)),
               "corr_baseline_ratio": float(np.corrcoef(n, r)[0, 1])}
        out[label][m] = rec
        print(f"{m:<12}{rec['frames']:>7}{100*rec['frac_worse']:>7.0f}%"
              f"{100*rec['frac_over_1p5']:>7.0f}%{rec['median_ratio']:>9.2f}"
              f"{rec['p10']:>7.2f}{rec['p90']:>7.2f}"
              f"{rec['corr_baseline_ratio']:>14.2f}")
    print()

(HERE / "results" / "summary.json").write_text(json.dumps(out, indent=2))
print(f"[h49] wrote {HERE / 'results' / 'summary.json'}")
