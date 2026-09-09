"""H48 — assemble the 2x2 and read it, with the verdict logic fixed in advance.

Committed BEFORE the results exist, which is this line's standing rule: the
reading must not be shaped by the numbers it will read.

    Aria content,      Aria lens        = H47  (measured, hard-coded below)
    ScanNet++ content, Aria lens        = A_*  (this experiment, forward)
    Aria content,      ScanNet++ lens   = R_*  (this experiment, reciprocal)
    ScanNet++ content, ScanNet++ lens   = H46  (measured, hard-coded below)

All figures are the RATIO OF MEANS (H33, H37): the mean of per-frame ratios is
not citable, and the runners record both so they cannot drift.
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
MODELS = ["da3:small", "da3:large", "vggt", "vggt_omega"]

# Measured cells, hard-coded so the comparison target cannot drift.
H47 = {"da3:small": 154.8, "da3:large": 261.4, "vggt": 45.7, "vggt_omega": 29.9}
H46 = {"da3:small": 13.5, "da3:large": 3.3, "vggt": -9.3, "vggt_omega": 1.6}


def load(prefix):
    per = {}
    for f in sorted(glob.glob(str(RES / f"{prefix}_*.json"))):
        d = json.loads(Path(f).read_text())
        for m, v in d["models"].items():
            per.setdefault(m, {"cost": [], "plumb": []})
            per[m]["cost"].append(
                v.get("mirror_cost_pct_ratio_of_means", v.get("mirror_cost_pct")))
            per[m]["plumb"].append(v["plumbing_max_rel_pct"])
    return per


fwd, rec = load("A"), load("R")
if not fwd and not rec:
    raise SystemExit("[h48] no results yet")

plumb = max([x for p in (fwd, rec) for m in p for x in p[m]["plumb"]] or [0])
print(f"[h48] plumbing bar: worst flip-twice deviation {plumb:.6f}% -> "
      f"{'PASS' if plumb < 0.01 else 'FAIL — nothing here is a measurement'}\n")

print(f"{'backbone':<12}{'Aria/Aria':>11}{'SN++ ct/Aria lens':>19}"
      f"{'Aria ct/SN++ lens':>19}{'SN++/SN++':>11}")
rows = {}
for m in MODELS:
    a = st.mean(fwd[m]["cost"]) if m in fwd else None
    r = st.mean(rec[m]["cost"]) if m in rec else None
    rows[m] = {"aria_aria": H47[m], "sca_content_aria_lens": a,
               "aria_content_sca_lens": r, "sca_sca": H46[m]}
    f = lambda v: f"{v:>+10.1f}%" if v is not None else f"{'--':>11}"
    print(f"{m:<12}{H47[m]:>+10.1f}%{f(a):>19}{f(r):>19}{H46[m]:>+10.1f}%")

# --- the reading, fixed in advance -----------------------------------------
# "follows the row" = the Aria-content cells are large and the ScanNet++-content
# cells are not, whichever lens. "follows the column" = the Aria-LENS cells are
# large. Judged on da3:small and da3:large, the two with a large effect to move.
BIG = 100.0   # per cent; H47's DA3 cells are 155 and 261, H46's are 13 and 3


def verdict():
    d = [m for m in ("da3:small", "da3:large")]
    have = all(rows[m]["sca_content_aria_lens"] is not None
               and rows[m]["aria_content_sca_lens"] is not None for m in d)
    if not have:
        return "INCOMPLETE — both resampled cells are needed"
    row_big = all(rows[m]["aria_content_sca_lens"] > BIG for m in d)
    col_big = all(rows[m]["sca_content_aria_lens"] > BIG for m in d)
    if row_big and not col_big:
        return "CONTENT — the effect travels with Aria's imagery, not its optics"
    if col_big and not row_big:
        return "LENS OR WARP — the effect travels with Aria's optics"
    if row_big and col_big:
        return ("BOTH CELLS LARGE — neither factor alone; the design cannot "
                "separate them and says so")
    return ("NEITHER CELL LARGE — the resampling itself destroys whatever "
            "carries the effect. Stated as a possible outcome in the protocol "
            "before the run; the 2x2 cannot answer the question.")


print(f"\nVERDICT: {verdict()}")
(RES / "summary.json").write_text(json.dumps(
    {"per_model": rows, "worst_plumbing_pct": plumb, "verdict": verdict(),
     "big_threshold_pct": BIG}, indent=2))
print(f"[h48] wrote {RES / 'summary.json'}")
