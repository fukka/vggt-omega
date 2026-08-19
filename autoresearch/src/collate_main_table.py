"""Collate every available eval JSON into the paper's main-table draft.

Reads whatever exists (local results + autoresearch/data pulls + future
#35-#38 deliveries once fetched into autoresearch/data/), emits a markdown
table with provenance per row. Missing rows print as pending — run it any
time; it fills in as tickets land.
"""

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Z = "near_rim(<=2m,>=38deg)"


def zone(d, key="zones"):
    z = d.get(key, {}).get(Z)
    return (z["before"], z["after"]) if z else None


def main() -> None:
    rows = []
    # rung 0/1, within-scene (seq131, local)
    H2 = ROOT / "experiments" / "h2-center-safe-adapter" / "results"
    r10 = json.load(open(H2 / "run_010_even_odd.json"))
    r11 = json.load(open(H2 / "run_011_even_odd.json"))
    rows.append(("frozen DA3-S (seq131 anchor)", r11["zones"][Z]["before"],
                 None, "run_011 before-arm"))
    rows.append(("rung0 48-param table", None, r10["zones"][Z]["after"],
                 "run_010_even_odd"))
    rows.append(("rung1 head (within-scene)", None, r11["zones"][Z]["after"],
                 "run_011_even_odd"))
    # six-seq + cross-scene means
    six = [json.load(open(p)) for p in
           sorted((ROOT / "data" / "h22-sixseq").glob("run_011_*_halves.json"))]
    if six:
        b = sum(d["zones"][Z]["before"] for d in six) / len(six)
        a = sum(d["zones"][Z]["after"] for d in six) / len(six)
        rows.append((f"rung1 head (6-seq mean, halves)", b, a, "ticket #29"))
    cross = []
    for p in sorted((ROOT / "data" / "h23-crossscene").glob("run_012_fold_*.json")):
        d = json.load(open(p))
        for ev in d["eval"].values():
            cross.append((ev["near_rim"]["before"], ev["near_rim"]["after"]))
    if cross:
        b = sum(c[0] for c in cross) / len(cross)
        a = sum(c[1] for c in cross) / len(cross)
        rows.append(("rung1 head (cross-scene, 6 folds)", b, a, "ticket #32"))
    om = [json.load(open(p)) for p in
          sorted((ROOT / "data" / "h24-omega").glob("run_013_*_halves.json"))]
    if om:
        b = sum(d["near_rim"]["before"] for d in om) / len(om)
        a = sum(d["near_rim"]["after"] for d in om) / len(om)
        rows.append(("rung1 head on VGGT-Omega (6-seq, halves)", b, a,
                     "ticket #33"))
    # pending slots (fill when fetched into data/)
    pending = [
        ("rung2 LoRA full (held-out scenes)", "ticket #35"),
        ("rung2 plain-LoRA control", "ticket #35"),
        ("rung3 peripheral attention (dense)", "ticket #36"),
        ("rung3 all-token control", "ticket #36"),
        ("frozen UniK3D/DAC/DA3-L/VGGT-Omega/DAv2 rows", "ticket #37"),
        ("RayTun3R per-scene adaptation", "ticket #38"),
    ]
    for pat, src in (("data/h5-train/eval_*.json", "#35"),
                     ("data/h6-train/eval_*.json", "#36"),
                     ("data/bench/*.json", "#37"),
                     ("data/bench/rt3r/*.json", "#38")):
        for p in sorted(ROOT.glob(pat)):
            d = json.load(open(p))
            zz = zone(d)
            if zz:
                rows.append((p.stem, zz[0], zz[1], src))

    print("| method | near-rim before | after | Δ | source |")
    print("|---|---|---|---|---|")
    for name, b, a, src in rows:
        bs = f"{b:.3f}" if b is not None else "—"
        as_ = f"{a:.3f}" if a is not None else "—"
        d = f"{(a - b) / b * 100:+.0f}%" if (a is not None and b) else "—"
        print(f"| {name} | {bs} | {as_} | {d} | {src} |")
    for name, src in pending:
        print(f"| {name} | ⏳ | ⏳ | ⏳ | {src} (pending) |")


if __name__ == "__main__":
    main()
