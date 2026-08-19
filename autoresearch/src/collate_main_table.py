"""Collate every available eval JSON into the paper's main-table draft (v2).

Emits ONE markdown table with the adaptation-data column the comparison
protocol requires. Missing rows print as pending — run any time; it fills
in as deliveries land in autoresearch/data/.

Rows and their schemas:
- run_010/run_011 (h2 results dir): zones{Z}{before/after}
- six-seq / cross-scene / omega pulls (data/h22*, h23*, h24*)
- Center-PH local anchor (bench results dir): zones{...}{absrel_covered}
- H5 evals (data/h5-train/eval_*.json): zones + pose (eval_lora schema)
- H6 evals (data/h6-train/eval_*.json): zones (eval_module schema)
- H6 KV probes (data/h6-train/probe_kv_*.json): report{kv}{zones}
- frozen rows (data/bench/meta.json): whole + near_rim per model/scene
- RayTun3R v2 rows (data/bench/rt3r/*_v2.json): joint/counts -> near_rim
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Z = "near_rim(<=2m,>=38deg)"
ZC = "center(<=11deg)"


def near_rim_from_joint(d, joint_key="joint"):
    t = np.array(d[joint_key])
    c = np.array(d["counts"])
    tm = d["theta_bin_mid_deg"]
    cells = [(i, j) for i in range(t.shape[0]) for j in range(min(2, t.shape[1]))
             if tm[i] >= 38]
    wg = sum(c[i, j] for i, j in cells)
    return (sum(t[i, j] * c[i, j] for i, j in cells) / wg) if wg else None


def main() -> None:
    rows = []  # (method, scene set, adaptation data, near_rim, center, pose, source)

    def add(m, a, nr, ce, po, src, scene="?"):
        rows.append((m, scene, a, nr, ce, po, src))

    H2 = ROOT / "experiments" / "h2-center-safe-adapter" / "results"
    r10 = json.load(open(H2 / "run_010_even_odd.json"))
    r11 = json.load(open(H2 / "run_011_even_odd.json"))
    add("frozen DA3-S (seq131 anchor)", "none",
        r11["zones"][Z]["before"], r11["zones"][ZC]["before"], None, "run_011", scene="seq131-odd")
    add("rung0 48-param table", "other-scene GT",
        r10["zones"][Z]["after"], r10["zones"][ZC]["after"], None, "run_010", scene="seq131-odd")
    add("rung1 head (within-scene)", "same-scene GT (frame split)",
        r11["zones"][Z]["after"], r11["zones"][ZC]["after"], None, "run_011", scene="seq131-odd")
    six = [json.load(open(p)) for p in
           sorted((ROOT / "data" / "h22-sixseq").glob("run_011_*_halves.json"))]
    if six:
        add("rung1 head (6-seq mean)", "same-scene GT (halves)",
            float(np.mean([d["zones"][Z]["after"] for d in six])),
            float(np.mean([d["zones"][ZC]["after"] for d in six])),
            None, "#29", scene="six-seq")
    cross = []
    for p in sorted((ROOT / "data" / "h23-crossscene").glob("run_012_fold_*.json")):
        for ev in json.load(open(p))["eval"].values():
            cross.append(ev["near_rim"]["after"])
    if cross:
        add("rung1 head (cross-scene folds)", "other-scene GT",
            float(np.mean(cross)), None, None, "#32", scene="five folds")

    cp = ROOT / "experiments" / "bench" / "results" / "centerph_seq131_odd.json"
    if cp.exists():
        d = json.load(open(cp))
        add("Center-PH (seq131, covered px only, 49.6% rim coverage)", "none",
            d["zones"][Z]["absrel_covered"], d["zones"][ZC]["absrel_covered"],
            None, "bench/centerph", scene="seq131-odd")

    # frozen rows (#37)
    bm = ROOT / "data" / "bench" / "meta.json"
    if bm.exists():
        res = json.load(open(bm))["results"]
        for m, scenes in res.items():
            add(f"frozen {m} (2 held-out mean)", "none",
                float(np.mean([s["near_rim_absrel"] for s in scenes.values()])),
                None, None, "#37", scene="held-out x2")

    # RayTun3R v2
    for p in sorted((ROOT / "data" / "bench" / "rt3r").glob("*_v2.json")):
        d = json.load(open(p))
        add(f"RayTun3R {p.stem}", "test-scene RGB (TTA)",
            near_rim_from_joint(d), None, None, "#38v2", scene="held-out")

    # H5 / H6 evals
    for p in sorted((ROOT / "data" / "h5-train").glob("eval_*.json")):
        d = json.load(open(p))
        po = d.get("pose", {})
        add(f"rung2 {p.stem}", "other-scene GT",
            d["zones"][Z]["after"], d["zones"][ZC]["after"],
            po.get("after", po) if po else None, "#35", scene=d["seq"])
        add(f"  (its before-arm)", "none",
            d["zones"][Z]["before"], d["zones"][ZC]["before"], None, "#35", scene=d["seq"])
    for p in sorted((ROOT / "data" / "h6-train").glob("eval_*.json")):
        d = json.load(open(p))
        add(f"rung3 {p.stem}", "other-scene GT (video)",
            d["zones"][Z]["after"], d["zones"][ZC]["after"], None, "#36", scene=d["seq"])
    for p in sorted((ROOT / "data" / "h6-train").glob("probe_kv_*.json")):
        d = json.load(open(p))
        for kv, r in d["report"].items():
            add(f"rung3 KV={kv} {d['seq']}", "other-scene GT (video)",
                r["zones"].get(Z), r["zones"].get(ZC), None, "#36 kv", scene=d["seq"])

    print("| method | scene set | adaptation data | near-rim | center | pose | src |")
    print("|---|---|---|---|---|---|---|")
    print("| NOTE: near-rim is NOT comparable across scene sets (held-out scenes have ~no <1m rim mass) | | | | | | |")
    for m, sc, a, nr, ce, po, src in rows:
        f = lambda x: f"{x:.3f}" if isinstance(x, float) else (str(x) if x else "—")
        print(f"| {m} | {sc} | {a} | {f(nr)} | {f(ce)} | {f(po)} | {src} |")
    pend = []
    if not list((ROOT / "data" / "h5-train").glob("eval_*.json")):
        pend.append("rung2 evals (#35, 8 JSON)")
    if not list((ROOT / "data" / "h6-train").glob("eval_*.json")):
        pend.append("rung3 evals (#36, 4 JSON + kv probes)")
    if not list((ROOT / "data" / "bench" / "rt3r").glob("*_v2.json")):
        pend.append("RayTun3R v2 rows (#38)")
    for x in pend:
        print(f"| ⏳ {x} | | | | | | pending |")


if __name__ == "__main__":
    main()
