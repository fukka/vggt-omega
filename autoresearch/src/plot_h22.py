"""H2.2 figure: uncorrected vs 48-param table vs feature head, key zones."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans GB", "PingFang SC",
                               "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "h2-center-safe-adapter" / "results"
OUT = ROOT / "to_human" / "assets"


def main() -> None:
    t10 = json.load(open(RES / "run_010_even_odd.json"))
    t11 = json.load(open(RES / "run_011_even_odd.json"))
    # zones: uncorrected / table / head  (near_center for run_010 recomputed
    # from its tables is not stored; use run_011's before as the shared
    # uncorrected reference and run_010's zone deltas where defined)
    zones = ["near_rim(<=2m,>=38deg)", "near_center(<=2m,<=11deg)",
             "center(<=11deg)", "far(>=3m)"]
    labels = ["近处边缘\n(≤2m, ≥38°)", "近处中心\n(≤2m, ≤11°)",
              "中心整体\n(≤11°)", "远处\n(≥3m)"]
    before = [t11["zones"][z]["before"] for z in zones]
    head = [t11["zones"][z]["after"] for z in zones]
    # table zones: run_010 stored 3 zones; near_center recompute from tables
    b10 = np.array(t10["before"]); a10 = np.array(t10["after"])
    cnt = np.array(t10["counts"]); tm = t10["theta_bin_mid_deg"]
    edges = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
    def zone_from_tables(pred, cells):
        w = np.array([cnt[i, j] for i, j in cells], float)
        v = np.array([pred[i, j] for i, j in cells])
        return float((v * w).sum() / w.sum())
    cells_nc = [(i, j) for i in range(8) for j in range(5)
                if tm[i] <= 11 and edges[j + 1] <= 2.0]
    cells_map = {
        "near_rim(<=2m,>=38deg)": [(i, j) for i in range(8) for j in range(5)
                                   if tm[i] >= 38 and edges[j + 1] <= 2.0],
        "near_center(<=2m,<=11deg)": cells_nc,
        "center(<=11deg)": [(i, j) for i in range(8) for j in range(5) if tm[i] <= 11],
        "far(>=3m)": [(i, j) for i in range(8) for j in range(5) if edges[j] >= 3.0],
    }
    table = [zone_from_tables(a10, cells_map[z]) for z in zones]

    x = np.arange(len(zones))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w, before, w, label="不修正(原始模型)", color="#7f8c8d")
    ax.bar(x, table, w, label="48个数字的查表修正", color="#e67e22")
    ax.bar(x + w, head, w, label="看图小头修正(2.5万参数)", color="#27ae60")
    for xi, vals in zip(x, zip(before, table, head)):
        for k, v in enumerate(vals):
            ax.text(xi + (k - 1) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("AbsRel(没见过的帧上,越低越好)")
    ax.set_title("同一批没见过的帧:三种做法的深度误差(DA3-Small,Aria seq131)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "feature_head_h22.png", dpi=140)
    print(f"wrote {OUT / 'feature_head_h22.png'}")


if __name__ == "__main__":
    main()
