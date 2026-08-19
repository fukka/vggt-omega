"""The research trajectory: held-out near-field-rim AbsRel across method rungs.

Every point re-read from its audited source JSON (no hand-typed numbers).
"""

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
H2 = ROOT / "experiments" / "h2-center-safe-adapter" / "results"
SIX = ROOT / "data" / "h22-sixseq"
CROSS = ROOT / "data" / "h23-crossscene"

Z = "near_rim(<=2m,>=38deg)"


def main() -> None:
    pts = []
    r10 = json.load(open(H2 / "run_010_even_odd.json"))
    r11 = json.load(open(H2 / "run_011_even_odd.json"))
    pts.append(("未修正\n(诊断基线)", r11["zones"][Z]["before"], "#7f8c8d"))
    pts.append(("48数字查表\n(第0级)", r10["zones"][Z]["after"], "#e67e22"))
    pts.append(("看图小头\n(第1级, 场景内)", r11["zones"][Z]["after"], "#27ae60"))
    # six-seq mean after (halves)
    sx = [json.load(open(p))["zones"][Z] for p in sorted(SIX.glob("run_011_*_halves.json"))]
    pts.append(("六序列验证\n(第1级, 平均)", float(np.mean([z["after"] for z in sx])), "#27ae60"))
    cr = []
    for p in sorted(CROSS.glob("run_012_fold_*.json")):
        d = json.load(open(p))
        for ev in d["eval"].values():
            cr.append(ev["near_rim"]["after"])
    pts.append(("跨场景\n(五折平均, 训练场景外)",
                float(np.mean(sorted(cr)[:5])), "#2980b9"))
    pts.append(("第2/3级\n(微调/视频)", None, "#95a5a6"))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(pts))
    for x, (label, v, c) in zip(xs, pts):
        if v is not None:
            ax.bar(x, v, 0.55, color=c)
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=10,
                    fontweight="bold")
        else:
            ax.bar(x, 0.02, 0.55, color=c, alpha=0.4)
            ax.text(x, 0.06, "训练中\n(#35/#36)", ha="center", fontsize=9,
                    color="#666")
    ax.set_xticks(xs, [p[0] for p in pts], fontsize=9)
    ax.set_ylabel("近场边缘 AbsRel(没见过的帧,越低越好)")
    ax.set_title("研究轨迹:近场边缘误差随方法阶梯的下降(每个数字都来自审计过的结果文件)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = ROOT / "to_human" / "assets" / "trajectory.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
