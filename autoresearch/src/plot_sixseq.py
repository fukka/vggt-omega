"""Six-sequence validation figure: near-rim AbsRel before/after per sequence."""

import glob
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
DATA = ROOT / "data" / "h22-sixseq"
OUT = ROOT / "to_human" / "assets"


def main() -> None:
    rows = []
    for p in sorted(glob.glob(str(DATA / "run_011_*_halves.json"))):
        d = json.load(open(p))
        z = d["zones"]["near_rim(<=2m,>=38deg)"]
        name = Path(p).stem.replace("run_011_Apartment_release_", "") \
            .replace("_M1292_halves", "")
        rows.append((name, z["before"], z["after"]))
    names = [r[0] for r in rows]
    before = [r[1] for r in rows]
    after = [r[2] for r in rows]

    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.bar(x - w / 2, before, w, label="修正前", color="#7f8c8d")
    ax.bar(x + w / 2, after, w, label="修正后(看图小头)", color="#27ae60")
    for xi, (b, a) in zip(x, zip(before, after)):
        ax.text(xi - w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi, max(b, a) + 0.06, f"{(a - b) / b * 100:+.0f}%",
                ha="center", fontsize=9, fontweight="bold", color="#1e8449")
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8.5)
    ax.set_ylabel("近处边缘 AbsRel(没见过的帧,越低越好)")
    ax.set_title("六条 ADT 序列独立验证(GPU,#29):近处边缘误差,修正前 vs 修正后\n"
                 "(前后帧划分;另一种划分结果相同方向)")
    ax.set_ylim(0, max(before) * 1.22)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "sixseq_h22.png", dpi=140)
    print(f"wrote {OUT / 'sixseq_h22.png'}")


if __name__ == "__main__":
    main()
