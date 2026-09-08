"""H37 - border cost for four backbones across thirteen held-out recordings.

Log x, because the four backbones span 45x. Each recording is a dot; the bar is
the thirteen-recording mean; the hollow marker is seq136, the single recording
every published number came from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RED, INK, GREY = "#0E7C86", "#A6412F", "#474E52", "#767E82"
ORDER = ["vggt_omega", "vggt", "da3:small", "da3:large"]
LABEL = {"vggt_omega": "VGGT-Ω", "vggt": "VGGT",
         "da3:small": "DA3-Small", "da3:large": "DA3-Large"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    d = json.loads(Path(a.summary).read_text())
    rows, held, stats = d["rows"], d["held_out"], d["stats"]

    fig, ax = plt.subplots(figsize=(9.4, 4.0), dpi=190)
    for i, m in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        v = [rows[s][m] for s in held]
        col = TEAL if m == "vggt_omega" else (RED if m == "da3:large" else GREY)
        ax.plot([min(v), max(v)], [y, y], color=col, lw=1.1, alpha=.45,
                solid_capstyle="round", zorder=1)
        ax.scatter(v, np.full(len(v), y), s=17, color=col, alpha=.55,
                   linewidths=0, zorder=2)
        ax.scatter([stats[m]["mean"]], [y], s=115, marker="|", color=col,
                   linewidths=2.6, zorder=4)
        ax.scatter([rows["seq136"][m]], [y], s=62, facecolors="none",
                   edgecolors=INK, linewidths=1.5, zorder=5)
        ax.text(min(v) * 0.72, y + 0.30, LABEL[m], color=col, fontsize=11,
                fontweight="bold", ha="right", va="center")
        ax.text(stats[m]["mean"], y - 0.34,
                f"{stats[m]['mean']:+.0f}% ± {stats[m]['sd']:.0f}",
                color=col, fontsize=9.4, ha="center", va="center")

    ax.set_xscale("log")
    ax.set_xlim(0.7, 900)
    ax.set_ylim(-0.72, 4.35)
    ax.set_yticks([])
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["+1%", "+3%", "+10%", "+30%", "+100%", "+300%"])
    ax.tick_params(colors=GREY, labelsize=9.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D5D8DA")
    ax.set_xlabel("Cost of one hard border beside the scored zone  (all-image AbsRel, log scale)",
                  color=INK, fontsize=10, labelpad=8)
    ax.scatter([], [], s=62, facecolors="none", edgecolors=INK, linewidths=1.5,
               label="seq136 — the published recording")
    ax.scatter([], [], s=17, color=GREY, linewidths=0, alpha=.55,
               label="each of 13 never-used recordings")
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.03),
              labelcolor=GREY, handletextpad=.5)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
