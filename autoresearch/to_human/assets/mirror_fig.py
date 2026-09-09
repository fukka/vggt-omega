"""H45 - what it costs to show these models a reflection of their own input.

D(flip(I)) should equal flip(D(I)) for a model with no preferred handedness.
Each dot is one recording; the bar is the thirteen-recording mean. Log axis,
because the two pretraining families are 4-6x apart.
"""
from __future__ import annotations

import argparse
import glob
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
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    S = json.loads(Path(a.summary).read_text())["per_model"]

    # Per-recording RATIO OF MEANS, matching the summary. The JSONs' own
    # b0_mirror_cost_pct is the mean of per-frame ratios and is not citable
    # (H33, H37); see h45-mirror-equivariance/correction.md.
    per: dict[str, dict[str, float]] = {m: dict(S[m]["per_seq"]) for m in S}

    fig, ax = plt.subplots(figsize=(9.4, 3.9), dpi=190)
    for i, m in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        v = list(per[m].values())
        col = TEAL if m.startswith("vggt") else RED
        ax.plot([min(v), max(v)], [y, y], color=col, lw=1.1, alpha=.45, zorder=1)
        ax.scatter(v, np.full(len(v), y), s=17, color=col, alpha=.55,
                   linewidths=0, zorder=2)
        ax.scatter([S[m]["mean"]], [y], s=120, marker="|", color=col,
                   linewidths=2.6, zorder=4)
        ax.text(min(v) * 0.78, y + 0.30, LABEL[m], color=col, fontsize=11,
                fontweight="bold", ha="right", va="center")
        ax.text(S[m]["mean"], y - 0.34, f"{S[m]['mean']:+.0f}% ± {S[m]['sd']:.0f}",
                color=col, fontsize=9.6, ha="center", va="center")

    ax.set_xscale("log")
    ax.set_xlim(15, 700)
    ax.set_ylim(-0.8, 3.7)
    ax.set_yticks([])
    ax.set_xticks([20, 50, 100, 200, 400])
    ax.set_xticklabels(["+20%", "+50%", "+100%", "+200%", "+400%"])
    ax.tick_params(colors=GREY, labelsize=9.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D5D8DA")
    ax.set_xlabel("extra depth error from reflecting the model's input, then "
                  "reflecting the answer back  (ratio of means, log scale)",
                  color=INK, fontsize=10, labelpad=8)
    ax.text(0.015, 0.93, "13 recordings · one dot each",
            transform=ax.transAxes, ha="left", color=GREY, fontsize=9.2)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
