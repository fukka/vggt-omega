"""H38 - what de-rotation removes, against what it charges.

Both bars are percentages of the same baseline (the model's own error on the
rolled view), averaged over the two signs and six recordings. The prize is the
roll penalty; the price is the `null` arm, which pays the identical resampling
and identical black corners while removing no roll at all.
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
    M = d["per_model"]

    fig, ax = plt.subplots(figsize=(9.4, 4.2), dpi=190)
    h = 0.34
    for i, m in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        # +30 only. H32's rule: the two signs differ (da3:small +45.4 vs
        # +27.8), so averaging them into one bar would hide exactly what that
        # rule exists to catch. +30 is the sign with the most to gain, i.e.
        # de-rotation's best case; -30 is in the table and is worse.
        prize = M[m]["30.0"]["roll_cost"][0]
        price = M[m]["30.0"]["null_vs_raw"][0]
        net = M[m]["30.0"]["derot_vs_raw"][0]
        ax.barh(y + h / 2 + .02, prize, height=h, color=TEAL, alpha=.85,
                edgecolor="none")
        ax.barh(y - h / 2 - .02, price, height=h, color=RED, alpha=.85,
                edgecolor="none")
        ax.text(prize + 4, y + h / 2 + .02, f"{prize:+.0f}%", color=TEAL,
                fontsize=9.5, va="center", fontweight="bold")
        ax.text(price + 4, y - h / 2 - .02, f"{price:+.0f}%", color=RED,
                fontsize=9.5, va="center", fontweight="bold")
        ax.text(-6, y, LABEL[m], color=INK, fontsize=11, ha="right",
                va="center", fontweight="bold")
        col = TEAL if net < 0 else RED
        ax.text(292, y + h / 2 + .02, f"net {net:+.0f}%", color=col,
                fontsize=10.5, ha="right", va="center", fontweight="bold")

    ax.set_xlim(-95, 300)
    ax.set_ylim(-0.72, 3.9)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_xticklabels(["0", "+50%", "+100%", "+150%", "+200%", "+250%"])
    ax.tick_params(colors=GREY, labelsize=9.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D5D8DA")
    ax.axvline(0, color="#D5D8DA", lw=1)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=TEAL, alpha=.85,
                             label="the prize — what a +30° roll costs you"),
                       Patch(facecolor=RED, alpha=.85,
                             label="the price — same crop and blur, no roll removed")],
              frameon=False, fontsize=9.5, loc="upper right",
              bbox_to_anchor=(1.0, 1.06), labelcolor=GREY, handlelength=1.2)
    ax.set_xlabel("percent of the model's own error on the rolled view  "
                  "(6 recordings, +30° — the sign with the most to gain)",
                  color=INK, fontsize=10, labelpad=8)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
