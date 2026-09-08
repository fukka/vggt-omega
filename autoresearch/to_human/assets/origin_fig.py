"""H43 - the same four curves, read against two different zeros.

Delta about the true (gravity-aligned) zero, against Delta about the
device-aligned zero the published sweeps used. Same curves, same recordings;
only where zero is put changes.
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
ORDER = ["da3:small", "da3:large", "vggt", "vggt_omega"]
LABEL = {"da3:small": "DA3-Small", "da3:large": "DA3-Large",
         "vggt": "VGGT", "vggt_omega": "VGGT-Ω"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    S = json.loads(Path(a.summary).read_text())["per_model"]

    fig, ax = plt.subplots(figsize=(9.4, 4.0), dpi=190)
    ys = np.arange(len(ORDER))[::-1]
    h = 0.34
    for y, m in zip(ys, ORDER):
        gv, dv = S[m]["d_grav"], S[m]["d_dev"]
        ax.barh(y + h / 2 + .02, gv, height=h, color=TEAL, alpha=.88)
        ax.barh(y - h / 2 - .02, dv, height=h, color=RED, alpha=.88)
        for v, yy in ((gv, y + h / 2 + .02), (dv, y - h / 2 - .02)):
            ax.text(v + (0.5 if v >= 0 else -0.5), yy, f"{v:+.1f}%",
                    ha="left" if v >= 0 else "right", va="center",
                    color=TEAL if yy > y else RED, fontsize=10,
                    fontweight="bold")
        ax.text(-13.8, y, LABEL[m], ha="right", va="center", color=INK,
                fontsize=11, fontweight="bold")

    ax.axvline(0, color="#C9CDD0", lw=1.2, zorder=1)
    ax.set_xlim(-13, 27)
    ax.set_ylim(-0.75, 3.95)
    ax.set_yticks([])
    ax.set_xticks([-10, -5, 0, 5, 10, 15, 20, 25])
    ax.set_xticklabels(["−10%", "−5%", "0", "+5%", "+10%", "+15%", "+20%", "+25%"])
    ax.tick_params(colors=GREY, labelsize=9.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D5D8DA")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=TEAL, alpha=.88,
                             label="measured about the TRUE zero (gravity-aligned)"),
                       Patch(facecolor=RED, alpha=.88,
                             label="measured about the DEVICE zero — what every published sweep used")],
              frameon=False, fontsize=9.4, loc="upper left",
              bbox_to_anchor=(0.0, 1.10), labelcolor=GREY, handlelength=1.2)
    ax.set_xlabel("cost at +22° minus cost at −22°   (positive = the +side is dearer)",
                  color=INK, fontsize=10, labelpad=8)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
