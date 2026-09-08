"""H42 - the residual-roll curve for all four backbones, same axes.

Every frame is rendered at roll_deg = psi + d, so the residual is exactly -d and
d = 0 is that frame's own gravity-aligned reference. Each curve is that
backbone's error divided by its OWN level error, so the four are directly
comparable in shape even though their absolute accuracies differ by 3x.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GREY = "#474E52", "#767E82"
STYLE = {
    "da3:small":  ("#A6412F", "DA3-Small",  "-"),
    "da3:large":  ("#C8795F", "DA3-Large",  "-"),
    "vggt":       ("#3E7CA6", "VGGT",       "-"),
    "vggt_omega": ("#0E7C86", "VGGT-Ω",     "-"),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    G = json.loads(Path(a.summary).read_text())["g"]

    fig, ax = plt.subplots(figsize=(9.6, 4.3), dpi=190)
    ax.axhline(1.0, color="#D5D8DA", lw=1, zorder=1)
    ax.axvline(0.0, color="#E4E7E9", lw=1, zorder=1)
    for m, (col, lab, ls) in STYLE.items():
        if m not in G:
            continue
        xs = sorted(float(k) for k in G[m])
        ys = [G[m][str(x)] if str(x) in G[m] else G[m][f"{x:g}"] for x in xs]
        ax.plot(xs, ys, color=col, lw=2.3, ls=ls, marker="o", ms=4.2, zorder=3)
        ax.text(xs[-1] + 0.7, ys[-1], lab, color=col, fontsize=10.5,
                fontweight="bold", va="center")

    ax.set_xlim(-24.5, 31)
    ax.set_xticks([-22, -15, -11, -8, -4, 0, 4, 8, 11, 15, 22])
    ax.set_xlabel("residual roll the backbone sees  (degrees; 0 = gravity-aligned)",
                  color=INK, fontsize=10, labelpad=7)
    ax.set_ylabel("error ÷ that backbone's own level error", color=INK, fontsize=10)
    ax.tick_params(colors=GREY, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#D5D8DA")
    ax.text(0.015, 0.95, "13 recordings · each curve normalised by its own "
            "backbone's level error", transform=ax.transAxes, color=GREY,
            fontsize=9.2)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
