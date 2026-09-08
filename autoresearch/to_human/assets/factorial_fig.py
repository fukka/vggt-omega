"""H40 - resampling alone, border alone, and both, at a matched 30 degrees.

Everything is measured on the same 60-degree crop; the rotation happens on a
canvas wide enough that the crop never loses a pixel, so "resampling alone"
really is two interpolations and nothing else.
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
ORDER = ["da3:small", "vggt_omega"]
LABEL = {"da3:small": "DA3-Small", "vggt_omega": "VGGT-Ω"}
BARS = [("rt", "two resamplings,\nno border", TEAL),
        ("masked", "a border,\nno extra resampling", RED),
        ("rt_masked", "both", "#7A4A3A")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    M = json.loads(Path(a.summary).read_text())["per_model"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), dpi=190)
    for ax, m in zip(axes, ORDER):
        xs = np.arange(len(BARS))
        vals = [M[m][k][0] for k, _, _ in BARS]
        errs = [M[m][k][1] for k, _, _ in BARS]
        cols = [c for _, _, c in BARS]
        ax.axhline(0, color="#C9CDD0", lw=1, zorder=1)
        ax.bar(xs, vals, width=.62, color=cols, alpha=.88, zorder=2)
        ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=INK, elinewidth=1.1,
                    capsize=4, alpha=.5, zorder=3)
        top = max(vals) if max(vals) > 0 else 1
        for x, v, e in zip(xs, vals, errs):
            ax.text(x, v + e + top * 0.045, f"{v:+.1f}%", ha="center",
                    color=INK, fontsize=10.5, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([l for _, l, _ in BARS], color=GREY, fontsize=9.3)
        ax.set_title(LABEL[m], color=INK, fontsize=11.5, pad=14, loc="left")
        ax.set_ylim(min(0, min(vals) - 3), top * 1.28)
        ax.tick_params(colors=GREY, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#D5D8DA")
    axes[0].set_ylabel("cost vs the un-rotated, un-masked view  (%)",
                       color=INK, fontsize=9.6)
    fig.tight_layout(pad=1.1)
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
