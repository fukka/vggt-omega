"""H39 - per-frame gain from gravity-aligned rendering, against the frame's roll.

One dot per frame. Positive is an improvement over the device-aligned render
the pipeline does today. The line is a binned mean.
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
PANEL = [("da3:small", "DA3-Small — single-image pretraining", TEAL),
         ("vggt_omega", "VGGT-Ω — multi-view pretraining", RED)]
BINS = np.array([0, 2, 4, 6, 8, 12, 30])


def collect(res: Path, model: str, better: str):
    roll, imp = [], []
    for f in sorted(glob.glob(str(res / "*_seq*.json"))):
        d = json.loads(Path(f).read_text())
        if model not in d["models"]:
            continue
        pf = d["models"][model]["per_frame"]
        r = np.abs(np.array(d["roll_deg"]))
        for k in range(len(r)):
            a, b = pf["device"][k], pf[better][k]
            if a is None or b is None or a <= 0:
                continue
            roll.append(r[k]); imp.append(100 * (1 - b / a))
    return np.array(roll), np.array(imp)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    S = json.loads(Path(a.summary).read_text())["per_model"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), dpi=190, sharey=True)
    for ax, (m, title, col) in zip(axes, PANEL):
        roll, imp = collect(Path(a.results), m, S[m]["better"])
        ax.axhline(0, color="#C9CDD0", lw=1, zorder=1)
        ax.scatter(roll, imp, s=13, color=col, alpha=.32, linewidths=0, zorder=2)
        cx, cy = [], []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            sel = (roll >= lo) & (roll < hi)
            if sel.sum() >= 8:
                cx.append(np.median(roll[sel])); cy.append(np.mean(imp[sel]))
        ax.plot(cx, cy, color=col, lw=2.4, marker="o", ms=5, zorder=4)
        ax.set_title(title, color=INK, fontsize=11, pad=9, loc="left")
        ax.set_xlabel("that frame's head roll  |ψ|  (degrees)", color=INK,
                      fontsize=9.8, labelpad=6)
        ax.set_xlim(-0.6, 24)
        ax.tick_params(colors=GREY, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#D5D8DA")
        ax.text(0.97, 0.06,
                f"pooled {S[m]['gain_pct']:+.2f}%  (better on {S[m]['gain_over_n']}/6 recordings)",
                transform=ax.transAxes, ha="right", color=col, fontsize=9.6,
                fontweight="bold")
    axes[0].set_ylim(-42, 52)
    axes[0].set_ylabel("gain from aligning the render to gravity  (%)",
                       color=INK, fontsize=9.8)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
