"""H32 - what real head roll actually costs.

The penalty curve measured densely where the roll actually happens, drawn over
the measured roll distribution. The expected cost is the product, integrated:
1.46% +- 1.09. Small - but 1.5% of frames carry a quarter of it, and positive
roll costs far more than negative.
"""
from __future__ import annotations

import argparse, glob, json, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RED, INK, GREY, AMBER = "#0E7C86", "#A6412F", "#474E52", "#767E82", "#C98A2E"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    p.add_argument("--dist", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    P = {}
    for f in glob.glob(str(Path(a.dir) / "dense_*.json")):
        s = re.search(r"dense_(seq\d+)\.json", f).group(1)
        A = json.loads(Path(f).read_text())["arms"]["pinhole"]
        b = A["0.0"]["common"]["all"]
        P[s] = {float(k): 100*(v["common"]["all"]/b-1) for k, v in A.items()}
    T13 = sorted(k for k in P if k != "seq136")
    angs = np.array(sorted(next(iter(P.values()))))
    M = np.array([[P[s][x] for x in angs] for s in T13])

    d = json.loads(Path(a.dist).read_text())
    cnt = np.array(d["hist_counts"], float); e = np.array(d["hist_edges"], float)
    mid = 0.5*(e[:-1]+e[1:]); w = cnt/cnt.sum()

    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=170)
    fig.patch.set_facecolor("#FFFFFF"); ax.set_facecolor("#FFFFFF")
    bx = ax.twinx()
    bx.bar(mid, 100*w, width=2.3, color=GREY, alpha=.28, lw=0, zorder=1)
    bx.set_ylabel("share of real frames at this tilt (%)", fontsize=9.5, color=GREY)
    bx.tick_params(colors=GREY, labelsize=9); bx.set_ylim(0, 26)
    for s in ("top",):
        bx.spines[s].set_visible(False)

    ax.axhline(0, color=GREY, lw=1.0, ls=":", zorder=2)
    ax.errorbar(angs, M.mean(0), yerr=M.std(0, ddof=1), fmt="o-", ms=6, lw=2.0,
                color=TEAL, ecolor=TEAL, elinewidth=1.6, capsize=4, zorder=4,
                label="extra depth error (13 recordings)")
    for x in (-20, 20):
        ax.axvline(x, color=RED, lw=1.0, ls="--", alpha=.6, zorder=3)
    ax.annotate("beyond ±20°:\n1.5% of frames,\n26% of the cost",
                xy=(24, 40), fontsize=9, color=RED, ha="left", va="center")
    ax.annotate("+30° costs 2.3× what −30° does —\nthe ± averaging we published hid this",
                xy=(30, 56.5), xytext=(2, 68), fontsize=8.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.set_xlabel("camera tilt (degrees)  ·  negative = counter-clockwise", fontsize=9.5)
    ax.set_ylabel("extra whole-image error (%)", fontsize=9.5, color=TEAL)
    ax.set_title("Expected cost of real head roll:  1.46% ± 1.09",
                 fontsize=11, color="#16191B", loc="left")
    ax.set_xlim(-34, 34); ax.set_ylim(-12, 88)
    ax.grid(True, color="#DFE3DF", lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#C2C8C4"); ax.spines["bottom"].set_color("#C2C8C4")
    ax.tick_params(colors=INK, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
