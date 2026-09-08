"""H36b - the rim penalty on a completely different fisheye domain.

Not our measurement: this is the by-theta breakdown from the depthfisheye work
stream's own validated evaluation on SynWoodScape (synthetic automotive
surround-view, ~190 degree lens, four cameras, 424M scored pixels). The model is
FINE-TUNED on that data, unlike the frozen models in the rest of this report, so
the numbers are not like-for-like — but the shape is the point.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RED, INK, GREY = "#0E7C86", "#A6412F", "#474E52", "#767E82"
TH = [6.8, 6.9, 20.5, 20.6, 20.7, 34.2, 34.4, 34.5, 34.6, 47.9, 48.1, 48.3, 48.4,
      61.6, 61.8, 62.1, 62.2, 75.3, 75.6, 75.9, 76.0, 89.0, 89.3, 89.7, 89.9,
      102.7, 103.1, 103.5, 103.7]
AR = [0.0092, 0.0123, 0.0144, 0.0149, 0.0166, 0.0157, 0.0166, 0.0152, 0.0365,
      0.0152, 0.0160, 0.0331, 0.0379, 0.0191, 0.0202, 0.0424, 0.0423, 0.0426,
      0.0507, 0.0547, 0.0542, 0.0617, 0.0835, 0.0789, 0.0675, 0.0704, 0.1649,
      0.0515, 0.0531]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    th, ar = np.array(TH), np.array(AR)
    # bin to the nominal ring angles so the four cameras pool visually
    rings = [6.85, 20.6, 34.4, 48.2, 61.9, 75.7, 89.5, 103.2]
    m, s = [], []
    for r in rings:
        sel = np.abs(th - r) < 3.0
        m.append(ar[sel].mean()); s.append(ar[sel].std(ddof=1) if sel.sum() > 1 else 0.0)
    m, s = np.array(m), np.array(s)

    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=170)
    fig.patch.set_facecolor("#FFFFFF"); ax.set_facecolor("#FFFFFF")
    ax.errorbar(rings, 100*m, yerr=100*s, fmt="o-", ms=7, lw=2.2, color=TEAL,
                ecolor=TEAL, elinewidth=1.6, capsize=5, label="SynWoodScape, 424M pixels")
    ax.axvspan(0, 11, color=GREY, alpha=.13, lw=0)
    ax.annotate("the centre", (5.5, 100*m[-1]*0.92), fontsize=8.8, color=GREY, ha="center")
    ax.annotate(f"×{m[-1]/m[0]:.1f} from centre to rim", (103.2, 100*m[-1]),
                textcoords="offset points", xytext=(-10, 14), fontsize=9.5,
                color=RED, ha="right", fontweight="bold")
    ax.set_xlabel("incidence angle  θ  (degrees from the lens axis)", fontsize=9.5)
    ax.set_ylabel("depth error (AbsRel, %)", fontsize=9.5)
    ax.set_title("The rim penalty is not an Aria artefact", fontsize=11, color="#16191B", loc="left")
    ax.grid(True, color="#DFE3DF", lw=0.7)
    ax.set_xlim(0, 112); ax.set_ylim(0, 9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#C2C8C4")
    ax.tick_params(colors=INK, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
