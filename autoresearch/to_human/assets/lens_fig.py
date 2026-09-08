"""H21 - the correction curve is not lens-specific, and which geometry you fit
on matters ~9x more than whether it matches.

Left: the four fitted curves, near-coincident in shape. Right: what each of them
buys when applied to the REAL Aria lens - the best curve for the Aria lens is
not the Aria lens's own.

All four arms are warped once from the real Aria camera, so the coefficients are
not comparable to the unwarped fit in the earlier sections; only comparisons
within this figure are meaningful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RED, AMBER, PURPLE = "#0E7C86", "#A6412F", "#C98A2E", "#6A5A8C"
COL = {"aria_kb4": TEAL, "equidistant": RED, "stereographic": AMBER, "equisolid": PURPLE}
NICE = {"aria_kb4": "the real lens (Aria)", "equidistant": "equidistant",
        "stereographic": "stereographic", "equisolid": "equisolid"}
TMAX = 54.83


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h21", required=True)
    p.add_argument("--seq", default="Apartment_release_clean_seq136_M1292")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    d = json.loads(Path(a.h21).read_text())
    L = d["lenses"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.1), dpi=170,
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    for f in (fig,):
        f.patch.set_facecolor("#FFFFFF")
    for x in (ax, bx):
        x.set_facecolor("#FFFFFF")
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            x.spines[s].set_color("#C2C8C4")
        x.tick_params(colors="#474E52", labelsize=9)

    e = np.linspace(0.0, TMAX, 9)
    mid = 0.5 * (e[:-1] + e[1:])
    for k in L:
        ax.plot(mid, d["coef_a"][k], color=COL[k], lw=2.2, marker="o", ms=4.5,
                label=NICE[k])
    ax.set_xlabel("incidence angle  θ  (degrees)", fontsize=10)
    ax.set_ylabel("fitted exponent  a(θ)", fontsize=10)
    ax.set_title("Four very different lens shapes, nearly the same curve",
                 fontsize=10.5, color="#16191B", loc="left")
    ax.grid(True, color="#DFE3DF", lw=0.7)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_ylim(1.45, 2.25)
    ax.annotate(f"mean pairwise gap {d['mean_pairwise_da']:.3f}\n"
                f"repeat-to-repeat noise 0.069",
                xy=(2, 2.16), fontsize=8.5, color="#767E82")

    M = np.array([d["matrix"][f"{a.seq}|{k}"] for k in L])
    j = L.index("aria_kb4")
    vals = M[:, j]
    order = np.argsort(vals)          # most negative (best) first
    ypos = np.arange(len(L))
    bx.barh(ypos, [vals[i] for i in order],
            color=[COL[L[i]] for i in order], height=.62)
    bx.set_yticks(ypos)
    bx.set_yticklabels([NICE[L[i]] + ("  ←own" if i == j else "") for i in order], fontsize=9)
    for y, i in zip(ypos, order):
        bx.text(vals[i] + 0.8, y, f"{vals[i]:.1f}%", va="center", ha="left",
                fontsize=8.8, color="#FFFFFF", fontweight="bold")
    bx.invert_yaxis()
    bx.set_xlabel("rim error change on the real Aria lens (%)", fontsize=9.5)
    bx.set_title("Which curve to use on the real lens",
                 fontsize=10.5, color="#16191B", loc="left")
    bx.set_xlim(-32, 0)
    bx.grid(True, axis="x", color="#DFE3DF", lw=0.7)

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
