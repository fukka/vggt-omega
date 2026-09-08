"""H24 - the ordering survives a different depth model; the headroom mostly does not.

Both panels plot the same thing for the two models: how good a curve fitted on
each geometry is, against how much of the frame that geometry gives the rim.
The shape is the same. The scale is not - and on the harder recording DA3-Large
sits at zero, where "which geometry" stops mattering because nothing helps.
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
NICE = {"orthographic": "ortho", "equisolid": "equisolid", "equidistant": "equidist",
        "stereographic": "stereo", "rectilinear": "rectilinear", "aria_kb4": "real lens"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--small", required=True)
    p.add_argument("--large", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    S = json.loads(Path(a.small).read_text()); Lg = json.loads(Path(a.large).read_text())
    L, rs = Lg["lenses"], Lg["rim_area_share"]
    order = sorted(range(len(L)), key=lambda i: rs[L[i]])
    seqs = Lg["eval_seqs"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), dpi=170, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")
    titles = ["Same room", "Same room, rearranged (the harder test)"]
    for ax, seq, title in zip(axes, seqs, titles):
        ax.set_facecolor("#FFFFFF")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#C2C8C4")
        ax.tick_params(colors=INK, labelsize=9)
        xs = [100 * rs[L[i]] for i in order]
        for J, col, mk, lab in ((S, TEAL, "o", "DA3-Small"), (Lg, RED, "s", "DA3-Large")):
            M = np.array([J["matrix"][f"{seq}|{k}"] for k in L])
            ax.plot(xs, [M[i].mean() for i in order], color=col, lw=2.0, marker=mk, ms=6, label=lab)
        ax.axhline(0, color=GREY, lw=1.0, ls=":")
        ax.grid(True, color="#DFE3DF", lw=0.7)
        ax.set_title(title, fontsize=10.5, color="#16191B", loc="left")
        ax.set_xlabel("share of the frame given to the rim (%)", fontsize=9.5)
        ax.set_xlim(66, 89)
    axes[0].set_ylabel("average error change when fitted here (%)", fontsize=9.5)
    axes[0].set_ylim(-34, 8)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[1].text(67, 6.0, "above this line the correction makes things worse",
                 fontsize=8.3, color=GREY, va="top")
    M = np.array([Lg["matrix"][f"{seqs[0]}|{k}"] for k in L])
    for i in order:
        axes[0].annotate(NICE[L[i]], (100 * rs[L[i]], M[i].mean()),
                         textcoords="offset points", xytext=(0, 9), ha="center",
                         fontsize=8, color=INK)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
