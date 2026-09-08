"""H22 - fit quality against how much of the frame the rim gets.

Left: row-mean gain against rim area share, six lens shapes, Spearman +1.000 on
both test recordings. The direction is the OPPOSITE of what H21's analysis
guessed: geometries that give the rim FEWER pixels make better fitting sets.

Right: how much a lens's own curve beats what row-and-column effects predict.
Near zero for the four realistic fisheye shapes; large only for the two extremes.
That is what "matching the lens buys nothing" means, and where it stops holding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RED, GREY, INK = "#0E7C86", "#A6412F", "#767E82", "#474E52"
NICE = {"orthographic": "orthographic", "equisolid": "equisolid",
        "equidistant": "equidistant", "stereographic": "stereographic",
        "rectilinear": "rectilinear", "aria_kb4": "the real lens"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h22", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    d = json.loads(Path(a.h22).read_text())
    L, rs = d["lenses"], d["rim_area_share"]
    seqs = d["eval_seqs"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=170,
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#FFFFFF")
    for x in (ax, bx):
        x.set_facecolor("#FFFFFF")
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            x.spines[s].set_color("#C2C8C4")
        x.tick_params(colors=INK, labelsize=9)

    order = sorted(range(len(L)), key=lambda i: rs[L[i]])
    for seq, col, mk, lab in ((seqs[0], TEAL, "o", "Same room"),
                              (seqs[1], RED, "s", "Same room, rearranged")):
        M = np.array([d["matrix"][f"{seq}|{k}"] for k in L])
        xs = [100 * rs[L[i]] for i in order]
        ys = [M[i].mean() for i in order]
        ax.plot(xs, ys, color=col, lw=1.8, marker=mk, ms=6, label=lab)
    for i in order:
        M = np.array([d["matrix"][f"{seqs[0]}|{k}"] for k in L])
        ax.annotate(NICE[L[i]], (100 * rs[L[i]], M[i].mean()),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.3, color=INK)
    ax.set_xlabel("share of the frame given to the outer half of the angle range (%)", fontsize=9.5)
    ax.set_ylabel("average error change when fitted here (%)", fontsize=9.5)
    ax.set_title("Fewer rim pixels in the fitting set → better curve",
                 fontsize=10.5, color="#16191B", loc="left")
    ax.grid(True, color="#DFE3DF", lw=0.7)
    ax.set_ylim(-36, 6); ax.set_xlim(66.5, 88.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    M = np.array([d["matrix"][f"{seqs[0]}|{k}"] for k in L])
    mu = M.mean(); r = M.mean(1) - mu; c = M.mean(0) - mu
    res = np.diag(M - (mu + r[:, None] + c[None, :]))
    sd = (M - (mu + r[:, None] + c[None, :])).std()
    ypos = np.arange(len(L))
    cols = [RED if abs(res[i]) > 2 * sd else GREY for i in order]
    bx.barh(ypos, [-res[i] for i in order], color=cols, height=.6)
    bx.axvline(2 * sd, color=INK, lw=1.0, ls=":")
    bx.text(2 * sd + 0.25, len(L) - 0.4, "noise", fontsize=8.5, color=INK)
    bx.set_yticks(ypos); bx.set_yticklabels([NICE[L[i]] for i in order], fontsize=9)
    bx.invert_yaxis()
    bx.set_xlabel("extra benefit from using this lens's OWN curve\n(percentage points)", fontsize=9.5)
    bx.set_title("Does matching the lens pay?", fontsize=10.5, color="#16191B", loc="left")
    bx.grid(True, axis="x", color="#DFE3DF", lw=0.7)

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
