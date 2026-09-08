"""H23 - doubling the source resolution changes nothing.

If the rim-density effect were about a warp inventing pixels, giving it a source
with twice the real detail should have pulled the geometries together. The two
lines sit on top of each other, so it is not that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RED, INK = "#0E7C86", "#A6412F", "#474E52"
NICE = {"orthographic": "orthographic", "equidistant": "equidistant",
        "aria_kb4": "the real lens", "rectilinear": "rectilinear"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    J = {k: json.loads((Path(a.dir) / f"{k}.json").read_text())
         for k in ("src504", "src1008")}
    L, rs = J["src504"]["lenses"], J["src504"]["rim_area_share"]
    seq = J["src504"]["eval_seqs"][0]
    order = sorted(range(len(L)), key=lambda i: rs[L[i]])

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=170)
    fig.patch.set_facecolor("#FFFFFF"); ax.set_facecolor("#FFFFFF")
    for k, col, mk, lab in (("src504", TEAL, "o", "source read at 504 px"),
                            ("src1008", RED, "s", "source read at 1008 px (2× the real detail)")):
        M = np.array([J[k]["matrix"][f"{seq}|{x}"] for x in L])
        ax.plot([100 * rs[L[i]] for i in order], [M[i].mean() for i in order],
                color=col, lw=2.0, marker=mk, ms=6.5, label=lab)
    M = np.array([J["src504"]["matrix"][f"{seq}|{x}"] for x in L])
    for i in order:
        ax.annotate(NICE[L[i]], (100 * rs[L[i]], M[i].mean()),
                    textcoords="offset points", xytext=(0, -18), ha="center",
                    fontsize=8.5, color=INK)
    ax.set_xlabel("share of the frame given to the outer half of the angle range (%)", fontsize=9.5)
    ax.set_ylabel("average error change when fitted here (%)", fontsize=9.5)
    ax.set_title("Twice the real detail, same result", fontsize=10.5,
                 color="#16191B", loc="left")
    ax.grid(True, color="#DFE3DF", lw=0.7)
    ax.set_xlim(66, 89); ax.set_ylim(-34, 2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C2C8C4")
    ax.tick_params(colors=INK, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
