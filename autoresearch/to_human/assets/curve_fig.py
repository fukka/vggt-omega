"""The fitted radial curve a(theta), for the report.

H18.2's 16 numbers and H19's failure are the same object seen twice, and the
table hides it. Plotted, the Apartment fit is a smooth shallow arch and the two
LiteOffice fits are not curves at all - one swings 1.87 -> 2.03 -> 0.99, the
other sits below 1.0 almost everywhere, which inverts the correction. That is
why native fitting loses, and it reads in one glance.

Each device is plotted against its OWN bin centres: Aria M1292 theta_max is
54.83 deg, the LiteOffice device 56.63 deg.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TMAX = {"apartment": 54.83, "dino": 56.63, "bowl": 56.63, "lite_both": 56.63}
STYLE = {
    "apartment": dict(color="#0E7C86", lw=2.6, marker="o", ms=5, zorder=3,
                      label="fitted on Apartment, 240 fr, walking"),
    "dino":      dict(color="#A6412F", lw=1.8, marker="s", ms=4, ls="--", zorder=2,
                      label="fitted on Room 2 seq A, 60 fr, near-static"),
    "bowl":      dict(color="#C98A2E", lw=1.8, marker="^", ms=4, ls="--", zorder=2,
                      label="fitted on Room 2 seq B, 60 fr, near-static"),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h19", required=True, help="native_unmatched.json")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    cells = json.loads(Path(a.h19).read_text())["cells"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=170)
    fig.patch.set_facecolor("#FFFFFF"); ax.set_facecolor("#FFFFFF")

    for key, st in STYLE.items():
        c = np.asarray(cells[f"{key}_radial"]["coef_a"])
        e = np.linspace(0.0, TMAX[key], len(c) + 1)
        ax.plot(0.5 * (e[:-1] + e[1:]), c, **st)

    ax.axhline(1.0, color="#767E82", lw=1.0, ls=":", zorder=1)
    ax.text(1.2, 1.005, "no correction", fontsize=8.5, color="#767E82", va="bottom")
    ax.set_xlabel("incidence angle  θ  (degrees from the lens axis)", fontsize=10)
    ax.set_ylabel("fitted exponent  a(θ)", fontsize=10)
    ax.set_xlim(0, 57); ax.set_ylim(0.10, 2.25)
    ax.grid(True, color="#DFE3DF", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C2C8C4")
    ax.tick_params(colors="#474E52", labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax.annotate("rim", xy=(48, 0.17), fontsize=9, color="#767E82")
    ax.annotate("centre", xy=(1.2, 0.17), fontsize=9, color="#767E82")

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
