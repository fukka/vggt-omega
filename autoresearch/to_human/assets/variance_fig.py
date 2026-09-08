"""H25 - the recording every in-room number was quoted on is an outlier.

Thirteen never-used held-out recordings from the same room and camera, scored
with the unchanged curve. seq136, the one this whole line reported on, sits 5.7
standard deviations off the rest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RED, AMBER, INK, GREY = "#0E7C86", "#A6412F", "#C98A2E", "#474E52", "#767E82"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h25", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    Z = json.loads(Path(a.h25).read_text())["zones"]
    gain = lambda n: 100 * (Z[n]["radial"]["near_rim"] / Z[n]["frozen"]["near_rim"] - 1)
    new = sorted(k for k in Z if "seq136" not in k and "seq132" not in k)
    g = np.array([gain(k) for k in new]); m, s = g.mean(), g.std(ddof=1)
    g136, g132 = gain("Apartment_release_clean_seq136_M1292"), \
                 gain("Apartment_release_decoration_seq132_M1292")

    fig, ax = plt.subplots(figsize=(8.4, 3.5), dpi=170)
    fig.patch.set_facecolor("#FFFFFF"); ax.set_facecolor("#FFFFFF")
    ax.axvspan(m - s, m + s, color=TEAL, alpha=.13, lw=0)
    ax.axvline(m, color=TEAL, lw=1.6)
    rng = np.random.default_rng(0)
    ax.scatter(g, .05 + rng.uniform(-.03, .03, len(g)), s=64, color=TEAL,
               zorder=3, label="13 never-used recordings")
    ax.scatter([g132], [.05], s=150, marker="s", color=AMBER, zorder=4,
               label='dec_seq132 — "the harder test"')
    ax.scatter([g136], [.05], s=190, marker="*", color=RED, zorder=5,
               label="seq136 — the number we published")
    ax.annotate(f"{m:.1f}% ± {s:.1f}\nthe typical recording", (m, .115),
                ha="center", fontsize=9, color=TEAL)
    ax.annotate(f"{g136:.1f}%\n5.7 σ out", (g136, .115), ha="center",
                fontsize=9, color=RED, fontweight="bold")
    ax.set_yticks([]); ax.set_ylim(-.02, .16)
    ax.set_xlim(-19, -2)
    ax.set_xlabel("rim error change from the 16-number curve (%) — negative is better", fontsize=9.5)
    ax.set_title("Every in-room number in this report was quoted on the red star",
                 fontsize=10.5, color="#16191B", loc="left")
    ax.grid(True, axis="x", color="#DFE3DF", lw=0.7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#C2C8C4")
    ax.tick_params(colors=INK, labelsize=9)
    ax.legend(frameon=False, fontsize=8.8, loc="lower left", ncol=1)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
