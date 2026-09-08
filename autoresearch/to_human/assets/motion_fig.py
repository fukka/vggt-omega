"""H20 - rim gain against how far the camera actually travelled.

H19's claim was binary: motion beats a lens match. H20 makes it a dose-response,
and three of the four points are the SAME device, same room, same 240-frame pool,
same 30-frame count and same teacher - the only variable is how far apart the
selected frames' camera positions are.

Camera spread is the mean distance of the per-frame camera centre from the
centroid, computed from GT pose as C = -R^T t.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RED, GREY = "#0E7C86", "#A6412F", "#767E82"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h20", required=True)
    p.add_argument("--h19", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    d = json.loads(Path(a.h20).read_text())
    h19 = json.loads(Path(a.h19).read_text())

    pts = [  # label, spread_m, gain, sd, same_device
        ("30 frames,\nmost spread out", d["arms"]["high"]["spread_m_mean"],
         d["arms"]["high"]["gain_mean"][2], d["arms"]["high"]["gain_sd"][2], True),
        ("all 240 frames", 1.768, h19["cells"]["apartment_radial"]["gain"][2], None, True),
        ("30 frames,\nleast spread out", d["arms"]["low"]["spread_m_mean"],
         d["arms"]["low"]["gain_mean"][2], d["arms"]["low"]["gain_sd"][2], True),
        ("the target camera's\nown 60 frames", d["motion"]["BlackCeramicBowl_seq030"]["spread_m"],
         h19["cells"]["bowl_radial"]["gain"][2], None, False),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=170)
    fig.patch.set_facecolor("#FFFFFF"); ax.set_facecolor("#FFFFFF")

    sd_ = [(x, y) for _, x, y, _, s in pts if s]
    ax.plot([x for x, _ in sd_], [y for _, y in sd_], color=TEAL, lw=1.6, ls="-", zorder=2)
    ax.plot([sd_[-1][0], pts[3][1]], [sd_[-1][1], pts[3][2]],
            color=GREY, lw=1.2, ls=":", zorder=1)

    # placed one by one: the automatic offsets collide on a log axis
    OFF = [(-14, -30, "right"), (-10, 12, "right"), (14, -6, "left"), (0, -26, "center")]
    for (lbl, x, y, sd, same), (dx, dy, ha) in zip(pts, OFF):
        c = TEAL if same else RED
        if sd is not None:
            ax.errorbar(x, y, yerr=sd, fmt="none", ecolor=c, elinewidth=1.4, capsize=4, zorder=3)
        ax.plot(x, y, "o" if same else "s", color=c, ms=9, zorder=4)
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8.8, color="#474E52", ha=ha)

    ax.axhline(0, color=GREY, lw=1.0, ls=":")
    ax.text(2.9, 0.7, "no better than\ndoing nothing", fontsize=8.5, color=GREY,
            va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1.0", "2.0"])
    ax.set_xlabel("how far the camera moved while the fitting frames were taken\n"
                  "(mean distance from the centre of the path, metres, log scale)", fontsize=10)
    ax.set_ylabel("rim error change on the new room (%)", fontsize=10)
    ax.set_xlim(0.085, 3.6); ax.set_ylim(-36, 8)
    ax.grid(True, color="#DFE3DF", lw=0.7, which="major")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C2C8C4")
    ax.tick_params(colors="#474E52", labelsize=9)

    ax.plot([], [], "o", color=TEAL, label="same camera, same room (only the motion differs)")
    ax.plot([], [], "s", color=RED, label="the other camera, in the room being tested")
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
