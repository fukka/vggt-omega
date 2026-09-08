"""H41 - the residual-roll curve, with an exact per-frame gravity-aligned zero.

Every frame is rendered at roll_deg = psi + d, so the residual the backbone sees
is exactly -d and d = 0 is that frame's own level reference. The chord shows the
Jensen construction the correction to H39b rests on: the midpoint of g(0) and
g(2psi) sits above g(psi) whenever the curve is convex.
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    S = json.loads(Path(a.summary).read_text())
    xs = sorted(float(k) for k in S["g"])
    ys = [S["g"][str(x)] if str(x) in S["g"] else S["g"][f"{x:g}"] for x in xs]

    sd = {}
    import glob
    for f in sorted(glob.glob(str(Path(a.results) / "*_seq*.json"))):
        d = json.loads(Path(f).read_text())
        g = d["models"]["da3:small"]["g"]
        for k, (m, s_, n) in g.items():
            sd.setdefault(-float(k), []).append(m)
    err = [np.std(sd[x]) if len(sd.get(x, [])) > 1 else 0.0 for x in xs]

    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=190)
    ax.axhline(1.0, color="#D5D8DA", lw=1, zorder=1)
    ax.fill_between(xs, np.array(ys) - np.array(err), np.array(ys) + np.array(err),
                    color=TEAL, alpha=.16, lw=0, zorder=2)
    ax.plot(xs, ys, color=TEAL, lw=2.4, marker="o", ms=5, zorder=4)

    # the Jensen construction at psi = 11
    psi = 11.0
    g0, gp, g2 = 1.0, S["g"]["11.0"], S["g"]["22.0"]
    ax.plot([0, 2 * psi], [g0, g2], color=RED, lw=1.6, ls="--", zorder=5)
    ax.plot([psi, psi], [gp, (g0 + g2) / 2], color=RED, lw=2.2, zorder=6)
    ax.scatter([psi, psi], [gp, (g0 + g2) / 2], s=34, color=RED, zorder=7)
    ax.annotate("the gap H39b called a price", xy=(psi, (gp + (g0 + g2) / 2) / 2),
                xytext=(psi + 2.2, 1.045), color=RED, fontsize=9.6,
                arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    ax.text(psi * 0.62, (g0 + (g0 + g2) / 2) / 2 + 0.035,
            "chord from 0° to 2ψ", color=RED, fontsize=9.4,
            rotation=17, rotation_mode="anchor", va="bottom")

    ax.set_xlim(-24.5, 25)
    ax.set_ylim(0.97, 1.42)
    ax.set_xticks([-22, -15, -11, -8, -4, 0, 4, 8, 11, 15, 22])
    ax.set_xlabel("residual roll the backbone sees  (degrees; 0 = gravity-aligned)",
                  color=INK, fontsize=10, labelpad=7)
    ax.set_ylabel("error ÷ that frame's own level error", color=INK, fontsize=10)
    ax.tick_params(colors=GREY, labelsize=9)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#D5D8DA")
    ax.text(0.015, 0.94, "DA3-Small · 13 recordings · shaded band is the "
            "recording-to-recording sd", transform=ax.transAxes, color=GREY,
            fontsize=9.2)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
