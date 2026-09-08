"""H39b - the price and the prize, against the frame's roll.

price S = (grav_p + grav_m)/2, the part that is the same for both rotation
signs. prize A = (grav_m - grav_p)/2, the roll actually removed. The operation
pays where the prize line is above the price line.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RED, INK, GREY = "#0E7C86", "#A6412F", "#474E52", "#767E82"
PANEL = [("da3:small", "DA3-Small — the prize wins"),
         ("vggt_omega", "VGGT-Ω — the price wins")]
BINS = np.array([0, 2, 4, 6, 8, 12, 30])


def collect(res: Path, model: str):
    roll, S, A = [], [], []
    for f in sorted(glob.glob(str(res / "*_seq*.json"))):
        d = json.loads(Path(f).read_text())
        if model not in d["models"]:
            continue
        pf = d["models"][model]["per_frame"]
        r = np.abs(np.array(d["roll_deg"]))
        for k in range(len(r)):
            dv, pv, mv = pf["device"][k], pf["grav_p"][k], pf["grav_m"][k]
            if None in (dv, pv, mv) or dv <= 0:
                continue
            gp, gm = 100 * (pv / dv - 1), 100 * (mv / dv - 1)
            roll.append(r[k]); S.append((gp + gm) / 2); A.append((gm - gp) / 2)
    return np.array(roll), np.array(S), np.array(A)


def binned(roll, y):
    cx, cy = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (roll >= lo) & (roll < hi)
        if sel.sum() >= 8:
            cx.append(np.median(roll[sel])); cy.append(np.mean(y[sel]))
    return cx, cy


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), dpi=190, sharey=True)
    for ax, (m, title) in zip(axes, PANEL):
        roll, S, A = collect(Path(a.results), m)
        ax.axhline(0, color="#C9CDD0", lw=1, zorder=1)
        for y, col, lab in ((A, TEAL, "prize — the roll removed"),
                            (S, RED, "price — resampling, both signs alike")):
            cx, cy = binned(roll, y)
            ax.plot(cx, cy, color=col, lw=2.5, marker="o", ms=5.5, zorder=3,
                    label=lab)
        ax.set_title(title, color=INK, fontsize=11, pad=9, loc="left")
        ax.set_xlabel("that frame's head roll  |ψ|  (degrees)", color=INK,
                      fontsize=9.8, labelpad=6)
        ax.set_xlim(-0.4, 15.5)
        ax.tick_params(colors=GREY, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#D5D8DA")
    axes[0].set_ylim(-3.5, 36)
    axes[0].set_ylabel("percent of that frame's own error", color=INK, fontsize=9.8)
    axes[0].legend(frameon=False, fontsize=9.4, loc="upper left",
                   labelcolor=GREY, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
