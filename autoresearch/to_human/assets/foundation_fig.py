"""H28 - the foundation numbers, re-measured across thirteen recordings.

Left: the rim/centre ratio. The published single-recording value sits in the
middle of the distribution, but the spread is much wider than the published
"2.0-2.6x in every model" implies.

Right: the roll penalty. Representative at 20 and 30 degrees; the published
40-degree figure is 2.3 sd high.
"""
from __future__ import annotations

import argparse, glob, json, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RED, INK, GREY = "#0E7C86", "#A6412F", "#474E52", "#767E82"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    D = Path(a.dir)

    RAT = {}
    for f in glob.glob(str(D / "ratio_*.json")):
        s = re.search(r"ratio_(seq\d+)\.json", f).group(1)
        RAT[s] = json.loads(Path(f).read_text())["results"]["da3:small"]["3"]["rim_over_center"]
    ROLL = {}
    for f in glob.glob(str(D / "roll_*.json")):
        s = re.search(r"roll_(seq\d+)\.json", f).group(1)
        A = json.loads(Path(f).read_text())["arms"]["pinhole"]
        b = A["0.0"]["common"]["all"]
        ROLL[s] = {ang: 100*(0.5*(A[f"{ang}.0"]["common"]["all"]+A[f"-{ang}.0"]["common"]["all"])/b-1)
                   for ang in (20, 30, 40)}
    T13 = sorted(k for k in RAT if k != "seq136")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.0), dpi=170,
                                 gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor("#FFFFFF")
    for x in (ax, bx):
        x.set_facecolor("#FFFFFF")
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            x.spines[s].set_color("#C2C8C4")
        x.tick_params(colors=INK, labelsize=9)

    v = np.array([RAT[s] for s in T13]); m, sd = v.mean(), v.std(ddof=1)
    ax.axhspan(2.0, 2.6, color=RED, alpha=.10, lw=0)
    ax.text(1.44, 2.63, 'the range we published\n"2.0–2.6× in every model"',
            fontsize=8.4, color=RED, ha="right", va="bottom")
    rng = np.random.default_rng(1)
    ax.scatter(1 + rng.uniform(-.10, .10, len(v)), v, s=60, color=TEAL, zorder=3,
               label="13 recordings")
    ax.scatter([1], [RAT["seq136"]], s=185, marker="*", color=RED, zorder=5,
               label="the one we published on")
    ax.errorbar([1.32], [m], yerr=[sd], fmt="o", ms=8, color=TEAL,
                ecolor=TEAL, elinewidth=2, capsize=6, zorder=4)
    ax.annotate(f"{m:.2f} ± {sd:.2f}", (1.32, m), textcoords="offset points",
                xytext=(12, 0), fontsize=9, color=TEAL, va="center")
    ax.set_xlim(.72, 1.62); ax.set_xticks([])
    ax.set_ylabel("rim error ÷ centre error", fontsize=9.5)
    ax.set_title("How much worse is the rim, really?", fontsize=10.5, color="#16191B", loc="left")
    ax.grid(True, axis="y", color="#DFE3DF", lw=0.7)
    ax.legend(frameon=False, fontsize=8.6, loc="lower left")

    angs = [20, 30, 40]
    M = np.array([[ROLL[s][x] for x in angs] for s in T13])
    mm, ss = M.mean(0), M.std(0, ddof=1)
    bx.errorbar(angs, mm, yerr=ss, fmt="o-", ms=7, lw=2.0, color=TEAL,
                ecolor=TEAL, elinewidth=1.8, capsize=6, label="13 recordings")
    bx.plot(angs, [ROLL["seq136"][x] for x in angs], "*--", ms=15, lw=1.5,
            color=RED, label="the one we published on")
    for x, y, s_ in zip(angs, mm, ss):
        bx.annotate(f"+{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(-6, -20), fontsize=9, color=TEAL, ha="right")
    bx.annotate("+131%\n2.3 σ high", (40, ROLL["seq136"][40]), textcoords="offset points",
                xytext=(-8, 6), fontsize=9, color=RED, ha="right", fontweight="bold")
    bx.set_xticks(angs); bx.set_xlabel("camera tilt (degrees)", fontsize=9.5)
    bx.set_ylabel("extra whole-image error (%)", fontsize=9.5)
    bx.set_title("The cost of tilting", fontsize=10.5, color="#16191B", loc="left")
    bx.grid(True, color="#DFE3DF", lw=0.7)
    bx.legend(frameon=False, fontsize=8.6, loc="upper left")

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
