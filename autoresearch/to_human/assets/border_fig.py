"""H29 - the border warning across thirteen recordings.

Left: the dose curve. The non-monotonic dip at 39.5% of the frame - the whole
evidence for "cost is set by proximity, not area" - holds on 13 of 13.

Right: the per-model border cost. The ordering holds on 13 of 13; the published
figures are the largest outliers found anywhere in this project.
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

    M, DS = {}, {}
    for f in glob.glob(str(D / "models_*.json")):
        s = re.search(r"models_(seq\d+)\.json", f).group(1)
        mm = json.loads(Path(f).read_text())["models"]
        M[s] = {b: 100*(mm[b]["pinhole_masked"]["0.0"]["all"]/mm[b]["pinhole"]["0.0"]["all"]-1)
                for b in mm}
    for f in glob.glob(str(D / "dose_*.json")):
        s = re.search(r"dose_(seq\d+)\.json", f).group(1)
        dd = json.loads(Path(f).read_text())["doses"]["0.0"]
        b = dd["0"]["all"]
        DS[s] = {int(w): (100*(dd[w]["all"]/b-1), dd[w]["black_frac"]) for w in dd if w != "0"}
    T13 = sorted(k for k in M if k != "seq136")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 4.1), dpi=170,
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#FFFFFF")
    for x in (ax, bx):
        x.set_facecolor("#FFFFFF")
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            x.spines[s].set_color("#C2C8C4")
        x.tick_params(colors=INK, labelsize=9)

    ws = sorted(next(iter(DS.values())))
    xs = [100*np.mean([DS[s][w][1] for s in T13]) for w in ws]
    mm = np.array([[DS[s][w][0] for w in ws] for s in T13])
    ax.errorbar(xs, mm.mean(0), yerr=mm.std(0, ddof=1), fmt="o-", ms=7, lw=2.0,
                color=TEAL, ecolor=TEAL, elinewidth=1.7, capsize=5, label="13 recordings")
    ax.plot(xs, [DS["seq136"][w][0] for w in ws], "*--", ms=13, lw=1.3, color=RED,
            label="the one we published on")
    i70 = ws.index(70)
    ax.annotate("39.5% of the frame black,\nand it costs the LEAST",
                (xs[i70], mm.mean(0)[i70]), textcoords="offset points", xytext=(-16, 26),
                ha="right", fontsize=8.8, color=TEAL,
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=1.0))
    ax.set_xlabel("share of the frame painted black (%)", fontsize=9.5)
    ax.set_ylabel("extra whole-image error (%)", fontsize=9.5)
    ax.set_title("More black is not worse — 13 of 13", fontsize=10.5, color="#16191B", loc="left")
    ax.grid(True, color="#DFE3DF", lw=0.7)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")

    labs, pos = ["DA3-Small", "DA3-Large"], np.arange(2)
    means = [np.mean([M[s][b] for s in T13]) for b in ("da3:small", "da3:large")]
    sds = [np.std([M[s][b] for s in T13], ddof=1) for b in ("da3:small", "da3:large")]
    bx.bar(pos, means, yerr=sds, color=TEAL, width=.5, capsize=6,
           error_kw=dict(ecolor=INK, elinewidth=1.4), label="13 recordings")
    bx.plot(pos, [M["seq136"][b] for b in ("da3:small", "da3:large")], "*",
            ms=17, color=RED, zorder=5, label="the one we published on")
    for i, b in enumerate(("da3:small", "da3:large")):
        bx.annotate(f"+{means[i]:.0f}%", (i, means[i]), textcoords="offset points",
                    xytext=(-34, -4), ha="right", fontsize=9, color=TEAL)
        bx.annotate(f"+{M['seq136'][b]:.0f}%", (i, M["seq136"][b]), textcoords="offset points",
                    xytext=(24, -4), fontsize=9, color=RED, fontweight="bold")
    bx.set_xticks(pos); bx.set_xticklabels(labs, fontsize=9.5)
    bx.set_ylabel("extra error from a border alongside (%)", fontsize=9.5)
    bx.set_title("Which model breaks — 13 of 13", fontsize=10.5, color="#16191B", loc="left")
    bx.set_ylim(0, 730); bx.set_xlim(-0.7, 1.7)
    bx.grid(True, axis="y", color="#DFE3DF", lw=0.7)
    bx.legend(frameon=False, fontsize=8.8, loc="upper left")

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
