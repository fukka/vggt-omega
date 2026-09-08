"""H35 - what real head roll costs, per depth model.

Integrated against 60,105 real frames. The multi-view-pretrained pair costs about
a quarter of what the single-image pair does, and the rare-tilt tail that carries
a quarter of DA3's cost carries only a tenth of VGGT-Omega's.
"""
from __future__ import annotations

import argparse, glob, json, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RED, INK, GREY = "#0E7C86", "#A6412F", "#474E52", "#767E82"
BB = ["da3:small", "da3:large", "vggt", "vggt_omega"]
NICE = {"da3:small": "DA3-Small", "da3:large": "DA3-Large",
        "vggt": "VGGT", "vggt_omega": "VGGT-Omega"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True); p.add_argument("--dist", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    P = {}
    for f in glob.glob(str(Path(a.dir) / "dense_*.json")):
        s = re.search(r"dense_(seq\d+)\.json", f).group(1)
        mm = json.loads(Path(f).read_text())["models"]
        P[s] = {b: {float(k): 100*(v["all"]/mm[b]["pinhole"]["0.0"]["all"]-1)
                    for k, v in mm[b]["pinhole"].items()} for b in BB}
    S = sorted(P)
    d = json.loads(Path(a.dist).read_text())
    cnt = np.array(d["hist_counts"], float); e = np.array(d["hist_edges"], float)
    mid = 0.5*(e[:-1]+e[1:]); w = cnt/cnt.sum()

    def integ(pen):
        xs = np.array(sorted(pen)); ys = np.array([pen[x] for x in xs])
        pp = np.maximum(np.interp(np.clip(mid, xs.min(), xs.max()), xs, ys), 0.0)
        return float((w*pp).sum()), pp

    ec, sd, tail = [], [], []
    for b in BB:
        vs = [integ(P[s][b])[0] for s in S]
        ec.append(np.mean(vs)); sd.append(np.std(vs, ddof=1))
        ts = []
        for s in S:
            tot, pp = integ(P[s][b])
            ts.append(100*(w[np.abs(mid) > 20]*pp[np.abs(mid) > 20]).sum()/tot)
        tail.append(np.mean(ts))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 3.9), dpi=170,
                                 gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor("#FFFFFF")
    for x in (ax, bx):
        x.set_facecolor("#FFFFFF")
        for s_ in ("top", "right"): x.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): x.spines[s_].set_color("#C2C8C4")
        x.tick_params(colors=INK, labelsize=9)
    pos = np.arange(len(BB))
    cols = [RED, RED, TEAL, TEAL]
    ax.bar(pos, ec, yerr=sd, color=cols, width=.6, capsize=5,
           error_kw=dict(ecolor=INK, elinewidth=1.3))
    for i, v in enumerate(ec):
        ax.annotate(f"{v:.2f}%", (i, v), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xticks(pos); ax.set_xticklabels([NICE[b] for b in BB], fontsize=9)
    ax.set_ylabel("expected extra error from real head roll (%)", fontsize=9.5)
    ax.set_title("What ignoring roll actually costs", fontsize=10.5, color="#16191B", loc="left")
    ax.set_ylim(0, 3.5); ax.grid(True, axis="y", color="#DFE3DF", lw=0.7)
    ax.annotate("trained on single photos", (0.5, 3.15), fontsize=8.6, color=RED, ha="center")
    ax.annotate("multi-view", (2.5, 3.15), fontsize=8.6, color=TEAL, ha="center")

    bx.bar(pos, tail, color=cols, width=.6)
    for i, v in enumerate(tail):
        bx.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9.5, color=INK, fontweight="bold")
    bx.set_xticks(pos); bx.set_xticklabels([NICE[b] for b in BB], fontsize=8.5, rotation=20)
    bx.set_ylabel("share of that cost from the 1.5%\nof frames beyond ±20° (%)", fontsize=9)
    bx.set_title("Does the rare tilt dominate?", fontsize=10.5, color="#16191B", loc="left")
    bx.set_ylim(0, 30); bx.grid(True, axis="y", color="#DFE3DF", lw=0.7)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
