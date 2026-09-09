"""H49 - per-frame mirror ratio, Aria against ScanNet++.

One bar per frame-ratio bin. The point is the SHAPE: on Aria the whole
distribution sits above 1, on ScanNet++ it straddles it.
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


def collect(pat, kn, km, model):
    r = []
    for f in sorted(glob.glob(pat)):
        d = json.loads(Path(f).read_text())
        v = d["models"].get(model)
        if not v:
            continue
        pf = v["per_frame"]
        a = pf[kn] if kn in pf else pf["normal"]
        b = pf[km] if km in pf else pf["mirror"]
        for x, y in zip(a, b):
            if x in (None, 0) or y is None:
                continue
            r.append(y / x)
    return np.array(r)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", required=True, help="autoresearch/experiments")
    p.add_argument("--model", default="da3:small")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    E = Path(a.exp)
    aria = collect(str(E / "h47-mirror-border/results/*_seq*.json"),
                   "normal|0.0", "mirror|0.0", a.model)
    sca = collect(str(E / "h46-mirror-external/results/R*_*.json"),
                  "normal", "mirror", a.model)

    fig, ax = plt.subplots(figsize=(9.4, 3.9), dpi=190)
    bins = np.linspace(0, 8, 41)
    ax.hist(np.clip(sca, 0, 8), bins=bins, density=True, color=TEAL, alpha=.75,
            label=f"ScanNet++, rectified 89°  (n={len(sca)})")
    ax.hist(np.clip(aria, 0, 8), bins=bins, density=True, color=RED, alpha=.6,
            label=f"Aria, border-free 60°  (n={len(aria)})")
    ax.axvline(1.0, color=INK, lw=1.4, ls="--")
    ax.text(1.06, ax.get_ylim()[1] * 0.92, "no change", color=INK, fontsize=9.4)
    ax.set_xlim(0, 8)
    ax.set_xlabel("that frame's error with the input reflected, ÷ its error without",
                  color=INK, fontsize=10, labelpad=7)
    ax.set_ylabel("share of frames", color=INK, fontsize=10)
    ax.set_yticks([])
    ax.tick_params(colors=GREY, labelsize=9.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D5D8DA")
    ax.legend(frameon=False, fontsize=9.6, labelcolor=GREY, loc="upper right")
    ax.set_title(f"{a.model} — 96% of Aria frames sit above 1, against 53% of ScanNet++ frames",
                 color=INK, fontsize=11.5, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
