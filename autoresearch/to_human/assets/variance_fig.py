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
    p.add_argument("--h26-dir", default=None,
                   help="H26 per-(arm,seq) eval JSONs; adds the adapter row")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    Z = json.loads(Path(a.h25).read_text())["zones"]
    gain = lambda n: 100 * (Z[n]["radial"]["near_rim"] / Z[n]["frozen"]["near_rim"] - 1)
    new = sorted(k for k in Z if "seq136" not in k and "seq132" not in k)
    g = np.array([gain(k) for k in new]); m, s = g.mean(), g.std(ddof=1)
    g136, g132 = gain("Apartment_release_clean_seq136_M1292"), \
                 gain("Apartment_release_decoration_seq132_M1292")

    rows = [("the 16-number curve", g, g136, g132, -16.6)]
    if a.h26_dir:
        import glob, re
        from collections import defaultdict
        RIM = "near_rim(<=2m,>=38deg)"
        A = defaultdict(dict)
        for f in glob.glob(str(Path(a.h26_dir) / "*.json")):
            d = json.loads(Path(f).read_text()); z = d["zones"][RIM]
            arm, sq = re.match(r"(.+)_seq(\d+)\.json$", Path(f).name).groups()
            A[arm][sq] = 100 * (z["after"] / z["before"] - 1)
        sqs = sorted(A["omega110"])
        ad = np.array([np.mean([A[f"omega110{x}"][q] for x in ("", "_s1", "_s2")]) for q in sqs])
        rows.append(("the 123,000-parameter adapter", ad, -51.5, None, -51.5))

    fig, axes = plt.subplots(len(rows), 1, figsize=(8.6, 2.35 * len(rows)), dpi=170)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#FFFFFF")
    for ax, (label, vals, pub, alt, _p) in zip(axes, rows):
        mm, ss = vals.mean(), vals.std(ddof=1)
        ax.set_facecolor("#FFFFFF")
        ax.axvspan(mm - ss, mm + ss, color=TEAL, alpha=.13, lw=0)
        ax.axvline(mm, color=TEAL, lw=1.6)
        rng = np.random.default_rng(0)
        ax.scatter(vals, .05 + rng.uniform(-.028, .028, len(vals)), s=58, color=TEAL, zorder=3)
        if alt is not None:
            ax.scatter([alt], [.05], s=140, marker="s", color=AMBER, zorder=4)
        ax.scatter([pub], [.05], s=185, marker="*", color=RED, zorder=5)
        ax.annotate(f"{mm:.1f}% ± {ss:.1f}", (mm, .112), ha="center", fontsize=9, color=TEAL)
        ax.annotate(f"{pub:.1f}%\n{abs(pub-mm)/ss:.1f} σ out", (pub, .112), ha="center",
                    fontsize=9, color=RED, fontweight="bold")
        ax.set_yticks([]); ax.set_ylim(-.02, .175)
        ax.set_title(label, fontsize=10, color="#16191B", loc="left")
        ax.grid(True, axis="x", color="#DFE3DF", lw=0.7)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#C2C8C4")
        ax.tick_params(colors=INK, labelsize=9)
    axes[-1].set_xlabel("rim error change (%) — negative is better", fontsize=9.5)
    axes[-1].scatter([], [], s=58, color=TEAL, label="13 never-used recordings")
    axes[-1].scatter([], [], s=140, marker="s", color=AMBER, label='dec_seq132 — "the harder test"')
    axes[-1].scatter([], [], s=185, marker="*", color=RED, label="the recording we published on")
    axes[-1].legend(frameon=False, fontsize=8.6, loc="upper right", ncol=3,
                    bbox_to_anchor=(1.0, 1.30))
    ax = axes[0]
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="#FFFFFF")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
