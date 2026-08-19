"""H2.0 figure: joint (theta x depth) AbsRel heatmap + uncontrolled curve."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "h2-center-safe-adapter" / "results"
OUT = ROOT / "to_human" / "assets"


def main() -> None:
    r = json.load(open(RES / "run_008b.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    t = r["theta_bin_mid_deg"]
    ax.plot(t, r["uncontrolled_absrel"], "o-", color="#c0392b")
    ax.set_xlabel("incidence angle θ (deg)")
    ax.set_ylabel("AbsRel")
    ax.set_title("Uncontrolled: the folklore curve\n(DA3-Small, Aria seq131, 28 frames)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    cell = np.array(r["joint_absrel"]).T          # rows = depth bands
    edges = r["depth_edges_m"]
    im = ax.imshow(cell, aspect="auto", cmap="RdYlGn_r",
                   norm=matplotlib.colors.LogNorm(vmin=0.12, vmax=2.0))
    ax.set_xticks(range(len(t)), [f"{x:.0f}°" for x in t])
    ax.set_yticks(range(len(edges) - 1),
                  [f"{edges[i]:g}–{edges[i+1]:g} m" for i in range(len(edges) - 1)])
    for i in range(cell.shape[0]):
        for j in range(cell.shape[1]):
            ax.text(j, i, f"{cell[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black")
    ax.set_xlabel("incidence angle θ")
    ax.set_title("Controlled: same GT-depth band across θ\nnear rim = disaster; far rim = better than center")
    fig.colorbar(im, ax=ax, label="AbsRel (log scale)")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "depth_baseline_h20.png", dpi=140)
    print(f"wrote {OUT / 'depth_baseline_h20.png'}")


if __name__ == "__main__":
    main()
