"""Aria replication figure: span (run_006) + masking (run_007)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "h1-rim-pose-value" / "results"
OUT = ROOT / "to_human" / "assets"


def main() -> None:
    r6 = json.load(open(RES / "run_006.json"))
    r7 = json.load(open(RES / "run_007.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    ax = axes[0]
    spans = [25, 35, 45, 54.8]
    for arm, color in (("real", "#c0392b"), ("synth", "#2980b9")):
        errs = [r6["span"][arm]["conds"][f"t{t:g}"]["median_rot_err_deg"]
                for t in spans]
        ax.plot(spans, errs, "o-", color=color, label=f"{arm} arm")
    ax.set_xlabel("admitted field θ ≤ T (deg)")
    ax.set_ylabel("median rotation error (deg)")
    ax.set_title("Aria seq131: span at fixed count\n(classical; wide better 8/11 pairs)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    conds = ["vanilla", "center_masked", "random_masked", "rim_masked"]
    labels = ["vanilla", "center\nmasked", "random\nmasked\n(=rim area)", "rim\nmasked"]
    vals = [r7["mask"]["conds"][c]["median_err_deg"] for c in conds]
    colors = ["#7f8c8d", "#2980b9", "#8e6bb5", "#c0392b"]
    ax.bar(labels, vals, color=colors)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.6, f"{v:.1f}°", ha="center")
    ax.set_ylabel("median rotation error (deg)")
    ax.set_title("Aria seq131: DA3-Small masking (59 pairs)\nrim still the most load-bearing region")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "aria_h13.png", dpi=140)
    print(f"wrote {OUT / 'aria_h13.png'}")


if __name__ == "__main__":
    main()
