"""Figures for the H1 family (runs 001-003). Writes PNGs next to --out."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "h1-rim-pose-value" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "to_human" / "assets"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    r2 = json.load(open(RES / "run_002.json"))
    r3 = json.load(open(RES / "run_003.json"))

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2))

    # Panel 1: H1 quartile bins (run_002)
    ax = axes[0]
    conds = ["q0", "q1", "q2", "q3"]
    th = [r2["real"]["conds"][c]["median_theta_deg"] for c in conds]
    for arm, color in (("real", "#c0392b"), ("synth", "#2980b9")):
        errs = [r2[arm]["conds"][c]["median_err_deg"] for c in conds]
        ax.plot(th, errs, "o-", color=color, label=f"{arm} arm")
    ax.set_xlabel("bin median incidence angle θ (deg)")
    ax.set_ylabel("median rotation error (deg)")
    ax.set_title("H1 (refuted): equal-count θ bins —\nno per-point rim advantage")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: H1.1 span curve (run_003)
    ax = axes[1]
    spans = [35, 45, 55, 65, 85]
    for arm, color in (("real", "#c0392b"), ("synth", "#2980b9")):
        errs = [r3[arm]["conds"][f"t{t}"]["median_rot_err_deg"] for t in spans]
        ax.plot(spans, errs, "o-", color=color, label=f"{arm} arm")
    ax.set_xlabel("admitted field θ ≤ T (deg)")
    ax.set_ylabel("median rotation error (deg)")
    ax.set_title("H1.1 (supported): span at FIXED count —\nwide wins 17/17 real pairs")
    ax.axvspan(55, 85, alpha=0.08, color="green")
    ax.text(70, 3.1, "peripheral\nband", ha="center", fontsize=9, color="green")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: translation direction
    ax = axes[2]
    for arm, color in (("real", "#c0392b"), ("synth", "#2980b9")):
        errs = [r3[arm]["conds"][f"t{t}"]["median_t_err_deg"] for t in spans]
        ax.plot(spans, errs, "o-", color=color, label=f"{arm} arm")
    ax.set_xlabel("admitted field θ ≤ T (deg)")
    ax.set_ylabel("median translation-direction error (deg)")
    ax.set_title("Translation direction:\nspan cuts error 43° → 16°")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: H1.2 masking (run_004)
    r4 = json.load(open(RES / "run_004.json"))
    ax = axes[3]
    conds4 = ["vanilla", "center_masked", "rim_masked"]
    labels = ["vanilla", "center\nmasked\n(39% px)", "rim\nmasked\n(61% px)"]
    vals = [r4["conds"][c]["median_err_deg"] for c in conds4]
    colors = ["#7f8c8d", "#2980b9", "#c0392b"]
    ax.bar(labels, vals, color=colors)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.2, f"{v:.1f}°", ha="center")
    ax.set_ylabel("median rotation error (deg)")
    ax.set_title("H1.2: DA3-Small pose is rim-driven —\ndelete the center, nothing happens")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out / "h1_family.png", dpi=140)
    print(f"wrote {out / 'h1_family.png'}")


if __name__ == "__main__":
    main()
