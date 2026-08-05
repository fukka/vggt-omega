"""Are we simply not training the adapter long enough?

**The question.** The paper never states how many optimisation steps it takes --
it is the one hyperparameter missing from Sec. 4.3, which otherwise pins down
everything (Adam, lr 1e-3, clip 1.0, 504 px, 20/8/20 bins, 30 three-frame
windows). We picked ``--iters 300``, which fits in about three minutes.

Appendix D is the clue that this may be far too few: a "typical full
train-and-evaluate run" is quoted at **2-3 hours per ScanNet++ scene** on an
A4000/A6000, and 180-250 GPU-hours across the paper. Even allowing that their
figure covers evaluating the full sequence and every baseline, three minutes of
fitting does not sit comfortably inside two hours. If their adapter sees
thousands of steps and ours sees 300, we would be reporting an undertrained
model and calling it a failed reproduction.

This is orthogonal to the FOV hypothesis in ``fov_sweep.py``: FOV asks *what the
methods are scored on*, this asks *how long ours was fitted*. Both are cheap, and
they can be wrong independently, so run both before concluding anything about the
method itself.

**Reading the result.** ``raytun3r``'s ``R_deg`` against iteration count:

* Still falling at 300 -> we were undertrained; re-run everything at the elbow.
* Flat from 300 onward -> adaptation length is not the problem, and the paper's
  2-3 h is dominated by evaluation and baselines rather than fitting.
* Rising -> the adapter is overfitting the 30-window adaptation set, which the
  L2/TV regularisers (Eq. 11-12) are supposed to prevent; check their weights
  before believing it.

Each point refits from scratch, so points are independent and can be spread over
GPUs. The default sweep is ~2.3 GPU-hours in total for one scene.

Usage::

    python -m raytun3r.experiments.iters_sweep \\
        --backbone da3 --variant small --weights pretrained \\
        --dataset scannetpp --path /netapp/datasets/scannetpp/data/3f15a9266d \\
        --out runs/iters-sweep/3f15a9266d

Results land in ``<out>/summary.json`` plus one subdirectory per iteration count
holding the usual ``adapter.pt`` / ``train_log.json`` / ``results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List

__all__ = ["main"]

#: 300 is the current default; the upper end is what Appendix D's 2-3 h per scene
#: would allow if fitting dominated it.
DEFAULT_ITERS = [300, 1000, 3000, 10000]

#: Held identical across every point so the step count is the only thing moving.
SHARED = ["--matcher", "ufm", "--stride", "10", "--max-size", "504",
          "--min-flow-px", "2.0", "--seed", "0"]


def _run(cmd: List[str], dry: bool) -> None:
    print("[iters-sweep] $ " + " ".join(cmd), flush=True)
    if dry:
        return
    subprocess.run(cmd, check=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser("raytun3r.experiments.iters_sweep", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="da3", choices=["vggt", "vggt_omega", "da3"])
    p.add_argument("--variant", default="small",
                   choices=["small", "base", "large", "giant"])
    p.add_argument("--weights", default="pretrained")
    p.add_argument("--dataset", default="scannetpp", choices=["scannetpp", "adt"])
    p.add_argument("--path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", default=",".join(str(i) for i in DEFAULT_ITERS),
                   help="comma-separated iteration counts")
    p.add_argument("--methods", default="vanilla,raytun3r,center_ph",
                   help="only raytun3r moves with iters; the others are fixed "
                        "reference lines, so the short list is the default")
    p.add_argument("--windows", type=int, default=30)
    p.add_argument("--eval-windows", type=int, default=100)
    p.add_argument("--max-fov", type=float, default=None)
    p.add_argument("--convention", default="range", choices=["range", "z"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running them")
    args = p.parse_args(argv)

    iters = [int(x) for x in args.iters.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)
    base = [sys.executable, "-m"]
    common = ["--backbone", args.backbone, "--weights", args.weights,
              "--dataset", args.dataset, "--path", args.path,
              "--device", args.device, "--convention", args.convention]
    if args.backbone == "da3":
        common += ["--variant", args.variant]
    if args.max_fov is not None:
        common += ["--max-fov", str(args.max_fov)]

    summary = {"_meta": {"backbone": args.backbone, "variant": args.variant,
                         "dataset": args.dataset, "path": args.path,
                         "iters": iters, "methods": args.methods,
                         "convention": args.convention, "max_fov": args.max_fov,
                         "shared": SHARED, "windows": args.windows}}

    for n in iters:
        tag = f"it{n}"
        run_dir = os.path.join(args.out, tag)
        os.makedirs(run_dir, exist_ok=True)

        _run(base + ["raytun3r.train"] + common + SHARED + [
            "--out", run_dir, "--iters", str(n), "--windows", str(args.windows),
        ], args.dry_run)

        _run(base + ["raytun3r.eval"] + common + SHARED + [
            "--adapter", os.path.join(run_dir, "adapter.pt"),
            "--methods", args.methods,
            "--windows", str(args.eval_windows),
            "--out", os.path.join(run_dir, "results.json"),
        ], args.dry_run)

        path = os.path.join(run_dir, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                summary[tag] = json.load(f)

    out = os.path.join(args.out, "summary.json")
    if not args.dry_run:
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[iters-sweep] wrote {out}")

    # The curve the experiment exists to draw.
    print("\n[iters-sweep] R_deg by adaptation length")
    print(f"{'iters':>7}  " + "  ".join(f"{m:>10}" for m in args.methods.split(",")))
    for n in iters:
        row = summary.get(f"it{n}", {})
        cells = []
        for m in args.methods.split(","):
            v = row.get(m, {}).get("R_deg")
            cells.append(f"{v:10.3f}" if isinstance(v, (int, float)) else f"{'-':>10}")
        print(f"{n:7d}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
