"""Does the field of view explain why the virtual-pinhole baselines win?

**The question.** On ScanNet++ at stride 10 with VGGT-1B, the first real runs put
Center-PH at 0.378 deg rotation against RayTun3R's 1.858 -- roughly 5x the wrong
way round from the paper's Tab. 1, where RayTun3R beats Center-PH. The leading
explanation is field of view: the paper describes ScanNet++'s DSLR as 115 deg,
but its released calibration implies ~170 deg diagonal, and the corners carry
real content. At 170 deg the backbone is far outside anything it was trained on,
and re-projecting to a 110 deg pinhole throws the hardest pixels away rather than
fixing them -- which would make Center-PH's win an artefact of *what it is scored
on*, not evidence about the method.

**The experiment.** ``--max-fov`` narrows ``theta_max``, which is what defines
``Omega``, which is what every loss (Eq. 8, 10) and every metric (Eq. 16-18) sums
over. The images are untouched: the model still sees the whole frame. So this
sweep isolates the scoring region from the input, and the prediction to test is
sharp:

* If FOV is the explanation, the RayTun3R-to-Center-PH gap closes as the cone
  narrows, and near 115 deg the paper's ordering should reappear.
* If the gap is flat in FOV, the explanation is elsewhere -- the backbone (VGGT
  is not the paper's primary; DA3-Small is), or the reproduction itself.

Either way the answer is informative, which is why this is worth a GPU slot
before any more full-dataset runs.

Usage (one scene, the default 5-point sweep, ~15 GPU-min per point)::

    python -m raytun3r.experiments.fov_sweep \\
        --backbone da3 --variant small --weights pretrained \\
        --dataset scannetpp --path /netapp/datasets/scannetpp/data/3f15a9266d \\
        --out runs/fov-sweep/3f15a9266d

Results land in ``<out>/summary.json`` plus one subdirectory per FOV holding the
usual ``adapter.pt`` / ``train_log.json`` / ``results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List

__all__ = ["main"]

#: The paper's stated ScanNet++ FOV, the calibration's real one, and points in
#: between. 110 matches the Center-PH virtual pinhole, so that row says what the
#: direct path scores on exactly the region Center-PH covers.
DEFAULT_FOVS = [110.0, 130.0, 150.0, 170.0]

#: Held identical across every point so FOV is the only thing that moves.
SHARED = ["--matcher", "ufm", "--stride", "10", "--max-size", "504",
          "--min-flow-px", "2.0", "--seed", "0"]


def _run(cmd: List[str], dry: bool) -> None:
    print("[fov-sweep] $ " + " ".join(cmd), flush=True)
    if dry:
        return
    subprocess.run(cmd, check=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser("raytun3r.experiments.fov_sweep", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="da3", choices=["vggt", "vggt_omega", "da3"])
    p.add_argument("--variant", default="small",
                   choices=["small", "base", "large", "giant"])
    p.add_argument("--weights", default="pretrained")
    p.add_argument("--dataset", default="scannetpp", choices=["scannetpp", "adt"])
    p.add_argument("--path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fovs", default=",".join(str(f) for f in DEFAULT_FOVS),
                   help="comma-separated total FOV in degrees")
    p.add_argument("--methods", default="vanilla,param_free,raytun3r,center_ph,multi_ph")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--windows", type=int, default=30)
    p.add_argument("--eval-windows", type=int, default=100)
    p.add_argument("--convention", default="range", choices=["range", "z"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running them")
    args = p.parse_args(argv)

    fovs = [float(x) for x in args.fovs.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)
    base = [sys.executable, "-m"]
    common = ["--backbone", args.backbone, "--weights", args.weights,
              "--dataset", args.dataset, "--path", args.path,
              "--device", args.device, "--convention", args.convention]
    if args.backbone == "da3":
        common += ["--variant", args.variant]

    summary = {"_meta": {"backbone": args.backbone, "variant": args.variant,
                         "dataset": args.dataset, "path": args.path,
                         "fovs": fovs, "methods": args.methods,
                         "convention": args.convention,
                         "shared": SHARED, "iters": args.iters}}

    for fov in fovs:
        tag = f"fov{int(round(fov))}"
        run_dir = os.path.join(args.out, tag)
        os.makedirs(run_dir, exist_ok=True)

        # The adapter is refitted per FOV on purpose: Omega enters the training
        # objective too, so reusing one adapter across the sweep would confound
        # "scored on a narrower cone" with "fitted on a narrower cone".
        _run(base + ["raytun3r.train"] + common + SHARED + [
            "--max-fov", str(fov), "--out", run_dir,
            "--iters", str(args.iters), "--windows", str(args.windows),
        ], args.dry_run)

        _run(base + ["raytun3r.eval"] + common + SHARED + [
            "--max-fov", str(fov),
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
        print(f"[fov-sweep] wrote {out}")

    # The comparison the experiment exists to make.
    print("\n[fov-sweep] R_deg by FOV")
    print(f"{'FOV':>6}  " + "  ".join(f"{m:>10}" for m in args.methods.split(",")))
    for fov in fovs:
        row = summary.get(f"fov{int(round(fov))}", {})
        cells = []
        for m in args.methods.split(","):
            v = row.get(m, {}).get("R_deg")
            cells.append(f"{v:10.3f}" if isinstance(v, (int, float)) else f"{'-':>10}")
        print(f"{fov:6.0f}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
