"""Run the full RayTun3R protocol over every ScanNet++ scene, then aggregate.

The paper reports a mean over a dataset. One scene is a single sample, and the
first runs of this reproduction were exactly that -- so nothing measured so far
can distinguish "the method does not reproduce" from "this scene is unusual".
This driver closes that gap.

**Parallelism.** RayTun3R is single-GPU by construction: it fits ~10k parameters
in about three minutes, and the paper's own selling point is that adaptation is
cheap. There is nothing to shard within a scene. The parallel axis is *scenes* --
one scene per worker, embarrassingly parallel -- so ``--workers N`` on an N-GPU
box, with ``CUDA_VISIBLE_DEVICES`` pinned per worker.

**Fitting is per scene, not global.** The method adapts to one camera on one
short segment; a single adapter shared across scenes would be a different method.
Each scene therefore gets its own fit, and the reported number is the mean over
per-scene evaluations, matching how the paper's tables read.

Usage::

    python -m raytun3r.experiments.scannetpp_all \\
        --backbone da3 --variant small --weights pretrained \\
        --root /netapp/datasets/scannetpp/data \\
        --out runs/rt3r/snpp-all-da3s --workers 4

Then ``summary.json`` holds per-scene results and the aggregate table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

__all__ = ["discover_scenes", "main"]

#: Reported per method, in the paper's own column order.
METRIC_KEYS = ["R_deg", "t_deg", "d_reproj", "AbsRel", "delta_1.25", "coverage"]


def discover_scenes(root: str) -> List[str]:
    """Scene directories that actually carry a DSLR fisheye capture.

    ScanNet++ ships scenes without ``dslr/nerfstudio/transforms.json``; those
    cannot be loaded and are skipped here rather than failing mid-sweep.
    """
    if not os.path.isdir(root):
        raise SystemExit(f"ScanNet++ root does not exist: {root}")
    out = []
    for name in sorted(os.listdir(root)):
        scene = os.path.join(root, name)
        if os.path.exists(os.path.join(scene, "dslr", "nerfstudio", "transforms.json")):
            out.append(scene)
    return out


def _one_scene(scene: str, args, gpu: Optional[int]) -> Dict:
    name = os.path.basename(scene.rstrip("/"))
    run_dir = os.path.join(args.out, name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")

    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    common = ["--backbone", args.backbone, "--weights", args.weights,
              "--dataset", "scannetpp", "--path", scene,
              "--device", "cuda", "--convention", args.convention,
              "--matcher", args.matcher, "--stride", str(args.stride),
              "--max-size", str(args.max_size), "--min-flow-px", str(args.min_flow_px),
              "--seed", str(args.seed)]
    if args.backbone == "da3":
        common += ["--variant", args.variant]
    if args.max_fov is not None:
        common += ["--max-fov", str(args.max_fov)]

    train = [sys.executable, "-m", "raytun3r.train"] + common + [
        "--out", run_dir, "--iters", str(args.iters), "--windows", str(args.windows)]
    evaluate = [sys.executable, "-m", "raytun3r.eval"] + common + [
        "--adapter", os.path.join(run_dir, "adapter.pt"),
        "--methods", args.methods, "--windows", str(args.eval_windows),
        "--out", os.path.join(run_dir, "results.json")]

    with open(log_path, "w") as log:
        for cmd in (train, evaluate):
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            if r.returncode != 0:
                # One bad scene must not take the sweep down; record and move on.
                print(f"[snpp-all] FAILED {name} (see {log_path})", flush=True)
                return {"scene": name, "error": f"exit {r.returncode}", "log": log_path}

    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    print(f"[snpp-all] done {name}", flush=True)
    return {"scene": name, "results": res}


def _aggregate(rows: List[Dict], methods: List[str]) -> Dict:
    """Mean and standard error per method, over scenes that succeeded.

    Standard error is reported because the headline question -- whether RayTun3R
    beats the virtual-pinhole baselines -- turns on gaps that a single scene
    cannot resolve. Without a spread, a mean over scenes is no more conclusive
    than the one-scene run it replaces.
    """
    agg: Dict[str, Dict[str, float]] = {}
    for m in methods:
        for key in METRIC_KEYS:
            vals = [r["results"][m][key] for r in rows
                    if "results" in r and m in r["results"] and key in r["results"][m]
                    and isinstance(r["results"][m][key], (int, float))
                    and not math.isnan(r["results"][m][key])]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                sem = math.sqrt(var / len(vals))
            else:
                sem = float("nan")
            agg.setdefault(m, {})[key] = mean
            agg[m][key + "_sem"] = sem
            agg[m]["n_scenes"] = len(vals)
    return agg


def main(argv=None) -> None:
    from ..backbones import BACKBONE_NAMES

    p = argparse.ArgumentParser("raytun3r.experiments.scannetpp_all", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="ScanNet++ data root holding scene dirs")
    p.add_argument("--out", required=True)
    p.add_argument("--backbone", default="da3", choices=BACKBONE_NAMES)
    p.add_argument("--variant", default="small",
                   choices=["small", "base", "large", "giant"])
    p.add_argument("--weights", default="pretrained")
    p.add_argument("--methods", default="vanilla,param_free,raytun3r,center_ph,multi_ph")
    p.add_argument("--convention", default="range", choices=["range", "z"])
    p.add_argument("--matcher", default="ufm", choices=["auto", "ufm", "raft", "sift"])
    p.add_argument("--max-fov", type=float, default=None)

    # Paper protocol; change only deliberately.
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--windows", type=int, default=30)
    p.add_argument("--eval-windows", type=int, default=100)
    p.add_argument("--stride", type=int, default=10,
                   help="10, not 1: at stride 1 the baseline is ~1 cm against ~3 m "
                        "of depth and translation direction is unobservable")
    p.add_argument("--min-flow-px", type=float, default=2.0)
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--workers", type=int, default=1, help="scenes in flight; one GPU each")
    p.add_argument("--gpus", default=None,
                   help="comma-separated device ids, e.g. '0,1,2,3'; defaults to 0..workers-1")
    p.add_argument("--limit", type=int, default=None, help="first N scenes only (smoke test)")
    args = p.parse_args(argv)

    scenes = discover_scenes(args.root)
    if args.limit:
        scenes = scenes[: args.limit]
    if not scenes:
        raise SystemExit(f"no ScanNet++ scenes with dslr/nerfstudio/transforms.json under {args.root}")
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(max(args.workers, 1)))
    os.makedirs(args.out, exist_ok=True)
    print(f"[snpp-all] {len(scenes)} scenes, {args.workers} worker(s), gpus={gpus}", flush=True)

    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one_scene, s, args, gpus[i % len(gpus)])
                   for i, s in enumerate(scenes)]
        for fut in futures:
            rows.append(fut.result())

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    failed = [r["scene"] for r in rows if "error" in r]
    summary = {
        "_meta": {"backbone": args.backbone, "variant": args.variant,
                  "root": args.root, "n_scenes": len(scenes),
                  "n_failed": len(failed), "failed": failed,
                  "protocol": {"iters": args.iters, "windows": args.windows,
                               "eval_windows": args.eval_windows, "stride": args.stride,
                               "min_flow_px": args.min_flow_px, "max_size": args.max_size,
                               "matcher": args.matcher, "convention": args.convention,
                               "max_fov": args.max_fov, "seed": args.seed}},
        "per_scene": rows,
        "aggregate": _aggregate(rows, methods),
    }
    out = os.path.join(args.out, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[snpp-all] wrote {out}")

    print(f"\n[snpp-all] mean over {len(scenes) - len(failed)} scenes "
          f"({len(failed)} failed)")
    head = f"{'method':>12}  " + "  ".join(f"{k:>16}" for k in METRIC_KEYS)
    print(head)
    for m in methods:
        a = summary["aggregate"].get(m, {})
        cells = []
        for k in METRIC_KEYS:
            if k in a:
                sem = a.get(k + "_sem", float("nan"))
                cells.append(f"{a[k]:8.3f}+-{sem:<6.3f}" if not math.isnan(sem)
                             else f"{a[k]:16.3f}")
            else:
                cells.append(f"{'-':>16}")
        print(f"{m:>12}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
