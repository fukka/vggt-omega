# Copyright (c) 2026.
"""Recombine a slambench run that was sharded across GPUs, by model.

    python tools/merge_slambench_shards.py --out eval_out/slam/step2 \
        eval_out/slam/step2.g0 eval_out/slam/step2.g1

**Why this is safe, and where it would stop being safe.** ``slambench.run``
intersects the support inside the per-model loop::

    for key in ready:                  # <- one model
        ...
        support = np.ones(len(pts), bool)
        for p in preds.values():       # <- its arms x contexts, nobody else's
            support &= np.isfinite(p) & (p > 0)

Every arm and every context of a model is scored on the points *that model's*
arms could all answer for. No other model enters that expression, and the split
is rebuilt identically from ``--egosynth-root`` in each shard. So splitting the
model list across two processes yields the same numbers as one process, and this
tool only has to staple the ``runs`` lists together.

That would stop being true the moment the intersection moved out of the loop —
so this tool does not assume it, it *checks* the things that would change:
every shard must agree on the digest, the root, and the whole ``config`` block
(baselines, datasets, takes, frames, contexts, stride, sigma, rect FOV, calib
root). Shards must also carry disjoint model sets; a model appearing twice means
one of the two was run with something this comparison cannot see, and that is an
error rather than a silent last-writer-wins.

Merging is recorded in the output: ``sharded_by`` and ``shards`` say the file
was assembled, by what, and from where. A merged results.json that looked
exactly like a single-process one would be the wrong artefact to publish.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slambench import models as M      # noqa: E402
from slambench import report as R      # noqa: E402

#: Everything that must be identical across shards for the merge to mean
#: anything. ``requested_models`` is deliberately absent -- differing is the
#: whole point -- and so is ``skipped_models``, which is unioned instead.
IDENTICAL = ("protocol", "digest", "egosynth_root", "n_frames", "datasets",
             "takes")


def _load(path: str) -> dict:
    p = path if path.endswith(".json") else os.path.join(path, "results.json")
    if not os.path.isfile(p):
        raise SystemExit(f"[merge] no results.json at {p}. A shard that died "
                         f"leaves its directory behind with only manifest.json "
                         f"-- check its log rather than merging without it.")
    with open(p) as fh:
        return json.load(fh)


def _run_key(r: dict) -> tuple:
    return (r["model"], r["dataset"], r["baseline"], int(r["context"]))


def merge(paths, out_dir: str) -> dict:
    shards = [(p, _load(p)) for p in paths]
    if len(shards) < 2:
        raise SystemExit("[merge] pass two or more shard directories")

    ref_path, ref = shards[0]
    for path, s in shards[1:]:
        for field in IDENTICAL:
            if s.get(field) != ref.get(field):
                raise SystemExit(
                    f"[merge] {path} and {ref_path} disagree on {field!r}:\n"
                    f"  {ref_path}: {ref.get(field)!r}\n"
                    f"  {path}: {s.get(field)!r}\n"
                    f"These are not shards of one run. If the digest differs, "
                    f"the splits differ and no column here is comparable.")
        if s.get("config") != ref.get("config"):
            diff = sorted(
                k for k in set(s["config"]) | set(ref["config"])
                if s["config"].get(k) != ref["config"].get(k))
            raise SystemExit(
                f"[merge] {path} and {ref_path} disagree on config: {diff}. "
                f"Every arm in a merged table has to have been asked the same "
                f"question.")

    # Disjoint model sets, checked on the runs themselves rather than on
    # requested_models: a model that was requested and skipped contributes no
    # rows, and two shards may legitimately both skip it.
    seen: dict = {}
    runs = []
    for path, s in shards:
        for r in s["runs"]:
            k = _run_key(r)
            if k in seen:
                raise SystemExit(
                    f"[merge] {k} appears in both {seen[k]} and {path}. The "
                    f"shards overlap, so one of these two numbers was produced "
                    f"under conditions this merge cannot compare. Re-shard "
                    f"with disjoint --models and re-run.")
            seen[k] = path
            runs.append(r)

    order = {k: i for i, k in enumerate(M.DEFAULT_MODELS)}
    runs.sort(key=lambda r: (order.get(r["model"], len(order)), r["model"],
                             r["dataset"], r["baseline"], int(r["context"])))

    requested, skipped = [], []
    for _path, s in shards:
        for k in s.get("requested_models", []):
            if k not in requested:
                requested.append(k)
        for d in s.get("skipped_models", []):
            if d not in skipped:
                skipped.append(d)
    requested.sort(key=lambda k: (order.get(k, len(order)), k))

    payload = dict(ref)
    payload["requested_models"] = requested
    payload["skipped_models"] = skipped
    payload["runs"] = runs
    payload["sharded_by"] = "model"
    payload["shards"] = [
        dict(path=os.path.relpath(p), models=sorted({r["model"] for r in s["runs"]}),
             rows=len(s["runs"])) for p, s in shards]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    R.write_all(payload, out_dir)
    models = sorted({r["model"] for r in runs})
    print(f"\n[merge] {len(shards)} shard(s) -> {out_dir}/results.json  "
          f"digest {payload['digest']}  {len(runs)} rows  "
          f"{len(models)} model(s): {', '.join(models)}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("shards", nargs="+", help="shard dirs (or results.json paths)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    merge(a.shards, a.out)


if __name__ == "__main__":
    main()
