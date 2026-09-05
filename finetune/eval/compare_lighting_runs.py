"""
compare_lighting_runs.py
========================
Does the 2x2 conclusion survive a change in the render's photometry?

The 2x2 contrast is WITHIN frame -- full and masked come from the same panorama,
so the lighting cannot manufacture the effect. It can change the effect's SIZE,
because a render far from the training distribution depresses everything the
model does. This compares two exp_rendered runs over the same windows: one on the
set rendered with the historical lighting, one on the set rendered with the
fitted lighting.

Reported per mode and projection:

  effect       mean AbsRel(full) - AbsRel(masked), and its window-clustered CI
  delta        effect_B - effect_A, clustered on the same windows, so the two
               runs are PAIRED frame by frame rather than compared as two
               independent means

A delta whose CI spans zero says the conclusion is robust to the lighting; one
that does not says the size of the effect depends on how the scene was lit, which
is itself worth knowing before the number is quoted as a property of VGGT.

    python -m finetune.eval.compare_lighting_runs --a runs/ev_clust96/results.json \\
                                                  --b runs/ev_clust96_lit/results.json
"""

import argparse, json, os
from typing import Dict, List, Optional

import numpy as np

SETTINGS = ("fisheye_full", "fisheye_masked", "persp_full", "persp_masked")


def cluster_ci(vals: Dict[str, float], group_of: Dict[str, str],
               n_boot: int = 10000, seed: int = 0) -> Optional[dict]:
    """Bootstrap over GROUPS. Frames in a window share a scene and are not
    independent draws; see cluster_bootstrap in exp_rendered.py."""
    keys = sorted(set(vals) & set(group_of))
    if len(keys) < 3:
        return None
    by: Dict[str, List[float]] = {}
    for k in keys:
        by.setdefault(group_of[k], []).append(vals[k])
    gk = list(by)
    if len(gk) < 2:
        return None
    arrs = [np.asarray(by[g], dtype=float) for g in gk]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(gk), size=(n_boot, len(gk)))
    boots = np.array([np.concatenate([arrs[j] for j in row]).mean() for row in draws])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    allv = np.concatenate(arrs)
    return {"mean": float(allv.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": len(allv), "n_groups": len(gk),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def base(d: str) -> str:
    """Key frames by sequence/frame, not by absolute path: the two runs live in
    different render roots, and comparing them requires the join to ignore that."""
    return os.path.join(os.path.basename(os.path.dirname(d)), os.path.basename(d))


def effects(res: dict, mode: str, proj: str):
    r_full = (res.get(mode, {}).get(f"{proj}_full") or {})
    r_mask = (res.get(mode, {}).get(f"{proj}_masked") or {})
    pf_f = {base(k): v for k, v in (r_full.get("_per_frame") or {}).items()}
    pf_m = {base(k): v for k, v in (r_mask.get("_per_frame") or {}).items()}
    go = {base(k): v for k, v in ((r_full.get("_group_of")
                                   or r_mask.get("_group_of") or {})).items()}
    common = sorted(set(pf_f) & set(pf_m))
    return {k: pf_f[k] - pf_m[k] for k in common}, go


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="results.json, run A (e.g. historical lighting)")
    ap.add_argument("--b", required=True, help="results.json, run B (e.g. fitted lighting)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    A, B = json.load(open(args.a)), json.load(open(args.b))
    modes = [m for m in A if m in B]

    L = ["", "=" * 96,
         f"2x2 effect under two lightings   A={args.label_a}   B={args.label_b}",
         "(effect = AbsRel full - masked; negative = true content helps. "
         "CIs resample WINDOWS.)",
         "=" * 96,
         f'{"mode":<9}{"proj":<9}{"effect A":>11}{"CI A":>22}'
         f'{"effect B":>11}{"CI B":>22}']
    deltas = []
    for mode in modes:
        for proj in ("fisheye", "persp"):
            ea, ga = effects(A, mode, proj)
            eb, gb = effects(B, mode, proj)
            ca, cb = cluster_ci(ea, ga), cluster_ci(eb, gb)
            if not (ca and cb):
                L.append(f"{mode:<9}{proj:<9}  (insufficient paired frames)")
                continue
            ci_a = f'[{ca["ci_lo"]:+.4f}, {ca["ci_hi"]:+.4f}]'
            ci_b = f'[{cb["ci_lo"]:+.4f}, {cb["ci_hi"]:+.4f}]'
            L.append(f'{mode:<9}{proj:<9}{ca["mean"]:>+11.4f}{ci_a:>22}'
                     f'{cb["mean"]:>+11.4f}{ci_b:>22}')
            common = sorted(set(ea) & set(eb))
            if len(common) >= 3:
                d = {k: eb[k] - ea[k] for k in common}
                cd = cluster_ci(d, gb or ga)
                if cd:
                    deltas.append((mode, proj, cd, len(common)))
    L.append("")
    L.append("change in the effect, PAIRED frame by frame across the two renders:")
    for mode, proj, cd, n in deltas:
        verdict = ("CHANGES with the lighting" if cd["excludes_zero"]
                   else "robust to the lighting (CI spans 0)")
        L.append(f'  {mode:<9}{proj:<9}delta {cd["mean"]:+.4f}  '
                 f'CI [{cd["ci_lo"]:+.4f}, {cd["ci_hi"]:+.4f}]  '
                 f'n={n}f/{cd["n_groups"]}w  {verdict}')
    txt = "\n".join(L)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
