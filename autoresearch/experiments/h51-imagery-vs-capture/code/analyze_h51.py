"""H51 — apply protocol.md's blind rule to the real and rendered arms.

    retention_m = mirror_cost(synth) / mirror_cost(real)

    CAPTURE       retention < 0.30 for at least 3 of 4 backbones
    IMAGERY       retention >= 0.60 for at least 3 of 4 backbones
    INCONCLUSIVE  anything else, including a split across backbones

The 0.30-0.60 band is deliberately undecided and is NOT split at a midpoint.
Gates first; a failure voids rather than softens. Both arms report the ratio of
means themselves (mirror_synth.py), so nothing is recomputed here from a
retired statistic -- the trap H50's analyser had to be fixed for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
MODELS = ("da3:small", "da3:large", "vggt", "vggt_omega")
CAPTURE_BELOW = 0.30      # protocol.md, fixed before the run
IMAGERY_ABOVE = 0.60      # protocol.md, fixed before the run
NEEDED = 3                # of 4 backbones
PLUMBING_MAX_PCT = 0.001


def arm(tag):
    cost, base, plumb, stems = {}, {}, 0.0, {}
    for f in sorted(RES.glob(f"{tag}_*.json")):
        d = json.load(open(f))
        stems[d["seq"]] = d.get("stems")
        for m, v in d.get("models", {}).items():
            cost.setdefault(m, []).append(v["mirror_cost_pct_ratio_of_means"])
            base.setdefault(m, []).append(v["normal_absrel"])
            plumb = max(plumb, v.get("plumbing_max_rel_pct", 0.0))
    return ({m: float(np.mean(v)) for m, v in cost.items()},
            {m: float(np.mean(v)) for m, v in base.items()}, plumb, stems)


def verdict(real, synth):
    missing = [m for m in MODELS if m not in real or m not in synth]
    if missing:
        return "INCOMPLETE", f"missing {', '.join(missing)}", {}
    ret = {m: synth[m] / real[m] for m in MODELS}
    low = sum(1 for m in MODELS if ret[m] < CAPTURE_BELOW)
    high = sum(1 for m in MODELS if ret[m] >= IMAGERY_ABOVE)
    txt = ", ".join(f"{m} {ret[m]:.2f}" for m in MODELS)
    if low >= NEEDED and high < NEEDED:
        return "CAPTURE", f"the effect largely does not survive rendering: {txt}", ret
    if high >= NEEDED and low < NEEDED:
        return "IMAGERY", f"the effect survives rendering: {txt}", ret
    return ("INCONCLUSIVE",
            f"neither threshold is met by {NEEDED} of 4: {txt}", ret)


def self_test():
    r = {m: 100.0 for m in MODELS}
    cases = [
        ("all four survive", r, {m: 80.0 for m in MODELS}, "IMAGERY"),
        ("three survive, one does not", r,
         {**{m: 80.0 for m in MODELS}, "vggt": 10.0}, "IMAGERY"),
        ("all four collapse", r, {m: 10.0 for m in MODELS}, "CAPTURE"),
        ("three collapse, one survives", r,
         {**{m: 10.0 for m in MODELS}, "vggt": 90.0}, "CAPTURE"),
        ("two and two", r, {"da3:small": 10.0, "da3:large": 10.0,
                            "vggt": 90.0, "vggt_omega": 90.0}, "INCONCLUSIVE"),
        ("all four land in the undecided band", r,
         {m: 45.0 for m in MODELS}, "INCONCLUSIVE"),
        ("exactly at the imagery bar counts as imagery", r,
         {m: 60.0 for m in MODELS}, "IMAGERY"),
        ("exactly at the capture bar does NOT count as capture", r,
         {m: 30.0 for m in MODELS}, "INCONCLUSIVE"),
        ("an arm is missing", r, {m: 80.0 for m in MODELS if m != "vggt"},
         "INCOMPLETE"),
    ]
    bad = 0
    for name, a, b, want in cases:
        got, why, _ = verdict(a, b)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:46s} -> {got}")
    print("  self-test:", "PASS" if not bad else f"{bad} FAILED")
    return bad == 0


def main():
    if "--self-test" in sys.argv:
        return 0 if self_test() else 1

    real, breal, preal, sreal = arm("real")
    synth, bsyn, psyn, ssyn = arm("synth")
    plumb = max(preal, psyn)
    print(f"\ngate  plumbing: worst {plumb:.6f}% "
          f"-> {'PASS' if plumb < PLUMBING_MAX_PCT else 'FAIL'}")
    # The pairing gate: the two arms must have walked the same frames.
    unpaired = [k for k in set(sreal) | set(ssyn) if sreal.get(k) != ssyn.get(k)]
    print(f"gate  pairing: {'PASS' if not unpaired else 'FAIL ' + str(unpaired)}")

    print(f"\n{'backbone':12s} {'real':>9s} {'synth':>9s} {'retention':>10s} "
          f"{'AbsRel real':>12s} {'AbsRel synth':>13s}")
    for m in MODELS:
        r = (synth[m] / real[m]) if (m in real and m in synth) else float("nan")
        print(f"{m:12s} {real.get(m, float('nan')):+9.1f} "
              f"{synth.get(m, float('nan')):+9.1f} {r:10.2f} "
              f"{breal.get(m, float('nan')):12.4f} {bsyn.get(m, float('nan')):13.4f}")

    v, why, ret = verdict(real, synth)
    if plumb >= PLUMBING_MAX_PCT or unpaired:
        v, why = "VOID", "a gate failed; the rule is not applied"
    print(f"\nVERDICT: {v} — {why}")
    (RES / "summary.json").write_text(json.dumps(
        {"real": real, "synth": synth, "retention": ret,
         "absrel_real": breal, "absrel_synth": bsyn,
         "verdict": v, "why": why,
         "capture_below": CAPTURE_BELOW, "imagery_above": IMAGERY_ABOVE},
        indent=2))
    print(f"[h51] wrote {RES / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
