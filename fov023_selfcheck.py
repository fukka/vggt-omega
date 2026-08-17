#!/usr/bin/env python3
"""Ticket 023's self-check: run A's four vanilla cells against #019's reference.

Views and protocols are not part of the split digest, so run A
(``--views fisheye --protocols radial``, five models, ``--n-frames 25``) carries
the same digest as section 0's main run and its four vanilla fisheye/radial cells
must be bit-identical to that run's.

The reference is ``results/fovbench-rectfix-393cab9/partA_6seq``, digest
``601fcb22767e`` -- 6 sequences x 50 frames, the #019 headline. It is **not**
``fovbench-main-22c108d``: that one is 25 frames over a SINGLE sequence, from
when only 1 of 20 ADT exports on the box carried all three streams, and the
6-sequence table superseded it.

**One row is expected to differ, and it is not a mystery.** The reference was
produced before ticket 021 opened the bf16 autocast that ``VGGTBackbone`` is
written to expect, so its ``vggt_1b`` column is fp32 and no current checkout
reproduces it. That is recorded in ``fovbench/README.md`` and in ticket 021. The
check therefore reports ``vggt_1b`` separately from the rows that must match
exactly: if any of those move at all, something changed that was not supposed to;
if ``vggt_1b`` moves by more than ticket 021's measured bound, the dtype is not
the whole story.

Usage:  fov023_selfcheck.py <run_a/results.json> <reference/results.json>
"""
import json
import sys

# The ones that must be bit-identical, and the one #021 deliberately moved.
MUST_MATCH = ("vggt_omega", "dav2_large", "da3_large", "da3_small")
EXPECTED_TO_MOVE = "vggt_1b"

# Ticket 021 measured <=0.51% AbsRel on slambench for the same dtype change.
# Anything much beyond that here is a different effect wearing its clothes.
BOUND_PCT = 2.0

KEYS = ("AbsRel", "SqRel", "RMSE", "RMSElog", "log10",
        "delta1", "delta2", "delta3", "scale_ratio", "n_valid_total")


def cells(doc):
    """{(model, stream, view, protocol): run} for the fisheye radial cells."""
    out = {}
    for r in doc["runs"]:
        if r["view"] != "fisheye" or r["protocol"] != "radial":
            continue
        out[(r["model"], r["stream"])] = r
    return out


def pct(new, ref):
    if ref == 0:
        return 0.0 if new == 0 else float("inf")
    return 100.0 * (new - ref) / abs(ref)


def main(run_path, ref_path):
    run = json.load(open(run_path))
    ref = json.load(open(ref_path))

    print(f"run A     {run_path}")
    print(f"reference {ref_path}")
    print(f"digest    run={run['digest']}  ref={ref['digest']}  "
          f"{'MATCH' if run['digest'] == ref['digest'] else '*** DIFFER ***'}")
    print(f"n_frames  run={run['n_frames']}  ref={ref['n_frames']}")
    print(f"sequences {'MATCH' if run['sequences'] == ref['sequences'] else '*** DIFFER ***'}")
    print()

    a, b = cells(run), cells(ref)
    shared = sorted(set(a) & set(b))
    if not shared:
        print("no shared fisheye/radial cells -- nothing to check")
        return 1

    verdict = 0
    for group, models in (("must match exactly", MUST_MATCH),
                          ("expected to move (#021 bf16 fix)", (EXPECTED_TO_MOVE,))):
        print(f"### {group}")
        for (model, stream) in shared:
            if model not in models:
                continue
            ra, rb = a[(model, stream)]["overall"], b[(model, stream)]["overall"]
            deltas = {k: pct(ra[k], rb[k]) for k in KEYS if k in ra and k in rb}
            worst_k = max(deltas, key=lambda k: abs(deltas[k]))
            exact = all(ra[k] == rb[k] for k in deltas)
            tag = "identical" if exact else f"worst {worst_k} {deltas[worst_k]:+.4f}%"
            print(f"  {model:12s} {stream:9s} AbsRel {rb['AbsRel']:.6f} -> "
                  f"{ra['AbsRel']:.6f} ({pct(ra['AbsRel'], rb['AbsRel']):+.4f}%)  {tag}")
            if model in MUST_MATCH and not exact:
                verdict = 1
            if model == EXPECTED_TO_MOVE and abs(pct(ra["AbsRel"], rb["AbsRel"])) > BOUND_PCT:
                print(f"    ^ beyond #021's bound of {BOUND_PCT}% -- investigate "
                      f"before reading the vggt360 row")
                verdict = 1
        print()

    print("SELF-CHECK", "PASS" if verdict == 0 else "FAIL")
    return verdict


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
