#!/usr/bin/env python3
"""Where every number in the ADT-FOV deck comes from — generated, then checked.

Run it to regenerate ``adt-fov-provenance.md``. It reads each run's own
``results.json`` off the ``results`` branch, so the split digest, frame count,
sequence list, context setting and model list in the table are the run's, not a
retyped copy.

It also enforces the rule that was broken once and cost a deck revision:

    Two figures a reader will compare must come from the same split digest AND
    the same run directory.

A run directory pins the code; a digest pins the frames. Same digest, different
directory means the pixels were the same and the code was not. Different digest
means the two curves are of different scenes, and a shape read across them says
nothing — which is exactly how the deck's Result page and its multi-frame pages
came to disagree: 300 frames over six sequences against 50 frames of seq131,
and seq131 has the flattest field of the six.

Exit code is non-zero when a group fails, so this can gate a deck build.

    python docs/decks/provenance.py            # write the table, check groups
    python docs/decks/provenance.py --check    # check only, write nothing
"""
import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "adt-fov-provenance.md")
DECK = "adt_fov_experiment_v5.pptx"

#: Every run the deck draws on, by the short name used below.
#: Path is relative to the root of the ``results`` branch.
RUNS = OrderedDict([
    ("6seq",     "results/fovbench-rectfix-393cab9/partA_6seq/results.json"),
    ("s131_n1",  "results/fovbench-rectfix-393cab9/partA_seq131/results.json"),
    ("s131_5s",  "results/fovbench-rectfix-393cab9/partB_5s/results.json"),
    ("s131_10s", "results/fovbench-rectfix-393cab9/partB_10s/results.json"),
])

#: Runs used only for the sequence-spread diagnostic, which is not in the deck.
#: They are the pre-rectfix build — the only one with the six sequences scored
#: separately. Kept out of the deck groups on purpose; see the note below.
AUX = OrderedDict([
    ("perseq_%s" % s, "results/fovbench-ctx-d351d94/perseq_seq%s/results.json" % s)
    for s in ("132", "133", "134", "135", "136")])

#: page -> (what is on it, which run, which generator produced the raster)
PAGES = [
    (1,  "Title: '300 frames over 6 sequences'", "6seq", "—"),
    (2,  "Camera table, KB4 coefficients", None,
         "finetune/aria_calibration.py — code, not a run"),
    (2,  "Fisheye frame (illustrative)", None,
         "mk_qual.py, seq131 frame_001750 — one frame, chosen for looks"),
    (3,  "Rectify: kept / discarded / mask", None,
         "mk_qual.py — closed-form over the KB4 LUT, no model, no data"),
    (4,  "AbsRel vs incidence angle, 3 heads", "6seq", "mk_charts.absrel_panel"),
    (4,  "Quoted bin values and rises", "6seq", "mk_charts, chart_numbers.json"),
    (5,  "Input sizes, patch grids, alignment", None,
         "fovbench/models.py + geometry.py + metrics.py — code"),
    (6,  "FOV x depth heatmaps", "PENDING",
         "mk_joint.py — empty until issue #24 part A lands on 6seq"),
    (6,  "GT depth per angle bin", "6seq", "mk_joint.depth_marginal"),
    (7,  "Multi-frame, raw fisheye", "PENDING",
         "mk_ctx.py — empty until issue #24 part B lands on 6seq"),
    (8,  "Multi-frame, rectified", "PENDING",
         "mk_ctx.py — empty until issue #24 part B lands on 6seq"),
    (9,  "Synthetic / real frame pair (illustrative)", None,
         "mk_qual.py, one seq131 frame"),
    (9,  "Vignetting: mean luminance -16% from 40 to 54 deg", None,
         "measured on the 9 local seq131 syn/real pairs — seq131 only"),
    (10, "Synthetic vs real, AbsRel vs angle", "6seq", "mk_charts.stream_panel"),
]

#: Figures a reader WILL compare, and therefore must be measured alike.
#: Each entry is (why they get compared, [run names]).
GROUPS = [
    ("p4 Result and p10 synthetic-vs-real: p10's subtitle says 'the same 300 "
     "frames as the Result page'", ["6seq", "6seq"]),
    ("p4 Result and p6 depth marginal: p6 explains p4's confound and must be "
     "the same scene", ["6seq", "6seq"]),
    # One group, not two: the frame-count contrast on p7/p8 and the rim
    # comparison against p4 both need the same frames, and `N=1` is p4's own
    # run. Splitting them is how the second one got missed.
    ("p7/p8 across frame counts AND against p4 Result: all four curves plot "
     "AbsRel against incidence angle, `N=1` is the Result run itself, and the "
     "pages are read for the same rim question",
     ["6seq", "6seq_3s", "6seq_5s", "6seq_10s"]),
]

#: Runs the deck is laid out for but that have not been measured yet. A page
#: naming one of these draws its axes empty; the group it belongs to is
#: reported as pending, not as a failure.
PENDING = {
    "6seq_3s":  "issue #24 part B — --context-frames 3 --context-stride 10",
    "6seq_5s":  "issue #24 part B — --context-frames 5 --context-stride 10",
    "6seq_10s": "issue #24 part B — --context-frames 10 --context-stride 10",
}

#: Superseded, kept only so a reader who finds these files knows why they are
#: not on a page any more.
RETIRED = {
    "s131_n1":  "results/fovbench-rectfix-393cab9/partA_seq131/results.json",
    "s131_5s":  "results/fovbench-rectfix-393cab9/partB_5s/results.json",
    "s131_10s": "results/fovbench-rectfix-393cab9/partB_10s/results.json",
}


def load(path):
    raw = subprocess.check_output(["git", "-C", REPO, "show", "results:" + path])
    return json.loads(raw)


def facts(name, path):
    p = load(path)
    cfg = p.get("config", {})
    seqs = p.get("sequences") or []
    return dict(
        name=name, path=path,
        run_dir=path.split("/")[1],
        digest=p.get("digest"),
        n_frames=p.get("n_frames"),
        n_seq=len(seqs),
        seqs=[s.split("_")[-2] for s in seqs],
        seq_dirs=list(seqs),
        ctx=cfg.get("context_frames", 1),
        stride=cfg.get("context_stride", 1),
        models=p.get("requested_models") or [])


def command(f):
    """The command that reproduces this run, from the run's own record."""
    root = "$ADT" if f["n_seq"] > 1 else "$ADT/%s" % f["seq_dirs"][0]
    parts = ["python -m fovbench.run", "--adt-root %s" % root,
             "--protocols radial", "--models %s" % ",".join(f["models"]),
             "--n-frames %d" % (f["n_frames"] // max(f["n_seq"], 1))]
    if f["ctx"] > 1:
        parts += ["--context-frames %d" % f["ctx"],
                  "--context-stride %d" % f["stride"]]
    return " \\\n  ".join(parts)


def status(known, names):
    """OK / PENDING / INCONSISTENT for one comparison group.

    A group with an unmeasured member is PENDING: nothing is on the page yet,
    so there is nothing to be wrong. It becomes OK or INCONSISTENT the moment
    the run lands, which is the point of listing it before it exists.
    """
    fs = [known[n] for n in names if n in known]
    waiting = [n for n in names if n in PENDING]
    digests = sorted({f["digest"] for f in fs})
    dirs = sorted({f["run_dir"] for f in fs})
    if len(digests) > 1 or len(dirs) > 1:
        return "INCONSISTENT", digests, dirs, waiting
    if waiting:
        return "PENDING", digests, dirs, waiting
    return "OK", digests, dirs, waiting


def check(known):
    """Every comparison group must agree on both frames and code."""
    bad = []
    for why, names in GROUPS:
        st, digests, dirs, waiting = status(known, names)
        if st == "PENDING":
            print("pending: %s" % why)
            for n in waiting:
                print("   %s <- %s" % (n, PENDING[n]))
        elif st == "INCONSISTENT":
            bad.append((why, names, digests, dirs))
            print("INCONSISTENT: %s" % why, file=sys.stderr)
            print("   runs   : %s" % ", ".join(names), file=sys.stderr)
            print("   digests: %s" % ", ".join(digests), file=sys.stderr)
            print("   run dir: %s" % ", ".join(dirs), file=sys.stderr)
    return bad


def render(known, bad):
    L = ["# ADT-FOV deck — where every number comes from",
         "",
         "Generated by `docs/decks/provenance.py` from each run's own",
         "`results.json` on the `results` branch. Do not edit by hand; re-run it.",
         "",
         "Deck: `%s`." % DECK,
         "",
         "## The rule",
         "",
         "> Two figures a reader will compare must come from the **same split",
         "> digest** and the **same run directory**.",
         "",
         "The digest pins the frames; the directory pins the code. `--n-frames`",
         "is *per sequence*, and context settings deliberately do not enter the",
         "digest (`fovbench/split.py`), so `N=1` and `N=10` on the same frames",
         "share one — that is the comparison the digest is designed to license.",
         "A **different** digest means different scenes, and a shape read across",
         "the two says nothing about the axis it appears to be about.",
         "",
         "## Runs",
         "",
         "| run | digest | frames | seqs | context | dir |",
         "|---|---|---|---|---|---|"]
    for f in known.values():
        L.append("| `%s` | `%s` | %d | %d (%s) | %s | `%s` |" % (
            f["name"], f["digest"], f["n_frames"], f["n_seq"],
            ", ".join(f["seqs"]),
            "1 frame" if f["ctx"] == 1 else "%d @ stride %d" % (f["ctx"], f["stride"]),
            f["run_dir"]))
    L += ["", "### Commands", ""]
    for f in known.values():
        L += ["`%s` — %s" % (f["name"], f["path"]), "", "```bash",
              command(f), "```", ""]
    L += ["## Page by page", "",
          "| page | what | run | generator |", "|---|---|---|---|"]
    for pg, what, run, gen in PAGES:
        L.append("| %d | %s | %s | %s |" % (
            pg, what, ("`%s`" % run) if run else "*not a run*", gen))
    L += ["", "## Comparison groups", ""]
    waiting_any = False
    for why, names in GROUPS:
        st, digests, dirs, waiting = status(known, names)
        waiting_any = waiting_any or bool(waiting)
        L += ["- **%s** — %s" % (st, why),
              "  runs `%s`; digest%s %s; dir%s %s" % (
                  "`, `".join(names), "" if len(digests) == 1 else "s",
                  ", ".join("`%s`" % d for d in digests) or "*none yet*",
                  "" if len(dirs) == 1 else "s",
                  ", ".join("`%s`" % d for d in dirs) or "*none yet*")]
        for n in waiting:
            L.append("  not measured yet: `%s` <- %s" % (n, PENDING[n]))
    if waiting_any:
        L += ["",
              "### Why pages 7 and 8 are blank",
              "",
              "They used to be `50` frames of **seq131 alone** while page 4 is",
              "`300` frames over **six sequences**, and both plot AbsRel against",
              "incidence angle. seq131 is not a representative sixth: its rim",
              "penalty is the lowest of the six in 5 of 6 model x view cells,",
              "and on the rectified view VGGT-Omega's is 0.97 — the rim",
              "marginally *better* than the centre. Going to 200 frames of",
              "seq131 moves it further from the pooled value, so this is the",
              "sequence, not the sample size.",
              "",
              "Rather than caption around it, the curves came off the pages. The",
              "axes stay at final geometry so the measured run drops in without",
              "moving the layout. Ticket 024 part B / issue #24 is the run.",
              "",
              "The superseded seq131 payloads:", ""]
        for n, p in RETIRED.items():
            L.append("- `%s` — `%s`" % (n, p))
    if bad:
        L += ["", "### Failing groups", ""]
        for why, names, digests, dirs in bad:
            L.append("- %s — digests %s, dirs %s" % (
                why, ", ".join("`%s`" % d for d in digests),
                ", ".join("`%s`" % d for d in dirs)))
    L += ["",
          "## Not covered by the digest rule", "",
          "Three claims in the deck are not benchmark runs and carry their own",
          "scope, stated on their own pages:", "",
          "- page 3's 34.0 % / 29.5 % are closed-form over the KB4 LUT at the",
          "  shipped output focal length — no data, no model.",
          "- pages 2, 3 and 9 show **one illustrative frame** of seq131. They",
          "  make no quantitative claim.",
          "- page 9's vignetting number (mean luminance -16 % from 40 to 54 deg)",
          "  *is* quantitative and is measured on the 9 local seq131 syn/real",
          "  pairs only. The page names the sequence in the sentence itself.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="check the groups and write nothing")
    a = ap.parse_args()
    known = OrderedDict()
    for name, path in RUNS.items():
        known[name] = facts(name, path)
    bad = check(known)
    if not a.check:
        with open(OUT, "w") as fh:
            fh.write(render(known, bad))
        print("wrote", OUT)
    print("%d comparison group(s) inconsistent" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
