#!/usr/bin/env python3
"""Derive ticket status from the `results` branch, which is what the tickets
README says status means: "Status is derived from what is on the results
branch, not from memory -- if a run is not there, the ticket is not done."

That rule was being applied by hand and drifted: on 2026-08-22 the board still
called #024 "open -- not started" four days after its artefacts were pushed.
This makes the rule executable.

Usage:
    python3 tools/ticket_status.py            # report
    python3 tools/ticket_status.py --stale    # only tickets whose Status line
                                              # disagrees with the branch (exit 1
                                              # if any -- usable as a CI gate)

Reads `origin/results` through git, so it needs no checkout of that branch.
Nothing is written; this only reports.

KNOWN LIMITATIONS -- read these before trusting a row:

* Some older `meta.json` files put an ISSUE number in the `ticket` field. E.g.
  `vanilla-repro-3f15a9266d` says `ticket: 9`, but it is ticket **004**'s run
  (issue 9). Those land in the "says done, no artefact" bucket, which is why
  that bucket is labelled `check` and not `STALE` -- it needs a human, and the
  real fix is correcting the meta, not loosening the matching here.
* Ticket numbers 026 and 028 are each used by TWO files, against the README's
  own rule that numbers are never reused. An artefact naming `28` is therefore
  attributed to both, and one of the two is a false STALE.
* A ticket can be legitimately done with no artefact (code-only work, e.g. 021).
  Absence of a results dir is not evidence of an unfinished ticket; it is only
  evidence that this tool cannot confirm it.

So: `STALE` (artefacts exist, ticket still says open) is the reliable signal and
is what `--stale` gates on. `check` is a prompt to look, not a verdict.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TICKETS = REPO / "docs/handoff/tickets"
RESULTS_REF = "origin/results"

# A ticket is "claimed done" by its own Status line if it says so. These are the
# phrasings actually in use; keep them literal rather than clever, so a new
# phrasing shows up as a mismatch to be looked at rather than being silently
# absorbed by a loose regex.
DONE_MARKERS = ("done", "complete", "run,", "fixed", "closed", "folded into")
OPEN_MARKERS = ("open", "not started", "not audited", "draft")


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout


def results_dirs():
    out = git("ls-tree", "--name-only", RESULTS_REF, "results/")
    return [d.rstrip("/") for d in out.split() if d.startswith("results/")]


def meta_of(d):
    raw = git("show", f"{RESULTS_REF}:{d}/meta.json")
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}          # present but unparseable -- distinct from absent


def ticket_numbers(meta):
    """Ticket numbers a results dir claims. meta['ticket'] is sometimes a path,
    sometimes a bare number; meta['issue'] is a GITHUB issue and is NOT a ticket
    number -- the two numbering schemes diverged and conflating them is how the
    board got wrong before."""
    if not meta:
        return set()
    nums = set()
    t = meta.get("ticket") or meta.get("tickets")
    items = t if isinstance(t, (list, tuple)) else ([] if t is None else [t])
    for s in items:
        for m in re.finditer(r"tickets/(\d{3})-", str(s)):
            nums.add(m.group(1))
        if re.fullmatch(r"\s*\d{1,3}\s*", str(s)):
            nums.add(f"{int(str(s).strip()):03d}")
    return nums


def ticket_status_line(path):
    for line in path.read_text(errors="replace").splitlines()[:12]:
        if line.startswith("**Status:**"):
            return re.sub(r"\s+", " ", line[len("**Status:**"):]).strip()
    return None


def claims_done(status):
    s = (status or "").lower()
    # order matters: "open" beats a stray "done" inside prose
    for m in OPEN_MARKERS:
        if s.startswith(m) or s.startswith("**" + m):
            return False
    return any(m in s for m in DONE_MARKERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", action="store_true",
                    help="only print disagreements; exit 1 if any")
    a = ap.parse_args()

    have = {}                      # ticket number -> [results dirs]
    unattributed = []
    for d in results_dirs():
        meta = meta_of(d)
        nums = ticket_numbers(meta)
        if not nums:
            unattributed.append((d, meta is None))
        for n in nums:
            have.setdefault(n, []).append(d.replace("results/", ""))

    rows, stale = [], []
    for path in sorted(TICKETS.glob("[0-9][0-9][0-9]-*.md")):
        num = path.name[:3]
        status = ticket_status_line(path)
        arts = have.get(num, [])
        said, real = claims_done(status), bool(arts)
        rows.append((num, path.name, said, real, arts, status))
        if said != real:
            stale.append(rows[-1])

    def show(rs):
        for num, name, said, real, arts, status in rs:
            if said == real:
                kind = "  ok  "
            elif real and not said:
                kind = "STALE "      # artefacts exist, ticket still says open
            else:
                kind = "check "      # says done, no artefacts: often code-only
            print(f"{kind} {num}  {name}")
            if arts:
                print(f"        -> {', '.join(arts)}")
            if said != real:
                print(f"        Status: {status}")

    dupes = {}
    for num, name, *_ in rows:
        dupes.setdefault(num, []).append(name)
    dupes = {k: v for k, v in dupes.items() if len(v) > 1}

    if a.stale:
        if stale or dupes:
            print(f"{len(stale)} ticket(s) disagree with {RESULTS_REF}:\n")
            show(stale)
            if dupes:
                print("\nduplicate ticket numbers (README: numbers are never reused):")
                for k, v in sorted(dupes.items()):
                    print(f"  {k}: {', '.join(v)}")
            return 1
        print(f"All ticket Status lines agree with {RESULTS_REF}.")
        return 0

    show(rows)
    n_stale = sum(1 for r in stale if r[3] and not r[2])
    print(f"\n{len(rows)} tickets, {len(stale)} disagree with {RESULTS_REF} "
          f"({n_stale} STALE: artefacts exist but the ticket still says open).")
    if dupes:
        print("\nduplicate ticket numbers (README: numbers are never reused):")
        for k, v in sorted(dupes.items()):
            print(f"  {k}: {', '.join(v)}")
    if unattributed:
        print("\nresults dirs naming no ticket (cannot be cross-checked):")
        for d, no_meta in unattributed:
            print(f"  {d.replace('results/','')}"
                  f"{'   (no meta.json)' if no_meta else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
