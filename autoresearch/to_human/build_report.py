"""Rebuild autoresearch/to_human/fisheye-rim-report.html from its fragments.

The report is assembled from four body fragments plus the shared <style> block,
with every figure inlined as a base64 data URI. That process lived only in a
session's scratchpad for most of its life, which meant the published page could
not be rebuilt by anyone else — and once bit me: an edit went into an
intermediate fragment AFTER it had been merged into the file the build reads,
so two rebuilds silently republished a claim that had already been retracted.

Hence the guards. They are cheap and each one exists because something got past
me:

  STALE       strings from claims that were withdrawn. If one reappears, a
              fragment has been resurrected from an old copy.
  ONE_PAGE    section 00 is titled "one page of conclusions". It reached 14 rows
              and 6,283 characters by being appended to once per work session.
              A summary degrades monotonically unless something re-reads it.
  tag balance the fragments are hand-edited HTML; an unbalanced <div> has
              slipped in twice.

Usage:
    python build_report.py --fragments <dir> [--figs <dir>] [-o <out.html>]

`--fragments` holds body1..body4.html and report_stripped.html (the latter only
for its <head>). Figure names in the fragments are `src="__FIG_<name>__"` and
resolve against --figs.
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

STALE = ("三分之二", "这一块是黑边", "发布 16 个数的径向标定",
         # H21 refuted this inference: matching the lens buys nothing (+0.84 pt,
         # inside noise), and the curve is a property of the ANGLE, not the lens.
         "所以它能带到另一个房间、另一台设备")
# Numbers a correction superseded, with how many times each is still ALLOWED to
# appear -- always somewhere that labels it as the old value ("as reported", or
# a quote of what was originally written). Count UP = it leaked into a new
# sentence. Count DOWN = a labelled comparison was lost. Counted on the
# tag-stripped text, so base64 image data cannot produce false matches.
#
# This exists because a correction on 2026-09-08 fixed one table and left the
# same numbers standing in four other places -- a callout, a comparison column,
# a recommendations bullet and an uncertainty row. Only re-reading caught it.
SUPERSEDED = {
    # counts verified against the built page on 2026-09-08, not guessed --
    # the first version of this table guessed two of them and the tripwire
    # caught itself on the first run, which is the behaviour wanted.
    "+254.2": 1,   # 03ae's "as reported" column
    "+350.1": 1,   # 03ae's "as reported" column
    "+67.0": 1,    # 03ae's "as reported" column
    "+58.0": 3,    # 03ae's column, plus two unrelated data-ladder numbers in
                   # body4 that happen to share the string
    "+254%": 0,    # must not appear anywhere; superseded by +214.5%
    "+350%": 1,    # only 03ae's quote of the wording it is correcting
    "+210.9": 0,   # H47, superseded by +154.8%
    "+41.9": 0,    # H47, superseded by +29.9%
    "+56.8": 0,    # H47, superseded by +45.7%
}
VOID = {"img", "br", "meta", "link", "input", "hr",
        "path", "rect", "circle", "line", "polyline", "polygon", "text"}
EXTRA_CSS = """<style>
.gal.five{grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:10px}
.gal.five .shot .tag{font-size:.68rem}
</style>
"""


class _Balance(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append((tag, self.getpos()))


def build(frag: Path, figs: Path, out: Path) -> None:
    head = (frag / "report_stripped.html").read_text().split("<body>")[0]
    head = head.replace("</head>", EXTRA_CSS + "</head>")
    body = "".join((frag / f"body{i}.html").read_text() for i in (1, 2, 3, 4))

    missing = []

    def inline(m):
        name = m.group(1)
        for ext in (".jpg", ".jpeg", ".png"):
            f = figs / f"{name}{ext}"
            if f.exists():
                return ('src="data:image/jpeg;base64,'
                        + base64.b64encode(f.read_bytes()).decode() + '"')
        missing.append(name)
        return m.group(0)

    html = head + re.sub(r'src="__FIG_([a-z0-9_]+)__"', inline, body)
    if missing:
        sys.exit(f"[build] no figure file for: {sorted(set(missing))}")

    for bad in STALE:
        if bad in html:
            sys.exit(f"[build] a withdrawn claim is back in the page: {bad!r}. "
                     f"A fragment has been resurrected from an old copy.")

    # Column-count check. These tables are hand-edited HTML and a header that
    # has drifted from its rows renders as a silently shifted column, which no
    # existing check would catch. Compare the FIRST header row's effective
    # width (colspan summed) against each body row's -- a two-row header's
    # second row belongs to the first row's groups and must not be added in.
    for tbl in re.findall(r"<table[^>]*>.*?</table>", html, re.S):
        head = re.search(r"<thead.*?</thead>", tbl, re.S)
        if not head:
            continue
        first = re.search(r"<tr\b[^>]*>(.*?)</tr>", head.group(0), re.S)
        if not first:
            continue

        def eff(row):
            n = 0
            for cell in re.finditer(r"<t[dh]\b([^>]*)>", row):
                cs = re.search(r"colspan\s*=\s*[\"']?(\d+)", cell.group(1))
                n += int(cs.group(1)) if cs else 1
            return n

        ncol = eff(first.group(1))
        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", tbl, re.S):
            if "<td" not in row:
                continue
            if eff(row) != ncol:
                cap = re.search(r"<caption[^>]*>(.*?)</caption>", tbl, re.S)
                cap = re.sub(r"<[^>]+>", "", cap.group(1))[:60] if cap else "?"
                print(f"[build] table column mismatch: header {ncol}, a body "
                      f"row {eff(row)} — {cap.strip()}", file=sys.stderr)
                break

    text = re.sub(r"<[^>]*>", "", html)

    for num, allowed in SUPERSEDED.items():
        got = text.count(num)
        if got != allowed:
            print(f"[build] superseded number {num!r} appears {got}x, expected "
                  f"{allowed}x — UP means it leaked into a new sentence, DOWN "
                  f"means a labelled comparison was lost. See SUPERSEDED.",
                  file=sys.stderr)
    m = re.search(r"一页纸的结论(.{0,20000}?)问题与协议", text, re.S)
    if m:
        n = len(m.group(1).replace(" ", "").replace("\n", ""))
        rows = html.count('<span class="who">', html.index("一页纸的结论"),
                          html.index("问题与协议"))
        if n > 2500 or rows > 9:
            sys.exit(f"[build] section 00 is {n} chars / {rows} rows. It is "
                     f"titled 'one page'; keep it under 2500 / 9. Re-read it "
                     f"whole instead of appending another row.")
        print(f"[build] section 00: {rows} rows, {n} chars")

    stripped = re.sub(r'src="data:[^"]*"', 'src=""', html)
    b = _Balance()
    b.feed(stripped)
    if b.errors or b.stack:
        sys.exit(f"[build] unbalanced HTML: stray closes {b.errors[:3]}, "
                 f"unclosed {b.stack[:3]}")

    out.write_text(html)
    # `\w?` only ever matched ONE trailing letter, so section 03aa was silently
    # dropped from the count and the id list — the check that exists to notice a
    # missing section quietly stopped seeing one.
    secs = re.findall(r'<span class="eyebrow">(\d+[a-z]*)</span>', stripped)
    print(f"[build] wrote {out} — {len(html)/1e6:.2f} MB, {len(secs)} sections: "
          f"{' '.join(secs)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fragments", required=True, type=Path)
    p.add_argument("--figs", type=Path, default=None)
    p.add_argument("-o", "--out", type=Path,
                   default=Path(__file__).resolve().parent / "fisheye-rim-report.html")
    a = p.parse_args(argv)
    build(a.fragments, a.figs or (a.fragments / "figs"), a.out)


if __name__ == "__main__":
    main()
