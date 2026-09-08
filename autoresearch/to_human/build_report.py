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

    text = re.sub(r"<[^>]*>", "", html)
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
    secs = re.findall(r'<span class="eyebrow">(\d+\w?)</span>', stripped)
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
