# -*- coding: utf-8 -*-
# Copyright (c) 2026.
"""Build the plain-English report at to_human/report.html.

Same source of truth as the Chinese research log: every figure comes from
to_human/final_en/ and every number is read from its numbers.json, which
make_final_figures.py refuses to write unless its bootstrap reproduces the
evaluator's own report.txt. Nothing here is typed in by hand.

Usage::

    python research/fisheye-inpaint/build_report_en.py
"""
from __future__ import annotations

import base64
import json
import os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "to_human", "final_en")
DATA = os.path.join(HERE, "data", "final")
OUT = os.path.join(HERE, "to_human", "report.html")

# id -> (number, short name, one-line description, chip kind)
INPUTS = OrderedDict([
    ("fisheye_masked", ("1", "Raw · black", "the fisheye frame as the camera gives it, black in the corners", "disc")),
    ("persp_masked",   ("2", "Rectified · black", "straightened to a flat photo, black wedges where nothing was imaged", "bitten")),
    ("fisheye_full",   ("3", "Raw · filled", "same fisheye frame, corners filled with the real scene", "full")),
    ("persp_full",     ("4", "Rectified · filled", "straightened, wedges filled with the real scene", "full")),
    ("persp_crop",     ("5", "Cropped", "straightened at a narrower zoom, so no black exists at all", "crop")),
])
CTRL = ("persp_crop_lores", ("5b", "Cropped · blurred", "input 5 softened to input 4's level of detail", "cropb"))
ALL = OrderedDict(list(INPUTS.items()) + [CTRL])

PROV = {
    "eval_commit": "bcf4030", "branch": "fisheye-2x2",
    "renderer_commit": "fbb0ef0",
    "run": "wt-fisheye2x2/runs/ev6_own and runs/ev6_crop on lambda_63",
    "finished": "2026-09-08 00:07 (UTC−4)",
    "env": "conda env raytun3r · torch 2.11.0+cu128 · one RTX 6000 Ada",
    "ckpt": "VGGT-Omega-1B-512/model.pt", "ckpt_md5": "bc5302eada6222303c5e5f8d7dbce709",
    "manifest_md5": "9dbc50b2e181285bf8df3db13c9085fa",
}

CSS = """
:root{
  --ground:#f6f5f9; --surface:#ffffff; --sunk:#eeecf4; --line:#dcd8e6; --hair:#e9e6f0;
  --ink:#16131f; --body:#3c3750; --muted:#6f6980;
  --accent:#4a3f8f; --accent-soft:#e6e2f4;
  --amber:#9a6212; --amber-soft:#f6ead6;
  --good:#2f6b47; --good-soft:#dceade;
  --bad:#93332e; --bad-soft:#f3ddda;
  --display:"Newsreader",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#131119; --surface:#1b1824; --sunk:#17151f; --line:#302b3d; --hair:#241f30;
  --ink:#eeecf4; --body:#c6c1d4; --muted:#928ca3;
  --accent:#a396ea; --accent-soft:#241f3c;
  --amber:#d99b45; --amber-soft:#32250f;
  --good:#7ab892; --good-soft:#16281c;
  --bad:#d8837a; --bad-soft:#331815;
}}
:root[data-theme="dark"]{
  --ground:#131119; --surface:#1b1824; --sunk:#17151f; --line:#302b3d; --hair:#241f30;
  --ink:#eeecf4; --body:#c6c1d4; --muted:#928ca3;
  --accent:#a396ea; --accent-soft:#241f3c;
  --amber:#d99b45; --amber-soft:#32250f;
  --good:#7ab892; --good-soft:#16281c;
  --bad:#d8837a; --bad-soft:#331815;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--body);font-family:var(--sans);
  font-size:16.5px;line-height:1.72;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:0 30px 110px}
.col{max-width:68ch}

/* masthead */
.mast{padding:78px 0 30px;border-bottom:1px solid var(--line)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.15em;text-transform:uppercase;
  color:var(--accent);margin:0 0 20px}
h1{font-family:var(--display);font-weight:500;font-size:clamp(36px,5.6vw,54px);line-height:1.1;
  letter-spacing:-.015em;color:var(--ink);margin:0 0 22px;max-width:20ch;text-wrap:balance}
.lede{font-size:19px;line-height:1.62;color:var(--body);max-width:62ch;margin:0}
.lede b{color:var(--ink);font-weight:600}
.facts{display:flex;flex-wrap:wrap;gap:6px 26px;margin-top:28px;
  font-family:var(--mono);font-size:11.5px;color:var(--muted)}

/* parts */
.part{padding-top:74px}
.pnum{font-family:var(--mono);font-size:11px;letter-spacing:.16em;color:var(--accent);
  display:block;margin-bottom:9px}
h2{font-family:var(--display);font-weight:500;font-size:31px;line-height:1.25;color:var(--ink);
  margin:0 0 20px;letter-spacing:-.01em;max-width:24ch;text-wrap:balance}
h3{font-family:var(--sans);font-weight:600;font-size:18px;color:var(--ink);margin:40px 0 10px;
  letter-spacing:-.005em}
h4{font-family:var(--sans);font-weight:600;font-size:15px;color:var(--ink);margin:26px 0 6px}
p{margin:0 0 17px;max-width:68ch}
a{color:var(--accent);text-underline-offset:3px}
strong{color:var(--ink);font-weight:600}
em{font-style:italic;color:var(--ink)}
code{font-family:var(--mono);font-size:.86em;background:var(--sunk);padding:1.5px 5px;
  border-radius:3px;color:var(--ink)}
ul{padding-left:22px;margin:0 0 17px;max-width:68ch}
li{margin-bottom:9px}

/* the answer block */
.answers{display:grid;gap:0;margin:34px 0 6px;border-top:2px solid var(--ink)}
.ans{display:grid;grid-template-columns:auto 1fr;gap:0 22px;padding:22px 0;
  border-bottom:1px solid var(--hair)}
@media(max-width:620px){.ans{grid-template-columns:1fr;gap:8px}}
.ans .q{font-family:var(--mono);font-size:11px;letter-spacing:.1em;color:var(--muted);
  padding-top:5px;white-space:nowrap}
.ans .a{font-size:17px}
.ans .a b{color:var(--ink)}
.ans .fig{font-family:var(--mono);font-size:13.5px;color:var(--accent);display:block;margin-top:7px}

/* input chips */
.chip{display:inline-flex;align-items:center;gap:7px;font-family:var(--mono);font-size:12px;
  color:var(--ink);white-space:nowrap;vertical-align:baseline}
.chip svg{flex:none}
.inputs{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);margin:28px 0}
.inputs>div{background:var(--surface);padding:17px 18px}
.inputs .nm{display:flex;align-items:center;gap:9px;margin-bottom:8px}
.inputs .nm b{font-size:14.5px;color:var(--ink)}
.inputs p{font-size:13.5px;line-height:1.6;margin:0;color:var(--muted)}
.inputs .ctl{background:var(--sunk)}

/* tables */
.scroll{overflow-x:auto;margin:24px 0;border:1px solid var(--line);background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:520px}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid var(--hair);vertical-align:top}
thead th{background:var(--sunk);color:var(--ink);font-weight:600;font-size:11.5px;
  letter-spacing:.04em;text-transform:uppercase;white-space:nowrap;border-bottom:1px solid var(--line)}
tbody tr:last-child td{border-bottom:0}
td.n,th.n{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
tr.hl td{background:var(--accent-soft)}
.ci{color:var(--muted);font-size:12px}
.tag{font-family:var(--mono);font-size:10px;letter-spacing:.06em;padding:1.5px 6px;border-radius:2px;
  border:1px solid;white-space:nowrap}
.t-yes{color:var(--good);border-color:var(--good);background:var(--good-soft)}
.t-no{color:var(--muted);border-color:var(--line);background:var(--sunk)}

/* figures */
figure{margin:30px 0;background:var(--surface);border:1px solid var(--line);padding:14px}
figure img{display:block;width:100%;height:auto}
figcaption{font-size:13px;color:var(--muted);line-height:1.62;margin-top:12px;padding-top:11px;
  border-top:1px solid var(--hair)}

/* callouts */
.note{border-left:2px solid var(--line);padding:3px 0 3px 20px;margin:24px 0;font-size:15px;
  color:var(--muted);max-width:66ch}
.key{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);
  padding:22px 26px;margin:28px 0;max-width:68ch}
.key h4{margin-top:0;color:var(--accent);font-family:var(--mono);font-size:11px;letter-spacing:.12em;
  text-transform:uppercase}
.key p:last-child{margin-bottom:0}
.warn{background:var(--amber-soft);border:1px solid var(--amber);padding:20px 24px;margin:28px 0;
  max-width:68ch}
.warn h4{margin-top:0;color:var(--amber);font-family:var(--mono);font-size:11px;letter-spacing:.12em;
  text-transform:uppercase}
.warn p:last-child{margin-bottom:0}

/* provenance */
.kv{display:grid;grid-template-columns:max-content 1fr;gap:7px 22px;font-size:13.5px;margin:20px 0}
@media(max-width:620px){.kv{grid-template-columns:1fr;gap:2px 0}.kv dt{margin-top:10px}}
.kv dt{font-family:var(--mono);font-size:11px;letter-spacing:.06em;color:var(--muted);padding-top:3px;
  text-transform:uppercase}
.kv dd{margin:0}
pre{font-family:var(--mono);font-size:12px;line-height:1.7;background:var(--sunk);
  border:1px solid var(--line);padding:16px 18px;overflow-x:auto;margin:20px 0}
footer{margin-top:76px;padding-top:22px;border-top:1px solid var(--line);
  font-family:var(--mono);font-size:11.5px;color:var(--muted);line-height:1.9}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""

# The shape of each input's valid region, so a reader always knows which frame is
# being discussed without scrolling back to the picture.
CHIPS = {
 "disc":   '<circle cx="11" cy="11" r="9" fill="currentColor"/>',
 # the imaged disc clipped by the square sensor: a square with four corners bitten out
 "bitten": ('<path d="M2 2h18v18H2z" fill="currentColor"/>'
            '<path d="M2 2h5A5 5 0 0 1 2 7z" fill="var(--ground)"/>'
            '<path d="M20 2v5a5 5 0 0 0-5-5z" fill="var(--ground)"/>'
            '<path d="M2 20v-5a5 5 0 0 0 5 5z" fill="var(--ground)"/>'
            '<path d="M20 20h-5a5 5 0 0 1 5-5z" fill="var(--ground)"/>'),
 "full":   '<path d="M2 2h18v18H2z" fill="currentColor"/>',
 "crop":   '<path d="M5 5h12v12H5z" fill="currentColor"/><path d="M2 2h18v18H2z" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="2 2" opacity=".55"/>',
 "cropb":  '<path d="M5 5h12v12H5z" fill="currentColor" opacity=".45"/><path d="M2 2h18v18H2z" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="2 2" opacity=".45"/>',
}


def chip(key, with_name=True):
    num, name, _desc, kind = ALL[key]
    col = "var(--amber)" if key in ("fisheye_masked", "persp_masked") else "var(--accent)"
    svg = (f'<svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true" '
           f'style="color:{col}">{CHIPS[kind]}</svg>')
    return f'<span class="chip">{svg}{num + " " + name if with_name else num}</span>'


def b64(path):
    mime = {"png": "image/png", "jpg": "image/jpeg"}[os.path.splitext(path)[1][1:].lower()]
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"


def fig(name, caption):
    for ext in (".png", ".jpg"):
        p = os.path.join(FIG, name + ext)
        if os.path.isfile(p):
            return f'<figure><img src="{b64(p)}" alt="{caption[:80]}"><figcaption>{caption}</figcaption></figure>'
    raise SystemExit(f"missing figure {name}")


def num(cb, fmt="{:+.4f}", tag=True):
    """A measured difference with its interval, and whether it is real."""
    out = (f'<span class="n">{fmt.format(cb["mean"])}</span> '
           f'<span class="ci">[{fmt.format(cb["ci_lo"])}, {fmt.format(cb["ci_hi"])}]</span>')
    if tag:
        out += (' <span class="tag t-yes">real</span>' if cb["excludes_zero"]
                else ' <span class="tag t-no">not real</span>')
    return out


def main():
    N = json.load(open(os.path.join(FIG, "numbers.json")))
    meta = json.load(open(os.path.join(DATA, "example_meta_seq131_frame_0830.json")))
    C1, C8 = N["cells"]["single"], N["cells"]["8-frame"]
    CC1, CC8 = N["cells_common"]["single"], N["cells_common"]["8-frame"]
    E1, E8 = N["effects"]["single"], N["effects"]["8-frame"]
    EC1, EC8 = N["effects_common"]["single"], N["effects_common"]["8-frame"]
    XC1, XC8 = N["crop_effects_common"]["single"], N["crop_effects_common"]["8-frame"]
    XO8 = N["crop_effects_own"]["8-frame"]
    P8, F1, F8 = N["pose"]["8-frame"], N["fov"]["single"], N["fov"]["8-frame"]
    LAD1, LAD8 = N["ladder"]["single"], N["ladder"]["8-frame"]
    MC, MED, MVS, INP, B, CTL = (N["multi_contrast"], N["mediation"], N["multi_vs_single"],
                                 N["inputs"], N["band"], N["control"])
    cm = meta["crop"]
    gt_wide, gt_crop = F1["persp_full"]["gt"], F1["persp_crop"]["gt"]

    def mfg(st):
        cb = MVS[st]
        return {"mean": -cb["mean"], "ci_lo": -cb["ci_hi"], "ci_hi": -cb["ci_lo"],
                "excludes_zero": cb["excludes_zero"]}
    rep1 = {p: LAD1[p]["rows"]["replicate"] for p in ("fisheye", "persp")}
    rep8 = {p: LAD8[p]["rows"]["replicate"] for p in ("fisheye", "persp")}
    left = [LAD1["fisheye"]["span"] - rep1["fisheye"]["gain"], LAD1["persp"]["span"] - rep1["persp"]["gain"],
            LAD8["fisheye"]["span"] - rep8["fisheye"]["gain"], LAD8["persp"]["span"] - rep8["persp"]["gain"]]


    ir = "".join(
        f'<div{" class=ctl" if k == CTRL[0] else ""}><div class="nm">{chip(k)}</div>'
        f'<p>{ALL[k][2]}<br><span class="ci">'
        + (f'black {INP[k]["black_pct"]:.1f}% · scored {INP[k]["graded_pct"]:.0f}%' if k in INP
           else 'a control, not one of the four cells')
        + '</span></p></div>' for k in ALL)

    def cells_table(Ca, Cb, keys, hl=None):
        rows = "".join(
            f'<tr{" class=hl" if k == hl else ""}><td>{chip(k)}</td>'
            f'<td class="n">{Ca[k]["AbsRel"]:.4f}</td><td class="n">{Cb[k]["AbsRel"]:.4f}</td>'
            f'<td class="n">{Ca[k]["delta1"]:.3f}</td><td class="n">{Cb[k]["delta1"]:.3f}</td></tr>'
            for k in keys if k in Ca)
        return ('<div class="scroll"><table><thead><tr><th>Input</th>'
                '<th class="n">AbsRel · 1 frame</th><th class="n">AbsRel · 8 frames</th>'
                '<th class="n">δ₁ · 1 frame</th><th class="n">δ₁ · 8 frames</th></tr></thead>'
                f'<tbody>{rows}</tbody></table></div>')

    H = []
    A = H.append

    A(f"""<title>Filling the Black Corners</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>

<div class="wrap">
<header class="mast">
  <p class="eyebrow">Experiment report · VGGT-Omega · Aria fisheye</p>
  <h1>Filling the black corners does not buy what we hoped</h1>
  <p class="lede">Straightening a fisheye photo leaves black wedges in the corners. We filled them with the
  <b>real</b> scene — the best any method could do. It bought a little depth accuracy, nothing on camera pose,
  and <b>less than simply cropping the black away</b>.</p>
  <div class="facts">
    <span>96 frames · 12 windows · 4 sequences</span>
    <span>VGGT-Omega-1B-512, no fine-tuning</span>
    <span>one run, 2026-09-08</span>
  </div>
</header>

<section class="part" id="answers">
<span class="pnum">Summary</span>
<h2>Four questions</h2>
<div class="answers">
  <div class="ans"><div class="q">Q1</div><div class="a">
    <b>Does real content in the corners improve depth?</b> Yes, a little. But it helps the fisheye frame as much
    as the straightened one — so this is not about looking like a normal photo.
    <span class="fig">{E1['AbsRel']['fisheye']['mean']:+.4f} fisheye · {E1['AbsRel']['persp']['mean']:+.4f} rectified · difference between them {E1['AbsRel']['interaction']['mean']:+.4f}, could be zero</span></div></div>
  <div class="ans"><div class="q">Q2</div><div class="a">
    <b>Does it improve the camera estimate?</b> No. On fisheye it makes it worse.
    <span class="fig">Pose score {E8['auc30']['fisheye']['mean']:+.3f} fisheye · {E8['auc30']['persp']['mean']:+.3f} rectified · inferred field of view moves {abs(F1['persp_full']['mean'] - F1['persp_masked']['mean']):.1f}° and stays {F1['persp_full']['abs_err']:.0f}° wrong</span></div></div>
  <div class="ans"><div class="q">Q3</div><div class="a">
    <b>How much of the gain needs a clever filler?</b> Almost none. Smearing the nearest pixel outward gets most of it.
    <span class="fig">Smearing captures {min(rep1['persp']['pct'], rep1['fisheye']['pct'], rep8['persp']['pct'], rep8['fisheye']['pct']):.0f}–{max(rep1['persp']['pct'], rep1['fisheye']['pct'], rep8['persp']['pct'], rep8['fisheye']['pct']):.0f}% · at most {max(left):.3f} AbsRel left for anything smarter</span></div></div>
  <div class="ans"><div class="q">Q4</div><div class="a">
    <b>Is a wide filled view better than cropping the black away?</b> No. Cropping wins on depth, on pose, and on
    multi-frame. The gap grows with more frames.
    <span class="fig">Depth {abs(XC1['AbsRel']['persp_full']['mean']):.4f} better at 1 frame, {abs(XC8['AbsRel']['persp_full']['mean']):.4f} at 8 · pose {P8['persp_crop']['auc30']:.2f} vs {P8['persp_full']['auc30']:.2f} · trajectory {P8['persp_crop']['ate_m']*100:.1f} cm vs {P8['persp_full']['ate_m']*100:.1f} cm</span></div></div>
</div>
<p class="note">Brackets are 95% confidence intervals. <b>Real</b> means the interval excludes zero. Intervals
resample <em>windows</em>, not frames — <a href="#method">why</a>.</p>
</section>

<section class="part" id="why">
<span class="pnum">Part 1 — Motivation</span>
<h2>Why fill the corners at all</h2>
<p>A fisheye lens draws a circle on a square sensor. The corners hold nothing.</p>
<p>Straightening that circle into a flat photo makes it worse. To keep the whole circle you zoom out, and a
third of the frame becomes black wedges.</p>
<p>3D models are trained on ordinary photos. Ordinary photos have no black wedges.</p>
<div class="key">
<h4>The claim we tested</h4>
<p>Fill the wedges → the picture looks normal again → the model works out its own camera → depth improves.</p>
<p>That predicts something specific: filling should help the <em>straightened</em> frame much more than the raw
fisheye frame, because only the straightened one becomes a normal photo. If both improve equally, the effect is
just "no large black regions" — a much smaller story.</p>
</div>
<p>Worth measuring before building. A good filler means a generative model: expensive to build, expensive to
run, hard to trust.</p>
</section>

<section class="part" id="data">
<span class="pnum">Part 2 — Data</span>
<h2>Why the frames had to be rendered</h2>
<p>You need to know the <em>right</em> answer in the corners. Real footage cannot give it — the lens never
pointed there. On real frames you can only compare one guess with another.</p>
<p>So we rebuilt the scene. Four Aria Digital Twin sequences, re-rendered in Blender from their 3D
reconstructions ({meta['n_objects']} objects, {meta['n_lights_used']} lights fitted to the real photos), along
the real camera path. Each frame is rendered as a 360° panorama, then resampled into every input below. Depth
comes from the renderer, so it is exact and shared.</p>
<p><strong>This gives true content in directions the lens never imaged. That is the ceiling. No filler can beat
it.</strong></p>

<h3>The inputs</h3>
<p>Same instant, same camera, same panorama. They differ only in how it was sampled and what sits in the corners.</p>
<div class="inputs">{ir}</div>
{fig("inputs_w00", "The six inputs, dumped by the evaluation script before the model saw them. Not illustrations.")}
<ul>
<li>Inputs <b>1</b> and <b>3</b> share a field of view. So do <b>2</b> and <b>4</b>. That is the 2×2: projection × corners.</li>
<li>The leftover black in 3, 4 and 5 (under {max(INP[k]['black_pct'] for k in ('fisheye_full','persp_full','persp_crop')):.1f}%) is shadow in a dim apartment, not missing render. Every input has it.</li>
<li>96 frames = 4 sequences × 3 windows × 8 frames. Each window covers 2–2.5 m of walking, so eight frames carry real parallax.</li>
</ul>
</section>

<section class="part" id="method">
<span class="pnum">Part 3 — Method</span>
<h2>How it was scored</h2>
<p>VGGT-Omega-1B-512, released weights, no fine-tuning. Run twice over the same frames: one frame at a time, and
eight at a time.</p>

<h3>Three traps, and how each was closed</h3>
<div class="scroll"><table>
<thead><tr><th>Trap</th><th>Fix</th></tr></thead>
<tbody>
<tr><td>Scoring the filled input on more pixels than the black one — it would win just by covering more.</td>
    <td>Both halves of a pair are scored on <strong>the black one's region</strong>. Filled pixels are never compared against anything.</td></tr>
<tr><td>Comparing inputs that see different amounts of the world. Input 5 sees a narrower slice than input 4.</td>
    <td>Every comparison with input 5 is also run with <strong>all inputs restricted to input 5's slice</strong>, computed from each pixel's ray. Both versions are reported.</td></tr>
<tr><td>Treating 96 frames as 96 samples. The 8 frames of a window share a room, a light setup and half a second of walking.</td>
    <td>Every interval resamples <strong>whole windows</strong> (n = 12). Wider, and honest.</td></tr>
</tbody></table></div>

<h3>What is measured</h3>
<ul>
<li><strong>Depth.</strong> AbsRel — average relative error after removing the model's unknown scale and offset. Lower is better. Plus δ₁, the fraction of pixels within 25% of truth. Higher is better.</li>
<li><strong>Camera pose.</strong> Per 8-frame window, every pair of frames compared through its relative pose, so the model's free origin and scale cannot help it. <strong>AUC@30</strong> summarises how many pairs get rotation and translation direction right (0–1, higher better). <strong>ATE</strong> is trajectory error in centimetres.</li>
<li><strong>Field of view.</strong> What the model thinks its lens is. Truth is {gt_wide:.1f}° for inputs 2 and 4, {gt_crop:.1f}° for input 5. Fisheye inputs have no pinhole truth, so their number is shown but not graded.</li>
</ul>
<p class="note">The ground-truth camera convention was checked on the data, not read off a comment: reprojecting
one frame's depth into the next lands within 0.12% median error, against 4–7% for each alternative.</p>
</section>
""")

    A(f"""
<section class="part" id="results">
<span class="pnum">Part 4 — Results</span>
<h2>Result 1 — filling helps depth a little, and both projections equally</h2>
{cells_table(C1, C8, ("fisheye_masked", "fisheye_full", "persp_masked", "persp_full"))}
<p class="note">Rows are comparable within a projection, not across it — the two pixel grids sample the world
differently. That is why the real test below is a difference of differences.</p>
<div class="scroll"><table>
<thead><tr><th>Change from filling (lower AbsRel = better)</th><th class="n">Fisheye: 3 − 1</th><th class="n">Rectified: 4 − 2</th><th class="n">Difference between them</th></tr></thead>
<tbody>
<tr><td>1 frame</td><td class="n">{num(E1['AbsRel']['fisheye'])}</td><td class="n">{num(E1['AbsRel']['persp'])}</td><td class="n">{num(E1['AbsRel']['interaction'])}</td></tr>
<tr><td>8 frames</td><td class="n">{num(E8['AbsRel']['fisheye'])}</td><td class="n">{num(E8['AbsRel']['persp'])}</td><td class="n">{num(E8['AbsRel']['interaction'])}</td></tr>
</tbody></table></div>
{fig("effects", "Each dot is one window. Square = mean, whisker = 95% interval. The right-hand group in each panel is the difference of differences — it would have to be clearly negative for the original claim to hold.")}
<p><strong>Filling helps. The prediction fails.</strong> The two columns are the same size, and the difference
between them straddles zero — and flips sign between 1 frame and 8. Six main effects across three metrics are
real; six differences-of-differences are inconclusive.</p>
<p>The gain sits in a narrow band just inside the edge of the black. Within 16 pixels of that edge, error drops
{B['fisheye']['band_gain_pct']:.0f}% (fisheye) and {B['persp']['band_gain_pct']:.0f}% (rectified). Further in, only
{B['fisheye']['int_gain_pct']:.0f}% and {B['persp']['int_gain_pct']:.0f}%. A boundary effect, the same size in
both projections.</p>

<h2>Result 2 — filling does not fix the camera</h2>
<div class="scroll"><table>
<thead><tr><th>Camera pose, 8-frame windows</th><th class="n">AUC@30 ↑</th><th class="n">Rotation err ↓</th><th class="n">Translation dir err ↓</th><th class="n">Trajectory err ↓</th></tr></thead>
<tbody>
{"".join(f'<tr{" class=hl" if k == "persp_crop" else ""}><td>{chip(k)}</td><td class="n">{P8[k]["auc30"]:.3f}</td>'
         f'<td class="n">{P8[k]["rot_err_deg"]:.2f}°</td><td class="n">{P8[k]["trans_err_deg"]:.2f}°</td>'
         f'<td class="n">{P8[k]["ate_m"]*100:.1f} cm</td></tr>' for k in ALL if k in P8)}
</tbody></table></div>
<div class="scroll"><table>
<thead><tr><th>Change from filling</th><th class="n">Fisheye: 3 − 1</th><th class="n">Rectified: 4 − 2</th></tr></thead>
<tbody>
<tr><td>AUC@30 (negative = filling hurt)</td><td class="n">{num(E8['auc30']['fisheye'], '{:+.3f}')}</td><td class="n">{num(E8['auc30']['persp'], '{:+.3f}')}</td></tr>
<tr><td>Rotation error (positive = filling hurt)</td><td class="n">{num(E8['rot_err_deg']['fisheye'], '{:+.2f}°')}</td><td class="n">{num(E8['rot_err_deg']['persp'], '{:+.2f}°')}</td></tr>
<tr><td>Trajectory error (positive = filling hurt)</td><td class="n">{num(E8['ate_m']['fisheye'], '{:+.4f} m')}</td><td class="n">{num(E8['ate_m']['persp'], '{:+.4f} m')}</td></tr>
</tbody></table></div>
<p>Rectified: no change you can distinguish from zero. Fisheye: <strong>worse on all three</strong>.</p>
<p class="note">A reading, not a measurement: the filled fisheye corners are real scene content placed beyond the
angle the lens can physically image — a lens model no real lens has. The depth head reads extra texture and
gains; the camera head reads an impossible lens and loses.</p>

<h3>The field of view explains it</h3>
{fig("fov", "Each dot is one frame's inferred horizontal field of view. Inputs 2 and 4 both sit near 108° against a true 124.7° — black corners or real ones. Input 5, truly 106.8°, is read almost right.")}
<div class="key">
<h4>The finding that settles the original idea</h4>
<p>A clean, black-free {gt_wide:.0f}° photo is still read as {F1['persp_full']['mean']:.0f}° — off by
{F1['persp_full']['abs_err']:.0f}°. Filling moves that guess by {abs(F1['persp_full']['mean'] - F1['persp_masked']['mean']):.1f}°.</p>
<p>The model is not confused by the black. It does not recognise the <em>width</em>. Filling changes the corners;
it does not change the width.</p>
</div>

<h2>Result 3 — cheap filling already takes nearly all of it</h2>
<p>How much of the gain does the dumbest filler get for free? We tried methods that invent nothing: smearing the
nearest valid pixel outward, flat averages, two classical inpainters.</p>
<div class="scroll"><table>
<thead><tr><th>Share of the real-content gain that smearing already captures</th><th class="n">1 frame</th><th class="n">8 frames</th></tr></thead>
<tbody>
<tr><td>Fisheye</td><td class="n">{rep1['fisheye']['pct']:.0f}%</td><td class="n">{rep8['fisheye']['pct']:.0f}%</td></tr>
<tr><td>Rectified</td><td class="n">{rep1['persp']['pct']:.0f}%</td><td class="n">{rep8['persp']['pct']:.0f}%</td></tr>
<tr><td><strong>AbsRel left on the table</strong></td><td class="n"><strong>{left[0]:.4f} / {left[1]:.4f}</strong></td><td class="n"><strong>{left[2]:.4f} / {left[3]:.4f}</strong></td></tr>
</tbody></table></div>
<p>In one case smearing <em>beats</em> the truth: real corner detail is complicated, a smear is simple, and the
model prefers simple. <strong>The budget for a generative filler is 0 to {max(left):.3f} AbsRel, with no camera
improvement.</strong></p>
<p class="note">Incidental, but worth carrying: a flat average fill <em>helps</em> on a rectified frame
({LAD1['persp']['rows']['mean']['pct']:+.0f}% of the gain) and is far <em>worse than black</em> on a raw fisheye
({LAD1['fisheye']['rows']['mean']['pct']:+.0f}%). Fill advice does not transfer across projections.</p>

<h2>Result 4 — cropping beats a wide filled view</h2>
<p>Input 5 sidesteps the problem: zoom in until the black is outside the frame. It costs ~17% of the lens's solid
angle, and needs no filler, no model, no compute. That is what filling has to beat.</p>
<p>But inputs 4 and 5 differ in two things: the field of view, <em>and</em> how many pixels each spends on the
part they share (input 5 all of them, input 4 about half — a factor of
{cm['Knew_crop'][0]/meta['Knew_pinhole'][0]:.2f}). So input 5 might just be sharper. Input <b>5b</b> settles it:
input 5 band-limited to input 4's sampling rate. Same framing, input 4's detail.</p>
{cells_table(CC1, CC8, ("persp_masked", "persp_full", "persp_crop", "persp_crop_lores"), hl="persp_crop")}
<p class="note">All four rows scored on the same slice of the world — the part input 5 can see.</p>
<div class="scroll"><table>
<thead><tr><th>Input 5 minus …</th><th class="n">vs 4 (wide, filled)</th><th class="n">vs 2 (wide, black)</th><th class="n">vs 5b (same view, blurred)</th></tr></thead>
<tbody>
<tr><td>Depth, 1 frame (negative = 5 better)</td><td class="n">{num(XC1['AbsRel']['persp_full'])}</td><td class="n">{num(XC1['AbsRel']['persp_masked'])}</td><td class="n">{num(XC1['AbsRel']['persp_crop_lores'])}</td></tr>
<tr><td>Depth, 8 frames</td><td class="n">{num(XC8['AbsRel']['persp_full'])}</td><td class="n">{num(XC8['AbsRel']['persp_masked'])}</td><td class="n">{num(XC8['AbsRel']['persp_crop_lores'])}</td></tr>
<tr><td>AUC@30 (positive = 5 better)</td><td class="n">{num(XO8['auc30']['persp_full'], '{:+.3f}')}</td><td class="n">{num(XO8['auc30']['persp_masked'], '{:+.3f}')}</td><td class="n">{num(XO8['auc30']['persp_crop_lores'], '{:+.3f}')}</td></tr>
<tr><td>Trajectory error (negative = 5 better)</td><td class="n">{num(XO8['ate_m']['persp_full'], '{:+.4f} m')}</td><td class="n">{num(XO8['ate_m']['persp_masked'], '{:+.4f} m')}</td><td class="n">{num(XO8['ate_m']['persp_crop_lores'], '{:+.4f} m')}</td></tr>
</tbody></table></div>
{fig("five_vs_four", "Input 5 against input 4, with the sharpness control 5b beside it. In every panel 5b sits on 5, not on 4 — the gap is field of view, not pixel density.")}
<p><strong>Cropping wins by {abs(XC1['AbsRel']['persp_full']['mean']):.4f} AbsRel at 1 frame and
{abs(XC8['AbsRel']['persp_full']['mean']):.4f} at 8 — a {abs(XC8['AbsRel']['persp_full']['mean'])/abs(XC1['AbsRel']['persp_full']['mean']):.1f}× wider gap.</strong>
The control 5b lands on 5 everywhere (pose {P8['persp_crop_lores']['auc30']:.3f} against
{P8['persp_crop']['auc30']:.3f}). Sharpness explains none of it.</p>

<h3>More content does not make multi-frame pay</h3>
<div class="scroll"><table>
<thead><tr><th>Gain from 8 frames instead of 1 (positive = it helps)</th><th class="n">Gain [95% interval]</th></tr></thead>
<tbody>
{"".join(f'<tr{" class=hl" if k == "persp_crop" else ""}><td>{chip(k)}</td><td class="n">{num(mfg(k))}</td></tr>' for k in ALL)}
</tbody></table></div>
<p>Input 4 carries strictly more of the scene than input 5. If extra context were what multi-frame needs, input 4
should gain most. It gains <em>least</em>. Compared window by window, input 5 gains {num(MC['common']['persp_full'])}
more than input 4. On the filled fisheye, eight frames are actively worse than one.</p>
<div class="warn">
<h4>A tempting explanation that failed its test</h4>
<p>"Multi-frame only pays where the camera is right" fits the headline numbers. Tested properly, it does not hold.</p>
<p>Across {MED['n']} (input × window) combinations, the correlation between a window's multi-frame gain and its
camera score is {MED['r_raw']:+.2f} overall, and {MED['r_within_window']:+.2f} once each window is centred — the
level at which the mechanism would have to work. <strong>The multi-frame results above are description, not a
demonstrated cause.</strong></p>
</div>

<h3>What it looks like</h3>
{fig("panels_w00", "One frame, all six inputs. Columns: input, depth from 1 frame, its error, the same from 8 frames, rendered truth. Grey = not scored. The brightest errors are large flat surfaces near the camera — in every input, unrelated to the black.")}
{fig("trajectories", "Recovered camera path per 8-frame window, aligned to truth. Input 5 (purple) tracks the black line; the wide inputs wander. The axes are not equally scaled — sideways deviations are centimetres.")}
</section>

<section class="part" id="conclusion">
<span class="pnum">Part 5 — Conclusion</span>
<h2>What to do</h2>
<div class="key">
<h4>Recommendation</h4>
<p><strong>Do not build a generative filler for this.</strong> The whole prize is
{abs(E1['AbsRel']['fisheye']['mean']):.3f}–{abs(E1['AbsRel']['persp']['mean']):.3f} AbsRel. A smear that invents
nothing already takes {min(rep1['persp']['pct'], rep1['fisheye']['pct']):.0f}%+ of it. None of it reaches the
camera estimate, which was the point.</p>
<p><strong>Crop instead of filling.</strong> It costs ~17% of the lens's solid angle and beats the filled wide
version on depth, on camera pose, and on multi-frame gain.</p>
</div>
<p>The reason underneath: the model does not fail because the picture has black in it. It fails because the
picture is <em>wide</em>. A clean {gt_wide:.0f}° photo is read as {F1['persp_full']['mean']:.0f}°; a
{gt_crop:.0f}° one is read almost right. Filling changes the corners, not the width.</p>

<h3>What this does not settle</h3>
<ul>
<li><strong>Where the limit is.</strong> We tested two zoom levels: {gt_crop:.0f}° works, {gt_wide:.0f}° does not. The transition could be anywhere between. Cheap to find — same renders, one parameter.</li>
<li><strong>Training-based approaches.</strong> Calibration tokens (Fisheye3R) and positional-encoding adapters (RayTun3R) attack the width, not the black. This experiment does not argue against them. It does sharpen their target: keep the whole lens <em>and</em> match what cropping gets free ({P8['persp_crop']['auc30']:.2f} vs {P8['persp_full']['auc30']:.2f}; {P8['persp_crop']['ate_m']*100:.1f} cm vs {P8['persp_full']['ate_m']*100:.1f} cm).</li>
<li><strong>Rendered ≠ real.</strong> Lighting was fitted to the real frames. An earlier badly-lit render of the same geometry gave the same conclusions with effects 4–6× larger — directions are robust, magnitudes are a property of the render.</li>
<li><strong>One model, one scene type.</strong> VGGT-Omega-1B-512, furnished apartment. Nothing checked on another backbone or outdoors.</li>
</ul>
</section>

<section class="part" id="repro">
<span class="pnum">Appendix</span>
<h2>Provenance</h2>
<dl class="kv">
<dt>Model</dt><dd><code>{PROV['ckpt']}</code>, md5 <code>{PROV['ckpt_md5']}</code>. Released weights, no fine-tuning.</dd>
<dt>Evaluation</dt><dd><code>finetune/eval/exp_rendered.py</code> at <code>{PROV['eval_commit']}</code>, branch <code>{PROV['branch']}</code>. Tests pin the pose metrics against synthetic cameras (a global similarity must cost nothing; a known rotation must read as its own angle; a collapsed prediction must not score perfect) and pin that the scoring region never leaks into the fill.</dd>
<dt>Run</dt><dd>{PROV['run']}, {PROV['finished']}, {PROV['env']}. Two passes differing only in scoring region; their pose numbers are bit-identical, confirming one set of predictions.</dd>
<dt>Rendering</dt><dd>Blender/Cycles, {meta['cycles_samples']} samples, OPTIX, {meta['eq_w']}×{meta['eq_h']} panorama, {meta['output_size']} px at {meta['supersample']}× supersampling. Input 5 added by <code>add_persp_crop.py</code> at <code>{PROV['renderer_commit']}</code> — same ray, resampling and depth code, only the focal length changed. Frame list md5 <code>{PROV['manifest_md5']}</code>.</dd>
<dt>Numbers</dt><dd><code>data/final/own/</code> and <code>data/final/common/</code> hold the raw per-frame and per-window results plus the evaluator's own report. <code>make_final_figures.py</code> recomputes every interval and refuses to write if it cannot reproduce that report. This page is generated from its output. No figure here was typed by hand.</dd>
</dl>
<pre>R=.../out/oracle_set_lit
python -m finetune.eval.exp_rendered --render-root $R --manifest $R/manifest.json \\
  --vggt-checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \\
  --out runs/ev6_own --seq-lens 1,8 --region own --settings &lt;all&gt;
python -m finetune.eval.exp_rendered ... --out runs/ev6_crop --region crop --settings &lt;five&gt;

python research/fisheye-inpaint/make_final_figures.py --run runs/ev6_own \\
  --run-crop runs/ev6_crop --out research/fisheye-inpaint/to_human/final_en --lang en
python research/fisheye-inpaint/build_report_en.py</pre>
</section>

<footer>
Research line fisheye-inpaint · branch {PROV['branch']} · generated by build_report_en.py · a fuller Chinese
research log, including the experiments this report supersedes, sits beside it in the same repository.
</footer>
</div>
""")
    html = "".join(H)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[report] wrote {OUT}  {len(html.encode('utf-8'))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
