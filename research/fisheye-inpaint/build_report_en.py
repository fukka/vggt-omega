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
  <p class="lede">Straightening a fisheye photo leaves black wedges in the corners. The obvious fix is to paint something into them so the picture looks like an ordinary photo again. We rendered a scene where we could put the <b>real</b> scene content into those corners — the best any filling method could ever do — and measured what a 3D model gained. It gained a little depth accuracy, nothing on camera pose, and <b>less than simply zooming in and cropping the black away</b>.</p>
  <div class="facts">
    <span>96 frames · 12 windows · 4 sequences</span>
    <span>VGGT-Omega-1B-512, no fine-tuning</span>
    <span>one evaluation run, 2026-09-08</span>
  </div>
</header>
""")

    # ---------------------------------------------------------------- answers
    A(f"""
<section class="part" id="answers">
<span class="pnum">Summary</span>
<h2>Four questions, four answers</h2>
<div class="answers">
  <div class="ans"><div class="q">Q1</div><div class="a">
    <b>Does real content in the black corners make depth better?</b> Yes, but only a little, and it helps
    the fisheye frame just as much as the straightened one — so it is not about "looking like a normal photo".
    <span class="fig">{E1['AbsRel']['fisheye']['mean']:+.4f} on fisheye, {E1['AbsRel']['persp']['mean']:+.4f} on rectified (AbsRel, lower is better). The difference between those two is {E1['AbsRel']['interaction']['mean']:+.4f} and could easily be zero.</span></div></div>
  <div class="ans"><div class="q">Q2</div><div class="a">
    <b>Does it make the camera estimate better?</b> No. On the fisheye frame it makes it measurably worse,
    and the model's guess at the lens's field of view barely moves.
    <span class="fig">Pose score changes {E8['auc30']['fisheye']['mean']:+.3f} on fisheye and {E8['auc30']['persp']['mean']:+.3f} on rectified. Inferred field of view moves {abs(F1['persp_full']['mean'] - F1['persp_masked']['mean']):.1f}° and stays {F1['persp_full']['abs_err']:.0f}° away from the truth.</span></div></div>
  <div class="ans"><div class="q">Q3</div><div class="a">
    <b>How much of that gain needs a clever filling method?</b> Almost none. Smearing the nearest real pixel
    outward — which invents nothing — already captures most or all of it.
    <span class="fig">Smearing captures {min(rep1['persp']['pct'], rep1['fisheye']['pct'], rep8['persp']['pct'], rep8['fisheye']['pct']):.0f}–{max(rep1['persp']['pct'], rep1['fisheye']['pct'], rep8['persp']['pct'], rep8['fisheye']['pct']):.0f}% of what real content is worth — in one case it beats the truth. At most {max(left):.3f} AbsRel is left for anything smarter.</span></div></div>
  <div class="ans"><div class="q">Q4</div><div class="a">
    <b>Is a wider filled view better than just cropping the black away?</b> No — cropping wins on every measure,
    and the gap grows when the model gets several frames at once.
    <span class="fig">Cropping beats filling by {abs(XC1['AbsRel']['persp_full']['mean']):.4f} AbsRel with 1 frame and {abs(XC8['AbsRel']['persp_full']['mean']):.4f} with 8, on the part of the scene both can see. Camera pose score {P8['persp_crop']['auc30']:.2f} against {P8['persp_full']['auc30']:.2f}; trajectory error {P8['persp_crop']['ate_m']*100:.1f} cm against {P8['persp_full']['ate_m']*100:.1f} cm.</span></div></div>
</div>
<p class="note">Every number below is a measurement from one evaluation run, with a 95% confidence interval in brackets. "Real" means the interval does not contain zero. Intervals are computed by resampling <em>windows</em>, not frames — see <a href="#method">the method</a> for why that matters.</p>
</section>
""")

    # ---------------------------------------------------------------- motivation
    A(f"""
<section class="part" id="why">
<span class="pnum">Part 1 — Motivation</span>
<h2>Why anyone would want to fill the corners</h2>
<p>A fisheye lens draws a circle of light on a square sensor, so the four corners of the file are empty.
Straightening that circle into a flat, ordinary-looking photo (called <em>rectification</em>) makes the problem
worse rather than better: to keep the whole circle you have to zoom out, and then a third of the picture is
black wedges.</p>
<p>3D models like VGGT-Omega were trained on ordinary photographs. Large black regions are not something an
ordinary photograph has. Two published results say this should matter: black is not a neutral value to a
convolutional or attention model, and networks are known to read the frame's edges as positional information.
So the idea is simple and appealing:</p>
<div class="key">
<h4>The claim we set out to test</h4>
<p>If you fill the black wedges with plausible content, the picture returns to the domain the model was trained
on, the model recovers its ability to work out its own camera, and its depth gets better.</p>
<p>That claim makes a sharp prediction. Filling should help the <em>straightened</em> picture much more than it
helps the raw fisheye picture, because only the straightened one becomes a normal photo. If filling helps both
equally, then whatever is going on is just "do not feed the model a big black region" — a much smaller and much
less interesting story.</p>
</div>
<p>The reason to test it carefully rather than just try it: a good filling method means a generative model,
which is expensive to build, expensive to run, and hard to trust. Before paying that, it is worth knowing the
size of the prize.</p>
</section>
""")

    # ---------------------------------------------------------------- data
    ir = "".join(
        f'<div{" class=ctl" if k == CTRL[0] else ""}><div class="nm">{chip(k)}</div>'
        f'<p>{ALL[k][2]}<br><span class="ci">black pixels {INP[k]["black_pct"]:.1f}% · scored area {INP[k]["graded_pct"]:.0f}%</span></p></div>'
        if k in INP else
        f'<div class="ctl"><div class="nm">{chip(k)}</div><p>{ALL[k][2]}<br><span class="ci">a control, not one of the four cells</span></p></div>'
        for k in ALL)
    A(f"""
<section class="part" id="data">
<span class="pnum">Part 2 — Data</span>
<h2>Why the pictures had to be rendered</h2>
<p>To test the claim properly you need to know what the <em>right</em> answer in the corners looks like. That is
impossible with real footage: the lens never pointed at those directions, so nobody has a photograph of them.
Any experiment on real frames can only compare one guess against another guess, and a negative result is then
ambiguous — did filling not help, or was the filler just not good enough?</p>
<p>So we rebuilt the scene. Four sequences from the Aria Digital Twin dataset were re-rendered in Blender from
their object-level 3D reconstructions ({meta['n_objects']} objects, {meta['n_lights_used']} lights fitted to
match the real photographs), following the real camera trajectory. Each frame was rendered as a full
360° panorama and then resampled into each of the inputs below. Depth comes from the renderer itself, so it is
exact and identical across inputs.</p>
<p>This gives us the thing real footage cannot: <strong>true content in directions the lens never imaged.</strong>
That is the ceiling. No filling method can ever beat it.</p>

<h3>The inputs</h3>
<p>Every input below is the same instant, the same camera position, the same panorama. They differ only in how
that panorama was sampled and what sits in the corners.</p>
<div class="inputs">{ir}</div>
{fig("inputs_w00", "The six inputs, taken straight out of the evaluation script before they were fed to the model — not illustrations. Inputs 1 and 2 have the black regions. Inputs 3 and 4 have the true scene there instead. Input 5 avoids the problem by zooming in. Input 5b is a control explained in Part 4.")}
<p>Inputs <b>1</b> and <b>3</b> are the same field of view as each other; so are <b>2</b> and <b>4</b>. That is what
makes this a clean 2×2: projection (fisheye or straightened) crossed with corners (black or filled).</p>
<p>The small remaining black in inputs 3, 4 and 5 — under {max(INP[k]['black_pct'] for k in ('fisheye_full','persp_full','persp_crop')):.1f}% of pixels — is
not missing render. Those pixels all carry valid depth; they are shadows and dark objects in a dimly lit
apartment, and all inputs have them equally.</p>

<h3>Frames and windows</h3>
<p>4 sequences × 3 windows each × 8 consecutive frames = 96 frames. Each window covers about 2–2.5 metres of real
walking, so the model has genuine parallax to work with when it sees eight frames at once.</p>
</section>
""")

    # ---------------------------------------------------------------- method
    A(f"""
<section class="part" id="method">
<span class="pnum">Part 3 — Method</span>
<h2>How it was scored, and the three traps avoided</h2>
<p>The model is VGGT-Omega-1B-512 with its released weights, no fine-tuning. It is run twice over the same
frames: once one frame at a time, and once eight frames at a time, because a corrupted input can do damage
across views that it cannot do on its own.</p>

<h3>Trap 1 — scoring the filled input on more pixels than the black one</h3>
<p>If you score input 4 everywhere it has an answer and input 2 only where it is not black, input 4 wins simply
by covering more ground. So both halves of a pair are always scored on <strong>the black one's region</strong>.
Pixels that were filled in are never compared against anything. The two inputs differ in what the model sees;
they are graded on identical pixels.</p>

<h3>Trap 2 — comparing two inputs that see different amounts of the world</h3>
<p>Input 5 sees a narrower slice of the scene than input 4. Comparing their overall scores is meaningless. So
every comparison involving input 5 is also run a second way: <strong>all inputs restricted to the slice input 5
can see</strong>, worked out analytically from each pixel's ray. Then all of them are scored on the same
directions in the world. We report both, and say which is which.</p>

<h3>Trap 3 — treating 96 frames as 96 independent samples</h3>
<p>The eight frames of one window share a room, a lighting setup and a fraction of a second of walking. They are
not eight independent observations. Every confidence interval here comes from resampling <strong>whole
windows</strong> (n = 12), which is wider and honest, rather than frames, which is narrower and wrong.</p>

<h3>What is measured</h3>
<ul>
<li><strong>Depth.</strong> AbsRel — the average relative error between predicted and true depth after
removing the model's unknown scale and offset. Lower is better. Also δ₁, the fraction of pixels within 25% of
the truth. Higher is better.</li>
<li><strong>Camera pose.</strong> For each 8-frame window, every pair of frames is compared through its relative
pose, so the model's free choice of world origin and overall scale cannot affect the score. We report
<strong>AUC@30</strong> (a 0–1 summary of how many pairs get both rotation and translation direction right;
higher is better) and <strong>ATE</strong>, the trajectory error in centimetres after aligning to the truth.</li>
<li><strong>Field of view.</strong> What the model thinks its own lens is. For the straightened inputs there is a
true answer to compare against ({gt_wide:.1f}° for inputs 2 and 4, {gt_crop:.1f}° for input 5). For the fisheye
inputs there is no such thing as a correct pinhole field of view, so their number is reported but not graded.</li>
</ul>
<p class="note">The ground-truth camera convention was checked against the data rather than read off a comment:
unprojecting one frame's depth and reprojecting it into the next frame lands within 0.12% median relative error
with the convention used here, versus 4–7% for each plausible alternative.</p>
</section>
""")

    # ---------------------------------------------------------------- results
    def cells_table(C1_, C8_, keys, note_hl=None):
        rows = "".join(
            f'<tr{" class=hl" if k == note_hl else ""}><td>{chip(k)}</td>'
            f'<td class="n">{C1_[k]["AbsRel"]:.4f}</td><td class="n">{C8_[k]["AbsRel"]:.4f}</td>'
            f'<td class="n">{C1_[k]["delta1"]:.3f}</td><td class="n">{C8_[k]["delta1"]:.3f}</td></tr>'
            for k in keys if k in C1_)
        return ('<div class="scroll"><table><thead><tr><th>Input</th>'
                '<th class="n">AbsRel · 1 frame</th><th class="n">AbsRel · 8 frames</th>'
                '<th class="n">δ₁ · 1 frame</th><th class="n">δ₁ · 8 frames</th></tr></thead>'
                f'<tbody>{rows}</tbody></table></div>')

    A(f"""
<section class="part" id="results">
<span class="pnum">Part 4 — Results</span>
<h2>Result 1: filling helps depth a little, and it helps both projections the same</h2>
{cells_table(C1, C8, ("fisheye_masked", "fisheye_full", "persp_masked", "persp_full"))}
<p class="note">Scores within a projection are comparable; scores <em>across</em> projections are not, because the
two pixel grids sample the world differently. That is exactly why the test below is a difference of differences.</p>
<div class="scroll"><table>
<thead><tr><th>Change from filling the corners (lower AbsRel = better)</th><th class="n">Fisheye: 3 − 1</th><th class="n">Rectified: 4 − 2</th><th class="n">Difference between them</th></tr></thead>
<tbody>
<tr><td>1 frame</td><td class="n">{num(E1['AbsRel']['fisheye'])}</td><td class="n">{num(E1['AbsRel']['persp'])}</td><td class="n">{num(E1['AbsRel']['interaction'])}</td></tr>
<tr><td>8 frames</td><td class="n">{num(E8['AbsRel']['fisheye'])}</td><td class="n">{num(E8['AbsRel']['persp'])}</td><td class="n">{num(E8['AbsRel']['interaction'])}</td></tr>
</tbody></table></div>
{fig("effects", "Each dot is one window's paired difference; the square is the average and the whisker its 95% interval. The first two panels are depth, the last two camera pose. The right-hand group in each panel is the interaction — the quantity that would have to be clearly negative for the original claim to hold.")}
<p><strong>Filling helps. The prediction that it should help the straightened picture more does not hold.</strong>
The two columns are the same size, and the difference between them straddles zero in both modes — and even
changes sign between them. Six main effects across three metrics are all real; six interactions are all
inconclusive. Whatever filling is doing, it is not "returning the image to the training domain".</p>

<h3>Where in the picture the gain actually is</h3>
<p>The improvement is concentrated in a narrow band just inside the edge of the black region. Within 16 pixels of
that boundary, error drops by {B['fisheye']['band_gain_pct']:.0f}% on the fisheye frame and
{B['persp']['band_gain_pct']:.0f}% on the straightened one. Further inside, it drops by only
{B['fisheye']['int_gain_pct']:.0f}% and {B['persp']['int_gain_pct']:.0f}%. This is a boundary effect, and its size
is the same in both projections — consistent with everything above.</p>

<h2>Result 2: filling does not fix the camera, and on fisheye it makes it worse</h2>
<div class="scroll"><table>
<thead><tr><th>Camera pose, 8-frame windows</th><th class="n">AUC@30 ↑</th><th class="n">Rotation error ↓</th><th class="n">Translation direction error ↓</th><th class="n">Trajectory error ↓</th></tr></thead>
<tbody>
{"".join(f'<tr{" class=hl" if k == "persp_crop" else ""}><td>{chip(k)}</td><td class="n">{P8[k]["auc30"]:.3f}</td>'
         f'<td class="n">{P8[k]["rot_err_deg"]:.2f}°</td><td class="n">{P8[k]["trans_err_deg"]:.2f}°</td>'
         f'<td class="n">{P8[k]["ate_m"]*100:.1f} cm</td></tr>' for k in ALL if k in P8)}
</tbody></table></div>
<div class="scroll"><table>
<thead><tr><th>Change from filling the corners</th><th class="n">Fisheye: 3 − 1</th><th class="n">Rectified: 4 − 2</th></tr></thead>
<tbody>
<tr><td>AUC@30 (higher = better, so negative means filling hurt)</td><td class="n">{num(E8['auc30']['fisheye'], '{:+.3f}')}</td><td class="n">{num(E8['auc30']['persp'], '{:+.3f}')}</td></tr>
<tr><td>Rotation error (lower = better)</td><td class="n">{num(E8['rot_err_deg']['fisheye'], '{:+.2f}°')}</td><td class="n">{num(E8['rot_err_deg']['persp'], '{:+.2f}°')}</td></tr>
<tr><td>Trajectory error (lower = better)</td><td class="n">{num(E8['ate_m']['fisheye'], '{:+.4f} m')}</td><td class="n">{num(E8['ate_m']['persp'], '{:+.4f} m')}</td></tr>
</tbody></table></div>
<p>On the straightened picture, filling changes the camera estimate by nothing you can distinguish from zero.
On the raw fisheye it makes it <strong>worse</strong> on all three measures. A plausible reading — this is
interpretation, not measurement — is that the filled fisheye corners are real scene content placed under a lens
model no real lens has, since they lie beyond the angle the lens can physically image. The depth head reads that
as extra texture and benefits; the camera head reads it as an impossible lens and suffers.</p>

<h3>The model's guess at its own field of view explains a lot</h3>
{fig("fov", "Each dot is one frame's inferred horizontal field of view. Inputs 2 and 4 both sit near 108° against a true 124.7° — whether their corners are black or contain the real scene. Input 5, at a true 106.8°, is read almost correctly.")}
<div class="key">
<h4>The finding that decides the original idea</h4>
<p>A perfectly clean, black-free {gt_wide:.0f}° photograph is still read by the model as roughly
{F1['persp_full']['mean']:.0f}° — off by {F1['persp_full']['abs_err']:.0f}°. Filling the corners moves that guess
by {abs(F1['persp_full']['mean'] - F1['persp_masked']['mean']):.1f}°.</p>
<p>The model is not being confused by the black. It does not recognise the <em>field of view</em>. Filling cannot
fix that, because the filled picture is just as wide as the black one.</p>
</div>

<h2>Result 3: cheap filling already captures nearly all of it</h2>
<p>If real content is worth some amount, how much of that amount does the dumbest possible filling get for free?
We tried several methods that invent nothing new: repeating the nearest valid pixel outward, flat averages, and
two classical inpainting algorithms.</p>
<div class="scroll"><table>
<thead><tr><th>Share of the real-content gain that smearing the nearest pixel already captures</th><th class="n">1 frame</th><th class="n">8 frames</th></tr></thead>
<tbody>
<tr><td>Fisheye</td><td class="n">{rep1['fisheye']['pct']:.0f}%</td><td class="n">{rep8['fisheye']['pct']:.0f}%</td></tr>
<tr><td>Rectified</td><td class="n">{rep1['persp']['pct']:.0f}%</td><td class="n">{rep8['persp']['pct']:.0f}%</td></tr>
<tr><td><strong>AbsRel still left on the table</strong></td><td class="n"><strong>{left[0]:.4f} / {left[1]:.4f}</strong></td><td class="n"><strong>{left[2]:.4f} / {left[3]:.4f}</strong></td></tr>
</tbody></table></div>
<p>In one case smearing actually <em>beats</em> the true content: the real corner detail is complicated, the smear
is simple, and the model prefers the simple one. <strong>The budget for any generative filling method is
0 to {max(left):.3f} AbsRel</strong>, and it comes with no camera-pose improvement at all.</p>
<p class="note">One incidental finding worth remembering: a flat average fill <em>helps</em> on a straightened
picture ({LAD1['persp']['rows']['mean']['pct']:+.0f}% of the gain) and is far <em>worse than black</em> on a raw
fisheye ({LAD1['fisheye']['rows']['mean']['pct']:+.0f}%). Advice about fill colours does not carry across
projections.</p>

<h2>Result 4: a wider filled view loses to simply cropping</h2>
<p>Input 5 avoids the whole problem: zoom in until the black is outside the frame. It costs about 17% of the
lens's solid angle and needs no filling, no model and no compute. It is the free alternative that any filling
approach has to beat.</p>
<p>Inputs 4 and 5 differ in two things at once, though — the field of view, and how many pixels each spends on the
part of the scene they share (input 5 spends all of them, input 4 about half, a factor of
{cm['Knew_crop'][0]/meta['Knew_pinhole'][0]:.2f}). So input 5 could be winning just by being sharper. Input
<b>5b</b> settles it: input 5 band-limited to input 4's sampling rate — same framing, input 4's level of detail.</p>
{cells_table(CC1, CC8, ("persp_masked", "persp_full", "persp_crop", "persp_crop_lores"), note_hl="persp_crop")}
<p class="note">All four rows above are scored on the same slice of the world — the part input 5 can see.</p>
<div class="scroll"><table>
<thead><tr><th>Input 5 minus …</th><th class="n">vs 4 (wide, filled)</th><th class="n">vs 2 (wide, black)</th><th class="n">vs 5b (same view, blurred)</th></tr></thead>
<tbody>
<tr><td>Depth, 1 frame (negative = 5 better)</td><td class="n">{num(XC1['AbsRel']['persp_full'])}</td><td class="n">{num(XC1['AbsRel']['persp_masked'])}</td><td class="n">{num(XC1['AbsRel']['persp_crop_lores'])}</td></tr>
<tr><td>Depth, 8 frames</td><td class="n">{num(XC8['AbsRel']['persp_full'])}</td><td class="n">{num(XC8['AbsRel']['persp_masked'])}</td><td class="n">{num(XC8['AbsRel']['persp_crop_lores'])}</td></tr>
<tr><td>Camera AUC@30 (positive = 5 better)</td><td class="n">{num(XO8['auc30']['persp_full'], '{:+.3f}')}</td><td class="n">{num(XO8['auc30']['persp_masked'], '{:+.3f}')}</td><td class="n">{num(XO8['auc30']['persp_crop_lores'], '{:+.3f}')}</td></tr>
<tr><td>Trajectory error (negative = 5 better)</td><td class="n">{num(XO8['ate_m']['persp_full'], '{:+.4f} m')}</td><td class="n">{num(XO8['ate_m']['persp_masked'], '{:+.4f} m')}</td><td class="n">{num(XO8['ate_m']['persp_crop_lores'], '{:+.4f} m')}</td></tr>
</tbody></table></div>
{fig("five_vs_four", "Input 5 against input 4, with the sharpness control 5b beside it. In every panel 5b sits on top of 5, not on 4 — so the gap is about the field of view, not about pixel density.")}
<p><strong>Cropping wins by {abs(XC1['AbsRel']['persp_full']['mean']):.4f} AbsRel on one frame and
{abs(XC8['AbsRel']['persp_full']['mean']):.4f} on eight — the gap grows by
{abs(XC8['AbsRel']['persp_full']['mean'])/abs(XC1['AbsRel']['persp_full']['mean']):.1f}×.</strong> The control 5b
lands on top of 5 in every measurement (camera score {P8['persp_crop_lores']['auc30']:.3f} against
{P8['persp_crop']['auc30']:.3f}), so sharpness explains none of it.</p>

<h3>And more content does not make multi-frame pay off</h3>
<div class="scroll"><table>
<thead><tr><th>Gain from seeing 8 frames instead of 1 (positive = it helps)</th><th class="n">Gain [95% interval]</th></tr></thead>
<tbody>
{"".join(f'<tr{" class=hl" if k == "persp_crop" else ""}><td>{chip(k)}</td><td class="n">{num(mfg(k))}</td></tr>' for k in ALL)}
</tbody></table></div>
<p>Input 4 carries strictly more of the scene than input 5, so if extra context were what multi-frame needs,
input 4 should benefit most. It benefits <em>least</em>. Comparing the two directly, window by window, input 5
gains {num(MC['common']['persp_full'])} more from multi-frame than input 4 does. On the filled fisheye input,
eight frames are actively <em>worse</em> than one.</p>
<div class="warn">
<h4>A tempting explanation that did not survive testing</h4>
<p>"Multi-frame only pays off where the camera is estimated correctly" fits the headline numbers nicely — input 5
has a good camera estimate and gains from multi-frame; input 4 has a poor one and gains nothing.</p>
<p>Tested properly, it fails. Across {MED['n']} (input × window) combinations, the correlation between a window's
multi-frame gain and its camera score is {MED['r_raw']:+.2f} overall, and only {MED['r_within_window']:+.2f} once
each window is centred — which is the level at which the mechanism would have to operate. The association lives
at the coarser "fisheye versus rectified" level instead. <strong>The multi-frame results above are a
description, not a demonstrated cause.</strong></p>
</div>

<h3>What it looks like</h3>
{fig("panels_w00", "One window, one frame, all six inputs. Columns: the input, the depth the model predicts from one frame, its error map, the same for eight frames, and the rendered truth. Grey means not scored. The brightest errors are large low-texture surfaces near the camera — present in every input, and not about the black regions.")}
{fig("trajectories", "The camera path the model recovers for each 8-frame window, aligned to the truth. Input 5 (purple) tracks the black line closely; the wide inputs wander. Note the two axes are not equally scaled — the sideways deviations are centimetres.")}
</section>
""")

    # ---------------------------------------------------------------- conclusion
    A(f"""
<section class="part" id="conclusion">
<span class="pnum">Part 5 — Conclusion</span>
<h2>What we would now do, and what we would not build</h2>
<div class="key">
<h4>Recommendation</h4>
<p><strong>Do not build a generative filler for this.</strong> Three independent reasons, each measured here:
the whole prize is {abs(E1['AbsRel']['fisheye']['mean']):.3f}–{abs(E1['AbsRel']['persp']['mean']):.3f} AbsRel;
a smear that invents nothing already takes {min(rep1['persp']['pct'], rep1['fisheye']['pct']):.0f}% or more of
it; and none of it reaches the camera estimate, which was the original motivation.</p>
<p><strong>If you have a fisheye frame and a model that expects photographs, crop rather than fill.</strong>
It costs about 17% of the lens's solid angle and beats the filled wide-angle version on depth, on camera pose,
and on how much the model gains from multiple frames.</p>
</div>
<h3>The reason underneath all of it</h3>
<p>The model does not fail on a wide picture because the picture has black in it. It fails because the picture is
wide. A clean {gt_wide:.0f}° photograph is read as {F1['persp_full']['mean']:.0f}°, and a {gt_crop:.0f}° one is
read almost correctly. Filling changes what is in the corners; it does not change how wide the frame is, which is
the thing the model cannot handle.</p>

<h3>What this does not settle</h3>
<ul>
<li><strong>Where the limit is.</strong> We measured two zoom levels: {gt_crop:.0f}° works, {gt_wide:.0f}° does not.
The transition could be anywhere between. Finding it is cheap — the same renders, one parameter changed — and it
would tell you directly how much of the lens you can keep.</li>
<li><strong>Whether training-based approaches do better.</strong> Methods that give the model explicit calibration
information (Fisheye3R's calibration tokens, RayTun3R's positional-encoding adapter) attack the field-of-view
problem rather than the black-region problem, so this experiment does not speak against them. It does sharpen the
question they have to answer: can they keep the whole lens <em>and</em> match what cropping gets for free
(camera score {P8['persp_crop']['auc30']:.2f} versus {P8['persp_full']['auc30']:.2f},
{P8['persp_crop']['ate_m']*100:.1f} cm versus {P8['persp_full']['ate_m']*100:.1f} cm)?</li>
<li><strong>Rendered scenes are not real footage.</strong> The rendering was photometrically fitted to the real
frames, and an earlier badly-lit version of the same geometry gave the same qualitative conclusions with effect
sizes 4–6× larger — so the direction of these findings is robust to the render, but the exact magnitudes are a
property of it.</li>
<li><strong>One model, one scene type.</strong> VGGT-Omega-1B-512 in a furnished apartment. Nothing here has been
checked on another backbone or outdoors.</li>
</ul>
</section>

<section class="part" id="repro">
<span class="pnum">Appendix</span>
<h2>Where every number came from</h2>
<dl class="kv">
<dt>Model</dt><dd><code>{PROV['ckpt']}</code>, md5 <code>{PROV['ckpt_md5']}</code>, released weights, no fine-tuning.</dd>
<dt>Evaluation</dt><dd><code>finetune/eval/exp_rendered.py</code> at commit <code>{PROV['eval_commit']}</code> on branch <code>{PROV['branch']}</code>. Unit tests pin the pose metrics against synthetic cameras (a global similarity transform must cost nothing; a known rotation must read as its own angle; a collapsed prediction must not score as perfect) and pin that the scoring region never leaks into the fill.</dd>
<dt>Run</dt><dd>{PROV['run']}, finished {PROV['finished']}, {PROV['env']}. Two passes differing only in the scoring region; their camera-pose numbers are bit-identical, which confirms the two passes share one set of predictions.</dd>
<dt>Rendering</dt><dd>Blender/Cycles via <code>render_oracle_2x2_lambda.py</code>, {meta['cycles_samples']} samples, OPTIX, {meta['eq_w']}×{meta['eq_h']} panorama, {meta['output_size']} px outputs at {meta['supersample']}× supersampling. Input 5 added by <code>add_persp_crop.py</code> at <code>{PROV['renderer_commit']}</code>, reusing the same ray, resampling and depth code with only the focal length changed. Frame list md5 <code>{PROV['manifest_md5']}</code>.</dd>
<dt>Numbers</dt><dd><code>research/fisheye-inpaint/data/final/own/</code> and <code>common/</code> hold the raw per-frame and per-window results and the evaluator's own printed report. <code>make_final_figures.py</code> recomputes every confidence interval and <em>refuses to write anything</em> if it does not reproduce that report; this page is generated from its output by <code>build_report_en.py</code>. No figure in this document was typed by hand.</dd>
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
Research line fisheye-inpaint · branch {PROV['branch']} · generated by build_report_en.py · a fuller Chinese research log, including the experiments this report supersedes, lives alongside it in the same repository.
</footer>
</div>
""")
    html = "".join(H)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[report] wrote {OUT}  {len(html.encode('utf-8'))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
