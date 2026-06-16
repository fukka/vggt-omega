# Copyright (c) 2026.
"""Per-frame depth export + a self-contained interactive depth viewer.

Consumed by ``load_and_visualize.py`` for three modes:

* ``export_depth_npy``  -> raw float depth ``depth/000000.npy`` per frame
* ``export_depth_vis``  -> colormapped ``depth_vis/000000.jpg`` per frame
* ``build_depth_viewer_html`` -> one self-contained ``depth_viewer.html`` with a
  frame scrubber, an input<->depth swipe-compare (+ side-by-side), a hover
  readout of the actual depth value, and the combined 3D point cloud
  (<model-viewer>) to orbit.

All inputs come from the saved ``predictions.npz`` (keys ``depth`` ``[S,H,W,1]``
and ``images`` ``[S,3,H,W]`` in ``[0,1]``) — no GPU, no checkpoint.

Optional deps are imported lazily: ``matplotlib`` (colormap) and ``Pillow`` (JPEG).
"""
from __future__ import annotations

import base64
import io
import json
import os

import numpy as np


# --------------------------------------------------------------------------- #
# array helpers
# --------------------------------------------------------------------------- #
def squeeze_depth(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth)
    if d.ndim == 4 and d.shape[-1] == 1:
        d = d[..., 0]
    return d.astype(np.float32)


def images_to_uint8(images: np.ndarray) -> np.ndarray:
    im = np.asarray(images)
    if im.ndim == 4 and im.shape[1] == 3:  # [S,3,H,W] -> [S,H,W,3]
        im = np.transpose(im, (0, 2, 3, 1))
    if im.dtype != np.uint8:
        im = (np.clip(im, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return im


def depth_range(depth: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    """Robust (percentile) min/max over finite, positive depth."""
    d = depth[np.isfinite(depth) & (depth > 0)]
    if d.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(np.percentile(d, lo)), float(np.percentile(d, hi))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _get_cmap(name: str):
    import matplotlib

    try:
        return matplotlib.colormaps[name]
    except Exception:  # older matplotlib
        from matplotlib import cm

        return cm.get_cmap(name)


def colorize_depth(depth2d: np.ndarray, vmin: float, vmax: float, cmap: str = "turbo") -> np.ndarray:
    """[H,W] depth -> [H,W,3] uint8. Invalid (<=0 / non-finite) pixels -> black."""
    cm = _get_cmap(cmap)
    d = depth2d.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    norm = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = (np.asarray(cm(norm))[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _resize_nearest(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    ys = np.linspace(0, arr.shape[0] - 1, out_h).round().astype(np.int64)
    xs = np.linspace(0, arr.shape[1] - 1, out_w).round().astype(np.int64)
    return arr[np.ix_(ys, xs)]


def _cmap_css(cmap: str, n: int = 12) -> str:
    cm = _get_cmap(cmap)
    stops = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b = (np.asarray(cm(t))[:3] * 255).astype(int)
        stops.append(f"rgb({r},{g},{b}) {t * 100:.0f}%")
    return "linear-gradient(to right, " + ", ".join(stops) + ")"


def _jpeg_data_uri(rgb_uint8: np.ndarray, quality: int = 85) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# mode 1 + 2: exporters
# --------------------------------------------------------------------------- #
def export_depth_npy(predictions: dict, out_dir: str) -> int:
    """Write raw float depth ``out_dir/000000.npy`` per frame. Returns count."""
    depth = squeeze_depth(predictions["depth"])
    os.makedirs(out_dir, exist_ok=True)
    for i in range(depth.shape[0]):
        np.save(os.path.join(out_dir, f"{i:06d}.npy"), depth[i])
    return depth.shape[0]


def export_depth_vis(
    predictions: dict,
    out_dir: str,
    cmap: str = "turbo",
    normalize: str = "global",
    quality: int = 92,
) -> int:
    """Write colormapped ``out_dir/000000.jpg`` per frame. Returns count."""
    from PIL import Image

    depth = squeeze_depth(predictions["depth"])
    os.makedirs(out_dir, exist_ok=True)
    g_vmin, g_vmax = depth_range(depth) if normalize == "global" else (None, None)
    for i in range(depth.shape[0]):
        vmin, vmax = (g_vmin, g_vmax) if normalize == "global" else depth_range(depth[i])
        rgb = colorize_depth(depth[i], vmin, vmax, cmap)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"{i:06d}.jpg"), quality=quality)
    return depth.shape[0]


# --------------------------------------------------------------------------- #
# mode 3: interactive viewer
# --------------------------------------------------------------------------- #
def build_depth_viewer_html(
    predictions: dict,
    out_path: str,
    glb_path: str | None = None,
    cmap: str = "turbo",
    normalize: str = "global",
    hover_max: int = 160,
    quality: int = 85,
    units: str = "",
    title: str = "VGGT-Ω depth viewer",
) -> str:
    depth = squeeze_depth(predictions["depth"])
    imgs = images_to_uint8(predictions["images"])
    S, H, W = depth.shape

    g_vmin, g_vmax = depth_range(depth)
    scale = min(1.0, hover_max / float(max(H, W))) if hover_max > 0 else 0.0
    gh = max(1, int(round(H * scale))) if scale else 0
    gw = max(1, int(round(W * scale))) if scale else 0

    input_uris, depth_uris, minmax = [], [], []
    grid = np.zeros((S, gh, gw), np.float32) if scale else None
    for i in range(S):
        vmin, vmax = (g_vmin, g_vmax) if normalize == "global" else depth_range(depth[i])
        minmax.append([round(vmin, 4), round(vmax, 4)])
        input_uris.append(_jpeg_data_uri(imgs[i], quality))
        depth_uris.append(_jpeg_data_uri(colorize_depth(depth[i], vmin, vmax, cmap), quality))
        if grid is not None:
            grid[i] = _resize_nearest(depth[i], gh, gw)

    glb_uri = None
    if glb_path and os.path.exists(glb_path):
        with open(glb_path, "rb") as f:
            glb_uri = "data:model/gltf-binary;base64," + base64.b64encode(f.read()).decode("ascii")

    grid_b64 = base64.b64encode(grid.astype("<f4").tobytes()).decode("ascii") if grid is not None else ""

    html = _TEMPLATE
    repl = {
        "__TITLE__": title,
        "__INPUTS__": json.dumps(input_uris),
        "__DEPTHS__": json.dumps(depth_uris),
        "__NFRAMES__": str(S),
        "__MINMAX__": json.dumps(minmax),
        "__NORMALIZE__": normalize,
        "__UNITS__": units,
        "__CMAP__": cmap,
        "__CMAP_CSS__": _cmap_css(cmap),
        "__GLB__": json.dumps(glb_uri),
        "__HAS_GRID__": "true" if grid is not None else "false",
        "__GRID_B64__": grid_b64,
        "__GRID_H__": str(gh),
        "__GRID_W__": str(gw),
    }
    for k, v in repl.items():
        html = html.replace(k, v)

    with open(out_path, "w") as f:
        f.write(html)
    return out_path


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
<style>
:root{--bg:#0b1220;--panel:#111a2b;--fg:#e2e8f0;--accent:#38bdf8;--muted:#64748b}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
  font-family:system-ui,Segoe UI,Roboto,sans-serif}
.toolbar{display:flex;align-items:center;gap:12px;padding:10px 14px;background:var(--panel);
  position:sticky;top:0;flex-wrap:wrap;border-bottom:1px solid #1e293b;z-index:5}
.toolbar h1{font-size:15px;margin:0 6px 0 0;font-weight:600}
button{background:#1e293b;color:var(--fg);border:1px solid #334155;border-radius:8px;
  padding:6px 10px;cursor:pointer;font-size:13px}
button.active{background:var(--accent);color:#04263a;border-color:var(--accent);font-weight:600}
input[type=range]{width:300px;accent-color:var(--accent)}
.frameLabel{font-variant-numeric:tabular-nums;min-width:78px;text-align:center}
.hint{color:var(--muted);font-size:12px}
.legend{display:flex;align-items:center;gap:8px;font-size:12px;margin-left:auto}
.bar{width:170px;height:12px;border-radius:6px;background:__CMAP_CSS__;border:1px solid #334155}
.wrap{display:flex;flex-wrap:wrap;gap:16px;padding:16px;align-items:flex-start}
.card{background:var(--panel);border:1px solid #1e293b;border-radius:12px;padding:10px}
.card h2{font-size:11px;margin:0 0 8px;color:#94a3b8;font-weight:600;letter-spacing:.05em;text-transform:uppercase}
.stage{position:relative;line-height:0;border-radius:8px;overflow:hidden;background:#000;cursor:crosshair}
.stage img{display:block;height:auto;user-select:none;-webkit-user-drag:none}
#cmpInput{max-width:62vw;width:auto}
#cmpDepthClip{position:absolute;top:0;left:0;height:100%;overflow:hidden;width:50%}
#cmpDepthClip img{position:absolute;top:0;left:0;height:100%;width:auto;max-width:none}
#handle{position:absolute;top:0;left:50%;width:2px;height:100%;background:var(--accent);cursor:ew-resize}
#handle::after{content:'';position:absolute;top:50%;left:50%;width:28px;height:28px;
  transform:translate(-50%,-50%);border-radius:50%;background:var(--accent);box-shadow:0 0 0 3px rgba(56,189,248,.25)}
.side{display:flex;gap:14px;flex-wrap:wrap}
.side .stage img{max-width:30vw}
.tooltip{position:fixed;pointer-events:none;background:#0f172a;border:1px solid var(--accent);
  border-radius:6px;padding:3px 7px;font-size:12px;display:none;z-index:9;font-variant-numeric:tabular-nums}
model-viewer{width:46vw;height:62vh;background:#0b1220;border-radius:8px}
.hidden{display:none!important}
</style></head>
<body>
<div class="toolbar">
  <h1>__TITLE__</h1>
  <button id="playBtn" title="Play/Pause (space)">&#9654;</button>
  <input id="slider" type="range" min="0" max="0" value="0" step="1"/>
  <span class="frameLabel" id="frameLabel">0 / 0</span>
  <span class="hint">&larr;/&rarr; step &middot; hover for depth value</span>
  <button id="modeCompare" class="active">Compare</button>
  <button id="modeSide">Side by side</button>
  <div class="legend"><span id="vmin">near</span><div class="bar"></div><span id="vmax">far</span>
    <span class="hint">__CMAP__</span></div>
</div>
<div class="wrap">
  <div class="card" id="compareCard">
    <h2>Input &#8644; Depth &nbsp;(drag the handle)</h2>
    <div class="stage" id="cmpStage">
      <img id="cmpInput" alt="input"/>
      <div id="cmpDepthClip"><img id="cmpDepth" alt="depth"/></div>
      <div id="handle"></div>
    </div>
  </div>
  <div class="card hidden" id="sideCard">
    <h2>Input &nbsp;|&nbsp; Depth</h2>
    <div class="side">
      <div class="stage" id="inStage"><img id="sideInput" alt="input"/></div>
      <div class="stage" id="dpStage"><img id="sideDepth" alt="depth"/></div>
    </div>
  </div>
  <div class="card" id="threeCard">
    <h2>3D combined point cloud &nbsp;(drag to orbit)</h2>
    <div id="threeHolder"></div>
  </div>
</div>
<div class="tooltip" id="tip"></div>
<script>
const INPUTS=__INPUTS__, DEPTHS=__DEPTHS__, NF=__NFRAMES__, MINMAX=__MINMAX__;
const UNITS="__UNITS__", GLB=__GLB__, HAS_GRID=__HAS_GRID__, GH=__GRID_H__, GW=__GRID_W__;
function b64ToF32(b){const s=atob(b),n=s.length,a=new Uint8Array(n);for(let i=0;i<n;i++)a[i]=s.charCodeAt(i);return new Float32Array(a.buffer);}
const GRID=HAS_GRID?b64ToF32("__GRID_B64__"):null;
let f=0,playing=false,timer=null;
const $=id=>document.getElementById(id);
const slider=$("slider");slider.max=NF-1;
function setFrame(i){f=Math.max(0,Math.min(NF-1,i|0));
  $("cmpInput").src=INPUTS[f];$("cmpDepth").src=DEPTHS[f];
  $("sideInput").src=INPUTS[f];$("sideDepth").src=DEPTHS[f];
  slider.value=f;$("frameLabel").textContent=(f+1)+" / "+NF;
  const m=MINMAX[f];$("vmin").textContent=m[0].toFixed(2)+UNITS;$("vmax").textContent=m[1].toFixed(2)+UNITS;}
slider.addEventListener("input",()=>setFrame(+slider.value));
function togglePlay(){playing=!playing;$("playBtn").innerHTML=playing?"&#10073;&#10073;":"&#9654;";
  if(playing)timer=setInterval(()=>setFrame((f+1)%NF),110);else clearInterval(timer);}
$("playBtn").addEventListener("click",togglePlay);
document.addEventListener("keydown",e=>{if(e.key==="ArrowRight")setFrame(f+1);
  else if(e.key==="ArrowLeft")setFrame(f-1);else if(e.key===" "){e.preventDefault();togglePlay();}});
function setMode(m){const c=m==="compare";$("compareCard").classList.toggle("hidden",!c);
  $("sideCard").classList.toggle("hidden",c);$("modeCompare").classList.toggle("active",c);
  $("modeSide").classList.toggle("active",!c);if(c)resetSplit();}
$("modeCompare").addEventListener("click",()=>setMode("compare"));
$("modeSide").addEventListener("click",()=>setMode("side"));
// swipe handle
const stage=$("cmpStage"),clip=$("cmpDepthClip"),handle=$("handle"),cmpDepth=$("cmpDepth");
let dragging=false;
function setSplit(px){const w=stage.getBoundingClientRect().width;const x=Math.max(0,Math.min(w,px));
  clip.style.width=x+"px";handle.style.left=x+"px";cmpDepth.style.width=w+"px";cmpDepth.style.height="";}
function resetSplit(){requestAnimationFrame(()=>setSplit(stage.getBoundingClientRect().width*0.5));}
handle.addEventListener("pointerdown",e=>{dragging=true;handle.setPointerCapture(e.pointerId);});
handle.addEventListener("pointermove",e=>{if(dragging)setSplit(e.clientX-stage.getBoundingClientRect().left);});
handle.addEventListener("pointerup",()=>dragging=false);
$("cmpInput").addEventListener("load",resetSplit);window.addEventListener("resize",resetSplit);
// hover readout
const tip=$("tip");
function hover(e,el){if(!GRID){return;}const r=el.getBoundingClientRect();
  const u=(e.clientX-r.left)/r.width,v=(e.clientY-r.top)/r.height;
  if(u<0||u>1||v<0||v>1){tip.style.display="none";return;}
  const gx=Math.min(GW-1,Math.floor(u*GW)),gy=Math.min(GH-1,Math.floor(v*GH));
  const z=GRID[(f*GH+gy)*GW+gx];tip.style.display="block";
  tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";
  tip.textContent=(z>0&&isFinite(z))?("z = "+z.toFixed(3)+UNITS):"no depth";}
["cmpStage","inStage","dpStage"].forEach(id=>{const el=$(id);
  el.addEventListener("mousemove",e=>hover(e,el));el.addEventListener("mouseleave",()=>tip.style.display="none");});
// 3D panel
if(GLB){const mv=document.createElement("model-viewer");mv.src=GLB;
  mv.setAttribute("camera-controls","");mv.setAttribute("touch-action","pan-y");
  mv.setAttribute("interaction-prompt","none");$("threeHolder").appendChild(mv);}
else{$("threeCard").innerHTML='<h2>3D combined point cloud</h2>'+
  '<p class="hint">Not embedded. Re-run with trimesh available (Python &ge;3.10) to include the orbitable point cloud.</p>';}
setFrame(0);resetSplit();
</script>
</body></html>
"""
