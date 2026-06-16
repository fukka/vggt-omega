# Copyright (c) 2026.
"""Per-frame depth export + a self-contained interactive depth viewer.

Consumed by ``load_and_visualize.py`` for three modes:

* ``export_depth_npy``  -> raw float depth ``depth/000000.npy`` per frame
* ``export_depth_vis``  -> colormapped ``depth_vis/000000.jpg`` per frame
* ``build_depth_viewer_html`` -> one self-contained ``depth_viewer.html`` with a
  frame scrubber, an input<->depth swipe-compare (+ side-by-side), a hover
  readout of the actual depth value, and a **per-frame 3D point cloud**
  (each frame's depth unprojected, rendered with Three.js, orbit by drag) that
  switches with the scrubber.

All inputs come from the saved ``predictions.npz`` (``depth`` ``[S,H,W,1]``,
``images`` ``[S,3,H,W]`` in ``[0,1]``, and ``world_points_from_depth``
``[S,H,W,3]`` — recomputed from depth + cameras if absent). No GPU, no trimesh.

Optional deps imported lazily: ``matplotlib`` (colormap) and ``Pillow`` (JPEG).
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


def _depth_edge_mask(depth2d: np.ndarray, rtol: float = 0.03, ksize: int = 3) -> np.ndarray:
    """Numpy port of visual_util.depth_edge: True at depth discontinuities
    (flying pixels) so they can be dropped from the 3D cloud."""
    d = depth2d
    pad = ksize // 2
    padded = np.pad(d, ((pad, pad), (pad, pad)), mode="edge")
    h, w = d.shape
    dmax = np.full_like(d, -np.inf)
    dmin = np.full_like(d, np.inf)
    for y in range(ksize):
        for x in range(ksize):
            win = padded[y : y + h, x : x + w]
            dmax = np.maximum(dmax, win)
            dmin = np.minimum(dmin, win)
    jump = (dmax - dmin) / np.maximum(np.abs(d), 1e-6)
    return jump > rtol


def _unproject(depth: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Fallback world points if ``world_points_from_depth`` is missing.
    ``depth [S,H,W]``, ``extrinsic [S,3,4]`` (cam-from-world), ``intrinsic [S,3,3]``."""
    S, H, W = depth.shape
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = np.broadcast_to(x[None], (S, H, W))
    y = np.broadcast_to(y[None], (S, H, W))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    cam = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(R, (0, 2, 1)), cam - t[:, None, None, :])


# --------------------------------------------------------------------------- #
# mode 1 + 2: exporters
# --------------------------------------------------------------------------- #
def export_depth_npy(predictions: dict, out_dir: str) -> int:
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
# per-frame point clouds (each frame's depth unprojected to world)
# --------------------------------------------------------------------------- #
def build_point_clouds(
    predictions: dict,
    points_per_frame: int = 15000,
    conf_thres: float = 50.0,
    filter_edges: bool = True,
):
    """Returns (pos_b64, col_b64, offsets, total) where positions are float32
    (centered + unit-scaled, all frames in a shared world frame) and colors are
    uint8 RGB. ``offsets[f] = [start, count]`` indexes the concatenated arrays.
    """
    depth = squeeze_depth(predictions["depth"])
    imgs = images_to_uint8(predictions["images"])
    S, H, W = depth.shape

    if "world_points_from_depth" in predictions:
        wp = np.asarray(predictions["world_points_from_depth"]).reshape(S, H, W, 3)
    else:
        wp = _unproject(depth, np.asarray(predictions["extrinsic"]), np.asarray(predictions["intrinsic"]))

    conf = predictions.get("depth_conf")
    cthr = None
    if conf is not None and conf_thres > 0:
        cflat = np.asarray(conf)[np.isfinite(conf)]
        cthr = float(np.percentile(cflat, conf_thres)) if cflat.size else None

    pos_list, col_list, offsets = [], [], []
    start = 0
    for i in range(S):
        P = wp[i].reshape(-1, 3)
        C = imgs[i].reshape(-1, 3)
        valid = np.isfinite(P).all(axis=1) & (depth[i].reshape(-1) > 0)
        if cthr is not None:
            valid &= np.asarray(conf)[i].reshape(-1) >= cthr
        if filter_edges:
            valid &= ~_depth_edge_mask(depth[i]).reshape(-1)
        idx = np.nonzero(valid)[0]
        if points_per_frame > 0 and idx.size > points_per_frame:
            sel = np.random.default_rng(i).choice(idx.size, points_per_frame, replace=False)
            idx = idx[np.sort(sel)]
        pos_list.append(P[idx])
        col_list.append(C[idx])
        offsets.append([start, int(idx.size)])
        start += int(idx.size)

    pos = np.concatenate(pos_list, 0).astype(np.float32) if pos_list else np.zeros((0, 3), np.float32)
    col = np.concatenate(col_list, 0).astype(np.uint8) if col_list else np.zeros((0, 3), np.uint8)

    if pos.shape[0]:
        center = np.median(pos, axis=0)
        dist = np.linalg.norm(pos - center, axis=1)
        scale = float(np.percentile(dist, 95)) or 1.0
        pos = ((pos - center) / scale).astype("<f4")

    pos_b64 = base64.b64encode(np.ascontiguousarray(pos, "<f4").tobytes()).decode("ascii")
    col_b64 = base64.b64encode(np.ascontiguousarray(col, np.uint8).tobytes()).decode("ascii")
    return pos_b64, col_b64, offsets, int(pos.shape[0])


# --------------------------------------------------------------------------- #
# mode 3: interactive viewer
# --------------------------------------------------------------------------- #
def build_depth_viewer_html(
    predictions: dict,
    out_path: str,
    cmap: str = "turbo",
    normalize: str = "global",
    hover_max: int = 160,
    quality: int = 85,
    units: str = "",
    title: str = "VGGT-Ω depth viewer",
    points_per_frame: int = 15000,
    conf_thres: float = 50.0,
    point_size: float = 0.01,
    filter_edges: bool = True,
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

    pc_pos, pc_col, pc_off, pc_total = build_point_clouds(
        predictions, points_per_frame=points_per_frame, conf_thres=conf_thres, filter_edges=filter_edges
    )
    grid_b64 = base64.b64encode(grid.astype("<f4").tobytes()).decode("ascii") if grid is not None else ""

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
        "__HAS_GRID__": "true" if grid is not None else "false",
        "__GRID_B64__": grid_b64,
        "__GRID_H__": str(gh),
        "__GRID_W__": str(gw),
        "__PC_POS__": pc_pos,
        "__PC_COL__": pc_col,
        "__PC_OFF__": json.dumps(pc_off),
        "__POINT_SIZE__": repr(float(point_size)),
    }
    html = _TEMPLATE
    for k, v in repl.items():
        html = html.replace(k, v)
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>
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
#threeHolder{width:46vw;height:62vh;min-width:320px;min-height:320px;border-radius:8px;background:#0b1220;
  position:relative}
#threeHolder canvas{display:block;border-radius:8px}
#threeMsg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  text-align:center;padding:20px;color:var(--muted);font-size:13px}
.hidden{display:none!important}
</style></head>
<body>
<div class="toolbar">
  <h1>__TITLE__</h1>
  <button id="playBtn" title="Play/Pause (space)">&#9654;</button>
  <input id="slider" type="range" min="0" max="0" value="0" step="1"/>
  <span class="frameLabel" id="frameLabel">0 / 0</span>
  <span class="hint">&larr;/&rarr; step &middot; hover for depth value &middot; drag the 3D to orbit</span>
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
    <h2>This frame's point cloud &nbsp;(drag to orbit, scroll to zoom)</h2>
    <div id="threeHolder"><div id="threeMsg">loading 3D&hellip;</div></div>
  </div>
</div>
<div class="tooltip" id="tip"></div>

<script>
/* ---- embedded data (globals so the 3D module can read them) ---- */
window.INPUTS=__INPUTS__; window.DEPTHS=__DEPTHS__; window.NF=__NFRAMES__;
window.MINMAX=__MINMAX__; window.UNITS="__UNITS__"; window.HAS_GRID=__HAS_GRID__;
window.GH=__GRID_H__; window.GW=__GRID_W__; window.GRID_B64="__GRID_B64__";
window.PC_POS="__PC_POS__"; window.PC_COL="__PC_COL__"; window.PC_OFF=__PC_OFF__;
window.POINT_SIZE=__POINT_SIZE__; window.__frame=0;

/* ---- 2D viewer (classic script; works even if Three.js/CDN fails) ---- */
(function(){
const INPUTS=window.INPUTS,DEPTHS=window.DEPTHS,NF=window.NF,MINMAX=window.MINMAX,UNITS=window.UNITS;
const $=id=>document.getElementById(id);
const slider=$("slider");slider.max=NF-1;
function b64ToF32(b){const s=atob(b),n=s.length,a=new Uint8Array(n);for(let i=0;i<n;i++)a[i]=s.charCodeAt(i);return new Float32Array(a.buffer);}
const GRID=window.HAS_GRID?b64ToF32(window.GRID_B64):null,GH=window.GH,GW=window.GW;
let f=0,playing=false,timer=null;
function setFrame(i){f=Math.max(0,Math.min(NF-1,i|0));window.__frame=f;
  $("cmpInput").src=INPUTS[f];$("cmpDepth").src=DEPTHS[f];
  $("sideInput").src=INPUTS[f];$("sideDepth").src=DEPTHS[f];
  slider.value=f;$("frameLabel").textContent=(f+1)+" / "+NF;
  const m=MINMAX[f];$("vmin").textContent=m[0].toFixed(2)+UNITS;$("vmax").textContent=m[1].toFixed(2)+UNITS;
  document.dispatchEvent(new CustomEvent("frame",{detail:f}));}
slider.addEventListener("input",()=>setFrame(+slider.value));
function togglePlay(){playing=!playing;$("playBtn").innerHTML=playing?"&#10073;&#10073;":"&#9654;";
  if(playing)timer=setInterval(()=>setFrame((f+1)%NF),110);else clearInterval(timer);}
$("playBtn").addEventListener("click",togglePlay);
document.addEventListener("keydown",e=>{if(e.key==="ArrowRight")setFrame(f+1);
  else if(e.key==="ArrowLeft")setFrame(f-1);else if(e.key===" "){e.preventDefault();togglePlay();}});
function setMode(c){$("compareCard").classList.toggle("hidden",!c);$("sideCard").classList.toggle("hidden",c);
  $("modeCompare").classList.toggle("active",c);$("modeSide").classList.toggle("active",!c);if(c)resetSplit();}
$("modeCompare").addEventListener("click",()=>setMode(true));
$("modeSide").addEventListener("click",()=>setMode(false));
const stage=$("cmpStage"),clip=$("cmpDepthClip"),handle=$("handle"),cmpDepth=$("cmpDepth");
let dragging=false;
function setSplit(px){const w=stage.getBoundingClientRect().width;const x=Math.max(0,Math.min(w,px));
  clip.style.width=x+"px";handle.style.left=x+"px";cmpDepth.style.width=w+"px";}
function resetSplit(){requestAnimationFrame(()=>setSplit(stage.getBoundingClientRect().width*0.5));}
handle.addEventListener("pointerdown",e=>{dragging=true;handle.setPointerCapture(e.pointerId);});
handle.addEventListener("pointermove",e=>{if(dragging)setSplit(e.clientX-stage.getBoundingClientRect().left);});
handle.addEventListener("pointerup",()=>dragging=false);
$("cmpInput").addEventListener("load",resetSplit);window.addEventListener("resize",resetSplit);
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
setFrame(0);resetSplit();
})();
</script>

<script type="module">
/* ---- per-frame 3D point cloud (isolated: if the CDN import fails, 2D still works) ---- */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
try{
  const holder=document.getElementById('threeHolder');
  const msg=document.getElementById('threeMsg'); if(msg) msg.remove();
  function b64ToBytes(b){const s=atob(b),n=s.length,a=new Uint8Array(n);for(let i=0;i<n;i++)a[i]=s.charCodeAt(i);return a;}
  const POS=new Float32Array(b64ToBytes(window.PC_POS).buffer);
  const COL=b64ToBytes(window.PC_COL);
  const OFF=window.PC_OFF;
  let w=holder.clientWidth||640,h=holder.clientHeight||480;
  const renderer=new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(w,h);renderer.setClearColor(0x0b1220,1);
  holder.appendChild(renderer.domElement);
  const scene=new THREE.Scene();
  const camera=new THREE.PerspectiveCamera(55,w/h,0.01,100);camera.position.set(0,0,2.6);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;controls.target.set(0,0,0);
  const geom=new THREE.BufferGeometry();
  const mat=new THREE.PointsMaterial({size:window.POINT_SIZE,vertexColors:true,sizeAttenuation:true});
  const pts=new THREE.Points(geom,mat);scene.add(pts);
  function setCloud(f){const o=OFF[f];if(!o||o[1]===0)return;const s=o[0]*3,n=o[1]*3;
    geom.setAttribute('position',new THREE.BufferAttribute(POS.subarray(s,s+n),3));
    geom.setAttribute('color',new THREE.BufferAttribute(COL.subarray(s,s+n),3,true));
    geom.computeBoundingSphere();}
  document.addEventListener('frame',e=>setCloud(e.detail));
  setCloud(window.__frame||0);
  new ResizeObserver(()=>{w=holder.clientWidth;h=holder.clientHeight;if(w&&h){renderer.setSize(w,h);
    camera.aspect=w/h;camera.updateProjectionMatrix();}}).observe(holder);
  (function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera);})();
}catch(err){
  const holder=document.getElementById('threeHolder');
  holder.innerHTML='<div id="threeMsg">3D unavailable: '+err+'<br/>(needs WebGL + internet for the Three.js CDN; the 2D views still work.)</div>';
}
</script>
</body></html>
"""
