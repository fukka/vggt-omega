# Copyright (c) 2026.
"""LOAD saved VGGT-Omega results and VISUALIZE them — no GPU, no checkpoint.

Pairs with ``save_demo_results.py``. Point it at the saved bundle (or a ``.glb``
/ ``predictions.npz`` directly) and it will:

* (re)build the GLB point cloud + cameras from ``predictions.npz`` using the
  same ``visual_util.predictions_to_glb`` as the Gradio demo — so you can adjust
  the confidence threshold / max points / masks **offline on any laptop**; and
* visualize it one of three ways:
    - ``--mode html``   (default) a self-contained ``viewer.html`` (GLB embedded
      as base64, rendered with Google's <model-viewer> — the same component the
      Gradio demo uses). Double-click it, or host it on any static server to get
      a shareable website **without any tunnel**.
    - ``--mode gradio`` a LOCAL Gradio ``Model3D`` viewer on 127.0.0.1 (no --share).
    - ``--mode show``   an offline trimesh window.

Examples
--------
    python load_and_visualize.py --results demo_results/run1
    python load_and_visualize.py --results demo_results/run1 --conf-thres 30 --max-points-k 2000 --rebuild
    python load_and_visualize.py --glb demo_results/run1/scene.glb --mode gradio
    python load_and_visualize.py --results demo_results/run1 --mode html --serve --port 8000
"""
from __future__ import annotations

import argparse
import base64
import os

import numpy as np

import depth_viewer


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_predictions(results: str):
    """``results`` is a bundle dir or a predictions.npz path. Returns (dict, target_dir)."""
    if results.endswith(".npz"):
        npz_path, target_dir = results, os.path.dirname(results) or "."
    else:
        npz_path, target_dir = os.path.join(results, "predictions.npz"), results
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"predictions.npz not found at {npz_path!r}")
    with np.load(npz_path) as loaded:
        predictions = {key: np.array(loaded[key]) for key in loaded.files}
    return predictions, target_dir


def glb_name(params: dict) -> str:
    return (
        f"scene_conf{params['conf_thres']}_black{params['mask_black_bg']}_"
        f"white{params['mask_white_bg']}_cam{params['show_cam']}_"
        f"sky{params['mask_sky']}_max{params['max_points'] // 1000}k.glb"
    )


def build_glb(predictions: dict, target_dir: str, params: dict, rebuild: bool) -> str:
    """(Re)build the GLB from predictions. Lazy-imports trimesh via visual_util."""
    glb_path = os.path.join(target_dir, glb_name(params))
    if os.path.exists(glb_path) and not rebuild:
        print(f"Using cached GLB: {glb_path}")
        return glb_path
    try:
        from visual_util import predictions_to_glb
    except ImportError as exc:
        raise SystemExit(
            f"Rebuilding the GLB needs `trimesh` (and visual_util): {exc}\n"
            f"Install with `pip install trimesh matplotlib scipy`, or pass an "
            f"already-built GLB via --glb."
        )
    scene = predictions_to_glb(
        predictions,
        conf_thres=params["conf_thres"],
        mask_black_bg=params["mask_black_bg"],
        mask_white_bg=params["mask_white_bg"],
        show_cam=params["show_cam"],
        mask_sky=params["mask_sky"],
        target_dir=target_dir,
        max_points=params["max_points"],
    )
    scene.export(file_obj=glb_path)
    print(f"Built GLB: {glb_path}")
    return glb_path


# --------------------------------------------------------------------------- #
# visualize: self-contained HTML (model-viewer)
# --------------------------------------------------------------------------- #
_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>VGGT-Ω reconstruction</title>
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
<style>
  html, body {{ margin: 0; height: 100%; background: #0f172a; color: #e2e8f0;
    font-family: system-ui, sans-serif; }}
  model-viewer {{ width: 100vw; height: 100vh; --poster-color: transparent; }}
  .hud {{ position: fixed; left: 12px; top: 10px; font-size: 13px; opacity: .85;
    background: rgba(15,23,42,.6); padding: 6px 10px; border-radius: 8px; }}
</style>
</head>
<body>
<div class="hud">VGGT-Ω · {title} · drag to orbit, scroll to zoom</div>
<model-viewer src="{src}" alt="VGGT-Omega reconstruction"
  camera-controls touch-action="pan-y" interaction-prompt="none"
  exposure="1.0" shadow-intensity="0" orientation="0deg 0deg 0deg"></model-viewer>
</body>
</html>
"""


def make_html(glb_path: str, embed: bool, title: str = "") -> str:
    if embed:
        with open(glb_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        src = f"data:model/gltf-binary;base64,{data}"
    else:
        src = os.path.basename(glb_path)
    html = _HTML_TEMPLATE.format(src=src, title=title or os.path.basename(glb_path))
    html_path = os.path.join(os.path.dirname(glb_path) or ".", "viewer.html")
    with open(html_path, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(html_path) / 1e6
    print(f"Wrote viewer: {html_path} ({size_mb:.1f} MB, {'self-contained' if embed else 'references ' + src})")
    return html_path


def serve_dir(directory: str, html_name: str, port: int) -> None:
    import http.server
    import socketserver

    os.chdir(directory)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/{html_name}"
        print(f"Serving {directory} at {url}  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


# --------------------------------------------------------------------------- #
# visualize: local gradio / trimesh
# --------------------------------------------------------------------------- #
def launch_gradio(glb_path: str, port: int, open_browser: bool) -> None:
    import gradio as gr

    with gr.Blocks(title="VGGT-Omega viewer") as demo:
        gr.Markdown("### VGGT-Ω reconstruction (local viewer — no --share)")
        gr.Model3D(value=glb_path, height=780, zoom_speed=0.2, pan_speed=0.2)
    # 127.0.0.1 keeps it local; no public tunnel involved.
    demo.launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=open_browser)


def show_trimesh(glb_path: str) -> None:
    import trimesh

    trimesh.load(glb_path).show()


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Load + visualize saved VGGT-Omega results (no GPU)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--results", help="Saved bundle dir or predictions.npz")
    src.add_argument("--glb", help="An already-built .glb to view directly (skips npz/trimesh)")

    p.add_argument("--mode", choices=["html", "gradio", "show"], default="html")
    p.add_argument("--rebuild", action="store_true", help="Rebuild GLB even if cached")
    # GLB params (mirror the gradio sliders); used when (re)building from npz
    p.add_argument("--conf-thres", type=float, default=50.0)
    p.add_argument("--max-points-k", type=int, default=1000)
    p.add_argument("--no-cam", action="store_true")
    p.add_argument("--mask-black-bg", action="store_true")
    p.add_argument("--mask-white-bg", action="store_true")
    p.add_argument("--mask-sky", action="store_true")
    # html options
    p.add_argument("--no-embed", action="store_true", help="Reference the GLB instead of embedding it")
    p.add_argument("--serve", action="store_true", help="Serve viewer.html over local HTTP")
    p.add_argument("--no-open", action="store_true", help="Do not auto-open a browser")
    p.add_argument("--port", type=int, default=8000)
    # depth modes (operate on --results predictions.npz)
    p.add_argument("--export-depth", action="store_true",
                   help="Write raw float depth .npy per frame to <results>/depth/")
    p.add_argument("--export-depth-visualized", action="store_true",
                   help="Write colormapped depth .jpg per frame to <results>/depth_vis/")
    p.add_argument("--visualize-depth", action="store_true",
                   help="Build an interactive depth viewer (frame scrub + input/depth swipe + hover value + 3D)")
    p.add_argument("--depth-cmap", default="turbo", help="Matplotlib colormap for depth (default: turbo)")
    p.add_argument("--depth-normalize", choices=["global", "per-frame"], default="global",
                   help="Depth color normalization: global (consistent across frames) or per-frame")
    p.add_argument("--depth-units", default="", help="Unit label for depth values in the viewer, e.g. 'm'")
    p.add_argument("--no-depth-hover", action="store_true",
                   help="Omit the embedded depth grid (smaller depth_viewer.html, no hover readout)")
    p.add_argument("--points-per-frame", type=int, default=15000,
                   help="Max points per frame in the 3D view (subsampled); lower = smaller html")
    p.add_argument("--point-size", type=float, default=0.01, help="3D point size (normalized units)")
    p.add_argument("--no-depth-edge-filter", action="store_true",
                   help="Keep flying pixels at depth discontinuities in the per-frame 3D cloud")
    return p.parse_args()


def run_depth_modes(args) -> None:
    """--export-depth / --export-depth-visualized / --visualize-depth (need --results)."""
    if not args.results:
        raise SystemExit(
            "--export-depth / --export-depth-visualized / --visualize-depth require "
            "--results (a predictions.npz bundle), not --glb."
        )
    predictions, target_dir = load_predictions(args.results)

    if args.export_depth:
        n = depth_viewer.export_depth_npy(predictions, os.path.join(target_dir, "depth"))
        print(f"Wrote {n} depth arrays -> {os.path.join(target_dir, 'depth')}/000000.npy ...")
    if args.export_depth_visualized:
        n = depth_viewer.export_depth_vis(
            predictions,
            os.path.join(target_dir, "depth_vis"),
            cmap=args.depth_cmap,
            normalize=args.depth_normalize,
        )
        print(f"Wrote {n} colormapped depths -> {os.path.join(target_dir, 'depth_vis')}/000000.jpg ...")
    if args.visualize_depth:
        html = depth_viewer.build_depth_viewer_html(
            predictions,
            os.path.join(target_dir, "depth_viewer.html"),
            cmap=args.depth_cmap,
            normalize=args.depth_normalize,
            hover_max=0 if args.no_depth_hover else 160,
            units=args.depth_units,
            points_per_frame=args.points_per_frame,
            conf_thres=max(0.0, float(args.conf_thres)),
            point_size=args.point_size,
            filter_edges=not args.no_depth_edge_filter,
        )
        print(f"Wrote depth viewer: {html} ({os.path.getsize(html) / 1e6:.1f} MB)")
        if args.serve:
            serve_dir(os.path.dirname(html) or ".", os.path.basename(html), args.port)
        elif not args.no_open:
            import webbrowser

            webbrowser.open("file://" + os.path.abspath(html))
            print("Opened in your browser. (Host depth_viewer.html on any static server to share it.)")
        else:
            print(f"Open {html} in a browser, or host it on a static server to share.")


def main():
    args = parse_args()

    if args.export_depth or args.export_depth_visualized or args.visualize_depth:
        run_depth_modes(args)
        return

    if args.glb:
        glb_path = args.glb
        if not os.path.isfile(glb_path):
            raise FileNotFoundError(glb_path)
    else:
        predictions, target_dir = load_predictions(args.results)
        params = {
            "conf_thres": max(2.0, float(args.conf_thres)),
            "max_points": int(args.max_points_k * 1000),
            "show_cam": not args.no_cam,
            "mask_black_bg": args.mask_black_bg,
            "mask_white_bg": args.mask_white_bg,
            "mask_sky": args.mask_sky,
        }
        glb_path = build_glb(predictions, target_dir, params, args.rebuild)

    if args.mode == "html":
        html_path = make_html(glb_path, embed=not args.no_embed)
        if args.serve:
            serve_dir(os.path.dirname(html_path) or ".", os.path.basename(html_path), args.port)
        elif not args.no_open:
            import webbrowser

            webbrowser.open("file://" + os.path.abspath(html_path))
            print("Opened in your default browser. (Host viewer.html on any static "
                  "server to share it as a website — no tunnel needed.)")
        else:
            print(f"Open {html_path} in a browser, or host it on a static server to share.")
    elif args.mode == "gradio":
        launch_gradio(glb_path, args.port if args.port != 8000 else 7860, open_browser=not args.no_open)
    elif args.mode == "show":
        show_trimesh(glb_path)


if __name__ == "__main__":
    main()
