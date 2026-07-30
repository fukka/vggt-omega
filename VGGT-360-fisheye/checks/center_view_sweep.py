# Copyright (c) 2026.
"""Center-view cropping/processing sweep — how does the CENTER view's FoV/ROI and
its pre-processing change VGGT's depth alignment?

Motivation
----------
``main_adt.py --n-ring 1`` reduces the layout to the single center view (tilt 0),
yet even that center view shows the "depth is a distorted version of the input"
misalignment.  Since the center view is a plain single image, the whole VGGT-360
machinery (multi-view attention, fusion, masks) is out of the loop — what remains
is: *how the center image is cropped/processed from the raw fisheye* × *original
VGGT's behaviour on that image*.  This script isolates exactly those two factors.

This is the **grid** front-end for ``depth_probe.py``: the view sources, the
backend loading and the edge-alignment diagnostic all live there, so a number
measured here is directly comparable to one measured by the probe CLI.  What
this script adds is the cross-product and its reporting (CSV, plot, grid).

Controlled experiment
---------------------
Everything is held fixed — the SAME raw ADT fisheye frame, the SAME optical
center (tilt 0, azimuth 0), the SAME original VGGT-1B, the SAME output
resolution — while sweeping two knobs:

  1. **coverage** ``--fovs`` : the center view's field of view, small -> large.
     A larger FoV covers more of the imaged cone (edge incidence = fov/2), so it
     ingests more of the distortion-heavy periphery.

  2. **processing** ``--modes`` : how the pixels are produced at that coverage —
     ``tangent`` (gnomonic, straight lines preserved, periphery stretched),
     ``raw_roi`` (same angular coverage, NO undistortion — the honest "is it the
     undistortion or the model?" control), or ``rectifier`` (the repo's
     validated wide ~85 deg pinhole of the whole frame; ignores ``--fovs``).

  Optional third axis ``--enhance`` {none,clahe,sharpen,blur} tests the
  "clearness" hypothesis (does sharpening/blurring the SAME crop move the depth
  alignment — i.e. is VGGT depth-edge localisation limited by image clarity?).

Measured per configuration
--------------------------
  * ``align%`` — fraction of RGB edges that have a depth edge within ~2 px,
    over the valid cone.  High = depth structure follows the image.
  * VGGT's inferred camera FoV (``pose_enc[7:9]``) vs the crop's TRUE FoV — the
    suspected mechanism: VGGT couples depth to its own FoV estimate, so a wrong
    estimate bends the geometry.  Watch whether the gap widens with coverage.
  * an ``RGB | depth | edge-overlay`` strip per config, a summary CSV, an
    ``align vs FoV`` plot, and an overlay grid (rows = mode/enhance, cols = FoV).

Use ``--render-only`` to dump the crops WITHOUT a model (no GPU needed — this is
what can be checked on a laptop).

Examples
--------
    # FoV sweep, tangent vs raw-ROI, on a sharp ADT frame
    python VGGT-360-fisheye/checks/center_view_sweep.py \
        --adt-root <ROOT> --rgb-subdir videos_synthetic --frame 6 \
        --fovs 40 60 80 100 120 --modes tangent raw_roi

    # add the clearness axis
    python VGGT-360-fisheye/checks/center_view_sweep.py --adt-root <ROOT> \
        --fovs 60 90 --modes tangent --enhance none clahe blur

    # geometry preview only, no model / no GPU
    python VGGT-360-fisheye/checks/center_view_sweep.py --adt-root <ROOT> \
        --fovs 40 80 120 --modes tangent raw_roi --render-only
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from depth_probe import (BACKENDS, EDGE_PERCENTILE, adt_frame, enhance_view,
                        load_backend, materialize, probe, view_raw_roi,
                        view_rectifier, view_tangent)


def build_view(mode, rgb, cam, fov, enh, out_size, supersample):
    """One sweep cell -> a tagged View (all rendering lives in depth_probe)."""
    if mode == "tangent":
        view = view_tangent(rgb, cam, fov_deg=fov, out_size=out_size,
                            supersample=supersample)
    elif mode == "raw_roi":
        view = view_raw_roi(rgb, cam, fov_deg=fov, out_size=out_size)
    elif mode == "rectifier":
        view = view_rectifier(rgb, out_size=out_size)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    view = enhance_view(view, enh)
    fov_tag = "rect" if fov is None else f"{int(fov)}"
    view.tag = f"{mode}_fov{fov_tag}_{enh}"
    return view


def main():
    ap = argparse.ArgumentParser(
        description="Center-view cropping/processing sweep for VGGT")
    ap.add_argument("--adt-root", required=True)
    ap.add_argument("--rgb-subdir", default="videos_synthetic",
                    help="videos_synthetic = sharp (recommended); videos_rgb = "
                         "real sensor (motion-blurred confound)")
    ap.add_argument("--frame", type=int, default=6)
    ap.add_argument("--fovs", type=float, nargs="+",
                    default=[40, 60, 80, 100, 120],
                    help="center-view FoVs to sweep (deg); edge incidence = "
                         "fov/2, so stay <= ~124 to keep the edge inside the "
                         "62.3deg cone")
    ap.add_argument("--modes", nargs="+", default=["tangent", "raw_roi"],
                    choices=["tangent", "raw_roi", "rectifier"])
    ap.add_argument("--enhance", nargs="+", default=["none"],
                    choices=["none", "clahe", "sharpen", "blur"])
    ap.add_argument("--persp-size", type=int, default=518)
    ap.add_argument("--crop-supersample", type=int, default=3)
    ap.add_argument("--backend", choices=BACKENDS, default="vggt1b",
                    help="which model to sweep (the probe's backend seam)")
    ap.add_argument("--model-path", default="facebook/VGGT-1B")
    ap.add_argument("--checkpoint", default=None,
                    help="local .pt for the vggt_omega backend")
    ap.add_argument("--image-resolution", type=int, default=512,
                    help="vggt_omega preprocessing size (divisible by 16); "
                         "ignored by the other backends")
    ap.add_argument("--edge-percentile", type=float, default=EDGE_PERCENTILE,
                    help="edge-strength percentile; changing it changes align% "
                         "and breaks comparability with other runs")
    ap.add_argument("--render-only", action="store_true",
                    help="dump crops + valid masks WITHOUT running the model "
                         "(no GPU)")
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(_HERE), "outputs",
                                         "center_view_sweep"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rgb, cam = adt_frame(args.adt_root, rgb_subdir=args.rgb_subdir,
                         frame=args.frame)
    Image.fromarray(rgb).save(os.path.join(args.out, "raw_fisheye.png"))

    # ---- enumerate configs: (mode, fov, enhance); rectifier ignores --fovs ----
    configs = [(mode, fov, enh)
               for mode in args.modes
               for fov in ([None] if mode == "rectifier" else args.fovs)
               for enh in args.enhance]

    if args.render_only:
        for (mode, fov, enh) in configs:
            view = build_view(mode, rgb, cam, fov, enh, args.persp_size,
                              args.crop_supersample)
            materialize(view, args.out)
            if view.valid is not None:
                Image.fromarray((view.valid * 255).astype(np.uint8)).save(
                    os.path.join(args.out, f"{view.tag}_valid.png"))
            tf = f"{view.true_fov:.0f}" if view.true_fov else "?"
            print(f"{view.tag:<32}  {tf:>6}   (render-only)")
        print(f"\nrendered {len(configs)} crops -> {args.out}/  (add a GPU + "
              "drop --render-only to score them)")
        return

    # Backend loaded ONCE, then every config is probed through it.
    backend = load_backend(args.backend, model_path=args.model_path,
                           checkpoint=args.checkpoint,
                           image_resolution=args.image_resolution)

    rows = []   # (tag, mode, fov, enh, true_fov, infer_h, infer_w, align, overlay)
    print("\nconfig                          true_fov  infer(h,w)      align")
    for (mode, fov, enh) in configs:
        view = build_view(mode, rgb, cam, fov, enh, args.persp_size,
                          args.crop_supersample)
        res = probe(backend, [view], args.out,
                    percentile=args.edge_percentile)[0]
        np.save(os.path.join(args.out, f"{res.tag}_depth.npy"), res.depth_z)
        ih, iw = res.fov_inferred or (float("nan"), float("nan"))
        # overlay is the last third of the RGB|depth|overlay strip
        strip = np.array(Image.open(res.strip_path).convert("RGB"))
        overlay = strip[:, 2 * (strip.shape[1] // 3):]
        rows.append((res.tag, mode, fov, enh, res.true_fov, ih, iw,
                     res.align, overlay))
        tf = f"{res.true_fov:.0f}" if res.true_fov else "?"
        print(f"{res.tag:<32}  {tf:>6}   {ih:5.1f},{iw:5.1f}   "
              f"{res.align * 100:5.1f}%")

    _write_summary(rows, args)
    print(f"\ndone — {len(rows)} configs. See {args.out}/ "
          "(summary.csv, align_vs_fov.png, grid.png, per-config *.png)")


def _write_summary(rows, args):
    with open(os.path.join(args.out, "summary.csv"), "w") as f:
        f.write("tag,mode,fov,enhance,true_fov,infer_fov_h,infer_fov_w,"
                "align_pct\n")
        for (tag, mode, fov, enh, tf, ih, iw, al, _ov) in rows:
            f.write(f"{tag},{mode},{fov},{enh},{tf},{ih:.2f},{iw:.2f},"
                    f"{al * 100:.2f}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # align vs FoV, one line per (mode, enhance) with a real FoV sweep
    fig, ax = plt.subplots(figsize=(7, 5))
    keyed = {}
    for (tag, mode, fov, enh, tf, ih, iw, al, _ov) in rows:
        if fov is None:
            continue
        keyed.setdefault((mode, enh), []).append((fov, al * 100))
    for (mode, enh), pts in sorted(keyed.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                label=f"{mode}/{enh} align%")
    ax.set_xlabel("center-view FoV (deg) — larger = more of the fisheye cone")
    ax.set_ylabel("edge alignment (%)")
    ax.set_title("Depth–input edge alignment vs center-view coverage")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "align_vs_fov.png"), dpi=140)
    plt.close(fig)

    # overlay grid: rows = (mode,enhance), cols = FoV order
    grid_rows = {}
    for (tag, mode, fov, enh, tf, ih, iw, al, ov) in rows:
        grid_rows.setdefault((mode, enh), []).append(
            (fov if fov else 1e9, tag, al, ov))
    if not grid_rows:
        return
    H = min(ov.shape[0] for r in grid_rows.values() for (_, _, _, ov) in r)
    thumbs = []
    for (mode, enh), r in sorted(grid_rows.items()):
        r.sort()
        row_imgs = []
        for (fov, tag, al, ov) in r:
            t = cv2.resize(ov, (H, H))
            cv2.putText(t, f"{tag} {al * 100:.0f}%", (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 60), 1,
                        cv2.LINE_AA)
            row_imgs.append(t)
        thumbs.append(np.concatenate(row_imgs, axis=1))
    wmax = max(t.shape[1] for t in thumbs)
    thumbs = [np.pad(t, ((0, 0), (0, wmax - t.shape[1]), (0, 0)),
                     constant_values=24) for t in thumbs]
    Image.fromarray(np.concatenate(thumbs, axis=0)).save(
        os.path.join(args.out, "grid.png"))


if __name__ == "__main__":
    main()
