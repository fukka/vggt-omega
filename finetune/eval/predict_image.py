# Copyright (c) 2026.
"""Qualitative single-image depth: run VGGT + monocular baselines on ONE image.

No ground-truth depth, no alignment, no metrics — just each model's depth
prediction, colourised for eyeballing. This is the GT-free counterpart to the
quantitative runners (:mod:`finetune.eval.run_eval` /
:mod:`finetune.eval.baselines.benchmark_adt`): point it at ANY image
(indoor/outdoor, a phone photo, a web image — not just ADT) to compare how the
models read the scene.

    python -m finetune.eval.predict_image --image photo.jpg \
        --models vggt,dav2_large,dpt_large,zoedepth_nyu \
        --vggt-checkpoint runs/ssi_r8/checkpoint_best.pt

    # every locally-available model (VGGT + all "ready" zoo models):
    python -m finetune.eval.predict_image --image photo.jpg --models all \
        --vggt-checkpoint runs/ssi_r8/checkpoint_best.pt

    python -m finetune.eval.predict_image --list      # registry + availability

Models
------
``vggt``                VGGT-Omega (metric). Needs ``--vggt-checkpoint``. NOTE:
                        VGGT is a multi-view model; on a single image it runs in
                        its degenerate 1-view mode (valid, but weaker than with
                        several views).
<zoo key / family>      anything in :mod:`finetune.eval.baselines.model_zoo`:
                        dav2_small/base/large, dav2_metric_indoor/outdoor,
                        dav3_large, dpt_large/hybrid/swin2_tiny, zoedepth_nyu/nk,
                        depth_pro, metric3d_vit_small/large, unik3d_vits/b/l.
                        A family substring (e.g. ``dav2``) expands to all its
                        variants.
``all``                 VGGT (if a checkpoint is given) + every zoo model that
                        reports ``ready``.

DAC is skipped — it is ERP/fisheye-native and needs a fisheye camera model, not a
plain image. UniK3D runs camera-free (it predicts its own rays).

Scale
-----
Relative models (DAv2/DAv3/MiDaS) output up-to-scale depth (``1/disparity``);
metric models (VGGT/ZoeDepth/Metric3D/UniK3D/DAv2-metric) output metres. Each
panel is percentile-normalised **independently**, so this compares depth
*structure*, not absolute scale — which is unknowable without GT for the relative
models (and camera-dependent for the metric ones on an unknown lens).

Outputs (under ``--out``; default ``eval_out/predict_image/<image_stem>/``)::

    <model>_depth.npy   raw predicted depth (native units)
    <model>_depth.png   colourised depth
    input.png           the (possibly resized) input image
    comparison.png      input + every model's depth in one grid
"""
from __future__ import annotations

import sys as _sys
import os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    __package__ = "finetune.eval"

import argparse
import dataclasses
import os
import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .baselines import model_zoo as zoo

_VGGT_KEY = "vggt"


# --------------------------------------------------------------------------- #
# Image IO + colourisation
# --------------------------------------------------------------------------- #

def load_image01(path: str, long_side: int = 768) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1], optionally resized so its longer
    side equals ``long_side`` (aspect preserved; 0 = keep native resolution)."""
    img = Image.open(path).convert("RGB")
    if long_side and max(img.size) != long_side:
        w, h = img.size
        s = long_side / float(max(w, h))
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def colorize_depth(depth: np.ndarray, cmap: str = "magma_r"
                   ) -> Tuple[np.ndarray, float, float]:
    """Colourise a depth map for display. Normalises to the 2nd–98th percentile of
    finite, positive depths; invalid pixels are black. Returns (rgb_u8, vmin, vmax)."""
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.zeros((*d.shape, 3), np.uint8), float("nan"), float("nan")
    vmin, vmax = (float(v) for v in np.percentile(d[valid], [2, 98]))
    dn = np.clip((d - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    rgb = plt.get_cmap(cmap)(dn)[..., :3]
    rgb[~valid] = 0.0
    return (rgb * 255).astype(np.uint8), vmin, vmax


# --------------------------------------------------------------------------- #
# VGGT (not in the zoo — loaded via run_eval's helpers)
# --------------------------------------------------------------------------- #

def predict_vggt(image_path: str, checkpoint: str, device,
                 image_resolution: int = 512) -> np.ndarray:
    """Run VGGT-Omega on a single image → planar-z depth (H, W) in metres."""
    import torch
    from .run_eval import _load_vggt_base, make_vggt_predict
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    model = _load_vggt_base(checkpoint, device)
    predict = make_vggt_predict(model, device)

    imgs = load_and_preprocess_images([image_path], image_resolution=image_resolution)
    imgs = imgs.unsqueeze(0)                       # [S,3,H,W] → [B=1, S=1, 3, H, W]
    depth_np, _pose = predict(imgs)                # [B, S, H, W]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(depth_np[0, 0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #

def resolve_models(models_arg: str, want_vggt_ready: bool
                   ) -> Tuple[bool, List[zoo.ModelSpec]]:
    """Parse ``--models`` into (run_vggt, [zoo specs]). ``all`` = vggt (if a
    checkpoint is available) + every non-DAC zoo model that reports ``ready``."""
    tokens = [t.strip() for t in models_arg.replace(",", " ").split() if t.strip()]

    if tokens == ["all"]:
        specs = [s for s in zoo.get_specs(None)
                 if s.kind != "dac" and zoo.status(s)[0] == "ready"]
        return want_vggt_ready, specs

    run_vggt = any(t.lower() == _VGGT_KEY for t in tokens)
    zoo_keys = [t for t in tokens if t.lower() != _VGGT_KEY]
    specs = zoo.get_specs(zoo_keys) if zoo_keys else []
    return run_vggt, specs


def print_registry() -> None:
    print(f"\n  {'key':22s} {'family':20s} {'type':9s} {'state':11s} detail")
    print("  " + "-" * 92)
    print(f"  {_VGGT_KEY:22s} {'VGGT-Omega':20s} {'metric':9s} "
          f"{'needs-ckpt':11s} pass --vggt-checkpoint <run>/checkpoint_best.pt")
    for s in zoo.get_specs(None):
        state, detail = zoo.status(s)
        note = " (ERP-only: skipped by predict_image)" if s.kind == "dac" else ""
        print(f"  {s.key:22s} {s.family:20s} {s.output_type:9s} {state:11s} {detail}{note}")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    import torch

    p = argparse.ArgumentParser(
        description="Qualitative single-image depth (VGGT + monocular baselines), no GT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", help="path to a single input image (any format PIL reads)")
    p.add_argument("--models", default="vggt,dav2_large",
                   help="comma/space list of zoo keys and/or 'vggt'; or 'all'")
    p.add_argument("--vggt-checkpoint", default=None,
                   help="VGGT-Omega checkpoint (required if 'vggt' is selected)")
    p.add_argument("--out", default=None,
                   help="output dir (default: eval_out/predict_image/<image_stem>)")
    p.add_argument("--long-side", type=int, default=768,
                   help="resize input so its longer side = this (0 = native res)")
    p.add_argument("--image-resolution", type=int, default=512,
                   help="VGGT preprocessing resolution")
    p.add_argument("--cmap", default="magma_r", help="matplotlib colormap for depth")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dav3-id", default=None, help="override the experimental DAv3 HF id")
    p.add_argument("--list", action="store_true", help="print model registry + availability, then exit")
    a = p.parse_args()

    if a.list:
        print_registry()
        return
    if not a.image:
        p.error("--image is required (or use --list)")
    if not os.path.isfile(a.image):
        p.error(f"image not found: {a.image}")

    device = torch.device(a.device)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    out_dir = a.out or os.path.join("eval_out", "predict_image", stem)
    os.makedirs(out_dir, exist_ok=True)

    run_vggt, specs = resolve_models(a.models, want_vggt_ready=bool(a.vggt_checkpoint))
    if a.dav3_id:  # let the user point at a real DAv3 id
        specs = [dataclasses.replace(s, ref=a.dav3_id, experimental=False)
                 if s.family == "Depth-Anything-V3" else s for s in specs]

    want_vggt = any(t.strip().lower() == _VGGT_KEY
                    for t in a.models.replace(",", " ").split()) or a.models.strip() == "all"
    if want_vggt and not a.vggt_checkpoint:
        print("[predict] 'vggt' selected but no --vggt-checkpoint given — skipping VGGT.")
        run_vggt = False

    print(f"[predict] image={a.image}  device={device}  out={out_dir}")
    print(f"[predict] VGGT={'yes' if run_vggt else 'no'}  "
          f"zoo models={[s.key for s in specs] or '(none)'}")

    # Working RGB (all monocular models run on this; each resizes its output back).
    rgb01 = load_image01(a.image, long_side=a.long_side)
    Image.fromarray((rgb01 * 255).astype(np.uint8)).save(os.path.join(out_dir, "input.png"))

    # (label, depth, output_type) for each successful prediction.
    results: List[Tuple[str, np.ndarray, str]] = []

    # ── VGGT ────────────────────────────────────────────────────────────────
    if run_vggt:
        print("\n[predict] --- VGGT-Omega ---")
        try:
            t0 = time.time()
            depth = predict_vggt(a.image, a.vggt_checkpoint, device, a.image_resolution)
            print(f"[predict]   ok ({time.time()-t0:.1f}s, {depth.shape})")
            results.append(("vggt", depth, "metric"))
        except Exception as exc:  # noqa: BLE001
            print(f"[predict]   FAILED: {type(exc).__name__}: {exc}")

    # ── Zoo models ──────────────────────────────────────────────────────────
    for s in specs:
        if s.kind == "dac":
            print(f"\n[predict] --- {s.key} --- skip: DAC is ERP/fisheye-native "
                  "(needs a fisheye camera model, not a plain image)")
            continue
        state, detail = zoo.status(s)
        if state != "ready":
            print(f"\n[predict] --- {s.key} --- skip ({state}): {detail}")
            continue
        print(f"\n[predict] --- {s.key} ({s.family} {s.size}, {s.output_type}) ---")
        try:
            ad = zoo.build_adapter(s, use_camera=False)   # camera-free (UniK3D predicts rays)
            ad.load(device)
            t0 = time.time()
            depth = ad.predict_frame(rgb01, None, stem)   # cam/frame unused by these adapters
            print(f"[predict]   ok ({time.time()-t0:.1f}s, {depth.shape})")
            results.append((s.key, np.asarray(depth, np.float32), s.output_type))
        except Exception as exc:  # noqa: BLE001
            print(f"[predict]   FAILED: {type(exc).__name__}: {exc}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not results:
        raise SystemExit("[predict] no model produced a prediction — check --models / "
                         "weights (run with --list to see availability).")

    # ── Save per-model outputs ──────────────────────────────────────────────
    for label, depth, _kind in results:
        np.save(os.path.join(out_dir, f"{label}_depth.npy"), depth)
        rgb, _vmin, _vmax = colorize_depth(depth, a.cmap)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"{label}_depth.png"))

    # ── Comparison grid: input + each model ─────────────────────────────────
    _save_comparison(rgb01, results, os.path.join(out_dir, "comparison.png"), a.cmap)

    print(f"\n[predict] done — {len(results)} model(s). Results in {out_dir}/")
    for label, _d, _k in results:
        print(f"[predict]   {label}_depth.png / .npy")


def _save_comparison(rgb01: np.ndarray, results, out_path: str, cmap: str) -> None:
    """Grid: input RGB + each model's colourised depth (independently normalised)."""
    n = len(results) + 1
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False)
    fig.patch.set_facecolor("#1a1a2e")

    ax = axes[0][0]
    ax.imshow(rgb01)
    ax.set_title("Input", color="white", fontsize=11)
    ax.axis("off")

    for i, (label, depth, kind) in enumerate(results, start=1):
        ax = axes[i // ncols][i % ncols]
        rgb, vmin, vmax = colorize_depth(depth, cmap)
        ax.imshow(rgb)
        unit = "m" if kind == "metric" else "rel"
        # Fold the range into the title — ax.axis("off") would hide an xlabel.
        title = f"{label}  ({kind})"
        if np.isfinite(vmin):
            title += f"\n[{vmin:.2f}–{vmax:.2f} {unit}]"
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")

    for j in range(n, nrows * ncols):        # blank any trailing cells
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Single-image depth — qualitative (no GT, per-panel normalised)",
                 color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[predict] comparison → {out_path}")


if __name__ == "__main__":
    main()
