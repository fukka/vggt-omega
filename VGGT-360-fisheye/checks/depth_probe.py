# Copyright (c) 2026.
"""Single-image depth probe: does a model's depth follow the image it was given?

Replaces three near-clone scripts (``vggt_single_image.py``,
``vggt_omega_single_image.py``, ``official_vggt_single.py``), which each carried
their own copy of the model load, the ADT view construction and the
edge-alignment diagnostic.  ``center_view_sweep.py`` drives the same functions
for its FoV x mode x enhance grid.

Two seams, each with three or more adapters already in use:

  * **view source** — where the image comes from: an arbitrary file, a gnomonic
    ``tangent`` crop, the repo's wide ``rectifier`` pinhole, or a ``raw_roi``
    (undistorted-free) crop of the same angular coverage.
  * **backend** — which model turns it into depth: ``vggt1b`` (this port's
    vendored ``vggt_visfeat``), ``vggt_omega`` (local ``.pt``), or ``official``
    (the pip ``vggt`` package, as an independent control).

Plain functions, wired by the caller.  ``load_backend`` returns an opaque handle
so a sweep loads the model **once** and probes many views through it.

Everything crosses the backend seam as a **file path**: VGGT-Omega and official
VGGT preprocess from disk, while ``vggt_visfeat`` wants PIL, so materialising
the view is the common denominator — and it exports the exact bytes each model
saw, which is what makes a cross-model comparison trustworthy.

How to localise "the depth is a distorted version of the input"
---------------------------------------------------------------
Run a NORMAL perspective photo and a fisheye-derived view through the same
backend:

    python VGGT-360-fisheye/checks/depth_probe.py --backend vggt1b \
        --images room.png --adt-root <ROOT> --view tangent

  * both low  => the model/weights are the problem (read the weight-check line).
  * photo high, fisheye low => the view construction (narrow FoV / periphery
    resampling / distribution), not the model.

Then swap ``--backend`` on the SAME inputs: if VGGT-Omega aligns where VGGT-1B
does not, it is the better fisheye backbone; if both curve, the behaviour is a
VGGT-family trait (camera/FoV coupling) rather than one checkpoint's flaw.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)          # VGGT-360-fisheye/  (fisheye utils, datasets)
_REPO = os.path.dirname(_PKG)          # repo root          (vggt_omega/, finetune/)
for _p in (_PKG, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.fisheye_cam import (aria_intrinsics, fisheye_ray_lut,
                               kb4_forward_theta)
from utils.fisheye_views import fisheye_to_persp

# Edge-strength percentile defining "an edge" in the alignment diagnostic.
# ONE number, so align% is comparable across views and backends — the three
# scripts this module replaces used 96 / 96 / 90 and were silently incomparable.
EDGE_PERCENTILE = 96.0

DEFAULT_OUT_SIZE = 518          # VGGT-1B's native token grid (patch 14 -> 37x37)

# Render size that lands exactly on each backend's token grid, so nothing is
# resampled between our renderer and the model:
#   vggt1b / official : patch 14, 518 -> 37x37 tokens
#   vggt_omega        : patch 16, 512 -> 32x32 tokens
# Rendering at the native size is preferable to cropping a 518 render down to
# 512: a crop silently shrinks the view's field of view by 512/518 (~1.2%),
# which would corrupt the true-vs-inferred FoV comparison this probe exists to
# make.  Render size is a free parameter, so just render at the right one.
NATIVE_SIZE = {"vggt1b": 518, "official": 518, "vggt_omega": 512}

_RECTIFIER_FOCAL_FRAC = 0.55    # finetune/data/rectify.py: focal_out = 0.55*max(H,W)


# --------------------------------------------------------------------------- #
# View sources
# --------------------------------------------------------------------------- #

@dataclass
class View:
    """One image to probe, plus what we honestly know about it.

    ``true_fov`` is the field of view in degrees **only when this module
    rendered the view** and therefore knows it exactly.  For a view loaded from
    a file it is ``None``: the FoV cannot be recovered from pixels, and claiming
    otherwise is what made an earlier tester report a "true FoV" it could not
    verify.  ``valid`` is the analytic cone mask where one applies, else None.
    """

    crop: np.ndarray                    # (H, W, 3) uint8
    tag: str
    true_fov: Optional[float] = None
    valid: Optional[np.ndarray] = None  # (H, W) float32 in {0,1}


def view_from_file(path: str, *, assume_fov: Optional[float] = None) -> View:
    """An arbitrary image off disk (a normal photo, or an exported crop)."""
    rgb = np.array(Image.open(path).convert("RGB"))
    return View(crop=rgb, tag=os.path.splitext(os.path.basename(path))[0],
                true_fov=assume_fov)


def view_tangent(rgb: np.ndarray, cam, *, fov_deg: float = 60.0,
                 azimuth: float = 0.0, tilt: float = 0.0,
                 out_size: int = DEFAULT_OUT_SIZE, supersample: int = 3) -> View:
    """Gnomonic tangent crop — straight 3D lines stay straight.

    The periphery is radially stretched and the effective resampling softens as
    ``fov_deg`` grows, which is the confound ``view_raw_roi`` controls for.
    """
    crop, valid = fisheye_to_persp(rgb, cam, azimuth, tilt, fov_deg,
                                   height=out_size, width=out_size,
                                   supersample=supersample)
    return View(crop=np.clip(crop, 0, 255).astype(np.uint8),
                tag=f"tangent{int(fov_deg)}",
                true_fov=fov_deg,                 # we rendered it -> known exactly
                valid=valid.astype(np.float32))


def view_raw_roi(rgb: np.ndarray, cam, *, fov_deg: float = 60.0,
                 out_size: int = DEFAULT_OUT_SIZE) -> View:
    """Centred square ROI of the RAW fisheye covering incidence <= fov/2.

    Same angular coverage as ``view_tangent`` at the same FoV, but only cropped
    and resized — no undistortion, so straight lines stay barrel-curved.  The
    clean "is it the undistortion, or is it the model on distorted input?"
    control.  ROI half-size is the KB4 image radius ``f * theta_d``.
    """
    _, cone = fisheye_ray_lut(cam)
    td = float(kb4_forward_theta(np.array(math.radians(fov_deg / 2.0)), cam.k))
    rx, ry = cam.fx * td, cam.fy * td
    size = (int(round(2 * rx)), int(round(2 * ry)))
    centre = (float(cam.cx), float(cam.cy))
    patch = cv2.getRectSubPix(rgb, size, centre)
    vpatch = cv2.getRectSubPix(cone.astype(np.float32), size, centre)
    crop = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(vpatch, (out_size, out_size),
                       interpolation=cv2.INTER_NEAREST)
    return View(crop=np.clip(crop, 0, 255).astype(np.uint8),
                tag=f"rawroi{int(fov_deg)}",
                true_fov=fov_deg,
                valid=(valid > 0.5).astype(np.float32))


def view_rectifier(rgb: np.ndarray, *, out_size: int = DEFAULT_OUT_SIZE,
                   preset: str = "aria-214-1") -> View:
    """The repo's validated wide (~85 deg) fisheye->pinhole of the WHOLE frame.

    A fixed reference that ignores any FoV argument: it tests whether the
    narrow tangent crop specifically is the problem.
    """
    from finetune.data.rectify import FisheyeRectifier
    crop = FisheyeRectifier(preset)(rgb.astype(np.float32) / 255.0) * 255.0
    crop = cv2.resize(np.clip(crop, 0, 255).astype(np.uint8),
                      (out_size, out_size))
    true_fov = float(np.degrees(2 * np.arctan(0.5 / _RECTIFIER_FOCAL_FRAC)))
    return View(crop=crop, tag="rectifier", true_fov=true_fov)


def enhance_view(view: View, mode: str) -> View:
    """Optional 'clearness' axis — does sharpening/blurring move alignment?"""
    if mode == "none":
        return view
    crop = view.crop
    if mode == "clahe":
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif mode == "sharpen":
        blur = cv2.GaussianBlur(crop, (0, 0), 1.5)
        crop = np.clip(1.5 * crop.astype(np.float32) - 0.5 * blur,
                       0, 255).astype(np.uint8)
    elif mode == "blur":
        crop = cv2.GaussianBlur(crop, (0, 0), 2.0)
    else:
        raise ValueError(f"unknown enhance mode: {mode!r}")
    return View(crop=crop, tag=f"{view.tag}_{mode}",
                true_fov=view.true_fov, valid=view.valid)


def materialize(view: View, out_dir: str) -> str:
    """Write the view's pixels to disk and return the path.

    Required, not incidental: VGGT-Omega and official VGGT preprocess from
    disk.  It also exports the EXACT input each backend consumed, so a
    cross-model comparison can be reproduced or re-fed elsewhere.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{view.tag}_input.png")
    Image.fromarray(view.crop).save(path)
    return path


def adt_frame(adt_root: str, *, rgb_subdir: str = "videos_synthetic",
              frame: int = 6) -> "tuple[np.ndarray, object]":
    """Fetch one raw ADT fisheye frame + its intrinsics (the shared setup)."""
    from datasets.adt import ADTFisheyeFrames, find_adt_sequences
    seqs = find_adt_sequences(adt_root, rgb_subdir=rgb_subdir)
    if not seqs:
        raise SystemExit(f"no ADT sequences with {rgb_subdir}/ under {adt_root}")
    ds = ADTFisheyeFrames(seqs[:1], rgb_subdir=rgb_subdir, max_frames=frame + 1)
    rgb = ds[min(frame, len(ds) - 1)]["rgb"]
    return rgb, aria_intrinsics(*rgb.shape[:2], rotated=True)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

@dataclass
class Backend:
    """An opaque, already-loaded model handle.

    Held by the caller so a sweep loads once and probes many views.  ``predict``
    maps a list of image PATHS to one ``(depth_z, fov_deg)`` per image.
    """

    name: str
    predict: Callable[[Sequence[str]], List["Prediction"]]


@dataclass
class Prediction:
    depth_z: np.ndarray                        # (H, W) planar z along the view axis
    fov_deg: Optional[Tuple[float, float]]     # model's INFERRED (fov_h, fov_w)


def _torch_device_dtype():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16
             if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
             else torch.float32)
    return torch, device, dtype


def _pose_fov(pose_row) -> Optional[Tuple[float, float]]:
    """pose_enc[7:9] = (fov_h, fov_w) in radians -> degrees.

    VGGT couples depth to this self-estimated camera; a large gap against a
    view's known ``true_fov`` is the suspected mechanism behind depth that bends
    away from the input.
    """
    if pose_row is None:
        return None
    return (float(np.degrees(pose_row[7])), float(np.degrees(pose_row[8])))


def _report_key_mismatch(missing, unexpected, *, ignore_prefix=None) -> None:
    """Loud weight check — a silently random-init layer alone bends depth."""
    if ignore_prefix:
        unexpected = [k for k in unexpected if not k.startswith(ignore_prefix)]
    print("=" * 68)
    if missing or unexpected:
        print(f"!!! WEIGHT CHECK FAILED — {len(missing)} model keys NOT in "
              f"checkpoint (LEFT AT RANDOM INIT), {len(unexpected)} checkpoint "
              f"keys unused.")
        if missing:
            print(f"    first missing: {sorted(missing)[:8]}")
        print("    => this alone can cause the distorted/curved depth.")
    else:
        print("weight check OK: all model keys matched (no random-init layers).")
    print("=" * 68)


def _load_vggt1b(model_path: str) -> Backend:
    """This port's vendored VGGT (``vggt_visfeat``) — the pipeline's own model."""
    torch, device, dtype = _torch_device_dtype()
    from vggt_visfeat.models.vggt import VGGT
    from vggt_visfeat.utils.load_fn2 import load_and_preprocess_images

    print(f"loading {model_path} on {device} ({dtype}) ...")
    model = VGGT.from_pretrained(model_path).to(device).eval()
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        ckpt = load_file(hf_hub_download(model_path, "model.safetensors"))
        mk = set(model.state_dict().keys())
        # ``track_head.*`` is deliberately absent from the model (tracker
        # removed) — not an error.
        _report_key_mismatch(mk - set(ckpt.keys()), set(ckpt.keys()) - mk,
                             ignore_prefix="track_head.")
    except Exception as e:
        print(f"weight check SKIPPED ({type(e).__name__}: {e}) — treat curvy "
              f"results with suspicion.")

    def predict(paths: Sequence[str]) -> List[Prediction]:
        # load_fn2 takes PIL objects, not paths.
        pil = [Image.open(p).convert("RGB") for p in paths]
        images = load_and_preprocess_images(pil).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type,
                                             dtype=dtype,
                                             enabled=device.type == "cuda"):
            pred, _ = model(images=images, save_attn=False)   # vanilla path
        depth = pred["depth"][0, ..., 0].float().cpu().numpy()   # [S,H,W]
        pose = pred.get("pose_enc")
        pose = None if pose is None else pose[0].float().cpu().numpy()
        return [Prediction(depth[i], _pose_fov(None if pose is None else pose[i]))
                for i in range(depth.shape[0])]

    return Backend(name="vggt1b", predict=predict)


def _load_vggt_omega(checkpoint: str, image_resolution: int = 512) -> Backend:
    """VGGT-Omega from a local ``.pt`` (native preprocessing, from disk).

    ``image_resolution`` is VGGT-Omega's own preprocessing size (must be
    divisible by its patch size of 16).  It is a genuine experimental variable:
    the input resolution changes how much of a wide view survives to the token
    grid, so it belongs alongside FoV when asking where a model starts to fail.
    """
    torch, device, dtype = _torch_device_dtype()
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    if not os.path.isfile(checkpoint):
        raise SystemExit(
            f"VGGT-Omega checkpoint not found: {checkpoint}\n"
            f"  pass --checkpoint <path to model.pt> (request access at "
            f"https://huggingface.co/facebook/VGGT-Omega)")
    print(f"loading VGGT-Omega from {checkpoint} on {device} ({dtype}) ...")
    model = VGGTOmega()
    sd = torch.load(checkpoint, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    _report_key_mismatch(missing, unexpected)
    model = model.to(device).eval()

    def predict(paths: Sequence[str]) -> List[Prediction]:
        # Each model gets its NATIVE preprocessing, else the comparison is unfair.
        imgs = load_and_preprocess_images(
            list(paths), image_resolution=image_resolution).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type,
                                             dtype=dtype,
                                             enabled=device.type == "cuda"):
            preds = model(imgs)
        depth = preds["depth"]
        if depth.ndim == 5:
            depth = depth.squeeze(-1)                    # [B,S,H,W,1]->[B,S,H,W]
        depth = depth[0].float().cpu().numpy()
        pose = preds.get("pose_enc")
        pose = None if pose is None else pose[0].float().cpu().numpy()
        return [Prediction(depth[i], _pose_fov(None if pose is None else pose[i]))
                for i in range(depth.shape[0])]

    return Backend(name="vggt_omega", predict=predict)


def _load_official(model_path: str) -> Backend:
    """The pip ``vggt`` package — an independent control on this port.

    If this and ``vggt1b`` disagree on the same input, the port's vendored copy
    has diverged from upstream; if they agree, the behaviour is upstream's.
    """
    torch, device, dtype = _torch_device_dtype()
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError as e:
        raise SystemExit(
            f"backend 'official' needs the pip package: pip install vggt  ({e})")
    print(f"loading official {model_path} on {device} ({dtype}) ...")
    model = VGGT.from_pretrained(model_path).to(device).eval()

    def predict(paths: Sequence[str]) -> List[Prediction]:
        images = load_and_preprocess_images(list(paths)).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type,
                                             dtype=dtype,
                                             enabled=device.type == "cuda"):
            pred = model(images)
        if isinstance(pred, tuple):
            pred = pred[0]
        depth = pred["depth"][0, ..., 0].float().cpu().numpy()
        pose = pred.get("pose_enc")
        pose = None if pose is None else pose[0].float().cpu().numpy()
        return [Prediction(depth[i], _pose_fov(None if pose is None else pose[i]))
                for i in range(depth.shape[0])]

    return Backend(name="official", predict=predict)


BACKENDS = ("vggt1b", "vggt_omega", "official")


def load_backend(name: str, *, model_path: str = "facebook/VGGT-1B",
                 checkpoint: Optional[str] = None,
                 image_resolution: int = 512) -> Backend:
    """Load one backend and return a reusable handle (load once, probe many).

    ``image_resolution`` applies to ``vggt_omega`` only — the other two
    backends' loaders fix their own input size (518 for VGGT-1B's token grid).
    """
    if name == "vggt1b":
        return _load_vggt1b(model_path)
    if name == "vggt_omega":
        return _load_vggt_omega(
            checkpoint or os.path.join(_REPO, "checkpoints",
                                       "VGGT-Omega-1B-512", "model.pt"),
            image_resolution=image_resolution)
    if name == "official":
        return _load_official(model_path)
    raise ValueError(f"unknown backend: {name!r} (choose from {BACKENDS})")


def predict_depth(backend: Backend, paths: Sequence[str], *,
                  multiview: bool = False) -> List[Prediction]:
    """Depth for each path.

    Default runs each image as an INDEPENDENT 1-view scene.  That matters: with
    ``multiview=True`` all paths go through one forward pass, so VGGT's
    cross-view attention can fuse them and a photo's depth then depends on what
    else was in the batch — meaningful only for views that genuinely share an
    optical centre, never for unrelated images.
    """
    if multiview:
        return backend.predict(list(paths))
    out: List[Prediction] = []
    for p in paths:
        out.extend(backend.predict([p]))
    return out


# --------------------------------------------------------------------------- #
# Diagnostic
# --------------------------------------------------------------------------- #

def edge_alignment(rgb: np.ndarray, depth_z: np.ndarray, *,
                   valid: Optional[np.ndarray] = None,
                   percentile: float = EDGE_PERCENTILE
                   ) -> "tuple[np.ndarray, float]":
    """``RGB | depth | overlay`` strip + the edge-alignment fraction.

    ``align`` = fraction of RGB edge pixels that have a depth edge within ~2 px.
    High => the depth's structure follows the image's.

    Edges are taken on **planar z**, so the radial "bowl" a flat surface has in
    range space is absent and a curved cyan contour means a genuine depth
    discontinuity.  Read the overlay as: red with no cyan => smooth iso-depth
    band on a flat surface (geometry, not a bug); red and cyan both curving
    together => the input itself is curved and depth follows it faithfully; red
    straight but cyan curved => the model is bending structure relative to its
    input, which is the real problem to chase.

    ``valid`` (the analytic cone mask) is eroded and applied before thresholding.
    Without it the zero-filled cone-clipped corners present a hard border that
    scores as a strong edge, and their zeros drag the percentile down — which is
    why the unmasked variants of this diagnostic were not comparable to the
    masked one.  At FoV 120 that region is ~19% of the crop.

    Scale to read the number against, measured on a real ADT tangent view at
    FoV 60 with synthetic "depth maps" of known quality:

        depth = blurred luminance (follows the image)   68%
        same, shifted 8 px (mislocalised)               38%
        radial bowl (wrong structure)                   19%
        smooth ramp (no structure at all)                0%
        white noise                                     52%   <- see below

    So the metric separates the failure modes that matter.  Its one blind spot
    is high-frequency noise: dense spurious edges satisfy the 2-px tolerance
    almost everywhere, so noise scores mid-range rather than low.  Real depth
    heads do not output white noise, but do not read a mid-range align% as
    proof of life — look at the overlay strip too.
    """
    H, W = depth_z.shape
    rgb = cv2.resize(rgb, (W, H))
    if valid is not None and valid.shape != (H, W):
        valid = cv2.resize(valid, (W, H), interpolation=cv2.INTER_NEAREST)
    vmask = None if valid is None else cv2.erode(
        (valid > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    d = depth_z.astype(np.float32)
    finite = d[np.isfinite(d)] if vmask is None else d[np.isfinite(d) & vmask]
    lo, hi = ((np.percentile(finite, 2), np.percentile(finite, 98))
              if finite.size else (0.0, 1.0))
    d8 = np.clip((d - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    dcol = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_VIRIDIS),
                        cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rgb_e = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3),
                     cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))
    dep_e = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0, 3),
                     cv2.Sobel(d, cv2.CV_32F, 0, 1, 3))
    if vmask is not None:
        rgb_e[~vmask] = 0.0
        dep_e[~vmask] = 0.0

    # Percentile over CONSIDERED pixels only, so the threshold means the same
    # thing whether or not a cone mask is present.
    def _thresh(e):
        pool = e[vmask] if vmask is not None else e.ravel()
        pool = pool[np.isfinite(pool)]
        return e > (np.percentile(pool, percentile) if pool.size else np.inf)

    rgb_m, dep_m = _thresh(rgb_e), _thresh(dep_e)

    ov = (dcol.astype(np.float32) * 0.45).astype(np.uint8)
    ov[rgb_m] = (255, 40, 40)      # RGB edges red
    ov[dep_m] = (0, 230, 230)      # depth (planar z) edges cyan
    if vmask is not None:
        dcol[~vmask] = 24
        ov[~vmask] = 24

    dep_dil = cv2.dilate(dep_m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    align = float((rgb_m & dep_dil).sum()) / max(int(rgb_m.sum()), 1)
    return np.concatenate([rgb, dcol, ov], axis=1), align


@dataclass
class ProbeResult:
    tag: str
    align: float
    true_fov: Optional[float]
    fov_inferred: Optional[Tuple[float, float]]
    input_path: str
    strip_path: str
    depth_z: np.ndarray


def probe(backend: Backend, views: Sequence[View], out_dir: str, *,
          percentile: float = EDGE_PERCENTILE,
          multiview: bool = False) -> List[ProbeResult]:
    """Materialize -> predict -> diagnose.  The wiring both CLIs share."""
    paths = [materialize(v, out_dir) for v in views]
    preds = predict_depth(backend, paths, multiview=multiview)
    results = []
    for view, path, pred in zip(views, paths, preds):
        strip, align = edge_alignment(view.crop, pred.depth_z, valid=view.valid,
                                      percentile=percentile)
        strip_path = os.path.join(out_dir, f"{view.tag}_{backend.name}.png")
        Image.fromarray(strip).save(strip_path)
        results.append(ProbeResult(
            tag=view.tag, align=align, true_fov=view.true_fov,
            fov_inferred=pred.fov_deg, input_path=path,
            strip_path=strip_path, depth_z=pred.depth_z))
    return results


def print_results(results: Sequence[ProbeResult], backend_name: str) -> None:
    print(f"\nbackend: {backend_name}")
    print("  inferred FoV vs the view's known render FoV ('true=unknown' means "
          "this module did not render it and cannot know):")
    for r in results:
        ref = f"true={r.true_fov:.0f}" if r.true_fov is not None else "true=unknown"
        got = ("n/a" if r.fov_inferred is None
               else f"fov_h={r.fov_inferred[0]:5.1f}  fov_w={r.fov_inferred[1]:5.1f}")
        print(f"    {r.tag:<28} {got}   ({ref})")
    print("\n  edge-alignment (RGB edges with a depth edge within ~2 px; "
          "high = depth structure follows the image):")
    for r in results:
        print(f"    {r.tag:<28} align = {r.align * 100:5.1f}%   -> {r.strip_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_views(args) -> List[View]:
    """Assemble the requested views: files first, then an ADT-derived view."""
    views = [view_from_file(p, assume_fov=args.assume_fov) for p in args.images]
    if args.adt_root:
        rgb, cam = adt_frame(args.adt_root, rgb_subdir=args.rgb_subdir,
                             frame=args.frame)
        if args.view == "tangent":
            v = view_tangent(rgb, cam, fov_deg=args.fov, out_size=args.out_size,
                             supersample=args.crop_supersample)
        elif args.view == "raw_roi":
            v = view_raw_roi(rgb, cam, fov_deg=args.fov, out_size=args.out_size)
        else:
            v = view_rectifier(rgb, out_size=args.out_size)
        v.tag = f"adt_{v.tag}_f{args.frame}"
        views.append(enhance_view(v, args.enhance))
    return views


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Single-image depth probe (model x view-source)")
    ap.add_argument("--backend", choices=BACKENDS, default="vggt1b")
    ap.add_argument("--model-path", default="facebook/VGGT-1B",
                    help="HF id for the vggt1b / official backends")
    ap.add_argument("--checkpoint", default=None,
                    help="local .pt for the vggt_omega backend")
    ap.add_argument("--image-resolution", type=int, default=512,
                    help="vggt_omega preprocessing size (divisible by 16); "
                         "ignored by the other backends, whose loaders fix "
                         "their own input size")
    ap.add_argument("--images", nargs="*", default=[],
                    help="arbitrary image paths (normal perspective photos)")
    ap.add_argument("--adt-root", default=None,
                    help="also build a view from a real ADT fisheye frame")
    ap.add_argument("--rgb-subdir", default="videos_synthetic",
                    help="videos_synthetic = sharp, GT-aligned (recommended); "
                         "videos_rgb = real sensor, MOTION-BLURRED (a confound)")
    ap.add_argument("--frame", type=int, default=6)
    ap.add_argument("--view", choices=["tangent", "raw_roi", "rectifier"],
                    default="tangent", help="how to build the ADT view")
    ap.add_argument("--fov", type=float, default=60.0,
                    help="render FoV for tangent / raw_roi (known exactly)")
    ap.add_argument("--assume-fov", type=float, default=None,
                    help="reference FoV to print for --images inputs, whose "
                         "real FoV cannot be recovered from pixels")
    ap.add_argument("--enhance", choices=["none", "clahe", "sharpen", "blur"],
                    default="none", help="'clearness' axis on the ADT view")
    ap.add_argument("--out-size", type=int, default=None,
                    help="render size; default = the backend's native token "
                         f"grid ({NATIVE_SIZE}), so the model resamples nothing")
    ap.add_argument("--crop-supersample", type=int, default=3,
                    help="anti-alias factor for the tangent render (1 = aliased)")
    ap.add_argument("--edge-percentile", type=float, default=EDGE_PERCENTILE,
                    help="edge-strength percentile; changing it changes align% "
                         "and breaks comparability with other runs")
    ap.add_argument("--multiview", action="store_true",
                    help="batch ALL inputs into one forward pass so cross-view "
                         "attention can fuse them. Only meaningful for views "
                         "sharing an optical centre — never for unrelated images.")
    ap.add_argument("--out", default=os.path.join(_PKG, "outputs", "depth_probe"))
    args = ap.parse_args()

    if args.out_size is None:
        args.out_size = NATIVE_SIZE[args.backend]
        print(f"render size {args.out_size} (native token grid for "
              f"{args.backend}; override with --out-size)")

    views = build_views(args)
    if not views:
        raise SystemExit("give --images and/or --adt-root")

    backend = load_backend(args.backend, model_path=args.model_path,
                           checkpoint=args.checkpoint,
                           image_resolution=args.image_resolution)
    out_dir = os.path.join(args.out, args.backend)
    results = probe(backend, views, out_dir,
                    percentile=args.edge_percentile, multiview=args.multiview)
    print_results(results, args.backend)
    print("\nRead: a normal photo should score high. If it scores LOW too, the "
          "model/weights are the problem (see the weight-check line), not the "
          "view construction. Re-run with a different --backend on the same "
          "inputs to separate a checkpoint flaw from a VGGT-family trait.")


if __name__ == "__main__":
    main()
